
use std::{borrow::Cow, cmp::Reverse, collections::{HashMap, HashSet}, sync::OnceLock};

use fancy_regex::Regex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const SPECIAL_TOKENS: &[&str] = &[
    "<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>",
];
const FALLBACK_ASSISTANT_END: usize = 50256;
const FALLBACK_PYTHON_START: usize = 50257;
const FALLBACK_PYTHON_END: usize = 50258;
const FALLBACK_OUTPUT_START: usize = 50259;
const FALLBACK_OUTPUT_END: usize = 50260;

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokenIds {
    pub bos: usize,
    pub assistant_end: usize,
    pub python_start: usize,
    pub python_end: usize,
    pub output_start: usize,
    pub output_end: usize,
}

#[derive(Debug, Clone, Copy)]
struct RenderTokenIds {
    bos: usize,
    user_start: usize,
    user_end: usize,
    assistant_start: usize,
    assistant_end: usize,
    python_start: usize,
    python_end: usize,
    output_start: usize,
    output_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePart {
    #[serde(rename = "type")]
    pub part_type: String, // "text", "python", "python_output"
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Simple(String),
    Parts(Vec<MessagePart>),
}

impl MessageContent {
    pub fn to_string_content(&self) -> String {
        match self {
            Self::Simple(s) => s.clone(),
            Self::Parts(parts) => parts.iter().map(|p| p.text.as_str()).collect(),
        }
    }

    fn expect_simple<'a>(&'a self, role: &str) -> &'a str {
        match self {
            Self::Simple(s) => s,
            Self::Parts(_) => panic!("{role} message cannot have multipart content"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// BPE merge ranks: maps a byte sequence to its token ID
    pub mergeable_ranks: HashMap<Vec<u8>, usize>,
    /// Mappings for special tokens
    pub special_tokens: HashMap<String, usize>,
    /// Inverse mappings for decoding base tokens
    #[serde(skip)]
    pub inverse_vocab: HashMap<usize, Vec<u8>>,
    /// Inverse mappings for decoding special tokens
    #[serde(skip)]
    pub inverse_special_tokens: HashMap<usize, String>,
}

#[derive(Serialize, Deserialize)]
struct TokenizerFile {
    version: u32,
    mergeable_ranks: Vec<(Vec<u8>, usize)>,
    special_tokens: HashMap<String, usize>,
}

impl BpeTokenizer {
    /// Get the compiled BPE split regex
    fn get_split_regex() -> &'static Regex {
        static REGEX: OnceLock<Regex> = OnceLock::new();
        REGEX.get_or_init(|| {
            Regex::new(concat!(
                r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|",
                r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
            ))
                .expect("Failed to compile BPE split pattern regex")
        })
    }

    /// Rebuild inverse mappings for decoding. Should be called after deserialization.
    pub fn build_inverse_mappings(&mut self) {
        self.inverse_vocab =
            self.mergeable_ranks.iter().map(|(k, &v)| (v, k.clone())).collect();
        // Also map single bytes to 0..256 if not present
        for i in 0..256 {
            self.inverse_vocab.entry(i).or_insert_with(|| vec![i as u8]);
        }
        self.inverse_special_tokens =
            self.special_tokens.iter().map(|(k, &v)| (v, k.clone())).collect();
    }

    /// Train a BPE tokenizer from an iterator of documents/texts.
    pub fn train_from_iterator<I, S>(text_iterator: I, vocab_size: usize) -> Self
        where I: IntoIterator<Item = S>, S: AsRef<str>, {
        assert!(vocab_size >= 256 + SPECIAL_TOKENS.len(),
            "vocabulary must contain all byte and special tokens");
        // 1. Split into unique words with counts
        let mut word_counts = HashMap::<Vec<u8>, usize>::new();
        let regex = Self::get_split_regex();

        for text in text_iterator {
            let text_ref = text.as_ref();
            for m in regex.find_iter(text_ref) {
                let m = m.expect("BPE split regex failed while training");
                let bytes = m.as_str().as_bytes().to_vec();
                *word_counts.entry(bytes).or_default() += 1;
            }
        }

        // 2. Initialize vocabulary with base bytes 0..256
        let mut mergeable_ranks = HashMap::<Vec<u8>, usize>::new();
        let mut temp_inverse = HashMap::<usize, Vec<u8>>::new();
        for b in 0..256 {
            let byte_seq = vec![b as u8];
            mergeable_ranks.insert(byte_seq.clone(), b);
            temp_inverse.insert(b, byte_seq);
        }

        // Represent words as a list of their token IDs
        // Initially, each byte is its own token
        let mut words: Vec<(usize, Vec<usize>)> = word_counts.into_iter()
            .map(|(word, count)| (count, word.into_iter().map(usize::from).collect())).collect();

        let vocab_size_no_special = vocab_size.saturating_sub(SPECIAL_TOKENS.len());
        let mut current_vocab_size = 256;

        while current_vocab_size < vocab_size_no_special {
            // Count frequencies of adjacent pairs
            let mut pair_counts = HashMap::<(usize, usize), usize>::new();
            for (count, tokens) in &words {
                for i in 0..tokens.len().saturating_sub(1) {
                    pair_counts.entry((tokens[i], tokens[i + 1]))
                        .and_modify(|frequency| *frequency += *count).or_insert(*count);
                }
            }

            let best_pair = pair_counts.iter()
                .min_by_key(|&(&(left, right), &count)| (Reverse(count), left, right))
                .map(|(&pair, _)| pair);

            if let Some(pair) = best_pair {
                let new_token_id = current_vocab_size;

                // Get the merged byte sequence for this pair in O(1)
                let mut merged_bytes = temp_inverse.get(&pair.0).unwrap().clone();
                merged_bytes.extend(temp_inverse.get(&pair.1).unwrap());

                temp_inverse.insert(new_token_id, merged_bytes.clone());
                mergeable_ranks.insert(merged_bytes, new_token_id);

                // Merge this pair in all word token lists
                for (_, tokens) in &mut words {
                    let mut i = 0;
                    let mut merged_tokens = Vec::with_capacity(tokens.len());
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                            merged_tokens.push(new_token_id);
                            i += 2;
                        } else {
                            merged_tokens.push(tokens[i]);
                            i += 1;
                        }
                    }
                    *tokens = merged_tokens;
                }

                current_vocab_size += 1;
            } else { break; } // No more pairs to merge
        }

        // Define special tokens starting from current_vocab_size
        let mut special_tokens = HashMap::new();
        for (i, &name) in SPECIAL_TOKENS.iter().enumerate() {
            special_tokens.insert(name.to_string(), current_vocab_size + i);
        }

        let mut tokenizer = BpeTokenizer {
            mergeable_ranks, special_tokens,
            inverse_vocab: HashMap::new(),
            inverse_special_tokens: HashMap::new(),
        };
        tokenizer.build_inverse_mappings();
        tokenizer
    }

    /// Encode a slice of bytes using the trained BPE merges
    fn encode_piece(&self, piece: &[u8]) -> Vec<usize> {
        if piece.is_empty() { return Vec::new(); }
        if piece.len() == 1 { return vec![piece[0] as usize]; }

        // Represent parts as (start_index, length)
        let mut parts: Vec<(usize, usize)> = (0..piece.len()).map(|i| (i, 1)).collect();

        loop {
            let best_pair_idx = (0..parts.len() - 1).filter_map(|i| {
                    let (start, len) = (parts[i].0, parts[i].1 + parts[i + 1].1);
                    self.mergeable_ranks.get(&piece[start..start + len]).map(|&rank| (i, rank))
                }).min_by_key(|&(_, rank)| rank).map(|(i, _)| i);

            if let Some(idx) = best_pair_idx {
                parts[idx].1 += parts[idx + 1].1;
                parts.remove(idx + 1);
            } else { break; }
        }

        parts.into_iter().map(|(start, len)| {
                let seq = &piece[start..start + len];
                if seq.len() == 1 { seq[0] as usize } else {
                    *self.mergeable_ranks.get(seq).expect("merged BPE token is missing its rank")
                }
            }).collect()
    }

    /// Encode a string into a list of token IDs, treating special tokens as ordinary text
    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        for m in Self::get_split_regex().find_iter(text) {
            let m = m.expect("BPE split regex failed while encoding");
            let piece = m.as_str().as_bytes();
            ids.extend(self.encode_piece(piece));
        }
        ids
    }

    /// Parallel batch BPE encoding of a list of strings
    pub fn encode_ordinary_batch(&self, texts: &[String]) -> Vec<Vec<usize>> {
        texts.par_iter().map(|text| self.encode_ordinary(text)).collect()
    }

    /// Encode a single special token via exact match
    pub fn encode_special(&self, text: &str) -> Option<usize> {
        self.special_tokens.get(text).copied()
    }

    fn required_special(&self, text: &str) -> usize {
        self.encode_special(text).unwrap_or_else(|| panic!("Missing {text} token"))
    }

    /// Get the Beginning of Sequence (BOS) token ID
    pub fn get_bos_token_id(&self) -> usize {
        *self
            .special_tokens
            .get("<|bos|>")
            .or_else(|| self.special_tokens.get("<|endoftext|>"))
            .expect("Failed to find BOS token in tokenizer")
    }

    pub fn special_token_ids(&self) -> SpecialTokenIds {
        let get = |token: &str, fallback|
            self.special_tokens.get(token).copied().unwrap_or(fallback);
        SpecialTokenIds {
            bos: self.get_bos_token_id(),
            assistant_end: get("<|assistant_end|>", FALLBACK_ASSISTANT_END),
            python_start: get("<|python_start|>", FALLBACK_PYTHON_START),
            python_end: get("<|python_end|>", FALLBACK_PYTHON_END),
            output_start: get("<|output_start|>", FALLBACK_OUTPUT_START),
            output_end: get("<|output_end|>", FALLBACK_OUTPUT_END),
        }
    }

    /// Return the vocabulary size
    pub fn get_vocab_size(&self) -> usize {
        self.mergeable_ranks.values().chain(self.special_tokens.values())
            .copied().max().map_or(0, |max_id| max_id + 1)
    }

    /// Return the special tokens mapping
    pub fn get_special_tokens(&self) -> &HashMap<String, usize> { &self.special_tokens }

    /// Decode a sequence of token IDs back into a UTF-8 string
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(special_str) = self.inverse_special_tokens.get(&id) {
                bytes.extend_from_slice(special_str.as_bytes());
            } else if let Some(token_bytes) = self.inverse_vocab.get(&id) {
                bytes.extend_from_slice(token_bytes);
            } else if id < 256 {
                bytes.push(id as u8);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn render_token_ids(&self) -> RenderTokenIds {
        RenderTokenIds {
            bos: self.get_bos_token_id(),
            user_start: self.required_special("<|user_start|>"),
            user_end: self.required_special("<|user_end|>"),
            assistant_start: self.required_special("<|assistant_start|>"),
            assistant_end: self.required_special("<|assistant_end|>"),
            python_start: self.required_special("<|python_start|>"),
            python_end: self.required_special("<|python_end|>"),
            output_start: self.required_special("<|output_start|>"),
            output_end: self.required_special("<|output_end|>"),
        }
    }

    fn normalize_messages(messages: &[ConversationMessage]) -> Cow<'_, [ConversationMessage]> {
        if !matches!(messages.first(), Some(message) if message.role == "system") {
            return Cow::Borrowed(messages);
        }

        assert!(messages.len() >= 2, "System message must be followed by a user message");
        assert_eq!(messages[1].role, "user", "System message must be followed by a user message");

        let system_content = messages[0].content.expect_simple("System");
        let user_content = messages[1].content.expect_simple("User");
        let mut normalized = messages.to_vec();
        normalized[1].content =
            MessageContent::Simple(format!("{}\n\n{}", system_content, user_content));
        normalized.remove(0);
        Cow::Owned(normalized)
    }

    fn render_messages(&self, messages: &[ConversationMessage], max_tokens: usize)
        -> (Vec<usize>, Vec<i32>) {
        let (mut ids, mut mask) = (Vec::new(), Vec::new());
        let tokens = self.render_token_ids();

        let mut add_tokens = |token_ids: &[usize], mask_val: i32| {
            mask.extend(std::iter::repeat_n(mask_val, token_ids.len()));
            ids.extend_from_slice(token_ids);
        };

        let messages = Self::normalize_messages(messages);
        assert!(!messages.is_empty(), "Conversation must have at least one message");
        add_tokens(&[tokens.bos], 0);

        for (i, message) in messages.iter().enumerate() {
            let expected_role = if i % 2 == 0 { "user" } else { "assistant" };
            assert_eq!(message.role, expected_role,
                "Message {} is from {} but should be from {}", i, message.role, expected_role);

            match &message.content {
                MessageContent::Simple(text) => {
                    let value_ids = self.encode_ordinary(text);
                    if message.role == "user" {
                        add_tokens(&[tokens.user_start], 0);
                        add_tokens(&value_ids, 0);
                        add_tokens(&[tokens.user_end], 0);
                    } else {
                        add_tokens(&[tokens.assistant_start], 0);
                        add_tokens(&value_ids, 1);
                        add_tokens(&[tokens.assistant_end], 1);
                    }
                }
                MessageContent::Parts(parts) => {
                    assert_eq!(
                        message.role, "assistant",
                        "Only assistant messages can have multipart content"
                    );
                    add_tokens(&[tokens.assistant_start], 0);

                    for part in parts {
                        let value_ids = self.encode_ordinary(&part.text);
                        match part.part_type.as_str() {
                            "text" => add_tokens(&value_ids, 1),
                            "python" => {
                                add_tokens(&[tokens.python_start], 1);
                                add_tokens(&value_ids, 1);
                                add_tokens(&[tokens.python_end], 1);
                            }
                            "python_output" => {
                                add_tokens(&[tokens.output_start], 0);
                                add_tokens(&value_ids, 0);
                                add_tokens(&[tokens.output_end], 0);
                            }
                            _ => panic!("Unknown part type: {}", part.part_type),
                        }
                    }

                    add_tokens(&[tokens.assistant_end], 1);
                }
            }
        }

        // 4. Truncate to max_tokens
        let final_len = std::cmp::min(ids.len(), max_tokens);
        ids.truncate(final_len);
        mask.truncate(final_len);

        (ids, mask)
    }

    /// Render a Chat conversation into sequence token IDs and attention target masks
    pub fn render_conversation(&self, conversation: &Conversation, max_tokens: usize)
        -> (Vec<usize>, Vec<i32>) {
        self.render_messages(&conversation.messages, max_tokens)
    }

    /// Render a Chat conversation priming the Assistant for completion (useful in RL)
    pub fn render_for_completion(&self, conversation: &Conversation) -> Vec<usize> {
        let (last, prompt) = conversation.messages.split_last()
            .expect("Conversation cannot be empty");
        assert_eq!(last.role, "assistant", "Last message must be from the Assistant");

        let (mut ids, _) = self.render_messages(prompt, usize::MAX);
        ids.push(self.required_special("<|assistant_start|>"));
        ids
    }

    /// Save the tokenizer state to a JSON file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut mergeable_ranks: Vec<_> = self.mergeable_ranks.iter()
            .map(|(bytes, &token)| (bytes.clone(), token)).collect();
        mergeable_ranks.sort_unstable_by_key(|(_, token)| *token);
        serde_json::to_writer_pretty(file, &TokenizerFile {
            version: 1, mergeable_ranks, special_tokens: self.special_tokens.clone(),
        })?;
        Ok(())
    }

    /// Load the tokenizer state from a JSON file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let saved: TokenizerFile = serde_json::from_reader(file)?;
        if saved.version != 1 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("unsupported tokenizer format version {}", saved.version)));
        }
        let rank_count = saved.mergeable_ranks.len();
        let mergeable_ranks: HashMap<_, _> = saved.mergeable_ranks.into_iter().collect();
        if mergeable_ranks.len() != rank_count ||
            !SPECIAL_TOKENS.iter().all(|token| saved.special_tokens.contains_key(*token)) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "tokenizer contains duplicate ranks or missing special tokens"));
        }
        let mut token_ids = HashSet::with_capacity(mergeable_ranks.len() + saved.special_tokens.len());
        if !mergeable_ranks.values().chain(saved.special_tokens.values())
            .all(|token| token_ids.insert(*token)) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "tokenizer contains duplicate token IDs"));
        }
        let mut tokenizer = Self {
            mergeable_ranks,
            special_tokens: saved.special_tokens,
            inverse_vocab: HashMap::new(), inverse_special_tokens: HashMap::new(),
        };
        tokenizer.build_inverse_mappings();
        Ok(tokenizer)
    }
}

#[cfg(test)] mod tests { use super::*;
    #[derive(Deserialize)]
    struct TokenizerParityFixture {
        schema_version: u32,
        source: FixtureSource,
        vocab_size: usize,
        corpus: Vec<String>,
        mergeable_ranks: Vec<FixtureRank>,
        special_tokens: HashMap<String, usize>,
        encoding_cases: Vec<EncodingCase>,
        conversation_cases: Vec<ConversationCase>,
    }

    #[derive(Deserialize)]
    struct FixtureSource { implementation: String, rustbpe: String, tiktoken: String }

    #[derive(Deserialize)]
    struct FixtureRank { bytes: Vec<u8>, id: usize }

    #[derive(Deserialize)]
    struct EncodingCase { text: String, ids: Vec<usize> }

    #[derive(Deserialize)]
    struct ConversationCase {
        name: String,
        max_tokens: usize,
        conversation: Conversation,
        ids: Vec<usize>,
        mask: Vec<i32>,
        completion_ids: Vec<usize>,
    }

    fn python_parity_fixture() -> TokenizerParityFixture {
        let fixture: TokenizerParityFixture = serde_json::from_str(
            include_str!("../data/fixtures/parity/tokenizer.json")).unwrap();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.source.implementation, "nanochat.tokenizer.RustBPETokenizer");
        assert_eq!(fixture.source.rustbpe, "0.1.0");
        assert_eq!(fixture.source.tiktoken, "0.11.0");
        fixture
    }

    fn tokenizer_from_fixture(fixture: &TokenizerParityFixture) -> BpeTokenizer {
        let mut tokenizer = BpeTokenizer {
            mergeable_ranks: fixture.mergeable_ranks.iter()
                .map(|rank| (rank.bytes.clone(), rank.id)).collect(),
            special_tokens: fixture.special_tokens.clone(),
            inverse_vocab: HashMap::new(), inverse_special_tokens: HashMap::new(),
        };
        tokenizer.build_inverse_mappings();
        tokenizer
    }

    #[test] fn test_python_tokenizer_training_parity() {
        let fixture = python_parity_fixture();
        let tokenizer = BpeTokenizer::train_from_iterator(&fixture.corpus, fixture.vocab_size);
        let expected = tokenizer_from_fixture(&fixture);
        assert_eq!(tokenizer.mergeable_ranks, expected.mergeable_ranks);
        assert_eq!(tokenizer.special_tokens, expected.special_tokens);
    }

    #[test] fn test_python_tokenizer_encoding_and_conversation_parity() {
        let fixture = python_parity_fixture();
        let tokenizer = tokenizer_from_fixture(&fixture);
        assert_eq!(tokenizer.get_vocab_size(), fixture.vocab_size);

        for case in fixture.encoding_cases {
            assert_eq!(tokenizer.encode_ordinary(&case.text), case.ids,
                "encoding parity failed for {:?}", case.text);
            assert_eq!(tokenizer.decode(&case.ids), case.text,
                "decoding parity failed for encoded fixture");
        }
        for case in fixture.conversation_cases {
            let (ids, mask) = tokenizer.render_conversation(&case.conversation, case.max_tokens);
            assert_eq!(ids, case.ids, "conversation ID parity failed for {}", case.name);
            assert_eq!(mask, case.mask, "conversation mask parity failed for {}", case.name);
            assert_eq!(tokenizer.render_for_completion(&case.conversation), case.completion_ids,
                "completion prompt parity failed for {}", case.name);
        }
    }

    #[test] fn test_bpe_training_and_encoding_roundtrip() {
        let corpus = vec![
            "Hello world! Hello system programming.",
            "This is a BPE tokenizer implementation in Rust.",
            "We are pair-programming to build nanochat-burn.",
            "Numbers: 123, 4567. Unicode: 你好世界 🌍.",
        ];

        // Train tokenizer with total vocabulary size 300 (which leaves 300 - 9 = 291 base tokens)
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 300);
        assert_eq!(tokenizer.get_vocab_size(), 300);

        let test_text = "Hello world! Building nanochat-burn in Rust is fun. 12345. 你好世界!";
        let encoded = tokenizer.encode_ordinary(test_text);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, test_text, "Roundtrip encoding/decoding did not match!");
    }

    #[test] fn test_chat_rendering() {
        let corpus = vec!["BOS user assistant python output system helpful result"];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 275);

        let conversation = Conversation {
            messages: vec![
                ConversationMessage {
                    role: "system".to_string(),
                    content: MessageContent::Simple("You are a helpful assistant.".to_string()),
                },
                ConversationMessage {
                    role: "user".to_string(),
                    content: MessageContent::Simple("Compute 2+2".to_string()),
                },
                ConversationMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Parts(vec![
                        MessagePart { part_type: "python".to_string(), text: "2 + 2".to_string() },
                        MessagePart {
                            part_type: "python_output".to_string(),
                            text: "4".to_string(),
                        },
                        MessagePart {
                            part_type: "text".to_string(),
                            text: "The result is 4.".to_string(),
                        },
                    ]),
                },
            ],
        };

        let (ids, mask) = tokenizer.render_conversation(&conversation, 1000);
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), mask.len());

        // Assert first token is BOS
        assert_eq!(ids[0], tokenizer.get_bos_token_id());
        assert_eq!(mask[0], 0);

        // Check that assistant SFT target tokens are masked with 1, and user prompt tokens are
        // masked with 0
        let decoded_tokens: Vec<(String, i32)> =
            ids.iter().zip(mask.iter()).map(|(&id, &m)| (tokenizer.decode(&[id]), m)).collect();

        // The character 'C' is from user prompt ("Compute"), it should have mask = 0
        let c_mask = decoded_tokens.iter().find(|(s, _)| s.contains('C')).map(|(_, m)| *m);
        assert_eq!(c_mask, Some(0));

        // The character 'T' is from assistant's text response ("The"), it should have mask = 1
        let t_mask = decoded_tokens.iter().find(|(s, _)| s.contains('T')).map(|(_, m)| *m);
        assert_eq!(t_mask, Some(1));
    }
}
