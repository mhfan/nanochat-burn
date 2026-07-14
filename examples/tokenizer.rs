use nanochat_burn::tokenizer::{BpeTokenizer, Conversation, ConversationMessage, MessageContent};

fn main() {
    let corpus = [
        "Rust makes systems programming explicit and safe.",
        "A tokenizer maps text bytes to model token identifiers.",
        "The assistant learns to predict the next token.",
    ];
    let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

    let text = "Rust maps text to tokens.";
    let ids = tokenizer.encode_ordinary(text);
    let decoded = tokenizer.decode(&ids);
    assert_eq!(decoded, text);

    let conversation = Conversation { messages: vec![
        ConversationMessage { role: "user".into(),
            content: MessageContent::Simple("What does a tokenizer emit?".into()) },
        ConversationMessage { role: "assistant".into(),
            content: MessageContent::Simple("Token identifiers.".into()) },
    ] };
    let (conversation_ids, mask) = tokenizer.render_conversation(&conversation, usize::MAX);
    assert_eq!(conversation_ids.len(), mask.len());
    let supervised_targets = mask.iter().filter(|&&value| value == 1).count();
    assert!(supervised_targets > 0);

    println!("vocabulary: {}", tokenizer.get_vocab_size());
    println!("text: {text}");
    println!("token IDs: {ids:?}");
    println!("decoded: {decoded}");
    println!("conversation tokens: {}", conversation_ids.len());
    println!("supervised targets: {supervised_targets}");
}
