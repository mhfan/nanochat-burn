
use std::{io, fs::{self, File}, path::{Path, PathBuf}};

use memmap2::Mmap;

use crate::tokenizer::{BpeTokenizer, Conversation, MessageContent};

pub struct Shard {
    pub path: PathBuf,
    pub mmap: Mmap,
    pub num_tokens: usize,
}

pub struct PretrainingDataset { pub shards: Vec<Shard>, }

impl PretrainingDataset {
    pub fn new<P: AsRef<Path>>(paths: &[P]) -> io::Result<Self> {
        let shards = paths.iter().map(|p| {
                let path = p.as_ref().to_path_buf();
                let mmap = unsafe { Mmap::map(&File::open(&path)?)? };
                let num_tokens = mmap.len() / 4;
                Ok(Shard { path, mmap, num_tokens })
        }).collect::<io::Result<_>>()?;
        Ok(Self { shards })
    }

    pub fn get_token(&self, shard_idx: usize, token_offset: usize) -> u32 {
        let shard = &self.shards[shard_idx];
        let byte_offset = token_offset * 4;
        let bytes = &shard.mmap[byte_offset..byte_offset + 4];
        u32::from_le_bytes(bytes.try_into().unwrap())
    }

    pub fn get_tokens(&self, shard_idx: usize, token_offset: usize, len: usize) -> Vec<u32> {
        let shard = &self.shards[shard_idx];
        let byte_start = token_offset * 4;
        let byte_end = (token_offset + len) * 4;
        let bytes = &shard.mmap[byte_start..byte_end];
        match bytemuck::try_cast_slice::<u8, u32>(bytes) {
            Ok(tokens_u32) => tokens_u32.to_vec(),
            Err(_) => bytes.chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect(),
        }
    }
}

pub struct SftDataset { pub conversations: Vec<Conversation>, }

impl SftDataset {
    pub fn new<P: AsRef<Path>>(jsonl_path: P) -> io::Result<Self> {
        use io::BufRead;
        let reader = io::BufReader::new(File::open(jsonl_path)?);
        let conversations = reader
            .lines()
            .filter_map(|line| match line {
                Ok(line) if line.trim().is_empty() => None,
                other => Some(other.and_then(|line| {
                    serde_json::from_str(line.trim()).map_err(io::Error::other)
                })),
            })
            .collect::<io::Result<_>>()?;
        Ok(Self { conversations })
    }

    pub fn get_corpus(&self) -> Vec<String> {
        self.conversations.iter().flat_map(|conv| conv.messages.iter())
            .flat_map(|msg| match &msg.content {
                MessageContent::Simple(s) => vec![s.clone()],
                MessageContent::Parts(parts) => parts.iter().map(|p| p.text.clone()).collect(),
            }).collect()
    }
}

pub fn pretokenize_text_to_bin<P: AsRef<Path>, Q: AsRef<Path>>(
    text_path: P, bin_path: Q, tokenizer: &BpeTokenizer) -> io::Result<()> {
    let content = fs::read_to_string(text_path)?;
    let tokens = tokenizer.encode_ordinary(&content);
    let tokens_u32: Vec<_> = tokens.into_iter().map(|tok| tok as u32).collect();
    let bytes = bytemuck::cast_slice::<u32, u8>(&tokens_u32);
    File::create(bin_path)?.write_all(bytes)?;
    use io::Write;
    Ok(())
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_bin_pretokenization_and_mmap_dataset() {
        let temp_dir = std::env::temp_dir();
        let text_path = temp_dir.join("test_pretrain.txt");
        let  bin_path = temp_dir.join("test_pretrain.bin");

        fs::write(&text_path, "System programming in Rust is elegant and fast.").unwrap();

        // Train tokenizer on the same corpus
        let corpus = vec!["System programming in Rust is elegant and fast."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        pretokenize_text_to_bin(&text_path, &bin_path, &tokenizer).unwrap();

        let dataset = PretrainingDataset::new(std::slice::from_ref(&bin_path)).unwrap();
        assert_eq!(dataset.shards.len(), 1);

        let num_tokens = dataset.shards[0].num_tokens;
        assert!(num_tokens > 0);

        let tokens = dataset.get_tokens(0, 0, num_tokens);
        let decoded = tokenizer.decode(&tokens.iter().map(|&t| t as usize).collect::<Vec<_>>());
        assert_eq!(decoded, "System programming in Rust is elegant and fast.");

        // Clean up
        let _ = fs::remove_file(text_path);
        let _ = fs::remove_file(bin_path);
    }
//}
