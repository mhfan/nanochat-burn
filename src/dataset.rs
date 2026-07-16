
use std::{fs::{self, File}, io::{self, BufWriter, Write}, path::{Path, PathBuf}};

use memmap2::Mmap;

use crate::{common::read_jsonl, tokenizer::{BpeTokenizer, Conversation, MessageContent}};

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
                // SAFETY: The read-only mapping owns its OS mapping after `File` is dropped and
                // this module never mutates the file through another handle.
                let mmap = unsafe { Mmap::map(&File::open(&path)?)? };
                if mmap.len() % std::mem::size_of::<u32>() != 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData,
                        format!("{} is not a valid u32 token shard", path.display())));
                }
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
        match bytemuck::try_cast_slice(bytes) {
            Ok(tokens_u32) => tokens_u32.iter().copied().map(u32::from_le).collect(),
            Err(_) => bytes.chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect(),
        }
    }
}

pub struct SftDataset { pub conversations: Vec<Conversation>, }

impl SftDataset {
    pub fn new<P: AsRef<Path>>(jsonl_path: P) -> io::Result<Self> {
        Ok(Self { conversations: read_jsonl(jsonl_path)? })
    }

    pub fn get_corpus(&self) -> Vec<String> {
        let mut corpus = Vec::new();
        for message in self.conversations.iter().flat_map(|conv| &conv.messages) {
            match &message.content {
                MessageContent::Simple(text) => corpus.push(text.clone()),
                MessageContent::Parts(parts) =>
                    corpus.extend(parts.iter().map(|part| part.text.clone())),
            }
        }
        corpus
    }
}

pub fn pretokenize_text_to_bin<P: AsRef<Path>, Q: AsRef<Path>>(
    text_path: P, bin_path: Q, tokenizer: &BpeTokenizer) -> io::Result<()> {
    let content = fs::read_to_string(text_path)?;
    let tokens = tokenizer.encode_ordinary(&content);
    let tokens_u32: Vec<_> = tokens.into_iter().map(|token| {
        u32::try_from(token).map(u32::to_le).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "token ID exceeds u32 storage")
        })
    }).collect::<io::Result<_>>()?;
    let bytes = bytemuck::cast_slice(&tokens_u32);
    File::create(bin_path)?.write_all(bytes)?;
    Ok(())
}

/// Tokenize and best-fit pack documents into fixed rows. Every row and document fragment starts
/// with BOS, so no training target is asked to infer an unrelated previous document. Documents
/// longer than one row are split without dropping their remaining tokens.
pub fn pretokenize_documents_to_bin<P: AsRef<Path>>(documents: &[String], bin_path: P,
    tokenizer: &BpeTokenizer, row_capacity: usize) -> io::Result<usize> {
    if row_capacity < 2 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput,
            "pretraining row capacity must be at least two"));
    }
    let bos = u32::try_from(tokenizer.get_bos_token_id()).map(u32::to_le).map_err(|_|
        io::Error::new(io::ErrorKind::InvalidData, "BOS token exceeds u32 storage"))?;
    let payload_capacity = row_capacity - 1;
    let mut buckets = vec![Vec::<Vec<u32>>::new(); row_capacity + 1];
    let mut remaining_documents = 0usize;

    for document in documents.iter().filter(|document| !document.trim().is_empty()) {
        let tokens = tokenizer.encode_ordinary(document);
        for chunk in tokens.chunks(payload_capacity) {
            let mut fragment = Vec::with_capacity(chunk.len() + 1);
            fragment.push(bos);
            fragment.extend(chunk.iter().map(|&token| u32::try_from(token).map(u32::to_le)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData,
                    "token ID exceeds u32 storage"))).collect::<io::Result<Vec<_>>>()?);
            buckets[fragment.len()].push(fragment);
            remaining_documents += 1;
        }
    }

    let mut writer = BufWriter::new(File::create(bin_path)?);
    let mut rows_written = 0;
    while remaining_documents > 0 {
        let mut row = Vec::with_capacity(row_capacity);
        while row.len() < row_capacity && remaining_documents > 0 {
            let remaining = row_capacity - row.len();
            let fitting = (1..=remaining).rev().find(|&len| !buckets[len].is_empty());
            // Splitting with only one slot left would consume a BOS and recreate the exact same
            // continuation forever. Use a BOS padding token for that single slot instead; no
            // document payload is consumed or discarded.
            if fitting.is_none() && remaining == 1 {
                row.push(bos);
                continue;
            }
            let selected = fitting.or_else(||
                (remaining + 1..=row_capacity).find(|&len| !buckets[len].is_empty()));
            let Some(length) = selected else { break; };
            let mut document = buckets[length].pop().expect("selected document bucket is empty");
            remaining_documents -= 1;
            if document.len() > remaining {
                let tail = document.split_off(remaining);
                let mut continuation = Vec::with_capacity(tail.len() + 1);
                continuation.push(bos);
                continuation.extend(tail);
                buckets[continuation.len()].push(continuation);
                remaining_documents += 1;
            }
            row.extend(document);
        }
        if row.len() != row_capacity { break; }
        writer.write_all(bytemuck::cast_slice(&row))?;
        rows_written += 1;
    }
    writer.flush()?;
    Ok(rows_written)
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_bin_pretokenization_and_mmap_dataset() {
        let temp_dir = std::env::temp_dir();
        let text_path = temp_dir.join("test_pretrain.txt");
        let bin_path = temp_dir.join("test_pretrain.bin");

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

    #[test] fn test_document_packing_starts_every_row_with_bos() {
        let temp_dir = std::env::temp_dir();
        let bin_path = temp_dir.join(format!(
            "nanochat-packed-documents-{}.bin", std::process::id()));
        let documents = vec![
            "alpha beta gamma delta epsilon".to_string(),
            "small document".to_string(),
            "rust burn tensor model training".to_string(),
        ];
        let tokenizer = BpeTokenizer::train_from_iterator(documents.clone(), 280);
        let row_capacity = 5;
        let rows = pretokenize_documents_to_bin(
            &documents, &bin_path, &tokenizer, row_capacity).unwrap();
        assert!(rows > 0);

        let dataset = PretrainingDataset::new(std::slice::from_ref(&bin_path)).unwrap();
        let tokens = dataset.get_tokens(0, 0, rows * row_capacity);
        let bos = tokenizer.get_bos_token_id() as u32;
        assert!(tokens.chunks_exact(row_capacity).all(|row| row[0] == bos));
        fs::remove_file(bin_path).ok();
    }
}
