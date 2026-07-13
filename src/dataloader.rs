
use std::path::PathBuf;
use tokio::sync::mpsc::{Receiver, channel};

use crate::{tokenizer::BpeTokenizer, dataset::{PretrainingDataset, SftDataset}};

pub struct Batch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub shard_idx: usize,
    pub token_offset: usize,
    pub epoch: usize,
}

pub struct DistributedDataLoader { receiver: Receiver<Batch>, }

#[derive(Debug, Clone, Copy)]
pub struct DistributedDataLoaderConfig {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub rank: usize,
    pub world_size: usize,
    pub start_shard_idx: usize,
    pub start_token_offset: usize,
    pub start_epoch: usize,
}

impl DistributedDataLoaderConfig {
    pub fn single_process(batch_size: usize, sequence_length: usize) -> Self {
        Self { batch_size, sequence_length, rank: 0, world_size: 1,
            start_shard_idx: 0, start_token_offset: 0, start_epoch: 0,
        }
    }

    fn validate(self) {
        assert!(self.batch_size > 0, "batch size must be greater than zero");
        assert!(self.sequence_length > 0, "sequence length must be greater than zero");
        assert!(self.world_size > 0, "world size must be greater than zero");
        assert!(self.rank < self.world_size, "rank must be smaller than world size");
    }
}

fn push_lm_rows(tokens: &[u32], batch_size: usize, sequence_length: usize, x: &mut Vec<i32>,
    y: &mut Vec<i32>) {
    for row in tokens.chunks_exact(sequence_length + 1).take(batch_size) {
        x.extend(row[..sequence_length].iter().map(|&t| t as i32));
        y.extend(row[1..].iter().map(|&t| t as i32));
    }
}

impl DistributedDataLoader {
    pub fn new(dataset_paths: Vec<PathBuf>, config: DistributedDataLoaderConfig) -> Self {
        config.validate();
        let DistributedDataLoaderConfig { batch_size, sequence_length, rank, world_size,
            start_shard_idx, start_token_offset, start_epoch } = config;
        let (sender, receiver) = channel(4);
        tokio::spawn(async move {
            let dataset = match PretrainingDataset::new(&dataset_paths) {
                Ok(ds) => ds,
                Err(e) => {
                    eprintln!("Failed to initialize PretrainingDataset: {:?}", e);
                    return;
                }
            };

            // 1. Determine shards assigned to this DDP rank
            let shards_assigned: Vec<usize> =
                (0..dataset.shards.len()).filter(|&i| i % world_size == rank).collect();
            if shards_assigned.is_empty() {
                eprintln!("DDP Rank {} has no shards assigned!", rank);
                return;
            }

            let num_needed = batch_size.checked_mul(sequence_length + 1)
                .expect("batch token count overflow");
            if shards_assigned.iter().all(|&idx| dataset.shards[idx].num_tokens < num_needed) {
                eprintln!("DDP Rank {} has no shard large enough for one batch: need {} tokens",
                    rank, num_needed);
                return;
            }

            // 2. Set up initial positioning
            let mut shard_pos =
                shards_assigned.iter().position(|&idx| idx == start_shard_idx).unwrap_or(0);
            let (mut offset, mut epoch) = (start_token_offset, start_epoch);

            loop {
                let active_shard_idx = shards_assigned[shard_pos];
                let shard = &dataset.shards[active_shard_idx];

                // Rollover to the next assigned shard if not enough tokens
                if offset + num_needed > shard.num_tokens {
                    offset = 0;
                    shard_pos = (shard_pos + 1) % shards_assigned.len();
                    if shard_pos == 0 { epoch += 1; }
                    continue;
                }

                // Slice tokens and format inputs/targets
                let tokens = dataset.get_tokens(active_shard_idx, offset, num_needed);
                let mut x = Vec::with_capacity(batch_size * sequence_length);
                let mut y = Vec::with_capacity(batch_size * sequence_length);

                push_lm_rows(&tokens, batch_size, sequence_length, &mut x, &mut y);

                let batch =
                    Batch { x, y, shard_idx: active_shard_idx, token_offset: offset, epoch };
                offset += num_needed;
                if sender.send(batch).await.is_err() { break; }
            }
        });
        DistributedDataLoader { receiver }
    }

    pub async fn next_batch(&mut self) -> Option<Batch> { self.receiver.recv().await }
}

pub struct SftBatch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub mask: Vec<i32>,
}

pub struct SftDataLoader {
    dataset: SftDataset,
    tokenizer: BpeTokenizer,
    batch_size: usize,
    sequence_length: usize,
    cur_idx: usize,
}

impl SftDataLoader {
    pub fn new(dataset: SftDataset, tokenizer: BpeTokenizer, batch_size: usize,
        sequence_length: usize) -> Self {
        assert!(batch_size > 0, "batch size must be greater than zero");
        assert!(sequence_length > 0, "sequence length must be greater than zero");
        SftDataLoader { dataset, tokenizer, batch_size, sequence_length, cur_idx: 0 }
    }

    pub fn next_batch(&mut self) -> Option<SftBatch> {
        if self.dataset.conversations.is_empty() { return None; }

        let mut x = Vec::with_capacity(self.batch_size * self.sequence_length);
        let mut y = Vec::with_capacity(self.batch_size * self.sequence_length);
        let mut mask = Vec::with_capacity(self.batch_size * self.sequence_length);
        let bos = self.tokenizer.get_bos_token_id();

        for _ in 0..self.batch_size {
            let conv = &self.dataset.conversations[self.cur_idx];
            self.cur_idx = (self.cur_idx + 1) % self.dataset.conversations.len();

            let (mut ids, mut m) =
                self.tokenizer.render_conversation(conv, self.sequence_length + 1);
            let pad_len = (self.sequence_length + 1).saturating_sub(ids.len());
            if pad_len > 0 {
                ids.extend(std::iter::repeat_n(bos, pad_len));
                m.extend(std::iter::repeat_n(0, pad_len));
            }

            x.extend(ids[..self.sequence_length].iter().map(|&id| id as i32));
            y.extend(ids[1..self.sequence_length + 1].iter().map(|&id| id as i32));
            mask.extend(m[1..self.sequence_length + 1].iter().copied());
        }
        Some(SftBatch { x, y, mask })
    }
}

#[cfg(test)] mod tests { use super::*;
    #[tokio::test] async fn test_distributed_dataloader_prefetch_and_sharding() {
        use crate::dataset::pretokenize_text_to_bin;
        let temp_dir = std::env::temp_dir();
        let t1_txt = temp_dir.join("t1.txt");
        let t1_bin = temp_dir.join("t1.bin");
        let t2_txt = temp_dir.join("t2.txt");
        let t2_bin = temp_dir.join("t2.bin");

        use std::fs;
        fs::write(&t1_txt, "Hello world! Contiguous tokens representation.").unwrap();
        fs::write(&t2_txt, "Rust is high throughput and safe memory management.").unwrap();

        let corpus = vec![
            "Hello world! Contiguous tokens representation.",
            "Rust is high throughput and safe memory management.",
        ];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        pretokenize_text_to_bin(&t1_txt, &t1_bin, &tokenizer).unwrap();
        pretokenize_text_to_bin(&t2_txt, &t2_bin, &tokenizer).unwrap();

        // World size = 2, Rank 0 gets t1.bin (shard 0), Rank 1 gets t2.bin (shard 1)
        let config = DistributedDataLoaderConfig { batch_size: 2, sequence_length: 2,
            rank: 0, world_size: 2, start_shard_idx: 0, start_token_offset: 0, start_epoch: 1,
        };
        let mut loader_r0 =
            DistributedDataLoader::new(vec![t1_bin.clone(), t2_bin.clone()], config);

        let batch = loader_r0.next_batch().await.unwrap();
        assert_eq!(batch.shard_idx, 0); // Rank 0 assigned shard 0
        assert_eq!(batch.epoch, 1);
        assert_eq!(batch.x.len(), 4); // B * T = 2 * 2 = 4

        // Clean up
        let _ = fs::remove_file(t1_txt);
        let _ = fs::remove_file(t1_bin);
        let _ = fs::remove_file(t2_txt);
        let _ = fs::remove_file(t2_bin);
    }
}
