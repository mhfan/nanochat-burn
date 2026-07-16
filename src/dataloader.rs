
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, channel};

use crate::dataset::PretrainingDataset;

pub struct Batch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub shard_idx: usize,
    pub token_offset: usize,
    pub epoch: usize,
    pub next_position: DataLoaderPosition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataLoaderPosition {
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

    pub fn with_position(mut self, position: DataLoaderPosition) -> Self {
        self.start_shard_idx = position.shard_idx;
        self.start_token_offset = position.token_offset;
        self.start_epoch = position.epoch;
        self
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
            let shards_assigned: Vec<_> =
                (0..dataset.shards.len()).filter(|&i| i % world_size == rank).collect();
            if shards_assigned.is_empty() {
                eprintln!("DDP Rank {} has no shards assigned!", rank);
                return;
            }
            if start_token_offset > 0 && !shards_assigned.contains(&start_shard_idx) {
                eprintln!("DDP Rank {} cannot resume from unassigned shard {}",
                    rank, start_shard_idx);
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
            if offset > dataset.shards[shards_assigned[shard_pos]].num_tokens {
                eprintln!("DDP Rank {} resume offset {} exceeds shard {} length",
                    rank, offset, shards_assigned[shard_pos]);
                return;
            }

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

                offset = offset.checked_add(num_needed).expect("dataset token offset overflow");
                let next_position =
                    DataLoaderPosition { shard_idx: active_shard_idx, token_offset: offset, epoch };
                let batch = Batch { x, y, shard_idx: active_shard_idx,
                    token_offset: offset - num_needed, epoch, next_position,
                };
                if sender.send(batch).await.is_err() { break; }
            }
        });
        DistributedDataLoader { receiver }
    }

    pub async fn next_batch(&mut self) -> Option<Batch> { self.receiver.recv().await }
}

#[cfg(test)] mod tests { use super::*;
    #[tokio::test] async fn test_distributed_dataloader_prefetch_and_sharding() {
        use crate::{dataset::pretokenize_text_to_bin, tokenizer::BpeTokenizer};
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

        let position = batch.next_position;
        let expected = loader_r0.next_batch().await.unwrap();
        let mut resumed = DistributedDataLoader::new(vec![t1_bin.clone(), t2_bin.clone()],
            config.with_position(position));
        let actual = resumed.next_batch().await.unwrap();
        assert_eq!(actual.x, expected.x);
        assert_eq!(actual.y, expected.y);
        assert_eq!(actual.shard_idx, expected.shard_idx);
        assert_eq!(actual.token_offset, expected.token_offset);
        assert_eq!(actual.epoch, expected.epoch);

        // Clean up
        let _ = fs::remove_file(t1_txt);
        let _ = fs::remove_file(t1_bin);
        let _ = fs::remove_file(t2_txt);
        let _ = fs::remove_file(t2_bin);
    }
}
