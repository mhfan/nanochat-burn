
use std::path::PathBuf;
use tokio::sync::mpsc::{channel, Receiver};
use crate::{tokenizer::BpeTokenizer, dataset::{PretrainingDataset, SftDataset}};

pub struct Batch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub shard_idx: usize,
    pub token_offset: usize,
    pub epoch: usize,
}

pub struct DistributedDataLoader { receiver: Receiver<Batch>, }

impl DistributedDataLoader {
    pub fn new(dataset_paths: Vec<PathBuf>, batch_size: usize,
        sequence_length: usize, rank: usize, world_size: usize,
        start_shard_idx: usize, start_token_offset: usize, start_epoch: usize,) -> Self {
        let (sender, receiver) = channel(4); // double buffering prefetching
        tokio::spawn(async move {
            let dataset = match PretrainingDataset::new(&dataset_paths) {
                Ok(ds) => ds,
                Err(e) => {
                    eprintln!("Failed to initialize PretrainingDataset: {:?}", e);
                    return;
                }
            };

            // 1. Determine shards assigned to this DDP rank
            let shards_assigned: Vec<usize> = (0..dataset.shards.len())
                .filter(|&i| i % world_size == rank).collect();
            if shards_assigned.is_empty() {
                eprintln!("DDP Rank {} has no shards assigned!", rank);
                return;
            }

            // 2. Set up initial positioning
            let mut shard_pos = shards_assigned.iter()
                .position(|&idx| idx == start_shard_idx).unwrap_or(0);
            let mut offset = start_token_offset;
            let mut epoch = start_epoch;
            let num_needed = batch_size * (sequence_length + 1);

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

                for row in 0..batch_size {
                    let start = row * (sequence_length + 1);
                    let row_tokens = &tokens[start..start + (sequence_length + 1)];
                    x.extend(row_tokens[..sequence_length].iter().map(|&t| t as i32));
                    y.extend(row_tokens[1..sequence_length + 1].iter().map(|&t| t as i32));
                }

                let batch = Batch { x, y, shard_idx: active_shard_idx, token_offset: offset, epoch };
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
    pub fn new(dataset: SftDataset, tokenizer: BpeTokenizer, batch_size: usize, sequence_length: usize) -> Self {
        SftDataLoader { dataset, tokenizer, batch_size, sequence_length, cur_idx: 0 }
    }

    pub fn next_batch(&mut self) -> Option<SftBatch> {
        if self.dataset.conversations.is_empty() { return None; }

        let mut x = Vec::with_capacity(self.batch_size * self.sequence_length);
        let mut y = Vec::with_capacity(self.batch_size * self.sequence_length);
        let mut mask = Vec::with_capacity(self.batch_size * self.sequence_length);
        let bos = self.tokenizer.get_bos_token_id();

        for _ in 0..self.batch_size {
            if self.cur_idx >= self.dataset.conversations.len() { self.cur_idx = 0; }
            let conv = &self.dataset.conversations[self.cur_idx];
            self.cur_idx += 1;

            let (mut ids, mut m) = self.tokenizer.render_conversation(conv, self.sequence_length + 1);
            let pad_len = (self.sequence_length + 1).saturating_sub(ids.len());
            if pad_len > 0 {
                ids.extend(std::iter::repeat(bos).take(pad_len));
                m.extend(std::iter::repeat(0).take(pad_len));
            }

            x.extend(ids[..self.sequence_length].iter().map(|&id| id as i32));
            y.extend(ids[1..self.sequence_length + 1].iter().map(|&id| id as i32));
            mask.extend(m[1..self.sequence_length + 1].iter().copied());
        }
        Some(SftBatch { x, y, mask })
    }
}

//#[cfg(test)] mod tests { use super::*;
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
            "Rust is high throughput and safe memory management."
        ];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        pretokenize_text_to_bin(&t1_txt, &t1_bin, &tokenizer).unwrap();
        pretokenize_text_to_bin(&t2_txt, &t2_bin, &tokenizer).unwrap();

        // World size = 2, Rank 0 gets t1.bin (shard 0), Rank 1 gets t2.bin (shard 1)
        let mut loader_r0 = DistributedDataLoader::new(
            vec![t1_bin.clone(), t2_bin.clone()],
            2, 2, 0, 2, 0, 0, 1
        );

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
//}
