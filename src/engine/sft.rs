
use std::{path::Path, time::Instant};
use burn::{prelude::ToElement, tensor::{Tensor, Int, backend::AutodiffBackend}};
use crate::{gpt::{Gpt, GptConfig}, tokenizer::BpeTokenizer,
    dataset::SftDataset, optim::MuonAdamW, engine::{get_lr_multiplier, get_weight_decay},
};

pub struct SftPacker {
    pub conversations: Vec<(Vec<usize>, Vec<i32>)>,
    pub cursor: usize,
}

impl SftPacker {
    pub fn new(dataset: &SftDataset, tokenizer: &BpeTokenizer) -> Self {
        let mut conversations = Vec::new();
        for conv in &dataset.conversations {
            let (ids, mask) = tokenizer.render_conversation(conv, usize::MAX);
            conversations.push((ids, mask));
        }
        SftPacker { conversations, cursor: 0 }
    }

    pub fn next_batch(&mut self, batch_size: usize, max_seq_len: usize,
        bos_token: usize,) -> (Vec<Vec<usize>>, Vec<Vec<i32>>, Vec<usize>) {
        let row_capacity = max_seq_len + 1;
        let mut rows = Vec::with_capacity(batch_size);
        let mut mask_rows = Vec::with_capacity(batch_size);
        let mut row_lengths = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let mut row = Vec::with_capacity(row_capacity);
            let mut mask_row = Vec::with_capacity(row_capacity);
            let mut content_len = row_capacity;
            let mut padded = false;

            while row.len() < row_capacity {
                let remaining = row_capacity - row.len();

                let mut best_idx = None;
                let mut best_len = 0;

                for (idx, (conv, _)) in self.conversations.iter().enumerate() {
                    let conv_len = conv.len();
                    if conv_len <= remaining && conv_len > best_len {
                        best_idx = Some(idx);
                        best_len = conv_len;
                    }
                }

                if let Some(idx) = best_idx {
                    let (conv, conv_mask) = self.conversations.remove(idx);
                    row.extend(conv);
                    mask_row.extend(conv_mask);
                } else {
                    content_len = row.len();
                    row.extend(std::iter::repeat(bos_token).take(remaining));
                    mask_row.extend(std::iter::repeat(0).take(remaining));
                    padded = true;
                    break;
                }

                if self.conversations.is_empty() {
                    self.cursor = 0;
                    break;
                }
            }

            if padded {
                row_lengths.push(content_len);
            } else {
                row_lengths.push(row_capacity);
            }

            rows.push(row);
            mask_rows.push(mask_row);
        }

        (rows, mask_rows, row_lengths)
    }
}

pub fn run_sft_training<B: AutodiffBackend>(device: &B::Device) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Supervised Fine-Tuning (SFT)     ");
    tracing::info!("=============================================");

    let sft_dataset_path = "data/sft_train.jsonl";
    if !Path::new(sft_dataset_path).exists() {
        tracing::error!("SFT dataset not found! Please run synthetic dataset generator first.");
        return;
    }

    let dataset = SftDataset::new(sft_dataset_path).expect("Failed to load SFT dataset");
    tracing::info!("Loaded SFT dataset with {} conversations", dataset.conversations.len());

    let mut corpus = Vec::new();
    for conv in &dataset.conversations {
        for msg in &conv.messages {
            match &msg.content {
                crate::tokenizer::MessageContent::Simple(s) => corpus.push(s.clone()),
                crate::tokenizer::MessageContent::Parts(parts) => {
                    for part in parts {
                        corpus.push(part.text.clone());
                    }
                }
            }
        }
    }

    tracing::info!("Training BpeTokenizer on {} SFT text fragments...", corpus.len());
    let mut tokenizer = BpeTokenizer::train_from_iterator(corpus, 1024);
    tokenizer.build_inverse_mappings();

    let config = GptConfig { sequence_len: 128, n_layer: 4, n_head: 4, n_kv_head: 2,
        n_embd: 64, window_pattern: "L".to_string(), vocab_size: tokenizer.get_vocab_size(),
    };

    let mut model: Gpt<B> = Gpt::new(config.clone(), device);
    let mut optimizer = MuonAdamW::new(model.config.n_layer);
    let mut packer = SftPacker::new(&dataset, &tokenizer);

    let batch_size = 4;
    let max_seq_len = config.sequence_len;
    let bos_token = tokenizer.get_bos_token_id();

    let num_iterations = 20;
    let warmup_steps = 5;
    let learning_rate = 1e-4;
    let weight_decay = 0.0;

    tracing::info!("Starting SFT training loop for {} iterations...", num_iterations);
    let start_time = Instant::now();
    let mut smooth_loss = 0.0;

    for step in 1..=num_iterations {
        if packer.conversations.is_empty() {
            packer = SftPacker::new(&dataset, &tokenizer);
        }

        let (rows, mask_rows, row_lengths) = packer.next_batch(batch_size, max_seq_len, bos_token);

        let mut flat_inputs = Vec::with_capacity(batch_size * max_seq_len);
        let mut flat_targets = Vec::with_capacity(batch_size * max_seq_len);

        for (i, row) in rows.iter().enumerate() {
            let content_len = row_lengths[i];
            let row_mask = &mask_rows[i];

            for j in 0..max_seq_len {
                flat_inputs.push(row[j] as i32);
            }

            for j in 1..=max_seq_len {
                let mask_val = row_mask[j];
                let is_padding = j >= content_len;
                if mask_val == 0 || is_padding {
                    flat_targets.push(-1);
                } else {
                    flat_targets.push(row[j] as i32);
                }
            }
        }

        let inputs_tensor = Tensor::<B, 1, Int>::from_data(flat_inputs.as_slice(), device)
            .reshape([batch_size, max_seq_len]);
        let targets_tensor = Tensor::<B, 1, Int>::from_data(flat_targets.as_slice(), device)
            .reshape([batch_size, max_seq_len]);

        let logits = model.forward(inputs_tensor, None);
        let loss = model.compute_loss(logits, targets_tensor);
        let grads = loss.backward();

        let lrm = get_lr_multiplier(step, num_iterations, warmup_steps, 0.5, 0.0);
        let lr = learning_rate * lrm;
        let wd = get_weight_decay(step, num_iterations, weight_decay);

        optimizer.step(&mut model, &grads, lr, step, wd);

        let loss_val = loss.into_scalar().to_f32();

        if step == 1 {
            smooth_loss = loss_val;
        } else {
            smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val;
        }

        if step % 5 == 0 || step == num_iterations {
            tracing::info!(
                "Step {:03}/{:03} | lr: {:.6} | Loss: {:.4} (smooth: {:.4})",
                step, num_iterations, lr, loss_val, smooth_loss
            );
        }
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   SFT Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_sft_packer() {
        use crate::tokenizer::BpeTokenizer;
        let dataset = SftDataset::new("data/sft_train.jsonl").unwrap();
        let corpus = vec!["Who are you? I am nanochat.", "Hello!"];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 512);
        let mut packer = SftPacker::new(&dataset, &tokenizer);

        let batch_size = 2;
        let max_seq_len = 32;
        let bos_token = tokenizer.get_bos_token_id();

        let (rows, mask_rows, row_lengths) = packer.next_batch(batch_size, max_seq_len, bos_token);
        assert_eq!(rows.len(), batch_size);
        assert_eq!(mask_rows.len(), batch_size);
        assert_eq!(row_lengths.len(), batch_size);

        assert_eq!(rows[0].len(), max_seq_len + 1);
    }
//}
