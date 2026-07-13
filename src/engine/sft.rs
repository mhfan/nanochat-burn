
use std::{iter, path::Path, time::Instant};

use burn::tensor::backend::AutodiffBackend;

use crate::{common::{int_tensor_2d, scalar_to_f32}, dataset::SftDataset,
    engine::{get_lr_multiplier, get_weight_decay}, gpt::{Gpt, GptConfig},
    optim::MuonAdamW, tokenizer::BpeTokenizer,
};

pub struct SftPacker {
    pub conversations: Vec<(Vec<usize>, Vec<i32>)>,
}

impl SftPacker {
    pub fn new(dataset: &SftDataset, tokenizer: &BpeTokenizer) -> Self {
        let mut conversations: Vec<_> = dataset.conversations.iter()
            .map(|conv| tokenizer.render_conversation(conv, usize::MAX)).collect();
        conversations.sort_by_key(|(conv, _)| conv.len());
        SftPacker { conversations }
    }

    pub fn next_batch(&mut self, batch_size: usize, max_seq_len: usize, bos_token: usize)
        -> (Vec<Vec<usize>>, Vec<Vec<i32>>, Vec<usize>) {
        let row_capacity = max_seq_len + 1;
        let mut rows = Vec::with_capacity(batch_size);
        let mut mask_rows = Vec::with_capacity(batch_size);
        let mut row_lengths = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let mut row = Vec::with_capacity(row_capacity);
            let mut mask_row = Vec::with_capacity(row_capacity);
            let (mut content_len, mut padded) = (row_capacity, false);

            while row.len() < row_capacity {
                let remaining = row_capacity - row.len();
                let idx_limit =
                    self.conversations.partition_point(|(conv, _)| conv.len() <= remaining);

                if idx_limit > 0 {
                    let (conv, conv_mask) = self.conversations.remove(idx_limit - 1);
                    mask_row.extend(conv_mask);
                    row.extend(conv);
                } else if row.is_empty() && !self.conversations.is_empty() {
                    let (mut conv, mut conv_mask) = self.conversations.remove(0);
                    conv.truncate(row_capacity);
                    conv_mask.truncate(row_capacity);
                    row.extend(conv);
                    mask_row.extend(conv_mask);
                } else {
                    content_len = row.len();
                    row.extend(iter::repeat_n(bos_token, remaining));
                    mask_row.extend(iter::repeat_n(0, remaining));
                    padded = true;
                    break;
                }
            }

            row_lengths.push(if padded { content_len } else { row_capacity });
            mask_rows.push(mask_row);
            rows.push(row);
        }

        (rows, mask_rows, row_lengths)
    }
}

fn flatten_sft_batch(rows: &[Vec<usize>], mask_rows: &[Vec<i32>], row_lengths: &[usize],
    max_seq_len: usize) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(rows.len(), mask_rows.len(), "SFT row/mask count mismatch");
    assert_eq!(rows.len(), row_lengths.len(), "SFT row/length count mismatch");
    let mut flat_inputs = Vec::with_capacity(rows.len() * max_seq_len);
    let mut flat_targets = Vec::with_capacity(rows.len() * max_seq_len);

    for ((row, row_mask), &content_len) in rows.iter().zip(mask_rows).zip(row_lengths) {
        flat_inputs.extend(row[..max_seq_len].iter().map(|&x| x as i32));
        flat_targets.extend((1..=max_seq_len)
            .map(|j| if row_mask[j] == 1 && j < content_len { row[j] as i32 } else { -1 }),
        );
    }

    (flat_inputs, flat_targets)
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

    let corpus = dataset.get_corpus();
    tracing::info!("Training BpeTokenizer on {} SFT text fragments...", corpus.len());

    let mut tokenizer = BpeTokenizer::train_from_iterator(corpus, 1024);
    tokenizer.build_inverse_mappings();

    let config = GptConfig { sequence_len: 128, n_layer: 4, n_head: 4, n_kv_head: 2,
        n_embd: 64, window_pattern: "L".to_string(),
        vocab_size: tokenizer.get_vocab_size(), quantization: None,
    };

    let mut model: Gpt<B> = Gpt::new(config.clone(), device);
    let mut optimizer = MuonAdamW::new(model.config.n_layer);
    let mut packer = SftPacker::new(&dataset, &tokenizer);

    let (batch_size, max_seq_len) = (4, config.sequence_len);
    let bos_token = tokenizer.get_bos_token_id();

    let (warmup_steps, num_iterations, learning_rate, weight_decay) = (5, 20, 1e-4, 0.0);
    tracing::info!("Starting SFT training loop for {} iterations...", num_iterations);
    let (start_time, mut smooth_loss) = (Instant::now(), 0.0);

    for step in 1..=num_iterations {
        if packer.conversations.is_empty() {
            packer = SftPacker::new(&dataset, &tokenizer);
        }
        let (rows, mask_rows, row_lengths) =
            packer.next_batch(batch_size, max_seq_len, bos_token);

        let (flat_inputs, flat_targets) =
            flatten_sft_batch(&rows, &mask_rows, &row_lengths, max_seq_len);

        let inputs_tensor = int_tensor_2d(flat_inputs, [batch_size, max_seq_len], device);
        let targets_tensor = int_tensor_2d(flat_targets, [batch_size, max_seq_len], device);

        let logits = model.forward(inputs_tensor, None);
        let loss = model.compute_loss(logits, targets_tensor);
        let grads = loss.backward();

        let lrm = get_lr_multiplier(step, num_iterations, warmup_steps, 0.5, 0.0);
        let wd = get_weight_decay(step, num_iterations, weight_decay);
        let lr = learning_rate * lrm;

        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        optimizer.step(&mut model, &grads_params, lr, step, wd);

        let loss_val = scalar_to_f32(loss.into_scalar());
        smooth_loss = if step == 1 { loss_val } else { 0.9 * smooth_loss + 0.1 * loss_val };

        if step % 5 == 0 || step == num_iterations {
            tracing::info!("Step {:03}/{:03} | lr: {:.6} | Loss: {:.4} (smooth: {:.4})",
                step, num_iterations, lr, loss_val, smooth_loss);
        }
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   SFT Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    let checkpoint_path = Path::new("data/sft_checkpoint.safetensors");
    tracing::info!("Saving SFT checkpoint to {:?}...", checkpoint_path);
    if let Err(e) = crate::checkpoint::save_gpt_to_safetensors(&model, checkpoint_path) {
        tracing::error!("Failed to save SFT checkpoint: {}", e);
    } else {
        tracing::info!("SFT checkpoint saved successfully!");
    }
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_sft_packer() {
        use crate::tokenizer::BpeTokenizer;
        let dataset = SftDataset::new("data/sft_train.jsonl").unwrap();
        let corpus = vec!["Who are you? I am nanochat.", "Hello!"];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 512);
        let mut packer = SftPacker::new(&dataset, &tokenizer);

        let (batch_size, max_seq_len) = (2, 32);
        let bos_token = tokenizer.get_bos_token_id();

        let (rows, mask_rows, row_lengths) = packer.next_batch(batch_size, max_seq_len, bos_token);
        assert_eq!(rows.len(), batch_size);
        assert_eq!(mask_rows.len(), batch_size);
        assert_eq!(row_lengths.len(), batch_size);

        assert_eq!(rows[0].len(), max_seq_len + 1);

        let mut oversized = SftPacker {
            conversations: vec![(vec![1; 64], vec![1; 64])],
        };
        let (rows, masks, _) = oversized.next_batch(1, 16, bos_token);
        assert_eq!(rows[0].len(), 17);
        assert_eq!(masks[0].len(), 17);
        assert!(oversized.conversations.is_empty());
    }
//}
