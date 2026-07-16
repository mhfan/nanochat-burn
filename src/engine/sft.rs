
use std::{iter, time::Instant};
use burn::tensor::backend::AutodiffBackend;

use crate::{artifact::{MetricRecord, TrainingStage, append_metric, load_artifact, path_from_env,
        reset_metrics, save_artifact, save_experiment_config},
    common::{int_tensor_2d, scalar_to_f32}, dataset::SftDataset,
    engine::{get_lr_multiplier, get_muon_momentum, get_weight_decay},
    experiment::ExperimentConfig,
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
        assert!(batch_size > 0, "batch size must be greater than zero");
        assert!(max_seq_len > 0, "sequence length must be greater than zero");
        let row_capacity = max_seq_len.checked_add(1).expect("sequence length overflow");
        let mut rows = Vec::with_capacity(batch_size);
        let mut mask_rows = Vec::with_capacity(batch_size);
        let mut row_lengths = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if self.conversations.is_empty() { break; }
            let mut row = Vec::with_capacity(row_capacity);
            let mut mask_row = Vec::with_capacity(row_capacity);
            let (mut content_len, mut padded) = (row_capacity, false);

            while row.len() < row_capacity {
                let remaining = row_capacity - row.len();
                let idx_limit =
                    self.conversations.partition_point(|(conv, _)| conv.len() <= remaining);

                if idx_limit > 0 {
                    let (conv, conv_mask) = self.conversations.remove(idx_limit - 1);
                    assert_eq!(conv.len(), conv_mask.len(), "SFT conversation/mask size mismatch");
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
        assert!(row.len() > max_seq_len && row_mask.len() > max_seq_len,
            "SFT row must contain sequence_length + 1 tokens");
        assert!(content_len <= row.len(), "SFT content length exceeds row length");
        flat_inputs.extend(row[..max_seq_len].iter().map(|&x| x as i32));
        flat_targets.extend((1..=max_seq_len)
            .map(|j| if row_mask[j] == 1 && j < content_len { row[j] as i32 } else { -1 }),
        );
    }

    (flat_inputs, flat_targets)
}

pub fn run_sft_training<B: AutodiffBackend>(device: &B::Device,
    experiment: &ExperimentConfig) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Supervised Fine-Tuning (SFT)     ");
    tracing::info!("=============================================");

    experiment.validate().unwrap_or_else(|error| panic!("invalid experiment config: {error}"));
    let config = &experiment.sft;
    if !config.dataset_path.exists() {
        tracing::error!("SFT dataset not found! Please run synthetic dataset generator first.");
        return;
    }

    let dataset = SftDataset::new(&config.dataset_path).expect("Failed to load SFT dataset");
    tracing::info!("Loaded SFT dataset with {} conversations", dataset.conversations.len());

    let input = path_from_env("NANOCHAT_INPUT_ARTIFACT", experiment.artifacts.pretrain.clone());
    let loaded = load_artifact::<B>(&input, device)
        .unwrap_or_else(|error| panic!("failed to load pretrain artifact {input:?}: {error}"));
    tracing::info!("Loaded {:?} artifact from {:?}", loaded.manifest.stage, input);
    let (mut model, tokenizer) = (loaded.model, loaded.tokenizer);
    let mut packer = SftPacker::new(&dataset, &tokenizer);

    let training_config = config.training.resolve(model.config.sequence_len)
        .unwrap_or_else(|error| panic!("invalid SFT training config: {error}"));
    let mut optimizer =
        MuonAdamW::with_kind(model.config.n_layer, training_config.optimizer);
    let (batch_size, max_seq_len) =
        (training_config.device_batch_size, training_config.sequence_length);
    let accumulation_steps = training_config.gradient_accumulation_steps();
    let bos_token = tokenizer.get_bos_token_id();
    let (warmup_steps, num_iterations, learning_rate, weight_decay) =
        (training_config.warmup_steps, training_config.num_iterations,
            training_config.learning_rate, training_config.weight_decay);
    tracing::info!("Starting SFT training loop for {} iterations...", num_iterations);
    let (start_time, mut smooth_loss) = (Instant::now(), 0.0);
    let output = path_from_env("NANOCHAT_OUTPUT_ARTIFACT", experiment.artifacts.sft.clone());
    reset_metrics(&output).unwrap_or_else(|error| panic!("failed to reset metrics: {error}"));

    for step in 1..=num_iterations {
        let step_start = Instant::now();
        let mut accumulator = burn::optim::GradientsAccumulator::new();
        let (mut loss_val, mut processed_tokens) = (0.0, 0usize);
        for _ in 0..accumulation_steps {
            if packer.conversations.is_empty() {
                packer = SftPacker::new(&dataset, &tokenizer);
            }
            let (rows, mask_rows, row_lengths) =
                packer.next_batch(batch_size, max_seq_len, bos_token);
            let actual_batch_size = rows.len();
            assert!(actual_batch_size > 0, "SFT dataset produced no trainable rows");
            let (flat_inputs, flat_targets) =
                flatten_sft_batch(&rows, &mask_rows, &row_lengths, max_seq_len);
            let inputs = int_tensor_2d(flat_inputs, [actual_batch_size, max_seq_len], device);
            let targets = int_tensor_2d(flat_targets, [actual_batch_size, max_seq_len], device);
            let loss = model.compute_loss(model.forward(inputs, None), targets) /
                accumulation_steps as f32;
            loss_val += scalar_to_f32(loss.clone().into_scalar());
            processed_tokens += actual_batch_size * max_seq_len;
            let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        }

        let schedule_step = step - 1;
        let lrm = get_lr_multiplier(schedule_step, num_iterations, warmup_steps,
            training_config.warmdown_ratio, training_config.final_lr_frac);
        let wd = get_weight_decay(schedule_step, num_iterations, weight_decay);
        let lr = learning_rate * lrm;
        let momentum = get_muon_momentum(
            schedule_step, num_iterations, training_config.warmdown_ratio);

        optimizer.step(&mut model, &accumulator.grads(), lr, step, wd, momentum);

        smooth_loss = if step == 1 { loss_val } else { 0.9 * smooth_loss + 0.1 * loss_val };

        if step % config.log_interval == 0 || step == num_iterations {
            tracing::info!("Step {:03}/{:03} | lr: {:.6} | Loss: {:.4} (smooth: {:.4})",
                step, num_iterations, lr, loss_val, smooth_loss);
        }
        append_metric(&output, &MetricRecord { stage: TrainingStage::Sft, step, loss: loss_val,
            smoothed_loss: Some(smooth_loss), learning_rate: Some(lr), reward: None,
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            bpb: None,
            tokens_per_second: Some(processed_tokens as f32 /
                step_start.elapsed().as_secs_f32().max(f32::EPSILON)),
            memory_bytes: crate::common::process_memory_bytes(),
            quality: None, kl: None, clip_fraction: None,
            response_length: None, acceptance_rate: None,
        }).unwrap_or_else(|error| panic!("failed to append SFT metric: {error}"));
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   SFT Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    save_artifact(&output, TrainingStage::Sft, &model, &tokenizer, Some(&training_config))
        .unwrap_or_else(|error| panic!("failed to save SFT artifact: {error}"));
    save_experiment_config(&output, experiment)
        .unwrap_or_else(|error| panic!("failed to save experiment config: {error}"));
    tracing::info!("SFT artifact saved to {:?}", output);
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_sft_packer() {
        use crate::tokenizer::{Conversation, ConversationMessage, MessageContent};
        let dataset = SftDataset { conversations: vec![Conversation { messages: vec![
            ConversationMessage { role: "user".to_string(),
                content: MessageContent::Simple("Who are you?".to_string()) },
            ConversationMessage { role: "assistant".to_string(),
                content: MessageContent::Simple("I am nanochat.".to_string()) },
        ]}] };
        let corpus = vec!["Who are you? I am nanochat.", "Hello!"];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 512);
        let mut packer = SftPacker::new(&dataset, &tokenizer);

        let (batch_size, max_seq_len) = (2, 32);
        let bos_token = tokenizer.get_bos_token_id();

        let (rows, mask_rows, row_lengths) = packer.next_batch(batch_size, max_seq_len, bos_token);
        assert_eq!(rows.len(), 1);
        assert_eq!(mask_rows.len(), rows.len());
        assert_eq!(row_lengths.len(), rows.len());

        assert_eq!(rows[0].len(), max_seq_len + 1);

        let mut oversized = SftPacker { conversations: vec![(vec![1; 64], vec![1; 64])] };
        let (rows, masks, _) = oversized.next_batch(1, 16, bos_token);
        assert_eq!(rows[0].len(), 17);
        assert_eq!(masks[0].len(), 17);
        assert!(oversized.conversations.is_empty());
    }
}
