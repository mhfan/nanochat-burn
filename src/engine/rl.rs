
use std::{path::Path, time::Instant};
use burn::{prelude::ToElement, tensor::{Tensor, Int, backend::AutodiffBackend}};
use crate::{gpt::{Gpt, GptConfig}, tokenizer::BpeTokenizer, dataset::SftDataset,
    optim::MuonAdamW, engine::inference::InferenceEngine,
};

fn extract_answer(text: &str) -> Option<i32> {
    // Find "#### " marker and parse the number following it
    let marker = "#### ";
    if let Some(idx) = text.rfind(marker) {
        let num_part = text[idx + marker.len()..].trim();
        // Remove commas or other non-digit chars
        let clean_num: String = num_part.chars().filter(|c| c.is_digit(10) || *c == '-').collect();
        clean_num.parse::<i32>().ok()
    } else { None }
}

pub fn run_rl_training<B: AutodiffBackend>(device: &B::Device) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Reinforcement Learning (RL)      ");
    tracing::info!("=============================================");

    let rl_dataset_path = "data/eval/gsm8k.jsonl";
    if !Path::new(rl_dataset_path).exists() {
        tracing::error!("GSM8K RL dataset not found! Please run synthetic dataset generator first.");
        return;
    }

    let dataset = SftDataset::new(rl_dataset_path).expect("Failed to load GSM8K RL dataset");
    tracing::info!("Loaded RL dataset with {} questions", dataset.conversations.len());

    // 1. Train/load tokenizer
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
    let mut tokenizer = BpeTokenizer::train_from_iterator(corpus, 1024);
    tokenizer.build_inverse_mappings();

    let output_start = *tokenizer.get_special_tokens().get("<|output_start|>").unwrap_or(&50259);
    let output_end = *tokenizer.get_special_tokens().get("<|output_end|>").unwrap_or(&50260);
    let assistant_end = *tokenizer.get_special_tokens().get("<|assistant_end|>").unwrap_or(&50256);

    // 2. Initialize Model and Optimizer
    let config = GptConfig { sequence_len: 256, n_layer: 4, n_head: 4, n_kv_head: 2,
        n_embd: 64, window_pattern: "L".to_string(), vocab_size: tokenizer.get_vocab_size(),
    };

    let mut model: Gpt<B> = Gpt::new(config.clone(), device);
    let mut optimizer = MuonAdamW::new(model.config.n_layer);

    let num_steps = 10;
    let batch_size = 2; // number of questions per step
    let num_samples = 4; // number of rollouts per question
    let learning_rate = 1e-5;

    tracing::info!("Starting RL training loop for {} steps...", num_steps);
    let start_time = Instant::now();
    let mut step = 0;

    while step < num_steps {
        step += 1;

        // Collect rollouts
        let mut all_rollouts = Vec::new();
        let mut all_masks = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_advantages = Vec::new();

        // Run rollouts on batch_size questions
        let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

        for q_idx in 0..batch_size {
            let conv_idx = (step * batch_size + q_idx) % dataset.conversations.len();
            let conversation = &dataset.conversations[conv_idx];

            // Render prompt for completion (keeps assistant start)
            let prompt_tokens = tokenizer.render_for_completion(conversation);
            let prompt_len = prompt_tokens.len();

            // Sample rollouts
            let rollouts = inference_engine.generate_batch(&prompt_tokens,
                num_samples, 128, 1.0, Some(50), 1.0, device,);

            // Get ground truth answer from conversation
            let last_msg = conversation.messages.last().unwrap();
            let ground_truth_text = match &last_msg.content {
                crate::tokenizer::MessageContent::Simple(s) => s.clone(),
                crate::tokenizer::MessageContent::Parts(parts) => {
                    parts.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join("")
                }
            };
            let gt_ans = extract_answer(&ground_truth_text);

            let mut question_rewards = Vec::with_capacity(num_samples);
            let mut question_rollouts = Vec::with_capacity(num_samples);
            let mut question_masks = Vec::with_capacity(num_samples);

            for rollout in rollouts {
                let generated_tokens = &rollout[prompt_len..];
                let decoded_text = tokenizer.decode(generated_tokens);
                let pred_ans = extract_answer(&decoded_text);

                // Compute reward
                let reward = if gt_ans.is_some() && pred_ans == gt_ans { 1.0f32 } else { 0.0f32 };
                question_rewards.push(reward);

                // Build target mask
                let mut mask = vec![0; rollout.len()];
                let mut in_output = false;
                for i in prompt_len..rollout.len() {
                    let t = rollout[i];
                    if t == output_start { in_output = true; }
                    if in_output { mask[i] = 0; } else { mask[i] = 1; }
                    if t == output_end { in_output = false; }
                }

                question_rollouts.push(rollout);
                question_masks.push(mask);
            }

            // Calculate advantages
            let mean_reward = question_rewards.iter().sum::<f32>() / (num_samples as f32);
            let advantages: Vec<f32> = question_rewards.iter().map(|&r| r - mean_reward).collect();

            all_rollouts.extend(question_rollouts);
            all_masks.extend(question_masks);
            all_rewards.extend(question_rewards);
            all_advantages.extend(advantages);
        }

        // Collate and pad rollouts to maximum length
        let max_len = all_rollouts.iter().map(|r| r.len()).max().unwrap_or(0);
        let num_sequences = all_rollouts.len();

        let mut flat_inputs = Vec::with_capacity(num_sequences * (max_len - 1));
        let mut flat_targets = Vec::with_capacity(num_sequences * (max_len - 1));

        for (i, rollout) in all_rollouts.iter().enumerate() {
            let mut padded_rollout = rollout.clone();
            let mut padded_mask = all_masks[i].clone();

            let pad_len = max_len - rollout.len();
            padded_rollout.extend(std::iter::repeat(assistant_end).take(pad_len));
            padded_mask.extend(std::iter::repeat(0).take(pad_len));

            // inputs are first max_len - 1
            for j in 0..(max_len - 1) { flat_inputs.push(padded_rollout[j] as i32); }

            // targets are shifted by 1
            for j in 1..max_len {
                let mask_val = padded_mask[j];
                let is_padding = j >= rollout.len();
                if mask_val == 0 || is_padding {
                    flat_targets.push(-1);
                } else {
                    flat_targets.push(padded_rollout[j] as i32);
                }
            }
        }

        let inputs_tensor = Tensor::<B, 1, Int>::from_data(flat_inputs.as_slice(), device)
            .reshape([num_sequences, max_len - 1]);
        let targets_tensor = Tensor::<B, 1, Int>::from_data(flat_targets.as_slice(), device)
            .reshape([num_sequences, max_len - 1]);

        // Forward and backward passes
        let logits = model.forward(inputs_tensor, None);
        let unreduced = model.compute_unreduced_loss(logits, targets_tensor.clone());

        // Reshape unreduced to [num_sequences, max_len - 1]
        let unreduced_2d = unreduced.reshape([num_sequences, max_len - 1]);

        // Broadcast advantages to [num_sequences, max_len - 1] and multiply
        let mut flat_adv_seq = Vec::with_capacity(num_sequences * (max_len - 1));
        for i in 0..num_sequences {
            let adv = all_advantages[i];
            flat_adv_seq.extend(std::iter::repeat(adv).take(max_len - 1));
        }
        let advantages_tensor = Tensor::<B, 1>::from_data(flat_adv_seq.as_slice(), device)
            .reshape([num_sequences, max_len - 1]);

        let pg_loss = (unreduced_2d * advantages_tensor).sum();
        let num_valid = targets_tensor.clone().not_equal_elem(-1).float().sum().clamp(1.0, 1e9);
        let loss = pg_loss / num_valid;

        let grads = loss.backward();

        // Update parameters
        let lrm = 1.0 - (step as f32 / num_steps as f32);
        let lr = learning_rate * lrm;
        optimizer.step(&mut model, &grads, lr, step, 0.0);

        let loss_val = loss.into_scalar().to_f32();
        let avg_reward = all_rewards.iter().sum::<f32>() / (all_rewards.len() as f32);

        tracing::info!(
            "Step {:02}/{:02} | Loss: {:.6} | Avg Reward: {:.2}%",
            step, num_steps, loss_val, avg_reward * 100.0
        );
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   RL Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");
}
