
use std::{path::Path, time::Instant};
use burn::tensor::{Tensor, backend::AutodiffBackend};

use crate::{common::{extract_answer, int_tensor_2d, scalar_to_f32}, dataset::SftDataset,
    engine::inference::{GenerationConfig, InferenceEngine, SamplingConfig},
    gpt::{Gpt, GptConfig}, optim::MuonAdamW, tokenizer::BpeTokenizer,
};

fn grpo_advantages(rewards: &[f32]) -> Vec<f32> {
    assert!(!rewards.is_empty(), "cannot calculate advantages for an empty reward set");
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance =
        rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32;
    let std_dev = variance.sqrt();
    rewards.iter().map(|&r| (r - mean) / (std_dev + 1e-4)).collect()
}

fn flatten_rollouts(rollouts: &[Vec<usize>], masks: &[Vec<u8>], max_len: usize,
    pad_token: usize) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(rollouts.len(), masks.len(), "rollout/mask count mismatch");
    assert!(max_len > 1, "rollout length must be greater than one");
    let row_len = max_len - 1;
    let mut flat_inputs = Vec::with_capacity(rollouts.len() * row_len);
    let mut flat_targets = Vec::with_capacity(rollouts.len() * row_len);

    for (rollout, mask) in rollouts.iter().zip(masks) {
        flat_inputs.extend((0..row_len).map(|idx|
            rollout.get(idx).copied().unwrap_or(pad_token) as i32));
        flat_targets.extend((1..max_len).map(|j| {
            if mask.get(j).copied().unwrap_or(0) == 1 {
                rollout.get(j).map(|&tok| tok as i32).unwrap_or(-1)
            } else { -1 }
        }));
    }

    (flat_inputs, flat_targets)
}

pub fn run_rl_training<B: AutodiffBackend>(device: &B::Device) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Reinforcement Learning (RL)      ");
    tracing::info!("=============================================");

    let rl_dataset_path = "data/eval/gsm8k.jsonl";
    if !Path::new(rl_dataset_path).exists() {
        tracing::error!(
            "GSM8K RL dataset not found! Please run synthetic dataset generator first."
        );
        return;
    }

    let dataset = SftDataset::new(rl_dataset_path).expect("Failed to load GSM8K RL dataset");
    tracing::info!("Loaded RL dataset with {} questions", dataset.conversations.len());

    // 1. Train/load tokenizer
    let corpus = dataset.get_corpus();
    let mut tokenizer = BpeTokenizer::train_from_iterator(corpus, 1024);
    tokenizer.build_inverse_mappings();

    let assistant_end = tokenizer.special_token_ids().assistant_end;

    // 2. Initialize Model and Optimizer
    let config = GptConfig { sequence_len: 256, n_layer: 4, n_head: 4, n_kv_head: 2, n_embd: 64,
        window_pattern: "L".to_string(), vocab_size: tokenizer.get_vocab_size(),
        quantization: None,
    };

    let mut model: Gpt<B> = Gpt::new(config.clone(), device);
    let mut optimizer = MuonAdamW::new(model.config.n_layer);

    let (num_steps, learning_rate) = (10, 1e-3);
    let batch_size = 2; // number of questions per step
    let num_samples = 4; // number of rollouts per question

    tracing::info!("Starting RL training loop for {} steps...", num_steps);
    let (start_time, mut step) = (Instant::now(), 0);

    while step < num_steps {
        step += 1;

        // Collect rollouts
        let mut all_masks = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_rollouts = Vec::new();
        let mut all_advantages = Vec::new();

        // Run rollouts on batch_size questions
        let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

        for q_idx in 0..batch_size {
            let conv_idx = (step * batch_size + q_idx) % dataset.conversations.len();
            let conversation = &dataset.conversations[conv_idx];

            // Render prompt for completion (keeps assistant start)
            let prompt_tokens = tokenizer.render_for_completion(conversation);
            let prompt_len = prompt_tokens.len();

            // Sample rollouts (returns both results and precise masks)
            tracing::info!("  Rollout for question {}/{} (conv {}), prompt len = {}...",
                q_idx + 1, batch_size, conv_idx, prompt_len);
            let generation = GenerationConfig { max_tokens: 128,
                sampling: SamplingConfig {
                    temperature: 1.0, top_k: Some(50), repetition_penalty: 1.0,
                },
            };
            let (rollouts, masks) =
                inference_engine.generate_batch(&prompt_tokens, num_samples, generation, device);

            // Get ground truth answer from conversation
            let last_msg = conversation.messages.last().unwrap();
            let ground_truth_text = last_msg.content.to_string_content();
            let gt_ans = extract_answer(&ground_truth_text);

            let mut question_rewards = Vec::with_capacity(num_samples);
            let mut question_rollouts = Vec::with_capacity(num_samples);
            let mut question_masks = Vec::with_capacity(num_samples);

            for (r_idx, rollout) in rollouts.into_iter().enumerate() {
                let generated_tokens = &rollout[prompt_len..];
                let decoded_text = tokenizer.decode(generated_tokens);
                let pred_ans = extract_answer(&decoded_text);

                // Compute reward
                let reward =
                    if gt_ans.is_some() && pred_ans == gt_ans { 1.0f32 } else { 0.0f32 };
                question_rewards.push(reward);

                // Use the precise token-level mask returned from InferenceEngine
                let mask = masks[r_idx].clone();

                question_rollouts.push(rollout);
                question_masks.push(mask);
            }

            let advantages = grpo_advantages(&question_rewards);

            all_rollouts.extend(question_rollouts);
            all_rewards.extend(question_rewards);
            all_advantages.extend(advantages);
            all_masks.extend(question_masks);
        }

        // Collate and pad rollouts to static maximum context length to eliminate JIT compiles
        // during backprop
        let (max_len, num_sequences) = (model.config.sequence_len, all_rollouts.len());

        let (flat_inputs, flat_targets) =
            flatten_rollouts(&all_rollouts, &all_masks, max_len, assistant_end);

        let inputs_tensor = int_tensor_2d(flat_inputs, [num_sequences, max_len - 1], device);
        let targets_tensor = int_tensor_2d(flat_targets, [num_sequences, max_len - 1], device);

        // Forward and backward passes
        tracing::info!("  Running training forward pass...");
        let logits = model.forward(inputs_tensor, None);
        let unreduced = model.compute_unreduced_loss(logits, targets_tensor.clone());

        // Reshape unreduced to [num_sequences, max_len - 1]
        let unreduced_2d = unreduced.reshape([num_sequences, max_len - 1]);

        // Broadcast advantages to [num_sequences, max_len - 1] and multiply
        let flat_adv_seq: Vec<f32> = all_advantages.iter()
            .flat_map(|&adv| std::iter::repeat_n(adv, max_len - 1)).collect();
        let advantages_tensor = Tensor::<B, 1>::from_data(flat_adv_seq.as_slice(), device)
            .reshape([num_sequences, max_len - 1]);

        let num_valid_val = scalar_to_f32(
            targets_tensor.clone().not_equal_elem(-1).float().sum().into_scalar(),
        ).max(1.0);
        let loss = (unreduced_2d * advantages_tensor).sum() / num_valid_val;

        tracing::info!("  Running training backward pass...");
        let grads = loss.backward();

        // Update parameters
        let lrm = 1.0 - (step as f32 / num_steps as f32);
        let lr = learning_rate * lrm;
        tracing::info!("  Running optimizer update...");

        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        optimizer.step(&mut model, &grads_params, lr, step, 0.0);
        tracing::info!("  Optimizer update completed!");

        let loss_val = scalar_to_f32(loss.into_scalar());
        let avg_reward = all_rewards.iter().sum::<f32>() / (all_rewards.len() as f32);

        tracing::info!("Step {:02}/{:02} | Loss: {:.6} | Avg Reward: {:.2}%",
            step, num_steps, loss_val, avg_reward * 100.0);
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   RL Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    let checkpoint_path = Path::new("data/rl_checkpoint.safetensors");
    tracing::info!("Saving RL checkpoint to {:?}...", checkpoint_path);
    if let Err(e) = crate::checkpoint::save_gpt_to_safetensors(&model, checkpoint_path) {
        tracing::error!("Failed to save RL checkpoint: {}", e);
    } else {
        tracing::info!("RL checkpoint saved successfully!");
    }
}

//#[cfg(test)] mod tests { use super::*;
    #[ignore] #[test] fn test_rl_training_loop() {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(tracing::Level::INFO).finish();
        let _ = tracing::subscriber::set_global_default(subscriber);

        let device = crate::common::init_device();
        run_rl_training::<crate::common::ModelAutodiffBackend>(&device);
    }
//}
