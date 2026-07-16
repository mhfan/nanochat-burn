
use std::{fs, io::Write, path::Path, time::Instant};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use serde::{Deserialize, Serialize};

use crate::{artifact::{MetricRecord, TrainingStage, append_metric, copy_metrics_through,
        load_artifact, load_resume_state, path_from_env, reset_metrics, save_artifact,
        save_experiment_config, save_resume_state, set_rollouts_file},
    common::{extract_answer, int_tensor_2d, scalar_to_f32}, dataset::SftDataset,
    engine::{TrainerState, get_muon_momentum,
        inference::{GenerationConfig, InferenceEngine, SamplingConfig, SamplingRng}},
    experiment::{ExperimentConfig, RlAlgorithm}, optim::MuonAdamW,
};

const ROLLOUTS_FILE: &str = "rollouts.jsonl";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutRecord {
    pub training_step: usize,
    pub policy_version: usize,
    pub tokens: Vec<usize>,
    pub sampled_mask: Vec<u8>,
    pub old_token_log_probs: Vec<f32>,
    pub reward: f32,
    pub advantage: f32,
}

fn group_normalized_advantages(rewards: &[f32]) -> Vec<f32> {
    assert!(!rewards.is_empty(), "cannot calculate advantages for an empty reward set");
    let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
    let variance =
        rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32;
    let std_dev = variance.sqrt();
    rewards.iter().map(|&r| (r - mean) / (std_dev + 1e-4)).collect()
}

fn append_rollouts(path: &Path, records: &[RolloutRecord]) -> Result<(), String> {
    let path = path.join(ROLLOUTS_FILE);
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)
        .map_err(|error| format!("failed to open {path:?}: {error}"))?;
    for record in records {
        serde_json::to_writer(&mut file, record)
            .map_err(|error| format!("failed to serialize rollout: {error}"))?;
        file.write_all(b"\n").map_err(|error| format!("failed to write rollout: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
fn clipped_surrogate(ratio: f32, advantage: f32, epsilon: f32) -> f32 {
    (ratio * advantage).min(ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage)
}

fn flatten_rollouts(rollouts: &[Vec<usize>], masks: &[Vec<u8>], max_len: usize,
    pad_token: usize) -> (Vec<i32>, Vec<i32>) {
    assert_eq!(rollouts.len(), masks.len(), "rollout/mask count mismatch");
    assert!(max_len > 1, "rollout length must be greater than one");
    let row_len = max_len - 1;
    let mut flat_inputs = Vec::with_capacity(rollouts.len() * row_len);
    let mut flat_targets = Vec::with_capacity(rollouts.len() * row_len);

    for (rollout, mask) in rollouts.iter().zip(masks) {
        assert_eq!(rollout.len(), mask.len(), "rollout/token-mask size mismatch");
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

fn fit_rollout_prompt(mut tokens: Vec<usize>, sequence_len: usize,
    generation_tokens: usize) -> Vec<usize> {
    assert!(sequence_len > 1, "RL requires at least two context positions");
    let generation_tokens = generation_tokens.min(sequence_len - 1);
    let max_prompt = sequence_len - generation_tokens;
    if tokens.len() > max_prompt { tokens.drain(..tokens.len() - max_prompt); }
    tokens
}

pub fn run_rl_training<B: AutodiffBackend>(device: &B::Device,
    experiment: &ExperimentConfig) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Reinforcement Learning (RL)      ");
    tracing::info!("=============================================");

    experiment.validate().unwrap_or_else(|error| panic!("invalid experiment config: {error}"));
    let config = &experiment.rl;
    if !config.dataset_path.exists() {
        tracing::error!(
            "GSM8K RL dataset not found! Please run synthetic dataset generator first."
        );
        return;
    }

    let dataset = SftDataset::new(&config.dataset_path).expect("Failed to load GSM8K RL dataset");
    if dataset.conversations.is_empty() {
        tracing::error!("GSM8K RL dataset is empty");
        return;
    }
    tracing::info!("Loaded RL dataset with {} questions", dataset.conversations.len());

    let input = path_from_env("NANOCHAT_INPUT_ARTIFACT", experiment.artifacts.sft.clone());
    let base = load_artifact::<B>(&input, device)
        .unwrap_or_else(|error| panic!("failed to load SFT artifact {input:?}: {error}"));
    assert_eq!(base.manifest.stage, TrainingStage::Sft,
        "RL reference model must come from an SFT artifact");
    tracing::info!("Loaded {:?} artifact from {:?}", base.manifest.stage, input);
    let reference_model = base.model.clone();
    let resume = std::env::var_os("NANOCHAT_RESUME_ARTIFACT").map(std::path::PathBuf::from);
    let output = std::env::var_os("NANOCHAT_OUTPUT_ARTIFACT").map(std::path::PathBuf::from)
        .or_else(|| resume.clone()).unwrap_or_else(|| experiment.artifacts.rl.clone());
    let (mut model, tokenizer, mut optimizer, mut step, mut sampling_rng,
        elapsed_before_resume) =
        if let Some(path) = resume.as_deref() {
            let resumed = load_artifact::<B>(path, device)
                .unwrap_or_else(|error| panic!("failed to load RL artifact {path:?}: {error}"));
            assert_eq!(resumed.manifest.stage, TrainingStage::ReinforcementLearning,
                "RL can only resume an RL artifact");
            let (optimizer, state) = load_resume_state::<B>(
                path, resumed.model.config.n_layer, device)
                .unwrap_or_else(|error| panic!("failed to load RL state: {error}"));
            let rng_state = state.rng_state.expect("RL state is missing sampling RNG state");
            if path != output {
                copy_metrics_through(path, &output, state.step)
                    .unwrap_or_else(|error| panic!("failed to copy RL metrics: {error}"));
                let source_rollouts = path.join(ROLLOUTS_FILE);
                if source_rollouts.is_file() {
                    fs::create_dir_all(&output).unwrap();
                    fs::copy(source_rollouts, output.join(ROLLOUTS_FILE)).unwrap();
                }
            }
            (resumed.model, resumed.tokenizer, optimizer, state.step,
                SamplingRng::new(rng_state), state.total_training_time_secs)
        } else {
            reset_metrics(&output)
                .unwrap_or_else(|error| panic!("failed to reset metrics: {error}"));
            match fs::remove_file(output.join(ROLLOUTS_FILE)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => panic!("failed to reset rollout records: {error}"),
            }
            let optimizer = MuonAdamW::new(base.model.config.n_layer);
            (base.model, base.tokenizer, optimizer, 0, SamplingRng::new(experiment.seed), 0.0)
        };
    let assistant_end = tokenizer.special_token_ids().assistant_end;

    let (num_steps, learning_rate) = (config.num_steps, config.learning_rate);
    let (batch_size, num_samples) = (config.batch_size, config.num_samples);

    tracing::info!("Starting RL training loop for {} steps...", num_steps);
    let start_time = Instant::now();

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
            let conv_idx = ((step - 1) * batch_size + q_idx) % dataset.conversations.len();
            let conversation = &dataset.conversations[conv_idx];

            // Render prompt for completion (keeps assistant start)
            let prompt_tokens = fit_rollout_prompt(tokenizer.render_for_completion(conversation),
                model.config.sequence_len, config.max_generation_tokens);
            let prompt_len = prompt_tokens.len();

            // Sample rollouts (returns both results and precise masks)
            tracing::info!("  Rollout for question {}/{} (conv {}), prompt len = {}...",
                q_idx + 1, batch_size, conv_idx, prompt_len);
            let generation = GenerationConfig { max_tokens: config.max_generation_tokens,
                sampling: SamplingConfig {
                    temperature: config.temperature, top_k: config.top_k,
                    repetition_penalty: config.repetition_penalty,
                },
                seed: sampling_rng.next_u64(),
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

            let advantages = group_normalized_advantages(&question_rewards);

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

        let target_values = flat_targets.clone();
        let inputs_tensor = int_tensor_2d(flat_inputs, [num_sequences, max_len - 1], device);
        let targets_tensor = int_tensor_2d(flat_targets, [num_sequences, max_len - 1], device);
        let flat_adv_seq: Vec<f32> = all_advantages.iter()
            .flat_map(|&adv| std::iter::repeat_n(adv, max_len - 1)).collect();
        let advantages_tensor = Tensor::<B, 1>::from_data(flat_adv_seq.as_slice(), device)
            .reshape([num_sequences, max_len - 1]);
        let valid_mask = targets_tensor.clone().not_equal_elem(-1).float();
        let num_valid_val = scalar_to_f32(valid_mask.clone().sum().into_scalar()).max(1.0);

        let old_log_probs = model.compute_unreduced_loss(
            model.forward(inputs_tensor.clone(), None), targets_tensor.clone())
            .reshape([num_sequences, max_len - 1]) * -1.0;
        let old_values = crate::common::tensor_data_to_f32_vec(
            old_log_probs.clone().into_data());
        let old_log_probs = Tensor::<B, 2>::from_data(old_log_probs.into_data(), device);

        let reference_log_probs = reference_model.compute_unreduced_loss(
            reference_model.forward(inputs_tensor.clone(), None), targets_tensor.clone())
            .reshape([num_sequences, max_len - 1]) * -1.0;
        let reference_values = crate::common::tensor_data_to_f32_vec(
            reference_log_probs.clone().into_data());
        let reference_log_probs =
            Tensor::<B, 2>::from_data(reference_log_probs.into_data(), device);

        let records = all_rollouts.iter().zip(&all_masks).zip(&all_rewards)
            .zip(&all_advantages).enumerate().map(
                |(index, (((tokens, sampled_mask), &reward), &advantage))| {
                    let start = index * (max_len - 1);
                    RolloutRecord { training_step: step,
                        policy_version: (step - 1) * config.update_epochs,
                        tokens: tokens.clone(), sampled_mask: sampled_mask.clone(),
                        old_token_log_probs: old_values[start..start + max_len - 1].to_vec(),
                        reward, advantage }
                }).collect::<Vec<_>>();
        append_rollouts(&output, &records)
            .unwrap_or_else(|error| panic!("failed to append rollout records: {error}"));

        let lrm = 1.0 - ((step - 1) as f32 / num_steps as f32);
        let lr = learning_rate * lrm;
        let (mut loss_val, mut avg_kl, mut clip_fraction) = (0.0, 0.0, 0.0);
        for epoch in 0..config.update_epochs {
            tracing::info!("  Running {:?} update epoch {}/{}...", config.algorithm,
                epoch + 1, config.update_epochs);
            let current_log_probs = model.compute_unreduced_loss(
                model.forward(inputs_tensor.clone(), None), targets_tensor.clone())
                .reshape([num_sequences, max_len - 1]) * -1.0;
            let ratio = (current_log_probs.clone() - old_log_probs.clone()).exp();
            let policy_objective = match config.algorithm {
                RlAlgorithm::GroupNormalizedReinforce =>
                    current_log_probs.clone() * advantages_tensor.clone(),
                RlAlgorithm::Grpo => {
                    let unclipped = ratio.clone() * advantages_tensor.clone();
                    let clipped = ratio.clone().clamp(
                        1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) *
                        advantages_tensor.clone();
                    Tensor::cat(vec![unclipped.unsqueeze_dim::<3>(2),
                        clipped.unsqueeze_dim::<3>(2)], 2)
                        .min_dim(2).reshape([num_sequences, max_len - 1])
                }
            };
            let log_ratio = reference_log_probs.clone() - current_log_probs.clone();
            let kl = log_ratio.clone().exp() - log_ratio - 1.0;
            let loss = (policy_objective * valid_mask.clone()).sum() *
                (-1.0 / num_valid_val) +
                (kl.clone() * valid_mask.clone()).sum() *
                    (config.kl_coefficient / num_valid_val);
            let current_values = crate::common::tensor_data_to_f32_vec(
                current_log_probs.clone().into_data());
            let ratios = crate::common::tensor_data_to_f32_vec(ratio.into_data());
            let valid_indices = target_values.iter().enumerate()
                .filter_map(|(index, &target)| (target >= 0).then_some(index))
                .collect::<Vec<_>>();
            avg_kl = valid_indices.iter().map(|&index| {
                let delta = reference_values[index] - current_values[index];
                delta.exp() - delta - 1.0
            }).sum::<f32>() / num_valid_val;
            clip_fraction = valid_indices.iter().filter(|&&index|
                (ratios[index] - 1.0).abs() > config.clip_epsilon).count() as f32 /
                num_valid_val;

            let grads = loss.backward();
            let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
            let optimizer_step = (step - 1) * config.update_epochs + epoch + 1;
            let total_optimizer_steps = num_steps * config.update_epochs;
            let momentum = get_muon_momentum(
                optimizer_step - 1, total_optimizer_steps, 1.0);
            optimizer.step(&mut model, &grads_params, lr, optimizer_step, 0.0, momentum);
            loss_val = scalar_to_f32(loss.into_scalar());
        }
        let avg_reward = all_rewards.iter().sum::<f32>() / (all_rewards.len() as f32);
        let response_length = all_masks.iter().map(|mask|
            mask.iter().filter(|&&sampled| sampled == 1).count()).sum::<usize>() as f32 /
            num_sequences as f32;

        tracing::info!("Step {:02}/{:02} | Loss: {:.6} | Reward: {:.2}% | KL: {:.6} | Clip: {:.2}%",
            step, num_steps, loss_val, avg_reward * 100.0, avg_kl, clip_fraction * 100.0);
        append_metric(&output, &MetricRecord {
            stage: TrainingStage::ReinforcementLearning, step, loss: loss_val,
            smoothed_loss: None, learning_rate: Some(lr), reward: Some(avg_reward),
            elapsed_secs: elapsed_before_resume + start_time.elapsed().as_secs_f64(),
            bpb: None, tokens_per_second: None,
            memory_bytes: crate::common::process_memory_bytes(), quality: None,
            kl: Some(avg_kl), clip_fraction: Some(clip_fraction),
            response_length: Some(response_length), acceptance_rate: None,
        }).unwrap_or_else(|error| panic!("failed to append RL metric: {error}"));
        save_rl_checkpoint(&output, &model, &tokenizer, &optimizer, step,
            sampling_rng.state(), elapsed_before_resume + start_time.elapsed().as_secs_f64(),
            experiment);
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   RL Training Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    save_rl_checkpoint(&output, &model, &tokenizer, &optimizer, step,
        sampling_rng.state(), elapsed_before_resume + elapsed.as_secs_f64(), experiment);
    tracing::info!("RL artifact saved to {:?}", output);
}

fn save_rl_checkpoint<B: AutodiffBackend>(output: &Path, model: &crate::gpt::Gpt<B>,
    tokenizer: &crate::tokenizer::BpeTokenizer, optimizer: &MuonAdamW<B>, step: usize,
    rng_state: u64, elapsed_secs: f64, experiment: &ExperimentConfig) {
    save_artifact(output, TrainingStage::ReinforcementLearning, model, tokenizer, None)
        .unwrap_or_else(|error| panic!("failed to save RL artifact: {error}"));
    let state = TrainerState { step, smooth_train_loss: 0.0,
        total_training_time_secs: elapsed_secs, dataloader_position: None,
        rng_state: Some(rng_state) };
    save_resume_state(output, optimizer, &state)
        .unwrap_or_else(|error| panic!("failed to save RL resume state: {error}"));
    set_rollouts_file(output, ROLLOUTS_FILE)
        .unwrap_or_else(|error| panic!("failed to register rollout records: {error}"));
    save_experiment_config(output, experiment)
        .unwrap_or_else(|error| panic!("failed to save experiment config: {error}"));
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_group_advantages_and_clipped_objective() {
        let advantages = group_normalized_advantages(&[0.0, 1.0, 2.0]);
        assert!(advantages[0] < 0.0 && advantages[2] > 0.0);
        assert!(advantages.iter().sum::<f32>().abs() < 1e-5);
        assert_eq!(clipped_surrogate(1.5, 2.0, 0.2), 2.4);
        assert_eq!(clipped_surrogate(0.5, -2.0, 0.2), -1.6);
        assert_eq!(fit_rollout_prompt((0..10).collect(), 8, 3), vec![5, 6, 7, 8, 9]);
    }
}
