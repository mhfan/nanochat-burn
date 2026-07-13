
use std::collections::VecDeque;
use burn::tensor::{Tensor, backend::Backend};

use crate::{common::int_tensor_2d, engine::calculator::use_calculator,
    gpt::{ForwardLayer, Gpt, KVCache}, tokenizer::BpeTokenizer,
};

#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}

impl SamplingConfig {
    pub const fn greedy() -> Self {
        Self { temperature: 0.0, top_k: None, repetition_penalty: 1.0 }
    }

    pub(crate) fn validate(self) {
        assert!(self.temperature.is_finite() && self.temperature >= 0.0,
            "temperature must be finite and non-negative");
        assert!(self.top_k != Some(0), "top_k must be greater than zero");
        assert!(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "repetition penalty must be finite and positive");
    }
}

impl Default for SamplingConfig {
    fn default() -> Self { Self { temperature: 1.0, top_k: None, repetition_penalty: 1.0 } }
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self { Self { max_tokens: 128, sampling: SamplingConfig::default() } }
}

/// Tracks self-regressive token generation state per sample
#[derive(Clone)]
pub struct GeneratorState<B: Backend> {
    pub cache: KVCache<B>,
    pub current_tokens: Vec<Vec<usize>>,
    pub forced_tokens: Vec<VecDeque<usize>>,
    pub in_python_block: Vec<bool>,
    pub python_expr_tokens: Vec<Vec<usize>>,
    pub completed: Vec<bool>,
    pub step: usize,
}

/// High-performance self-regressive inference engine
pub struct InferenceEngine<B: Backend, L: ForwardLayer<B> = burn::nn::Linear<B>> {
    pub model: Gpt<B, L>,
    pub tokenizer: BpeTokenizer,
}

impl<B: Backend, L: ForwardLayer<B>> InferenceEngine<B, L> {
    pub fn new(model: Gpt<B, L>, tokenizer: BpeTokenizer) -> Self { Self { model, tokenizer } }

    pub(crate) fn record_generated_token(&self, state: &mut GeneratorState<B>, sample: usize,
        token: usize) {
        let special = self.tokenizer.special_token_ids();
        state.current_tokens[sample].push(token);
        if token == special.assistant_end || token == special.bos {
            state.completed[sample] = true;
        }

        if token == special.python_start {
            state.in_python_block[sample] = true;
            state.python_expr_tokens[sample].clear();
        } else if token == special.python_end && state.in_python_block[sample] {
            state.in_python_block[sample] = false;
            let expression = self.tokenizer.decode(&state.python_expr_tokens[sample]);
            if let Some(result) = use_calculator(&expression) {
                let forced = &mut state.forced_tokens[sample];
                forced.push_back(special.output_start);
                forced.extend(self.tokenizer.encode_ordinary(&result));
                forced.push_back(special.output_end);
            }
        } else if state.in_python_block[sample] {
            state.python_expr_tokens[sample].push(token);
        }
    }

    /// Run prefill phase over the prompt sequence across all batch items
    pub fn prefill(&self, prompt_tokens: &[usize], num_samples: usize,
        device: &B::Device) -> (GeneratorState<B>, Tensor<B, 2>) {
        let prompt_len = prompt_tokens.len();
        assert!(prompt_len > 0, "prompt must contain at least one token");
        assert!(num_samples > 0, "num_samples must be greater than zero");
        assert!(prompt_len <= self.model.config.sequence_len,
            "prompt length exceeds model sequence length");
        assert!(prompt_tokens.iter().all(|&token| token < self.model.config.vocab_size &&
            i32::try_from(token).is_ok()), "prompt contains an invalid token ID");
        let batch_idx_data: Vec<i32> = std::iter::repeat_n(prompt_tokens, num_samples)
            .flatten().map(|&t| t as i32).collect();

        let idx = int_tensor_2d(batch_idx_data, [num_samples, prompt_len], device);

        let head_dim = self.model.config.n_embd / self.model.config.n_head;
        let mut cache = KVCache::new_allocated(
            self.model.config.n_layer, num_samples, self.model.config.sequence_len,
            self.model.config.n_kv_head, head_dim, device,
        );
        let logits_3d = self.model.forward_with_cache(idx, &mut cache, 0);

        // Extract the logits at the last token position
        let last_logits = logits_3d.slice([
                0..num_samples, (prompt_len - 1)..prompt_len,
                0..self.model.config.vocab_size,
            ]).reshape([num_samples, self.model.config.vocab_size]);

        let state = GeneratorState { cache,
            current_tokens: vec![prompt_tokens.to_vec(); num_samples],
            forced_tokens: vec![VecDeque::new(); num_samples],
            in_python_block: vec![false; num_samples],
            python_expr_tokens: vec![Vec::new(); num_samples],
            completed: vec![false; num_samples],
            step: prompt_len,
        };

        (state, last_logits)
    }

    /// Perform a single self-regressive token generation step
    pub fn step_generation(&self, state: &mut GeneratorState<B>, logits: Tensor<B, 2>,
        sampling: SamplingConfig, device: &B::Device) -> (Vec<usize>, Vec<u8>, Tensor<B, 2>) {
        let num_samples = state.current_tokens.len();
        assert!(num_samples > 0, "generation state must contain at least one sample");
        assert_eq!(state.forced_tokens.len(), num_samples, "forced-token batch size mismatch");
        assert_eq!(state.in_python_block.len(), num_samples, "tool-state batch size mismatch");
        assert_eq!(state.python_expr_tokens.len(), num_samples,
            "tool-expression batch size mismatch");
        assert_eq!(state.completed.len(), num_samples, "completion-state batch size mismatch");
        assert!(state.step < self.model.config.sequence_len,
            "generation exceeded model sequence length");

        let special_tokens = self.tokenizer.special_token_ids();

        // Sample candidate tokens
        let sampled_tokens = sample_next_token(logits, sampling, &state.current_tokens);

        let mut next_token_column = Vec::with_capacity(num_samples);
        let mut is_sampled_mask = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            if state.completed[i] {
                next_token_column.push(special_tokens.bos);
                is_sampled_mask.push(0);
                continue;
            }

            let next_tok_opt = state.forced_tokens[i].pop_front();
            let next_tok = next_tok_opt.unwrap_or(sampled_tokens[i]);

            next_token_column.push(next_tok);
            is_sampled_mask.push(if next_tok_opt.is_some() { 0 } else { 1 });
            self.record_generated_token(state, i, next_tok);
        }

        let next_idx = int_tensor_2d(
            next_token_column.iter().map(|&token| token as i32).collect(),
            [num_samples, 1], device);

        let next_logits_3d =
            self.model.forward_with_cache(next_idx, &mut state.cache, state.step);
        state.step += 1;

        let next_logits = next_logits_3d
            .slice([0..num_samples, 0..1, 0..self.model.config.vocab_size])
            .reshape([num_samples, self.model.config.vocab_size]);

        (next_token_column, is_sampled_mask, next_logits)
    }

    /// Non-streaming batch generation interface returning (results, masks)
    pub fn generate_batch(&self, prompt_tokens: &[usize], num_samples: usize,
        config: GenerationConfig, device: &B::Device) -> (Vec<Vec<usize>>, Vec<Vec<u8>>) {
        config.sampling.validate();
        let (mut state, mut cur_logits) = self.prefill(prompt_tokens, num_samples, device);

        let special_tokens = self.tokenizer.special_token_ids();

        let mut results = vec![prompt_tokens.to_vec(); num_samples];
        let mut masks = vec![vec![0; prompt_tokens.len()]; num_samples];
        let mut completed = vec![false; num_samples];

        let available_tokens = self.model.config.sequence_len.saturating_sub(prompt_tokens.len());
        let max_tokens = config.max_tokens.min(available_tokens);
        for step_idx in 0..max_tokens {
            if completed.iter().all(|&c| c) { break; }
            if step_idx > 0 && step_idx % 20 == 0 {
                tracing::info!("    Generated {}/{} tokens...", step_idx, max_tokens);
            }
            let (token_column, token_masks, next_logits) =
                self.step_generation(&mut state, cur_logits, config.sampling, device);
            cur_logits = next_logits;

            for (i, (&token, &mask)) in token_column.iter().zip(&token_masks).enumerate() {
                if completed[i] { continue; }
                if token == special_tokens.assistant_end || token == special_tokens.bos {
                    completed[i] = true;
                } else {
                    results[i].push(token);
                    masks[i].push(mask);
                }
            }
        }

        (results, masks)
    }
}

/// Dynamic sample selection based on temperature, top_k, and repetition penalties
pub fn sample_next_token<B: Backend>(logits: Tensor<B, 2>, sampling: SamplingConfig,
    generated_tokens: &[Vec<usize>]) -> Vec<usize> {
    sampling.validate();
    let [batch_size, vocab_size] = logits.shape().dims();
    assert!(vocab_size > 0, "cannot sample from an empty vocabulary");
    assert_eq!(generated_tokens.len(), batch_size, "token history batch size mismatch");

    let logits_vec = crate::common::tensor_data_to_f32_vec(logits.into_data());
    let mut sampled_ids = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let start = b * vocab_size;
        let mut sample_logits = logits_vec[start..start + vocab_size].to_vec();

        // 1. Repetition penalty
        if sampling.repetition_penalty != 1.0 {
            let unique_history: std::collections::HashSet<_> =
                generated_tokens[b].iter().copied().collect();
            for &t in &unique_history {
                if t < vocab_size {
                    let val = sample_logits[t];
                    sample_logits[t] = if val > 0.0 {
                        val / sampling.repetition_penalty
                    } else {
                        val * sampling.repetition_penalty
                    };
                }
            }
        }

        // 2. Argmax if temperature is 0
        if sampling.temperature == 0.0 {
            let (mut max_val, mut max_idx) = (sample_logits[0], 0);
            for (idx, &val) in sample_logits.iter().enumerate() {
                if val.total_cmp(&max_val).is_gt() { (max_val, max_idx) = (val, idx); }
            }
            sampled_ids.push(max_idx);
            continue;
        }

        // 3. Scale by temperature
        for val in sample_logits.iter_mut() { *val /= sampling.temperature; }

        // 4. Top-K filtering
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        if let Some(k) = sampling.top_k {
            let k = k.min(vocab_size);
            if k < vocab_size {
                indices.select_nth_unstable_by(k,
                    |&i, &j| sample_logits[j].total_cmp(&sample_logits[i]));
                for &idx in &indices[k..] { sample_logits[idx] = -1e9; }
            }
        }

        // 5. Stable Softmax & Multinomial sampling
        let max_logit = sample_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_logits: Vec<f32> =
            sample_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        assert!(sum_exp.is_finite() && sum_exp > 0.0, "sampling probabilities are invalid");
        for v in exp_logits.iter_mut() { *v /= sum_exp; }

        let (mut cum_sum, mut chosen_idx) = (0.0f32,
            indices.iter().copied().find(|&idx| exp_logits[idx] > 0.0).unwrap_or(indices[0]));
        let r: f32 = rand::random();
        for &idx in &indices {
            cum_sum += exp_logits[idx];
            if r < cum_sum {
                chosen_idx = idx;
                break;
            }
        }
        sampled_ids.push(chosen_idx);
    }

    sampled_ids
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_inference_engine_instantiation() {
        let device = crate::common::init_device();
        let corpus = vec!["Interactive chat agent with Tool-Use integration."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        let config = crate::gpt::GptConfig { sequence_len: 8, n_layer: 1, n_head: 2,
            n_kv_head: 1, n_embd: 32, quantization: None,
            window_pattern: "L".to_string(), vocab_size: tokenizer.get_vocab_size(),
        };

        use crate::common::ModelBackend;
        let gpt: Gpt<ModelBackend> = Gpt::new(config.clone(), &device);
        let engine = InferenceEngine::new(gpt, tokenizer);

        let prompt_tokens = vec![1, 2, 3];
        let (state, logits) = engine.prefill(&prompt_tokens, 2, &device);

        assert_eq!(state.current_tokens[0], prompt_tokens);
        let dims: [usize; 2] = logits.shape().dims();
        assert_eq!(dims, [2, engine.model.config.vocab_size]);
    }

    #[test] fn test_inference_step_generation() {
        let device = crate::common::init_device();
        let corpus = vec!["Interactive chat agent with Tool-Use integration."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        let config = crate::gpt::GptConfig { sequence_len: 32, n_layer: 1, n_head: 2,
            n_kv_head: 1, n_embd: 32, window_pattern: "L".to_string(),
            vocab_size: tokenizer.get_vocab_size(), quantization: None,
        };

        use crate::common::ModelBackend;
        let gpt: Gpt<ModelBackend> = Gpt::new(config.clone(), &device);
        let engine = InferenceEngine::new(gpt, tokenizer);

        let prompt_tokens = vec![1, 2, 3];
        let config = GenerationConfig { max_tokens: 5,
            sampling: SamplingConfig {
                temperature: 1.0, top_k: Some(5), repetition_penalty: 1.0,
            },
        };
        let (results, masks) = engine.generate_batch(&prompt_tokens, 2, config, &device);

        assert_eq!(results.len(), 2);
        assert_eq!(masks.len(), 2);
        assert!(results[0].len() >= 3);
    }

    #[test] fn test_tool_state_queues_calculator_output() {
        let device = crate::common::init_device();
        let tokenizer = BpeTokenizer::train_from_iterator(["2 + 2"], 280);
        let config = crate::gpt::GptConfig { sequence_len: 16, n_layer: 1, n_head: 2,
            n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(),
            vocab_size: tokenizer.get_vocab_size(), quantization: None,
        };
        let model = Gpt::<crate::common::ModelBackend>::new(config, &device);
        let engine = InferenceEngine::new(model, tokenizer);
        let (mut state, _) = engine.prefill(&[1], 1, &device);
        let special = engine.tokenizer.special_token_ids();

        engine.record_generated_token(&mut state, 0, special.python_start);
        for token in engine.tokenizer.encode_ordinary("2 + 2") {
            engine.record_generated_token(&mut state, 0, token);
        }
        engine.record_generated_token(&mut state, 0, special.python_end);

        assert_eq!(state.forced_tokens[0].front(), Some(&special.output_start));
        assert_eq!(state.forced_tokens[0].back(), Some(&special.output_end));
    }
}
