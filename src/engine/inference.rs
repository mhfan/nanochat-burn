
use std::collections::VecDeque;
use burn::tensor::{Tensor, TensorData, Shape, backend::Backend, Int};
use crate::{gpt::{Gpt, KVCache, ForwardLayer}, tokenizer::BpeTokenizer, engine::calculator::use_calculator};

/// Tracks self-regressive token generation state per sample
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

    /// Run prefill phase over the prompt sequence across all batch items
    pub fn prefill(&self, prompt_tokens: &[usize], num_samples: usize,
        device: &B::Device,) -> (GeneratorState<B>, Tensor<B, 2>) {
        let prompt_len = prompt_tokens.len();
        let mut batch_idx_data = Vec::with_capacity(num_samples * prompt_len);
        for _ in 0..num_samples { for &t in prompt_tokens { batch_idx_data.push(t as i32); } }

        let idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(batch_idx_data, Shape::new([num_samples, prompt_len])),
            device,
        );

        let head_dim = self.model.config.n_embd / self.model.config.n_head;
        let mut cache = KVCache::new_allocated(self.model.config.n_layer, num_samples,
            self.model.config.sequence_len, self.model.config.n_kv_head, head_dim, device,);
        let logits_3d = self.model.forward_with_cache(idx, &mut cache, 0);

        // Extract the logits at the last token position
        let last_logits = logits_3d
            .slice([0..num_samples, (prompt_len - 1)..prompt_len, 0..self.model.config.vocab_size])
            .reshape([num_samples, self.model.config.vocab_size]);

        let current_tokens = vec![prompt_tokens.to_vec(); num_samples];
        let forced_tokens = vec![VecDeque::new(); num_samples];
        let in_python_block = vec![false; num_samples];
        let python_expr_tokens = vec![Vec::new(); num_samples];
        let completed = vec![false; num_samples];

        let state = GeneratorState { cache, current_tokens, forced_tokens,
            in_python_block, python_expr_tokens, completed, step: prompt_len,
        };

        (state, last_logits)
    }

    /// Perform a single self-regressive token generation step
    pub fn step_generation(&self, state: &mut GeneratorState<B>, logits: Tensor<B, 2>,
        temperature: f32, top_k: Option<usize>, repetition_penalty: f32,
        device: &B::Device,) -> (Vec<usize>, Vec<u8>, Tensor<B, 2>) {
        let num_samples = state.current_tokens.len();

        let assistant_end = *self.tokenizer.get_special_tokens().get("<|assistant_end|>").unwrap_or(&50256);
        let bos = self.tokenizer.get_bos_token_id();

        // Sample candidate tokens
        let sampled_tokens = sample_next_token(logits, temperature, top_k,
            repetition_penalty, &state.current_tokens,);

        let python_start = *self.tokenizer.get_special_tokens().get("<|python_start|>").unwrap_or(&50257);
        let python_end = *self.tokenizer.get_special_tokens().get("<|python_end|>").unwrap_or(&50258);
        let output_start = *self.tokenizer.get_special_tokens().get("<|output_start|>").unwrap_or(&50259);
        let output_end = *self.tokenizer.get_special_tokens().get("<|output_end|>").unwrap_or(&50260);

        let mut next_token_column = Vec::with_capacity(num_samples);
        let mut is_sampled_mask = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            if state.completed[i] {
                next_token_column.push(bos);
                is_sampled_mask.push(0);
                continue;
            }

            let is_forced = !state.forced_tokens[i].is_empty();
            let next_tok = if is_forced {
                state.forced_tokens[i].pop_front().unwrap()
            } else { sampled_tokens[i] };

            next_token_column.push(next_tok);
            state.current_tokens[i].push(next_tok);
            is_sampled_mask.push(if is_forced { 0 } else { 1 });

            if next_tok == assistant_end || next_tok == bos {
                state.completed[i] = true;
            }

            // Built-in Tool-Use State Machine
            if next_tok == python_start {
                state.in_python_block[i] = true;
                state.python_expr_tokens[i].clear();
            } else if next_tok == python_end && state.in_python_block[i] {
                state.in_python_block[i] = false;
                let expr_str = self.tokenizer.decode(&state.python_expr_tokens[i]);
                if let Some(res) = use_calculator(&expr_str) {
                    let res_tokens = self.tokenizer.encode_ordinary(&res);
                    state.forced_tokens[i].push_back(output_start);
                    for &t in &res_tokens { state.forced_tokens[i].push_back(t); }
                    state.forced_tokens[i].push_back(output_end);
                }
            } else if state.in_python_block[i] {
                state.python_expr_tokens[i].push(next_tok);
            }
        }

        let next_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                next_token_column.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                Shape::new([num_samples, 1]),
            ),
            device,
        );

        let next_logits_3d = self.model.forward_with_cache(next_idx, &mut state.cache, state.step);
        state.step += 1;

        let next_logits = next_logits_3d
            .slice([0..num_samples, 0..1, 0..self.model.config.vocab_size])
            .reshape([num_samples, self.model.config.vocab_size]);

        (next_token_column, is_sampled_mask, next_logits)
    }

    /// Non-streaming batch generation interface returning (results, masks)
    pub fn generate_batch(&self, prompt_tokens: &[usize], num_samples: usize,
        max_tokens: usize, temperature: f32, top_k: Option<usize>,
        repetition_penalty: f32, device: &B::Device,) -> (Vec<Vec<usize>>, Vec<Vec<u8>>) {
        let (mut state, mut cur_logits) = self.prefill(prompt_tokens, num_samples, device);

        let assistant_end = *self.tokenizer.get_special_tokens().get("<|assistant_end|>").unwrap_or(&50256);
        let bos = self.tokenizer.get_bos_token_id();

        let mut results = vec![prompt_tokens.to_vec(); num_samples];
        let mut masks = vec![vec![0; prompt_tokens.len()]; num_samples];
        let mut completed = vec![false; num_samples];

        for step_idx in 0..max_tokens {
            if completed.iter().all(|&c| c) { break; }
            if step_idx > 0 && step_idx % 20 == 0 {
                tracing::info!("    Generated {}/{} tokens...", step_idx, max_tokens);
            }
            let (token_column, token_masks, next_logits) = self.step_generation(&mut state, cur_logits,
                temperature, top_k, repetition_penalty, device,);
            cur_logits = next_logits;

            for i in 0..num_samples {
                if !completed[i] {
                    let token = token_column[i];
                    let mask = token_masks[i];
                    if token == assistant_end || token == bos {
                        completed[i] = true;
                    } else {
                        results[i].push(token);
                        masks[i].push(mask);
                    }
                }
            }
        }

        (results, masks)
    }
}

/// Dynamic sample selection based on temperature, top_k, and repetition penalties
pub fn sample_next_token<B: Backend>(logits: Tensor<B, 2>, temperature: f32,
    top_k: Option<usize>, repetition_penalty: f32,
    generated_tokens: &[Vec<usize>],) -> Vec<usize> {
    let shape: [usize; 2] = logits.shape().dims();
    let (batch_size, vocab_size) = (shape[0], shape[1]);

    let logits_vec = crate::common::tensor_data_to_f32_vec(logits.into_data());
    let mut sampled_ids = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let start = b * vocab_size;
        let mut sample_logits = logits_vec[start..start + vocab_size].to_vec();

        // 1. Repetition penalty
        if repetition_penalty != 1.0 {
            let mut unique_history = std::collections::HashSet::new();
            for &t in &generated_tokens[b] { unique_history.insert(t); }
            for &t in &unique_history {
                if t < vocab_size {
                    let val = sample_logits[t];
                    if val > 0.0 {
                        sample_logits[t] = val / repetition_penalty;
                    } else {
                        sample_logits[t] = val * repetition_penalty;
                    }
                }
            }
        }

        // 2. Argmax if temperature is 0
        if temperature == 0.0 {
            let (mut max_val, mut max_idx) = (sample_logits[0], 0);
            for (idx, &val) in sample_logits.iter().enumerate() {
                if val > max_val { (max_val, max_idx) = (val, idx); }
            }
            sampled_ids.push(max_idx);
            continue;
        }

        // 3. Scale by temperature
        for val in sample_logits.iter_mut() { *val /= temperature; }

        // 4. Top-K filtering
        let mut indices: Vec<usize> = (0..vocab_size).collect();
        if let Some(k) = top_k {
            let k = k.min(vocab_size);
            indices.sort_by(|&i, &j| sample_logits[j].partial_cmp(&sample_logits[i]).unwrap());
            for &idx in &indices[k..] { sample_logits[idx] = -1e9; }
        }

        // 5. Stable Softmax & Multinomial sampling
        let max_logit = sample_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_logits: Vec<f32> = sample_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        for v in exp_logits.iter_mut() { *v /= sum_exp; }

        let mut cum_sum = 0.0f32;
        let r: f32 = rand::random();
        let mut chosen_idx = indices[0];
        for &idx in &indices {
            cum_sum += exp_logits[idx];
            if r <= cum_sum { chosen_idx = idx; break; }
        }
        sampled_ids.push(chosen_idx);
    }

    sampled_ids
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_inference_engine_instantiation() {
        let device = crate::common::init_device();
        let corpus = vec!["Interactive chat agent with Tool-Use integration."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        let config = crate::gpt::GptConfig { sequence_len: 8,
            n_layer: 1, n_head: 2, n_kv_head: 1, n_embd: 32, quantization: None,
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

        let config = crate::gpt::GptConfig { sequence_len: 32,
            n_layer: 1, n_head: 2, n_kv_head: 1, n_embd: 32,
            window_pattern: "L".to_string(), vocab_size: tokenizer.get_vocab_size(),
            quantization: None,
        };

        use crate::common::ModelBackend;
        let gpt: Gpt<ModelBackend> = Gpt::new(config.clone(), &device);
        let engine = InferenceEngine::new(gpt, tokenizer);

        let prompt_tokens = vec![1, 2, 3];
        let (results, masks) = engine.generate_batch(&prompt_tokens, 2, 5, 1.0, Some(5), 1.0, &device);

        assert_eq!(results.len(), 2);
        assert_eq!(masks.len(), 2);
        assert!(results[0].len() >= 3);
    }
//}
