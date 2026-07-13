
use burn::tensor::{Int, Tensor, backend::Backend};

use crate::{common::int_tensor_2d,
    engine::inference::{GenerationConfig, GeneratorState, InferenceEngine, SamplingConfig,
        sample_next_token},
    gpt::{ForwardLayer, Gpt}, tokenizer::BpeTokenizer,
};

fn token_tensor<B: Backend>(tokens: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    int_tensor_2d(tokens.iter().map(|&token| token as i32).collect(), [1, tokens.len()], device)
}

#[derive(Debug, Clone, Copy)]
pub struct SpeculativeConfig {
    pub generation: GenerationConfig,
    pub draft_tokens: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self { Self { generation: GenerationConfig::default(), draft_tokens: 4 } }
}

/// Coordinates speculative decoding state
pub struct SpeculativeState<B: Backend> {
    pub target_state: GeneratorState<B>,
    pub draft_state: GeneratorState<B>,
    pub current_tokens: Vec<usize>,
    pub step: usize,
}

pub struct SpeculativeInferenceEngine<B: Backend,
    LTarget: ForwardLayer<B> = burn::nn::Linear<B>,
    LDraft:  ForwardLayer<B> = burn::nn::Linear<B>> {
    pub target_engine: InferenceEngine<B, LTarget>,
    pub draft_engine: InferenceEngine<B, LDraft>,
    pub tokenizer: BpeTokenizer,
}

impl<B: Backend, LTarget: ForwardLayer<B>, LDraft: ForwardLayer<B>>
    SpeculativeInferenceEngine<B, LTarget, LDraft> {
    pub fn new(target_model: Gpt<B, LTarget>, draft_model: Gpt<B, LDraft>,
        tokenizer: BpeTokenizer) -> Self {
        assert_eq!(target_model.config.vocab_size, draft_model.config.vocab_size,
            "target and draft models must use the same vocabulary");
        Self {
            target_engine: InferenceEngine::new(target_model, tokenizer.clone()),
            draft_engine: InferenceEngine::new(draft_model, tokenizer.clone()),
            tokenizer,
        }
    }

    /// Prefill prompt sequence for both models, initializing states
    pub fn prefill(&self, prompt_tokens: &[usize], device: &B::Device)
        -> (SpeculativeState<B>, Tensor<B, 2>) {
        let (draft_state, _) = self.draft_engine.prefill(prompt_tokens, 1, device);
        let (target_state, target_logits) =
            self.target_engine.prefill(prompt_tokens, 1, device);

        let state = SpeculativeState {
            target_state, draft_state,
            current_tokens: prompt_tokens.to_vec(),
            step: prompt_tokens.len(),
        };

        (state, target_logits)
    }

    fn sync_draft_state(&self, state: &mut SpeculativeState<B>, device: &B::Device) {
        let (draft_state, _) = self.draft_engine.prefill(&state.current_tokens, 1, device);
        state.draft_state = draft_state;
    }

    /// Perform speculative decoding steps: draft K tokens,
    /// evaluate in parallel, and verify losslessly
    pub fn step_speculative(&self, state: &mut SpeculativeState<B>, target_logits: Tensor<B, 2>,
        draft_tokens_per_step: usize, sampling: SamplingConfig, device: &B::Device)
        -> (Vec<usize>, Tensor<B, 2>, bool) {
        let special_tokens = self.tokenizer.special_token_ids();
        let sequence_len = self.target_engine.model.config.sequence_len
            .min(self.draft_engine.model.config.sequence_len);
        let draft_tokens_per_step = draft_tokens_per_step
            .min(sequence_len.saturating_sub(state.step + 1));

        // 1. Autoregressively draft K tokens using the fast Draft Model
        let last_tok = *state.current_tokens.last().unwrap();
        let mut draft_tokens = Vec::with_capacity(draft_tokens_per_step);
        let draft_vocab_size = self.draft_engine.model.config.vocab_size;
        let mut cur_draft_logits = self.draft_engine.model
            .forward_with_cache(token_tensor(&[last_tok], device),
                &mut state.draft_state.cache, state.step - 1).reshape([1, draft_vocab_size]);

        let mut temp_draft_state = state.draft_state.clone();
        temp_draft_state.step = state.step;

        for _ in 0..draft_tokens_per_step {
            if temp_draft_state.completed[0] { break; }
            let (sampled_toks, _, next_logits) = self.draft_engine.step_generation(
                &mut temp_draft_state, cur_draft_logits, sampling, device,
            );
            draft_tokens.push(sampled_toks[0]);
            cur_draft_logits = next_logits;
        }

        if draft_tokens.is_empty() {
            // Nothing drafted, fall back to single target model step
            let (sampled_toks, _, next_logits) = self.target_engine.step_generation(
                &mut state.target_state, target_logits, sampling, device,
            );
            let token = sampled_toks[0];
            state.current_tokens.push(token);
            state.step += 1;
            self.sync_draft_state(state, device);

            let is_finished =
                token == special_tokens.assistant_end || token == special_tokens.bos;
            return (vec![token], next_logits, is_finished);
        }

        // 2. Parallelly evaluate all K draft tokens in the Target Model
        let draft_len = draft_tokens.len();
        let target_input = token_tensor(&draft_tokens, device);

        let target_logits_3d = self.target_engine.model
            .forward_with_cache(target_input, &mut state.target_state.cache, state.step);

        // 3. Lossless verification check
        let mut accepted_tokens = Vec::new();
        let mut final_next_logits = target_logits.clone();
        let (mut is_finished, mut accepted_count) = (false, 0);

        for i in 0..draft_len {
            let draft_tok = draft_tokens[i];

            // The target prediction for draft_tok (at position L + i) is sampled from
            // target_logits (for i == 0) or target_logits_3d (for i > 0)
            let target_logits_for_verify = if i == 0 { target_logits.clone() } else {
                let vocab_size = self.target_engine.model.config.vocab_size;
                target_logits_3d.clone().slice([0..1, (i - 1)..i]).reshape([1, vocab_size])
            };

            let target_pred_toks = sample_next_token(
                target_logits_for_verify, sampling, &state.target_state.current_tokens);
            let target_pred_tok = target_pred_toks[0];

            if draft_tok == target_pred_tok {
                accepted_tokens.push(draft_tok);
                state.current_tokens.push(draft_tok);
                state.target_state.current_tokens[0].push(draft_tok);
                accepted_count += 1;

                if draft_tok == special_tokens.assistant_end || draft_tok == special_tokens.bos {
                    is_finished = true;
                    break;
                }

                // If this is the last drafted token and it is accepted, we also accept the
                // target's prediction for the next token
                if i == draft_len - 1 {
                    let vocab_size = self.target_engine.model.config.vocab_size;
                    let next_target_logits = target_logits_3d.clone()
                        .slice([0..1, i..(i + 1)]).reshape([1, vocab_size]);

                    let last_pred_toks = sample_next_token(
                        next_target_logits.clone(), sampling, &state.target_state.current_tokens);
                    let last_pred_tok = last_pred_toks[0];

                    accepted_tokens.push(last_pred_tok);
                    state.current_tokens.push(last_pred_tok);
                    state.target_state.current_tokens[0].push(last_pred_tok);
                    accepted_count += 1;

                    let next_logits_3d = self.target_engine.model.forward_with_cache(
                        token_tensor(&[last_pred_tok], device),
                        &mut state.target_state.cache,
                        state.step + draft_len,
                    );
                    final_next_logits = next_logits_3d.reshape([1, vocab_size]);

                    if last_pred_tok == special_tokens.assistant_end ||
                        last_pred_tok == special_tokens.bos { is_finished = true; }
                }
            } else {
                // Reject draft_tok, accept target_pred_tok instead
                accepted_tokens.push(target_pred_tok);
                state.current_tokens.push(target_pred_tok);
                state.target_state.current_tokens[0].push(target_pred_tok);
                accepted_count += 1;

                // Run correction forward pass for target_pred_tok to overwrite cache at
                // position L + i and get next logits.
                let correction_logits_3d = self.target_engine.model.forward_with_cache(
                    token_tensor(&[target_pred_tok], device),
                    &mut state.target_state.cache, state.step + i,
                );
                let vocab_size = self.target_engine.model.config.vocab_size;
                final_next_logits = correction_logits_3d.reshape([1, vocab_size]);

                if target_pred_tok == special_tokens.assistant_end ||
                    target_pred_tok == special_tokens.bos { is_finished = true; }
                break;
            }
        }

        // 4. Synchronize state pointers and rebuild the draft cache from the accepted sequence.
        let final_len = state.step + accepted_count;
        state.target_state.step = final_len;
        state.step = final_len;
        self.sync_draft_state(state, device);

        (accepted_tokens, final_next_logits, is_finished)
    }

    /// High-level generation loop returning the fully generated token sequence
    pub fn generate(&self, prompt_tokens: &[usize], config: SpeculativeConfig,
        device: &B::Device) -> Vec<usize> {
        let (mut state, mut cur_logits) = self.prefill(prompt_tokens, device);
        let max_total_len = (prompt_tokens.len() + config.generation.max_tokens)
            .min(self.target_engine.model.config.sequence_len)
            .min(self.draft_engine.model.config.sequence_len);

        while state.current_tokens.len() < max_total_len {
            let remaining = max_total_len - state.current_tokens.len();
            let draft_tokens = config.draft_tokens.min(remaining.saturating_sub(1));
            let (_, next_logits, is_finished) = self.step_speculative(&mut state,
                cur_logits, draft_tokens, config.generation.sampling, device);
            cur_logits = next_logits;
            if is_finished { break; }
        }

        state.current_tokens.truncate(max_total_len);
        state.current_tokens
    }
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_speculative_decoding_lossless() {
        let device = crate::common::init_device();
        let corpus =
            vec!["Rust is extremely elegant, ultra-fast, and lossless speculative decoding works!"];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);

        let target_config = crate::gpt::GptConfig { sequence_len: 16,
            vocab_size: tokenizer.get_vocab_size(), n_layer: 1, n_head: 4, n_kv_head: 1, n_embd: 32,
            window_pattern: "L".to_string(), quantization: None,
        };
        let draft_config = crate::gpt::GptConfig { sequence_len: 16,
            vocab_size: tokenizer.get_vocab_size(), n_layer: 1, n_head: 4, n_kv_head: 1, n_embd: 32,
            window_pattern: "L".to_string(), quantization: None,
        };

        use crate::common::ModelBackend;
        let target_model: Gpt<ModelBackend> = Gpt::new(target_config.clone(), &device);
        let draft_model: Gpt<ModelBackend> = Gpt::new(draft_config.clone(), &device);

        let spec_engine = SpeculativeInferenceEngine::new(target_model.clone(),
            draft_model.clone(), tokenizer.clone());
        let target_engine = InferenceEngine::new(target_model, tokenizer.clone());

        let prompt_tokens = vec![1, 2, 3];

        // 1. Generate with pure Target Model (deterministic greedy: temperature = 0.0)
        let (mut target_state, mut target_logits) =
            target_engine.prefill(&prompt_tokens, 1, &device);
        let mut target_res = prompt_tokens.clone();
        for _ in 0..3 {
            let (toks, _, next_logits) = target_engine
                .step_generation(&mut target_state, target_logits,
                    SamplingConfig::greedy(), &device);
            target_res.push(toks[0]);
            target_logits = next_logits;
            let special_tokens = tokenizer.special_token_ids();
            if toks[0] == special_tokens.assistant_end ||
                toks[0] == special_tokens.bos { break; }
        }

        // 2. Generate with Speculative Decoding (deterministic greedy: temperature = 0.0, K = 2)
        let config = SpeculativeConfig { draft_tokens: 2,
            generation: GenerationConfig { max_tokens: 3, sampling: SamplingConfig::greedy() },
        };
        let spec_res = spec_engine.generate(&prompt_tokens, config, &device);

        // 3. Verify Lossless Parity (outputs must be mathematically identical!)
        assert_eq!(spec_res, target_res,
            "Speculative decoding did not maintain lossless parity with target model!");
    }
//}
