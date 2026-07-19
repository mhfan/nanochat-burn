use std::collections::VecDeque;

use burn::{nn::Linear, tensor::{Tensor, backend::Backend}};

use super::{calculator::use_calculator,
    inference::{DeviceSampler, GenerationConfig, InferenceEngine, SamplingRng, TokenSampler},
    scheduler::{ContinuousBatchScheduler, RequestId}};
use crate::{common::int_tensor_2d, gpt::{ForwardLayer, KVCache}, tokenizer::BpeTokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason { Stop, Length }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationStep {
    pub request_id: RequestId,
    pub token: Option<usize>,
    pub finish_reason: Option<FinishReason>,
}

struct RequestState<B: Backend> {
    slot: usize,
    current_tokens: Vec<usize>,
    forced_tokens: VecDeque<usize>,
    in_python_block: bool,
    python_expr_tokens: Vec<usize>,
    completed: bool,
    step: usize,
    rng: SamplingRng,
    logits: Option<Tensor<B, 2>>,
}

struct ScheduledGeneration<B: Backend> {
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    state: Option<RequestState<B>>,
    generated: usize,
}

pub struct DynamicGenerationEngine<B: Backend, L: ForwardLayer<B> = Linear<B>> {
    engine: InferenceEngine<B, L>,
    device: B::Device,
    scheduler: ContinuousBatchScheduler<ScheduledGeneration<B>>,
    cache: KVCache<B>,
    free_slots: Vec<usize>,
    last_decode_batch_size: usize,
}

impl<B: Backend, L: ForwardLayer<B>> DynamicGenerationEngine<B, L> {
    pub fn new(engine: InferenceEngine<B, L>, device: B::Device, capacity: usize) -> Self {
        assert!(capacity > 0, "generation capacity must be positive");
        let config = &engine.model.config;
        let cache = KVCache::new_allocated(config.n_layer, capacity, config.sequence_len,
            config.n_kv_head, config.n_embd / config.n_head, &device);
        Self { engine, device, scheduler: ContinuousBatchScheduler::new(capacity), cache,
            free_slots: (0..capacity).rev().collect(), last_decode_batch_size: 0 }
    }

    pub fn submit(&mut self, prompt_tokens: Vec<usize>, config: GenerationConfig) -> RequestId {
        config.sampling.validate();
        self.scheduler.submit(ScheduledGeneration {
            prompt_tokens, config, state: None, generated: 0,
        })
    }

    pub fn cancel(&mut self, request_id: RequestId) -> bool {
        let Some(request) = self.scheduler.cancel(request_id) else { return false; };
        Self::recycle_request(request, &mut self.cache, &mut self.free_slots);
        true
    }

    pub fn step(&mut self) -> Vec<GenerationStep> {
        let request_ids = self.scheduler.schedule_iteration();
        for &request_id in &request_ids { self.initialize(request_id); }

        let mut events = Vec::with_capacity(request_ids.len());
        let mut finished = Vec::new();
        let mut decode_ids = Vec::with_capacity(request_ids.len());
        let mut next_tokens = Vec::with_capacity(request_ids.len());
        let mut slots = Vec::with_capacity(request_ids.len());
        let mut steps = Vec::with_capacity(request_ids.len());

        for request_id in request_ids {
            let request = self.scheduler.get_mut(request_id).expect("scheduled request missing");
            let state = request.state.as_mut().expect("request was not initialized");
            if request.generated >= request.config.max_tokens ||
                state.step >= self.engine.model.config.sequence_len {
                events.push(GenerationStep { request_id, token: None,
                    finish_reason: Some(FinishReason::Length) });
                finished.push(request_id);
                continue;
            }

            let logits = state.logits.take().expect("active request has no logits");
            let sampled = DeviceSampler.sample(logits, request.config.sampling,
                std::slice::from_ref(&state.current_tokens), &mut state.rng)[0];
            let forced = state.forced_tokens.pop_front();
            let token = forced.unwrap_or(sampled);
            record_generated_token(&self.engine.tokenizer, state, token);
            decode_ids.push(request_id);
            next_tokens.push(token as i32);
            slots.push(state.slot);
            steps.push(state.step);
        }

        self.last_decode_batch_size = decode_ids.len();
        if !decode_ids.is_empty() {
            let batch_size = decode_ids.len();
            let idx = int_tensor_2d(next_tokens, [batch_size, 1], &self.device);
            let next_logits = self.engine.model.forward_with_cache_rows(
                idx, &mut self.cache, &slots, &steps)
                .reshape([batch_size, self.engine.model.config.vocab_size]);

            for (source, request_id) in decode_ids.into_iter().enumerate() {
                let request = self.scheduler.get_mut(request_id)
                    .expect("decoded request missing");
                let state = request.state.as_mut().expect("decoded request has no state");
                state.logits = Some(next_logits.clone().slice([
                    source..source + 1, 0..self.engine.model.config.vocab_size]));
                state.step += 1;
                request.generated += 1;
                let finish_reason = if state.completed { Some(FinishReason::Stop) } else if
                    request.generated >= request.config.max_tokens ||
                    state.step >= self.engine.model.config.sequence_len {
                    Some(FinishReason::Length)
                } else { None };
                events.push(GenerationStep { request_id,
                    token: (!state.completed).then(|| state.current_tokens[state.step - 1]),
                    finish_reason });
                if finish_reason.is_some() { finished.push(request_id); }
            }
        }

        for request_id in finished {
            let request = self.scheduler.complete(request_id).expect("finished request missing");
            Self::recycle_request(request, &mut self.cache, &mut self.free_slots);
        }
        events.sort_unstable_by_key(|event| event.request_id);
        events
    }

    pub fn tokenizer(&self) -> &BpeTokenizer { &self.engine.tokenizer }
    pub fn active_len(&self) -> usize { self.scheduler.active_len() }
    pub fn waiting_len(&self) -> usize { self.scheduler.waiting_len() }

    fn initialize(&mut self, request_id: RequestId) {
        if self.scheduler.get(request_id).is_some_and(|request| request.state.is_some()) {
            return;
        }
        let slot = self.free_slots.pop().expect("scheduler admitted more requests than slots");
        let request = self.scheduler.get_mut(request_id).expect("scheduled request missing");
        let prompt_tokens = std::mem::take(&mut request.prompt_tokens);
        let prompt_len = prompt_tokens.len();
        assert!(prompt_len > 0, "prompt must contain at least one token");
        assert!(prompt_len <= self.engine.model.config.sequence_len,
            "prompt length exceeds model sequence length");
        assert!(prompt_tokens.iter().all(|&token|
            token < self.engine.model.config.vocab_size && i32::try_from(token).is_ok()),
            "prompt contains an invalid token ID");
        let idx = int_tensor_2d(
            prompt_tokens.iter().map(|&token| token as i32).collect(), [1, prompt_len], &self.device);
        let logits = self.engine.model.forward_with_cache_rows(
            idx, &mut self.cache, &[slot], &[0])
            .slice([0..1, prompt_len - 1..prompt_len, 0..self.engine.model.config.vocab_size])
            .reshape([1, self.engine.model.config.vocab_size]);
        request.state = Some(RequestState { slot, current_tokens: prompt_tokens,
            forced_tokens: VecDeque::new(), in_python_block: false,
            python_expr_tokens: Vec::new(), completed: false, step: prompt_len,
            rng: SamplingRng::new(request.config.seed), logits: Some(logits) });
    }

    fn recycle_request(mut request: ScheduledGeneration<B>, cache: &mut KVCache<B>,
        free_slots: &mut Vec<usize>) {
        if let Some(state) = request.state.take() {
            cache.release_request(state.slot);
            free_slots.push(state.slot);
        }
    }
}

fn record_generated_token<B: Backend>(tokenizer: &BpeTokenizer, state: &mut RequestState<B>,
    token: usize) {
    let special = tokenizer.special_token_ids();
    state.current_tokens.push(token);
    if token == special.assistant_end || token == special.bos { state.completed = true; }
    if token == special.python_start {
        state.in_python_block = true;
        state.python_expr_tokens.clear();
    } else if token == special.python_end && state.in_python_block {
        state.in_python_block = false;
        let expression = tokenizer.decode(&state.python_expr_tokens);
        if let Some(result) = use_calculator(&expression) {
            state.forced_tokens.push_back(special.output_start);
            state.forced_tokens.extend(tokenizer.encode_ordinary(&result));
            state.forced_tokens.push_back(special.output_end);
        }
    } else if state.in_python_block {
        state.python_expr_tokens.push(token);
    }
}

#[cfg(test)] mod tests { use super::*;
    use crate::{common::{TestBackend}, gpt::{Gpt, GptConfig}};

    fn service(capacity: usize) -> DynamicGenerationEngine<TestBackend> {
        let device = Default::default();
        let tokenizer = BpeTokenizer::train_from_iterator(["dynamic request scheduling"], 280);
        let config = GptConfig { sequence_len: 16, n_layer: 1, n_head: 2, n_kv_head: 1,
            n_embd: 16, window_pattern: "L".into(), vocab_size: tokenizer.get_vocab_size(),
            features: Default::default(), quantization: None };
        let model = Gpt::new(config, &device);
        DynamicGenerationEngine::new(InferenceEngine::new(model, tokenizer), device, capacity)
    }

    #[test] fn test_dynamic_admission_batches_and_reuses_slots() {
        let mut service = service(2);
        let generation = GenerationConfig { max_tokens: 1, ..Default::default() };
        let (first, second) =
            (service.submit(vec![1, 2], generation), service.submit(vec![3, 4, 5], generation));

        let events = service.step();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].request_id, first);
        assert_eq!(events[1].request_id, second);
        assert!(events.iter().all(|event| event.finish_reason == Some(FinishReason::Length)));
        assert_eq!(service.last_decode_batch_size, 2);
        assert_eq!(service.free_slots.len(), 2);
        assert_eq!(service.cache.allocator.available(), service.cache.allocator.capacity());

        service.submit(vec![6, 7], generation);
        service.step();
        assert_eq!(service.last_decode_batch_size, 1);
        assert_eq!(service.free_slots.len(), 2);
    }

    #[test] fn test_cancel_reclaims_active_slot() {
        let mut service = service(1);
        let request = service.submit(vec![1, 2], GenerationConfig {
            max_tokens: 4, ..Default::default()
        });
        assert_eq!(service.step()[0].finish_reason, None);
        assert!(service.cancel(request));
        assert_eq!(service.active_len(), 0);
        assert_eq!(service.free_slots, vec![0]);
        assert_eq!(service.cache.allocator.available(), service.cache.allocator.capacity());
    }
}
