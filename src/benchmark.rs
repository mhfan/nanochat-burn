use std::{mem::size_of, time::Instant};

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{common::DeviceMemoryUsage,
    engine::inference::{DeviceSampler, InferenceEngine, SamplingConfig,
        SamplingRng, TokenSampler}, engine::speculative::{SpeculativeConfig,
        SpeculativeInferenceEngine}, gpt::{ForwardLayer, GptConfig}};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceBenchmark {
    pub batch_size: usize,
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    pub iterations: usize,
    pub prefill_latency_ms: f64,
    pub time_to_first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    #[serde(default)]
    pub median_decode_step_ms: f64,
    pub cache_bytes: u64,
    #[serde(default)]
    pub peak_device_bytes_in_use: Option<u64>,
    #[serde(default)]
    pub peak_device_bytes_reserved: Option<u64>,
    pub model_bytes: u64,
    pub quantization_bits: Option<usize>,
    pub quantization_max_error: Option<f32>,
    pub estimated_quantized_linear_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeculativeBenchmark {
    pub iterations: usize,
    pub generated_tokens: usize,
    pub measured_tokens: usize,
    pub draft_tokens_per_step: usize,
    pub acceptance_rate: f32,
    pub target_tokens_per_second: f64,
    pub speculative_tokens_per_second: f64,
    pub speedup: f64,
}

pub fn benchmark_speculative<B: Backend, LTarget: ForwardLayer<B>, LDraft: ForwardLayer<B>>(
    engine: &SpeculativeInferenceEngine<B, LTarget, LDraft>, prompt: &[usize],
    generated_tokens: usize, draft_tokens_per_step: usize, iterations: usize,
    device: &B::Device) -> SpeculativeBenchmark {
    assert!(iterations > 0, "benchmark iterations must be positive");
    let generation = crate::engine::inference::GenerationConfig {
        max_tokens: generated_tokens, sampling: SamplingConfig::greedy(), seed: 42,
    };
    let target_start = Instant::now();
    let target_outputs = (0..iterations).map(|_| engine.target_engine
        .generate_batch(prompt, 1, generation, device).0.remove(0)).collect::<Vec<_>>();
    let target_secs = target_start.elapsed().as_secs_f64();
    let measured_tokens = target_outputs.iter()
        .map(|tokens| tokens.len().saturating_sub(prompt.len())).sum::<usize>();

    let speculative_start = Instant::now();
    let mut proposed = 0;
    let mut accepted = 0;
    for expected in target_outputs {
        let (actual, stats) = engine.generate_with_stats(prompt, SpeculativeConfig {
            generation, draft_tokens: draft_tokens_per_step,
        }, device);
        assert_eq!(actual, expected, "speculative benchmark requires lossless output parity");
        proposed += stats.proposed_draft_tokens;
        accepted += stats.accepted_draft_tokens;
    }
    let speculative_secs = speculative_start.elapsed().as_secs_f64();
    let tokens = measured_tokens.max(1) as f64;
    let target_tokens_per_second = tokens / target_secs;
    let speculative_tokens_per_second = tokens / speculative_secs;
    SpeculativeBenchmark { iterations, generated_tokens, measured_tokens, draft_tokens_per_step,
        acceptance_rate: if proposed == 0 { 0.0 } else { accepted as f32 / proposed as f32 },
        target_tokens_per_second, speculative_tokens_per_second,
        speedup: speculative_tokens_per_second / target_tokens_per_second,
    }
}

pub fn benchmark_inference<B: Backend, L: ForwardLayer<B>>(
    engine: &InferenceEngine<B, L>, prompt: &[usize], batch_size: usize,
    decode_tokens: usize, warmup: usize, iterations: usize, model_bytes: u64,
    quantization_bits: Option<usize>, quantization_max_error: Option<f32>,
    device: &B::Device, memory_usage: &impl Fn() -> Option<DeviceMemoryUsage>)
    -> InferenceBenchmark {
    assert!(!prompt.is_empty(), "benchmark prompt must not be empty");
    assert!(batch_size > 0, "benchmark batch size must be positive");
    assert!(iterations > 0, "benchmark iterations must be positive");
    assert!(prompt.len() + decode_tokens <= engine.model.config.sequence_len,
        "benchmark prompt and decode tokens exceed model capacity");
    let history = vec![prompt.to_vec(); batch_size];

    for _ in 0..warmup {
        let (_, logits) = engine.prefill(prompt, batch_size, device);
        let mut rng = SamplingRng::new(0);
        DeviceSampler.sample(logits, SamplingConfig::greedy(), &history, &mut rng);
    }

    let mut prefill_secs = 0.0;
    let mut ttft_secs = 0.0;
    for _ in 0..iterations {
        let start = Instant::now();
        let (_, logits) = engine.prefill(prompt, batch_size, device);
        let _ = logits.clone().into_data();
        prefill_secs += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let (_, logits) = engine.prefill(prompt, batch_size, device);
        let mut rng = SamplingRng::new(0);
        DeviceSampler.sample(logits, SamplingConfig::greedy(), &history, &mut rng);
        ttft_secs += start.elapsed().as_secs_f64();
    }

    let start = Instant::now();
    let mut generated = 0;
    for _ in 0..iterations {
        let (mut state, mut logits) = engine.prefill(prompt, batch_size, device);
        for _ in 0..decode_tokens {
            let (_, _, next) = engine.step_generation(
                &mut state, logits, SamplingConfig::greedy(), device);
            logits = next;
            generated += batch_size;
        }
        let _ = logits.into_data();
    }
    let decode_tokens_per_second = generated as f64 / start.elapsed().as_secs_f64();

    // Throughput keeps the decode pipeline asynchronous; this second pass synchronizes every
    // step to expose user-visible inter-token latency instead of reporting host submission time.
    let mut decode_step_ms = Vec::with_capacity(iterations * decode_tokens);
    for _ in 0..iterations {
        let (mut state, mut logits) = engine.prefill(prompt, batch_size, device);
        for _ in 0..decode_tokens {
            let start = Instant::now();
            let (_, _, next) = engine.step_generation(
                &mut state, logits, SamplingConfig::greedy(), device);
            let _ = next.clone().into_data();
            decode_step_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            logits = next;
        }
    }
    decode_step_ms.sort_by(f64::total_cmp);
    let median_decode_step_ms = decode_step_ms.get(decode_step_ms.len() / 2).copied()
        .unwrap_or(0.0);

    // Keep allocator synchronization outside timed sections.
    let mut peak_memory = memory_usage();
    let (mut state, mut logits) = engine.prefill(prompt, batch_size, device);
    update_peak_memory(&mut peak_memory, memory_usage());
    for _ in 0..decode_tokens {
        let (_, _, next) = engine.step_generation(
            &mut state, logits, SamplingConfig::greedy(), device);
        logits = next;
        update_peak_memory(&mut peak_memory, memory_usage());
    }

    InferenceBenchmark { batch_size, prompt_tokens: prompt.len(), decode_tokens, iterations,
        prefill_latency_ms: prefill_secs * 1000.0 / iterations as f64,
        time_to_first_token_ms: ttft_secs * 1000.0 / iterations as f64,
        decode_tokens_per_second, median_decode_step_ms,
        cache_bytes: estimate_cache_bytes::<B>(
        &engine.model.config, batch_size),
        peak_device_bytes_in_use: peak_memory.map(|usage| usage.bytes_in_use),
        peak_device_bytes_reserved: peak_memory.map(|usage| usage.bytes_reserved),
        model_bytes, quantization_bits,
        quantization_max_error,
        estimated_quantized_linear_bytes: quantization_bits.map(|bits|
            estimate_quantized_linear_bytes(&engine.model.config, bits)),
    }
}

fn update_peak_memory(peak: &mut Option<DeviceMemoryUsage>, sample: Option<DeviceMemoryUsage>) {
    if let Some(sample) = sample {
        let peak = peak.get_or_insert(sample);
        peak.bytes_in_use = peak.bytes_in_use.max(sample.bytes_in_use);
        peak.bytes_reserved = peak.bytes_reserved.max(sample.bytes_reserved);
    }
}

fn estimate_cache_bytes<B: Backend>(config: &GptConfig, batch_size: usize) -> u64 {
    let head_dim = config.n_embd / config.n_head;
    (2 * config.n_layer * batch_size * config.sequence_len * config.n_kv_head * head_dim *
        size_of::<B::FloatElem>()) as u64
}

fn estimate_quantized_linear_bytes(config: &GptConfig, bits: usize) -> u64 {
    let head_dim = config.n_embd / config.n_head;
    let kv_dim = config.n_kv_head * head_dim;
    let block = config.n_embd * config.n_embd * 2 + config.n_embd * kv_dim * 2 +
        config.n_embd * config.n_embd * 8;
    let output = config.n_embd * config.vocab_size;
    ((config.n_layer * block + output) * bits).div_ceil(8) as u64
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_peak_memory_combines_allocator_samples() {
        let mut peak = Some(DeviceMemoryUsage { bytes_in_use: 10, bytes_reserved: 20 });
        update_peak_memory(&mut peak,
            Some(DeviceMemoryUsage { bytes_in_use: 15, bytes_reserved: 18 }));
        update_peak_memory(&mut peak,
            Some(DeviceMemoryUsage { bytes_in_use: 12, bytes_reserved: 30 }));
        assert_eq!(peak, Some(DeviceMemoryUsage { bytes_in_use: 15, bytes_reserved: 30 }));
    }
}
