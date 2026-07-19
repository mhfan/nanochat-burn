use std::{fs, path::PathBuf};

use burn::tensor::{Int, Tensor, TensorData};
use nanochat_burn::{artifact::{inference_artifact_path, load_artifact},
    benchmark::{InferenceBenchmark, InferenceBenchmarkConfig, benchmark_inference},
    common::{DeviceMemoryUsage, InferBackend, device_memory_usage, init_device},
    engine::inference::InferenceEngine, experiment::ArtifactPaths, gpt::{ForwardLayer, Gpt},
};

use super::{CliResult, next_value, write_report};

struct Args {
    artifact: PathBuf,
    output: PathBuf,
    batches: Vec<usize>,
    prompt_tokens: usize,
    decode_tokens: usize,
    warmup: usize,
    iterations: usize,
    quantization: Option<usize>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args {
        artifact: inference_artifact_path(&ArtifactPaths::default()),
        output: PathBuf::from("runs/benchmarks/inference.json"),
        batches: vec![1, 2, 4], prompt_tokens: 32, decode_tokens: 32, warmup: 1,
        iterations: 3, quantization: None,
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => parsed.artifact = PathBuf::from(next_value(&mut args, &arg)?),
            "--output" => parsed.output = PathBuf::from(next_value(&mut args, &arg)?),
            "--batches" => parsed.batches = next_value(&mut args, &arg)?.split(',')
                .map(|value| value.parse().map_err(|_| "invalid batch size".to_string()))
                .collect::<Result<_, _>>()?,
            "--prompt-tokens" => parsed.prompt_tokens = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid prompt token count".to_string())?,
            "--decode-tokens" => parsed.decode_tokens = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid decode token count".to_string())?,
            "--warmup" => parsed.warmup = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid warmup count".to_string())?,
            "--iterations" => parsed.iterations = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid iteration count".to_string())?,
            "--quantization" => parsed.quantization = Some(next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid quantization bits".to_string())?),
            _ => return Err(format!("unknown inference benchmark argument: {arg}")),
        }
    }
    if parsed.batches.is_empty() || parsed.batches.contains(&0) {
        return Err("batch sizes must be positive".into());
    }
    if parsed.prompt_tokens == 0 || parsed.iterations == 0 {
        return Err("prompt tokens and iterations must be positive".into());
    }
    if parsed.quantization.is_some_and(|bits| !matches!(bits, 4 | 8)) {
        return Err("quantization must be 4 or 8 bits".into());
    }
    Ok(parsed)
}

fn benchmark_model<B: burn::tensor::backend::Backend, L: ForwardLayer<B>>(model: Gpt<B, L>,
    tokenizer: nanochat_burn::tokenizer::BpeTokenizer, args: &Args, model_bytes: u64,
    quantization_error: Option<f32>, device: &B::Device,
    memory_usage: &impl Fn() -> Option<DeviceMemoryUsage>) -> Vec<InferenceBenchmark> {
    let engine = InferenceEngine::new(model, tokenizer.clone());
    let bos = tokenizer.get_bos_token_id();
    let prompt = vec![bos; args.prompt_tokens];
    args.batches.iter().map(|&batch| benchmark_inference(&engine, &prompt,
        InferenceBenchmarkConfig {
            batch_size: batch, decode_tokens: args.decode_tokens,
            warmup: args.warmup, iterations: args.iterations, model_bytes,
            quantization_bits: args.quantization,
            quantization_max_error: quantization_error,
        }, device, memory_usage)).collect()
}

pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult {
    let (args, device) = (parse_args(args)?, init_device());
    let artifact = load_artifact(&args.artifact, &device)
        .map_err(|error| format!("failed to load {:?}: {error}", args.artifact))?;
    let model_bytes = fs::metadata(args.artifact.join(&artifact.manifest.model_file))
        .map_err(|error| format!("failed to stat model: {error}"))?.len();

    let reports = if let Some(bits) = args.quantization {
        let prompt = Tensor::<InferBackend, 2, Int>::from_data(TensorData::new(
            vec![artifact.tokenizer.get_bos_token_id() as i32; args.prompt_tokens],
            [1, args.prompt_tokens]), &device);
        let baseline = nanochat_burn::common::tensor_data_to_f32_vec(
            artifact.model.forward(prompt.clone()).into_data());
        let quantized = artifact.model.quantize(bits, 0);
        let quantized_logits = nanochat_burn::common::tensor_data_to_f32_vec(
            quantized.forward(prompt).into_data());
        let error = baseline.into_iter().zip(quantized_logits)
            .map(|(left, right)| (left - right).abs()).fold(0.0, f32::max);
        benchmark_model(quantized, artifact.tokenizer, &args, model_bytes, Some(error), &device,
            &|| device_memory_usage(&device))
    } else {
        benchmark_model(artifact.model, artifact.tokenizer, &args, model_bytes, None, &device,
            &|| device_memory_usage(&device))
    };

    write_report(&args.output, &reports)?;
    for report in &reports {
        println!("batch={} prefill={:.2}ms ttft={:.2}ms tpot={:.2}ms decode={:.2} tok/s cache={:.2}MiB device={}",
            report.batch_size, report.prefill_latency_ms, report.time_to_first_token_ms,
            report.median_decode_step_ms, report.decode_tokens_per_second,
            report.cache_bytes as f64 / 1_048_576.0,
            report.peak_device_bytes_reserved.map_or_else(|| "n/a".into(),
                |bytes| format!("{:.2}MiB", bytes as f64 / 1_048_576.0)));
    }
    println!("Benchmark saved to {}", args.output.display());
    Ok(())
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_benchmark_args() {
        let args = parse_args(["--batches".into(), "1,3".into(),
            "--quantization".into(), "4".into()]).unwrap();
        assert_eq!(args.batches, vec![1, 3]);
        assert_eq!(args.quantization, Some(4));
        assert!(parse_args(["--quantization".into(), "3".into()]).is_err());
    }
}
