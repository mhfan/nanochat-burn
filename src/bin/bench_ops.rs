use std::{fs, hint::black_box, path::PathBuf, time::Instant};

use burn::tensor::{Bool, Distribution, Element, Shape, Tensor, TensorData, activation,
    backend::Backend};
use nanochat_burn::{common::{ModelBackend, init_device, scalar_to_f32},
    gpt::{apply_rotary_emb, rms_norm, scaled_dot_product_attention_burn,
        scaled_dot_product_attention_reference},
};
use serde::Serialize;

#[derive(Debug)]
struct Args {
    output: PathBuf,
    batch_size: usize,
    sequence_len: usize,
    n_head: usize,
    head_dim: usize,
    warmup: usize,
    iterations: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self { output: "runs/benchmarks/operators.json".into(), batch_size: 2,
            sequence_len: 256, n_head: 2, head_dim: 8, warmup: 3, iterations: 20 }
    }
}

#[derive(Debug, Serialize)]
struct OperatorBenchmark {
    backend: String,
    batch_size: usize,
    sequence_len: usize,
    n_head: usize,
    head_dim: usize,
    warmup: usize,
    iterations: usize,
    rms_norm_latency_ms: f64,
    rope_latency_ms: f64,
    softmax_latency_ms: f64,
    attention_reference_latency_ms: f64,
    attention_masked_latency_ms: f64,
    attention_causal_latency_ms: f64,
    attention_masked_speedup: f64,
    attention_causal_speedup: f64,
    attention_masked_max_error: f32,
    attention_causal_max_error: f32,
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{option} requires a value"))
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let value = match arg.as_str() {
            "--output" => {
                parsed.output = next_value(&mut args, &arg)?.into();
                continue;
            }
            "--batch" | "--sequence" | "--heads" | "--head-dim" | "--warmup" |
            "--iterations" => next_value(&mut args, &arg)?,
            _ => return Err(format!("unknown operator benchmark argument: {arg}")),
        };
        let value = value.parse::<usize>()
            .map_err(|_| format!("{arg} requires a non-negative integer"))?;
        match arg.as_str() {
            "--batch" => parsed.batch_size = value,
            "--sequence" => parsed.sequence_len = value,
            "--heads" => parsed.n_head = value,
            "--head-dim" => parsed.head_dim = value,
            "--warmup" => parsed.warmup = value,
            "--iterations" => parsed.iterations = value,
            _ => unreachable!(),
        }
    }
    if parsed.batch_size == 0 || parsed.sequence_len == 0 || parsed.n_head == 0 ||
        parsed.head_dim == 0 || parsed.iterations == 0 {
        return Err("batch, sequence, heads, head-dim and iterations must be positive".into());
    }
    if !parsed.head_dim.is_multiple_of(2) {
        return Err("head-dim must be even for RoPE".into());
    }
    Ok(parsed)
}

fn measure<B: Backend, const D: usize>(device: &B::Device, warmup: usize, iterations: usize,
    mut operation: impl FnMut() -> Tensor<B, D>) -> f64 {
    for _ in 0..warmup {
        let output = operation();
        B::sync(device).expect("failed to synchronize benchmark backend");
        black_box(output);
    }
    B::sync(device).expect("failed to synchronize benchmark backend");
    let start = Instant::now();
    for _ in 0..iterations {
        let output = operation();
        B::sync(device).expect("failed to synchronize benchmark backend");
        black_box(output);
    }
    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
}

fn causal_mask<B: Backend>(sequence_len: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
    let values = (0..sequence_len).flat_map(|row|
        (0..sequence_len).map(move |column| column > row)).collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, Shape::new([1, 1, sequence_len, sequence_len])),
        device)
}

fn benchmark<B: Backend>(args: &Args, device: &B::Device) -> OperatorBenchmark {
    B::seed(device, 42);
    let shape = [args.batch_size, args.sequence_len, args.n_head, args.head_dim];
    let input = Tensor::<B, 4>::random(shape, Distribution::Normal(0.0, 1.0), device);
    let half_dim = args.head_dim / 2;
    let cos = Tensor::<B, 4>::random([1, args.sequence_len, 1, half_dim],
        Distribution::Uniform(-1.0, 1.0), device);
    let sin = Tensor::<B, 4>::random([1, args.sequence_len, 1, half_dim],
        Distribution::Uniform(-1.0, 1.0), device);
    let scores = Tensor::<B, 4>::random(
        [args.batch_size, args.n_head, args.sequence_len, args.sequence_len],
        Distribution::Normal(0.0, 1.0), device);
    let (q, k, v) = (input.clone().swap_dims(1, 2), input.clone().swap_dims(1, 2),
        input.clone().swap_dims(1, 2));
    let mask = causal_mask(args.sequence_len, device);
    let eps = B::FloatElem::dtype().finfo().expect("float backend dtype").epsilon as f32;

    let rms_norm_latency_ms = measure(device, args.warmup, args.iterations,
        || rms_norm(input.clone(), eps));
    let rope_latency_ms = measure(device, args.warmup, args.iterations,
        || apply_rotary_emb(input.clone(), cos.clone(), sin.clone()));
    let softmax_latency_ms = measure(device, args.warmup, args.iterations,
        || activation::softmax(scores.clone(), 3));
    let attention_reference_latency_ms = measure(device, args.warmup, args.iterations, ||
        scaled_dot_product_attention_reference(q.clone(), k.clone(), v.clone(), mask.clone()));
    let attention_masked_latency_ms = measure(device, args.warmup, args.iterations, ||
        scaled_dot_product_attention_burn(
            q.clone(), k.clone(), v.clone(), Some(mask.clone()), false));
    let attention_causal_latency_ms = measure(device, args.warmup, args.iterations, ||
        scaled_dot_product_attention_burn(q.clone(), k.clone(), v.clone(), None, true));

    let reference = scaled_dot_product_attention_reference(
        q.clone(), k.clone(), v.clone(), mask.clone());
    let masked = scaled_dot_product_attention_burn(
        q.clone(), k.clone(), v.clone(), Some(mask), false);
    let causal = scaled_dot_product_attention_burn(q, k, v, None, true);
    let attention_masked_max_error = scalar_to_f32(
        (reference.clone() - masked).abs().max().into_scalar());
    let attention_causal_max_error = scalar_to_f32(
        (reference - causal).abs().max().into_scalar());

    OperatorBenchmark { backend: B::name(device), batch_size: args.batch_size,
        sequence_len: args.sequence_len, n_head: args.n_head, head_dim: args.head_dim,
        warmup: args.warmup, iterations: args.iterations, rms_norm_latency_ms,
        rope_latency_ms, softmax_latency_ms, attention_reference_latency_ms,
        attention_masked_latency_ms, attention_causal_latency_ms,
        attention_masked_speedup: attention_reference_latency_ms / attention_masked_latency_ms,
        attention_causal_speedup: attention_reference_latency_ms / attention_causal_latency_ms,
        attention_masked_max_error, attention_causal_max_error }
}

fn main() {
    let args = parse_args(std::env::args().skip(1)).unwrap_or_else(|error| panic!("{error}"));
    let device = init_device();
    let report = benchmark::<ModelBackend>(&args, &device);
    if let Some(parent) = args.output.parent() && !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&args.output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("backend={} shape=[{}, {}, {}, {}]", report.backend, report.batch_size,
        report.sequence_len, report.n_head, report.head_dim);
    println!("RMSNorm={:.4}ms RoPE={:.4}ms Softmax={:.4}ms",
        report.rms_norm_latency_ms, report.rope_latency_ms, report.softmax_latency_ms);
    println!("Attention reference={:.4}ms masked={:.4}ms ({:.2}x) causal={:.4}ms ({:.2}x)",
        report.attention_reference_latency_ms, report.attention_masked_latency_ms,
        report.attention_masked_speedup, report.attention_causal_latency_ms,
        report.attention_causal_speedup);
    println!("Attention max error: masked={:.6} causal={:.6}",
        report.attention_masked_max_error, report.attention_causal_max_error);
    println!("Benchmark saved to {}", args.output.display());
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_operator_benchmark_args() {
        let args = parse_args(["--batch".into(), "4".into(), "--head-dim".into(),
            "16".into(), "--warmup".into(), "0".into()]).unwrap();
        assert_eq!(args.batch_size, 4);
        assert_eq!(args.head_dim, 16);
        assert_eq!(args.warmup, 0);
        assert!(parse_args(["--head-dim".into(), "7".into()]).is_err());
        assert!(parse_args(["--iterations".into(), "0".into()]).is_err());
    }
}
