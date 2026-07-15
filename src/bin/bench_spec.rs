use std::{fs, path::PathBuf};

use nanochat_burn::{artifact::load_artifact, benchmark::benchmark_speculative,
    common::{ModelBackend, init_device}, engine::speculative::SpeculativeInferenceEngine};

fn main() {
    let mut args = std::env::args().skip(1);
    let target = PathBuf::from(args.next().unwrap_or_else(|| "runs/sft".into()));
    let draft = PathBuf::from(args.next().unwrap_or_else(|| "runs/pretrain".into()));
    let output = PathBuf::from(args.next()
        .unwrap_or_else(|| "runs/benchmarks/speculative.json".into()));
    if args.next().is_some() {
        panic!("usage: bench_spec [target-artifact] [draft-artifact] [output]");
    }
    let device = init_device();
    let target_artifact = load_artifact::<ModelBackend>(&target, &device)
        .unwrap_or_else(|error| panic!("failed to load target {target:?}: {error}"));
    let draft_artifact = load_artifact::<ModelBackend>(&draft, &device)
        .unwrap_or_else(|error| panic!("failed to load draft {draft:?}: {error}"));
    assert_eq!(target_artifact.tokenizer.get_vocab_size(),
        draft_artifact.tokenizer.get_vocab_size(), "target/draft tokenizer mismatch");
    let tokenizer = target_artifact.tokenizer;
    let prompt = vec![tokenizer.get_bos_token_id(); 16];
    let engine = SpeculativeInferenceEngine::new(
        target_artifact.model, draft_artifact.model, tokenizer);
    let report = benchmark_speculative(&engine, &prompt, 32, 4, 3, &device);
    if let Some(parent) = output.parent() && !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("acceptance={:.2}% target={:.2} tok/s speculative={:.2} tok/s speedup={:.3}x",
        report.acceptance_rate * 100.0, report.target_tokens_per_second,
        report.speculative_tokens_per_second, report.speedup);
    println!("Benchmark saved to {}", output.display());
}
