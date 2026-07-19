use std::path::PathBuf;

use nanochat_burn::{artifact::load_artifact, benchmark::benchmark_speculative,
    common::{InferBackend, init_device}, engine::speculative::SpeculativeInferenceEngine};

use super::{CliResult, next_value, write_report};

struct Args {
    target: PathBuf,
    draft: PathBuf,
    output: PathBuf,
    prompt_tokens: usize,
    decode_tokens: usize,
    draft_tokens: usize,
    iterations: usize,
}

impl Default for Args {
    fn default() -> Self { Self {
            target: "runs/sft".into(), draft: "runs/pretrain".into(),
            output: "runs/benchmarks/speculative.json".into(), prompt_tokens: 16,
            decode_tokens: 32, draft_tokens: 4, iterations: 3
    } }
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let (mut parsed, mut args) = (Args::default(), args.into_iter());
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--draft"  => parsed.draft  = next_value(&mut args, &arg)?.into(),
            "--target" => parsed.target = next_value(&mut args, &arg)?.into(),
            "--output" => parsed.output = next_value(&mut args, &arg)?.into(),
            "--prompt-tokens" => parsed.prompt_tokens = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid prompt token count".to_string())?,
            "--decode-tokens" => parsed.decode_tokens = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid decode token count".to_string())?,
            "--draft-tokens" => parsed.draft_tokens = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid draft token count".to_string())?,
            "--iterations" => parsed.iterations = next_value(&mut args, &arg)?.parse()
                .map_err(|_| "invalid iteration count".to_string())?,
            _ => return Err(format!("unknown speculative benchmark argument: {arg}")),
        }
    }
    if parsed.prompt_tokens == 0 || parsed.decode_tokens == 0 || parsed.draft_tokens == 0 ||
        parsed.iterations == 0 {
        return Err("prompt, decode, draft tokens and iterations must be positive".into());
    }
    Ok(parsed)
}

pub(super) fn run(args: impl IntoIterator<Item = String>) -> CliResult {
    let args = parse_args(args)?;
    let device = init_device();
    let target_artifact = load_artifact::<InferBackend>(&args.target, &device)
        .map_err(|error| format!("failed to load target {:?}: {error}", args.target))?;
    let draft_artifact = load_artifact(&args.draft, &device)
        .map_err(|error| format!("failed to load draft {:?}: {error}", args.draft))?;
    if target_artifact.tokenizer.get_vocab_size() != draft_artifact.tokenizer.get_vocab_size() {
        return Err("target/draft tokenizer mismatch".into());
    }
    let tokenizer = target_artifact.tokenizer;
    let prompt = vec![tokenizer.get_bos_token_id(); args.prompt_tokens];
    let engine = SpeculativeInferenceEngine::new(
        target_artifact.model, draft_artifact.model, tokenizer);
    let report = benchmark_speculative(&engine, &prompt, args.decode_tokens, args.draft_tokens,
        args.iterations, &device);
    write_report(&args.output, &report)?;
    println!("acceptance={:.2}% target={:.2} tok/s speculative={:.2} tok/s speedup={:.3}x",
        report.acceptance_rate * 100.0, report.target_tokens_per_second,
        report.speculative_tokens_per_second, report.speedup);
    println!("Benchmark saved to {}", args.output.display());
    Ok(())
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_speculative_benchmark_args() {
        let args = parse_args(["--target".into(), "target".into(), "--draft-tokens".into(),
            "8".into(), "--iterations".into(), "2".into()]).unwrap();
        assert_eq!(args.target, PathBuf::from("target"));
        assert_eq!(args.draft_tokens, 8);
        assert_eq!(args.iterations, 2);
        assert!(parse_args(["--decode-tokens".into(), "0".into()]).is_err());
    }
}
