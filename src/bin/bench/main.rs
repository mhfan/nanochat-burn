mod infer;
mod ops;
mod speculative;

use std::{fs, path::Path};

use serde::Serialize;

type CliResult = Result<(), String>;

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{option} requires a value"))
}

fn write_report(path: &Path, report: &impl Serialize) -> CliResult {
    if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to serialize benchmark report: {error}"))?;
    fs::write(path, json)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: bench <command> [options]\n\ncommands:\n  infer        prefill, decode and quantized inference\n  speculative  target/draft speculative decoding\n  ops          RMSNorm, RoPE, Softmax and attention operators"
}

fn dispatch(args: impl IntoIterator<Item = String>) -> CliResult {
    let mut args = args.into_iter();
    match args.next().as_deref() {
        Some("infer") => infer::run(args),
        Some("speculative") => speculative::run(args),
        Some("ops") => ops::run(args),
        Some("--help") | Some("-h") => { println!("{}", usage()); Ok(()) }
        Some(command) => Err(format!("unknown benchmark command: {command}\n\n{}", usage())),
        None => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = dispatch(std::env::args().skip(1)) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_dispatch_rejects_unknown_command() {
        assert!(dispatch(["unknown".into()]).is_err());
    }
}
