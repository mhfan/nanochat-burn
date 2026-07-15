use std::path::PathBuf;

use nanochat_burn::report::{build_report, write_report};

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(PathBuf, Vec<PathBuf>), String> {
    let (mut output, mut runs) = (PathBuf::from("runs/report.json"), Vec::new());
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--output" {
            output = args.next().map(PathBuf::from)
                .ok_or_else(|| "--output requires a path".to_string())?;
        } else if arg.starts_with('-') {
            return Err(format!("unknown report argument: {arg}"));
        } else { runs.push(PathBuf::from(arg)); }
    }
    if runs.is_empty() {
        runs = ["runs/pretrain", "runs/sft", "runs/rl"].into_iter()
            .map(PathBuf::from).filter(|path| path.join("manifest.json").is_file()).collect();
    }
    if runs.is_empty() { return Err("no artifact runs found".into()); }
    Ok((output, runs))
}

fn main() {
    let (output, runs) = parse_args(std::env::args().skip(1))
        .unwrap_or_else(|error| panic!("{error}"));
    let report = build_report(&runs).unwrap_or_else(|error| panic!("{error}"));
    write_report(&report, &output).unwrap_or_else(|error| panic!("{error}"));

    println!("stage\talgorithm\tstep\tloss\tbpb\ttokens/s\tmodel MiB\tquality");
    for run in &report.runs {
        println!("{:?}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}", run.stage,
            run.rl_algorithm.map_or_else(|| "-".into(), |value| format!("{value:?}")),
            display(run.step), display(run.loss), display(run.bpb),
            display(run.tokens_per_second), run.model_bytes as f64 / 1_048_576.0,
            display(run.quality));
    }
    println!("Report saved to {}", output.display());
}

fn display(value: Option<impl std::fmt::Display>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_report_args() {
        let (output, runs) = parse_args([
            "--output".into(), "out.json".into(), "runs/a".into()]).unwrap();
        assert_eq!(output, PathBuf::from("out.json"));
        assert_eq!(runs, vec![PathBuf::from("runs/a")]);
        assert!(parse_args(["--output".into()]).is_err());
    }
}
