use std::{fs, path::{Path, PathBuf}};

use serde::{Deserialize, Serialize};

use crate::{artifact::{ArtifactManifest, MetricRecord, TrainingStage}, engine::eval::EvalReport,
    experiment::{ExperimentConfig, RlAlgorithm}};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub path: PathBuf,
    pub stage: TrainingStage,
    pub rl_algorithm: Option<RlAlgorithm>,
    pub step: Option<usize>,
    pub loss: Option<f32>,
    pub bpb: Option<f32>,
    pub tokens_per_second: Option<f32>,
    pub model_bytes: u64,
    pub quality: Option<f32>,
    pub reward: Option<f32>,
    pub kl: Option<f32>,
    pub clip_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentReport { pub runs: Vec<RunSummary> }

pub fn summarize_run(path: impl AsRef<Path>) -> Result<RunSummary, String> {
    let path = path.as_ref();
    let manifest: ArtifactManifest = read_json(path.join("manifest.json"))?;
    let metrics = read_jsonl::<MetricRecord>(path.join(&manifest.metrics_file))?;
    let latest = metrics.last();
    let bpb = metrics.iter().rev().find_map(|metric| metric.bpb);
    let eval_path = path.join("eval.json");
    let quality = if eval_path.is_file() {
        read_json::<EvalReport>(eval_path)?.aggregate
    } else { None };
    let model_bytes = fs::metadata(path.join(&manifest.model_file))
        .map_err(|error| format!("failed to stat model in {path:?}: {error}"))?.len();
    let rl_algorithm = manifest.experiment_file.as_ref().map(|filename| {
        let config_path = path.join(filename);
        let contents = fs::read_to_string(&config_path)
            .map_err(|error| format!("failed to read {config_path:?}: {error}"))?;
        toml::from_str::<ExperimentConfig>(&contents)
            .map(|experiment| experiment.rl.algorithm)
            .map_err(|error| format!("failed to parse {config_path:?}: {error}"))
    }).transpose()?.filter(|_| manifest.stage == TrainingStage::ReinforcementLearning);
    Ok(RunSummary { path: path.to_path_buf(), stage: manifest.stage, rl_algorithm,
        step: latest.map(|metric| metric.step), loss: latest.map(|metric| metric.loss), bpb,
        tokens_per_second: latest.and_then(|metric| metric.tokens_per_second), model_bytes,
        quality, reward: latest.and_then(|metric| metric.reward),
        kl: latest.and_then(|metric| metric.kl),
        clip_fraction: latest.and_then(|metric| metric.clip_fraction),
    })
}

pub fn build_report(paths: &[PathBuf]) -> Result<ExperimentReport, String> {
    paths.iter().map(summarize_run).collect::<Result<Vec<_>, _>>()
        .map(|runs| ExperimentReport { runs })
}

pub fn write_report(report: &ExperimentReport, path: impl AsRef<Path>) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create report directory {parent:?}: {error}"))?;
    }
    let encoded = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to serialize experiment report: {error}"))?;
    fs::write(path, encoded).map_err(|error| format!("failed to write report {path:?}: {error}"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T, String> {
    let bytes = fs::read(&path).map_err(|error| format!("failed to read {path:?}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("failed to parse {path:?}: {error}"))
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<Vec<T>, String> {
    let contents = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {path:?}: {error}"))?;
    contents.lines().enumerate().filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| serde_json::from_str(line)
            .map_err(|error| format!("failed to parse {} line {}: {error}",
                path.display(), index + 1))).collect()
}
