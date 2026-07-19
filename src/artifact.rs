use std::{fs, env, io::Write, path::{Path, PathBuf}, time::{SystemTime, UNIX_EPOCH}};

use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::{checkpoint::{load_safetensors_to_gpt, save_gpt_to_safetensors},
    engine::{TrainerState, TrainingConfig}, experiment::{ArtifactPaths, ExperimentConfig},
    gpt::{Gpt, GptConfig}, optim::MuonAdamW, tokenizer::BpeTokenizer,
};

pub const PRETRAIN_ARTIFACT: &str = "runs/pretrain";
pub const SFT_ARTIFACT: &str = "runs/sft";
pub const RL_ARTIFACT: &str = "runs/rl";
const SCHEMA_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MODEL_FILE: &str = "model.safetensors";
const OPTIMIZER_FILE: &str = "optimizer.safetensors";
const TRAINER_STATE_FILE: &str = "trainer-state.json";
const EXPERIMENT_FILE: &str = "experiment.toml";
const METRICS_FILE: &str = "metrics.jsonl";

impl Default for ArtifactPaths {
    fn default() -> Self {
        Self { pretrain: PRETRAIN_ARTIFACT.into(), sft: SFT_ARTIFACT.into(),
            rl: RL_ARTIFACT.into() }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStage { Pretrain, Sft, ReinforcementLearning }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub stage: TrainingStage,
    pub created_unix_secs: u64,
    pub config_file: String,
    pub tokenizer_file: String,
    pub model_file: String,
    #[serde(default = "default_metrics_file")]
    pub metrics_file: String,
    #[serde(default)]
    pub optimizer_file: Option<String>,
    #[serde(default)]
    pub trainer_state_file: Option<String>,
    #[serde(default)]
    pub experiment_file: Option<String>,
    #[serde(default)]
    pub rollouts_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConfig {
    pub model: GptConfig,
    pub training: Option<TrainingConfig>,
}

pub struct LoadedArtifact<B: Backend> {
    pub model: Gpt<B>,
    pub tokenizer: BpeTokenizer,
    pub manifest: ArtifactManifest,
    pub config: ArtifactConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub stage: TrainingStage,
    pub step: usize,
    pub loss: f32,
    pub smoothed_loss: Option<f32>,
    pub learning_rate: Option<f32>,
    pub reward: Option<f32>,
    pub elapsed_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpb: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kl: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_fraction: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_length: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_rate: Option<f32>,
}

fn default_metrics_file() -> String { METRICS_FILE.to_string() }

pub fn path_from_env(variable: &str, default: impl Into<PathBuf>) -> PathBuf {
    env::var_os(variable).map(PathBuf::from).unwrap_or_else(|| default.into())
}

pub fn inference_artifact_path(paths: &ArtifactPaths) -> PathBuf {
    select_inference_artifact([paths.rl.clone(), paths.sft.clone(), paths.pretrain.clone()])
}

fn select_inference_artifact(paths: [PathBuf; 3]) -> PathBuf {
    if let Some(path) = env::var_os("NANOCHAT_ARTIFACT") { return PathBuf::from(path); }
    let fallback = paths[0].clone();
    paths.into_iter().find(|path| path.join(MANIFEST_FILE).is_file()).unwrap_or(fallback)
}

pub fn reset_metrics(root: impl AsRef<Path>) -> Result<(), String> {
    let root = root.as_ref();
    fs::create_dir_all(root)
        .map_err(|error| format!("failed to create artifact directory {root:?}: {error}"))?;
    match fs::remove_file(root.join(METRICS_FILE)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to reset metrics: {error}")),
    }
}

pub fn append_metric(root: impl AsRef<Path>, metric: &MetricRecord) -> Result<(), String> {
    let path = root.as_ref().join(METRICS_FILE);
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)
        .map_err(|error| format!("failed to open {path:?}: {error}"))?;
    serde_json::to_writer(&mut file, metric)
        .map_err(|error| format!("failed to serialize metric: {error}"))?;
    file.write_all(b"\n").map_err(|error| format!("failed to write metric: {error}"))
}

pub fn copy_metrics_through(source: impl AsRef<Path>, destination: impl AsRef<Path>,
    completed_step: usize) -> Result<(), String> {
    let source = source.as_ref();
    let manifest: ArtifactManifest = read_json(source.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;
    let path = source.join(&manifest.metrics_file);
    let contents = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {path:?}: {error}"))?;
    let mut retained = String::new();
    for (index, line) in contents.lines().enumerate() {
        let metric: MetricRecord = serde_json::from_str(line)
            .map_err(|error| format!("failed to parse metric line {}: {error}", index + 1))?;
        if metric.step <= completed_step {
            retained.push_str(line);
            retained.push('\n');
        }
    }

    let destination = destination.as_ref();
    fs::create_dir_all(destination)
        .map_err(|error| format!("failed to create artifact directory {destination:?}: {error}"))?;
    fs::write(destination.join(METRICS_FILE), retained)
        .map_err(|error| format!("failed to restore metrics: {error}"))
}

pub fn save_artifact<B: Backend>(root: impl AsRef<Path>, stage: TrainingStage,
    model: &Gpt<B>, tokenizer: &BpeTokenizer, training: Option<&TrainingConfig>)
    -> Result<(), String> {
    let root = root.as_ref();
    fs::create_dir_all(root)
        .map_err(|error| format!("failed to create artifact directory {root:?}: {error}"))?;
    match fs::remove_file(root.join(MANIFEST_FILE)) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to invalidate existing artifact: {error}")),
    }

    let config = ArtifactConfig { model: model.config.clone(), training: training.cloned() };
    write_json(root.join(CONFIG_FILE), &config)?;
    tokenizer.save(root.join(TOKENIZER_FILE))
        .map_err(|error| format!("failed to save tokenizer: {error}"))?;
    save_gpt_to_safetensors(model, &root.join(MODEL_FILE))?;

    let created_unix_secs = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?.as_secs();
    let manifest = ArtifactManifest {
        schema_version: SCHEMA_VERSION, stage, created_unix_secs,
        config_file: CONFIG_FILE.to_string(), tokenizer_file: TOKENIZER_FILE.to_string(),
        model_file: MODEL_FILE.to_string(), metrics_file: METRICS_FILE.to_string(),
        optimizer_file: None, trainer_state_file: None, experiment_file: None,
        rollouts_file: None,
    };
    write_json(root.join(MANIFEST_FILE), &manifest)
}

pub fn set_rollouts_file(root: impl AsRef<Path>, filename: &str) -> Result<(), String> {
    let root = root.as_ref();
    if filename.is_empty() || !root.join(filename).is_file() {
        return Err(format!("rollout file {filename:?} does not exist in {root:?}"));
    }
    let mut manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;
    manifest.rollouts_file = Some(filename.to_string());
    write_json(root.join(MANIFEST_FILE), &manifest)
}

pub fn save_resume_state<B: AutodiffBackend>(root: impl AsRef<Path>,
    optimizer: &MuonAdamW<B>, trainer: &TrainerState) -> Result<(), String> {
    let root = root.as_ref();
    let mut manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;

    optimizer.save_state(root.join(OPTIMIZER_FILE))?;
    write_json(root.join(TRAINER_STATE_FILE), trainer)?;
    manifest.optimizer_file = Some(OPTIMIZER_FILE.to_string());
    manifest.trainer_state_file = Some(TRAINER_STATE_FILE.to_string());
    write_json(root.join(MANIFEST_FILE), &manifest)
}

pub fn load_resume_state<B: AutodiffBackend>(root: impl AsRef<Path>, n_layer: usize,
    device: &B::Device) -> Result<(MuonAdamW<B>, TrainerState), String> {
    let root = root.as_ref();
    let manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;
    let (optimizer_file, trainer_state_file) =
        match (&manifest.optimizer_file, &manifest.trainer_state_file) {
            (Some(optimizer), Some(trainer)) => (optimizer, trainer),
            (None, None) => return Err("artifact does not contain resumable training state".into()),
            _ => return Err("artifact contains incomplete resumable training state".into()),
        };
    let optimizer = MuonAdamW::load_state(root.join(optimizer_file), n_layer, device)?;
    let trainer = read_json(root.join(trainer_state_file))?;
    Ok((optimizer, trainer))
}

pub fn save_experiment_config(root: impl AsRef<Path>, config: &ExperimentConfig)
    -> Result<(), String> {
    let root = root.as_ref();
    let mut manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;
    config.save(root.join(EXPERIMENT_FILE))?;
    manifest.experiment_file = Some(EXPERIMENT_FILE.to_string());
    write_json(root.join(MANIFEST_FILE), &manifest)
}

pub fn load_experiment_config(root: impl AsRef<Path>) -> Result<ExperimentConfig, String> {
    let root = root.as_ref();
    let manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;
    let file = manifest.experiment_file
        .ok_or_else(|| "artifact does not contain an experiment config".to_string())?;
    ExperimentConfig::load(root.join(file))
}

pub fn load_artifact<B: Backend>(root: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedArtifact<B>, String> {
    let root = root.as_ref();
    let manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    validate_manifest(&manifest)?;

    let config: ArtifactConfig = read_json(root.join(&manifest.config_file))?;
    config.model.validate().map_err(|error| format!("invalid model config: {error}"))?;
    let tokenizer = BpeTokenizer::load(root.join(&manifest.tokenizer_file))
        .map_err(|error| format!("failed to load tokenizer: {error}"))?;
    if tokenizer.get_vocab_size() != config.model.vocab_size {
        return Err(format!("tokenizer vocabulary size {} does not match model vocabulary size {}",
            tokenizer.get_vocab_size(), config.model.vocab_size));
    }

    let mut model = Gpt::new(config.model.clone(), device);
    load_safetensors_to_gpt(&mut model, &root.join(&manifest.model_file), device)?;
    Ok(LoadedArtifact { model, tokenizer, manifest, config })
}

fn validate_manifest(manifest: &ArtifactManifest) -> Result<(), String> {
    if manifest.schema_version == SCHEMA_VERSION { Ok(()) } else {
        Err(format!("artifact schema {} is unsupported; expected {SCHEMA_VERSION}",
            manifest.schema_version))
    }
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), String> {
    let file = fs::File::create(&path)
        .map_err(|error| format!("failed to create {path:?}: {error}"))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|error| format!("failed to serialize {path:?}: {error}"))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T, String> {
    let file = fs::File::open(&path)
        .map_err(|error| format!("failed to open {path:?}: {error}"))?;
    serde_json::from_reader(file)
        .map_err(|error| format!("failed to parse {path:?}: {error}"))
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_artifact_roundtrip() {
        use burn::tensor::Tensor;
        use crate::{common::{TrainBackend, InferBackend,
                tensor_data_to_f32_vec}, dataloader::DataLoaderPosition,
            experiment::DEFAULT_EXPERIMENT_CONFIG, optim::{AdamWState, OptimizerKind},
        };

        let device = Default::default();
        let tokenizer = BpeTokenizer::train_from_iterator(["artifact roundtrip"], 280);
        let config = GptConfig { sequence_len: 8, vocab_size: tokenizer.get_vocab_size(),
            n_layer: 1, n_head: 2, n_kv_head: 1, n_embd: 16,
            window_pattern: "L".to_string(), features: Default::default(), quantization: None,
        };
        let model = Gpt::<InferBackend>::new(config.clone(), &device);
        let root = env::temp_dir().join(format!(
            "nanochat-artifact-test-{}", std::process::id()));

        reset_metrics(&root).unwrap();
        append_metric(&root, &MetricRecord { stage: TrainingStage::Pretrain, step: 1,
            loss: 1.25, smoothed_loss: Some(1.25), learning_rate: Some(1e-3), reward: None,
            elapsed_secs: 0.5, bpb: None, tokens_per_second: Some(128.0), memory_bytes: None,
            quality: None, kl: None, clip_fraction: None, response_length: None,
            acceptance_rate: None,
        }).unwrap();
        save_artifact(&root, TrainingStage::Pretrain, &model, &tokenizer, None).unwrap();
        let mut optimizer = MuonAdamW::<TrainBackend>::new(
            config.n_layer, OptimizerKind::MuonAdamW);
        optimizer.wte = Some(AdamWState {
            exp_avg: Tensor::from_data([[1.0, 2.0]], &device),
            exp_avg_sq: Tensor::from_data([[3.0, 4.0]], &device),
        });
        let trainer = TrainerState { step: 3, smooth_train_loss: 0.75,
            total_training_time_secs: 1.5,
            dataloader_position: Some(DataLoaderPosition {
                shard_idx: 1, token_offset: 42, epoch: 2,
            }),
            rng_state: Some(99),
        };
        save_resume_state(&root, &optimizer, &trainer).unwrap();
        let experiment = ExperimentConfig::load(DEFAULT_EXPERIMENT_CONFIG).unwrap();
        save_experiment_config(&root, &experiment).unwrap();
        append_metric(&root, &MetricRecord { stage: TrainingStage::Pretrain, step: 4,
            loss: 0.5, smoothed_loss: Some(0.5), learning_rate: Some(1e-3), reward: None,
            elapsed_secs: 2.0, bpb: Some(1.2), tokens_per_second: Some(140.0),
            memory_bytes: None, quality: None, kl: None, clip_fraction: None,
            response_length: None, acceptance_rate: None,
        }).unwrap();
        copy_metrics_through(&root, &root, trainer.step).unwrap();
        let loaded = load_artifact::<InferBackend>(&root, &device).unwrap();
        let (optimizer, restored_trainer) =
            load_resume_state::<TrainBackend>(&root, config.n_layer, &device).unwrap();
        let restored_experiment = load_experiment_config(&root).unwrap();

        assert_eq!(loaded.manifest.stage, TrainingStage::Pretrain);
        assert_eq!(loaded.manifest.optimizer_file.as_deref(), Some(OPTIMIZER_FILE));
        assert_eq!(loaded.manifest.experiment_file.as_deref(), Some(EXPERIMENT_FILE));
        assert_eq!(loaded.config.model.sequence_len, config.sequence_len);
        assert_eq!(loaded.tokenizer.encode_ordinary("artifact"),
            tokenizer.encode_ordinary("artifact"));
        let metrics = fs::read_to_string(root.join(&loaded.manifest.metrics_file)).unwrap();
        assert_eq!(metrics.lines().count(), 1);
        let metric: MetricRecord = serde_json::from_str(metrics.trim()).unwrap();
        assert_eq!(metric.step, 1);
        assert_eq!(metric.loss, 1.25);
        assert_eq!(restored_trainer, trainer);
        assert_eq!(restored_experiment, experiment);
        assert_eq!(tensor_data_to_f32_vec(optimizer.wte.unwrap().exp_avg.into_data()),
            vec![1.0, 2.0]);
        fs::remove_dir_all(root).ok();
    }
}
