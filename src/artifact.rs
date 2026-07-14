use std::{fs, io::Write, path::{Path, PathBuf}, time::{SystemTime, UNIX_EPOCH}};

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{checkpoint::{load_safetensors_to_gpt, save_gpt_to_safetensors},
    engine::TrainingConfig, gpt::{Gpt, GptConfig}, tokenizer::BpeTokenizer,
};

pub const PRETRAIN_ARTIFACT: &str = "runs/pretrain";
pub const SFT_ARTIFACT: &str = "runs/sft";
pub const RL_ARTIFACT: &str = "runs/rl";
const SCHEMA_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const MODEL_FILE: &str = "model.safetensors";
const METRICS_FILE: &str = "metrics.jsonl";

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
}

fn default_metrics_file() -> String { METRICS_FILE.to_string() }

pub fn path_from_env(variable: &str, default: &str) -> PathBuf {
    std::env::var_os(variable).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

pub fn inference_artifact_path() -> PathBuf {
    if let Some(path) = std::env::var_os("NANOCHAT_ARTIFACT") { return PathBuf::from(path); }
    [RL_ARTIFACT, SFT_ARTIFACT, PRETRAIN_ARTIFACT].into_iter()
        .map(PathBuf::from).find(|path| path.join(MANIFEST_FILE).is_file())
        .unwrap_or_else(|| PathBuf::from(RL_ARTIFACT))
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
    };
    write_json(root.join(MANIFEST_FILE), &manifest)
}

pub fn load_artifact<B: Backend>(root: impl AsRef<Path>, device: &B::Device)
    -> Result<LoadedArtifact<B>, String> {
    let root = root.as_ref();
    let manifest: ArtifactManifest = read_json(root.join(MANIFEST_FILE))?;
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(format!("artifact schema {} is unsupported; expected {SCHEMA_VERSION}",
            manifest.schema_version));
    }

    let config: ArtifactConfig = read_json(root.join(&manifest.config_file))?;
    config.model.validate().map_err(|error| format!("invalid model config: {error}"))?;
    let tokenizer = BpeTokenizer::load(root.join(&manifest.tokenizer_file))
        .map_err(|error| format!("failed to load tokenizer: {error}"))?;
    if tokenizer.get_vocab_size() != config.model.vocab_size {
        return Err(format!("tokenizer vocabulary size {} does not match model vocabulary size {}",
            tokenizer.get_vocab_size(), config.model.vocab_size));
    }

    let mut model = Gpt::<B>::new(config.model.clone(), device);
    load_safetensors_to_gpt(&mut model, &root.join(&manifest.model_file), device)?;
    Ok(LoadedArtifact { model, tokenizer, manifest, config })
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
        use crate::common::{ModelBackend, init_device};

        let device = init_device();
        let tokenizer = BpeTokenizer::train_from_iterator(["artifact roundtrip"], 280);
        let config = GptConfig { sequence_len: 8, vocab_size: tokenizer.get_vocab_size(),
            n_layer: 1, n_head: 2, n_kv_head: 1, n_embd: 16,
            window_pattern: "L".to_string(), quantization: None,
        };
        let model = Gpt::<ModelBackend>::new(config.clone(), &device);
        let root = std::env::temp_dir().join(format!(
            "nanochat-artifact-test-{}", std::process::id()));

        reset_metrics(&root).unwrap();
        append_metric(&root, &MetricRecord { stage: TrainingStage::Pretrain, step: 1,
            loss: 1.25, smoothed_loss: Some(1.25), learning_rate: Some(1e-3), reward: None,
            elapsed_secs: 0.5,
        }).unwrap();
        save_artifact(&root, TrainingStage::Pretrain, &model, &tokenizer, None).unwrap();
        let loaded = load_artifact::<ModelBackend>(&root, &device).unwrap();

        assert_eq!(loaded.manifest.stage, TrainingStage::Pretrain);
        assert_eq!(loaded.config.model.sequence_len, config.sequence_len);
        assert_eq!(loaded.tokenizer.encode_ordinary("artifact"),
            tokenizer.encode_ordinary("artifact"));
        let metrics = fs::read_to_string(root.join(&loaded.manifest.metrics_file)).unwrap();
        let metric: MetricRecord = serde_json::from_str(metrics.trim()).unwrap();
        assert_eq!(metric.step, 1);
        assert_eq!(metric.loss, 1.25);
        fs::remove_dir_all(root).ok();
    }
}
