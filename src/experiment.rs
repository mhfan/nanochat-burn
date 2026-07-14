use std::{fs, path::{Path, PathBuf}};

use serde::{Deserialize, Serialize};

use crate::{engine::TrainingConfig, gpt::GptConfig};

pub const EXPERIMENT_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_EXPERIMENT_CONFIG: &str = "configs/mini.toml";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    pub schema_version: u32,
    pub seed: u64,
    pub artifacts: ArtifactPaths,
    pub pretrain: PretrainConfig,
    pub sft: SftConfig,
    pub rl: RlConfig,
    pub eval: EvalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPaths {
    pub pretrain: PathBuf,
    pub sft: PathBuf,
    pub rl: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PretrainConfig {
    pub corpus: PretrainCorpus,
    pub text_path: PathBuf,
    pub token_path: PathBuf,
    pub checkpoint_interval: usize,
    pub model: GptConfig,
    pub training: StageTrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PretrainCorpus {
    Synthetic { repeats: usize },
    Text { path: PathBuf, repeats: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SftConfig {
    pub dataset_path: PathBuf,
    pub log_interval: usize,
    pub training: StageTrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StageTrainingConfig {
    pub num_iterations: usize,
    pub warmup_steps: usize,
    pub warmdown_ratio: f32,
    pub final_lr_frac: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub device_batch_size: usize,
    pub total_batch_size: usize,
    #[serde(default)]
    pub sequence_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RlConfig {
    pub dataset_path: PathBuf,
    pub num_steps: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_samples: usize,
    pub max_generation_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalConfig {
    pub max_generation_tokens: usize,
    pub tasks: Vec<EvalTaskConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvalTaskConfig {
    pub name: String,
    pub path: PathBuf,
    pub kind: EvalTaskKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalTaskKind { Categorical, Generative, HumanEval }

impl ExperimentConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read experiment config {path:?}: {error}"))?;
        let config: Self = toml::from_str(&contents)
            .map_err(|error| format!("failed to parse experiment config {path:?}: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() && !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create config directory {parent:?}: {error}"))?;
        }
        let contents = toml::to_string_pretty(self)
            .map_err(|error| format!("failed to serialize experiment config: {error}"))?;
        fs::write(path, contents)
            .map_err(|error| format!("failed to write experiment config {path:?}: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION {
            return Err(format!("experiment schema {} is unsupported; expected {}",
                self.schema_version, EXPERIMENT_SCHEMA_VERSION));
        }
        self.artifacts.validate()?;
        self.pretrain.validate()?;
        self.sft.validate()?;
        self.rl.validate()?;
        self.eval.validate()
    }
}

impl ArtifactPaths {
    fn validate(&self) -> Result<(), String> {
        validate_path(&self.pretrain, "pretrain artifact path")?;
        validate_path(&self.sft, "sft artifact path")?;
        validate_path(&self.rl, "rl artifact path")
    }
}

impl PretrainConfig {
    fn validate(&self) -> Result<(), String> {
        if self.model.vocab_size <= 256 {
            return Err("pretrain model vocab_size must include byte and special tokens".into());
        }
        self.corpus.validate()?;
        validate_path(&self.text_path, "pretrain text_path")?;
        validate_path(&self.token_path, "pretrain token_path")?;
        self.model.validate().map_err(|error| format!("invalid pretrain model: {error}"))?;
        self.training.resolve(self.model.sequence_len)
            .map_err(|error| format!("invalid pretrain training: {error}"))?;
        Ok(())
    }
}

impl PretrainCorpus {
    pub fn repeats(&self) -> usize {
        match self { Self::Synthetic { repeats } | Self::Text { repeats, .. } => *repeats }
    }

    fn validate(&self) -> Result<(), String> {
        if self.repeats() == 0 { return Err("pretrain corpus repeats must be positive".into()); }
        if let Self::Text { path, .. } = self {
            validate_path(path, "pretrain corpus path")?;
        }
        Ok(())
    }
}

impl SftConfig {
    fn validate(&self) -> Result<(), String> {
        validate_path(&self.dataset_path, "sft dataset_path")?;
        if self.log_interval == 0 { return Err("sft log_interval must be positive".into()); }
        self.training.validate().map_err(|error| format!("invalid sft training: {error}"))?;
        if self.training.total_batch_size != self.training.device_batch_size {
            return Err("SFT currently requires total_batch_size to equal device_batch_size".into());
        }
        Ok(())
    }
}

impl StageTrainingConfig {
    pub fn resolve(&self, model_sequence_len: usize) -> Result<TrainingConfig, String> {
        let sequence_length = self.sequence_length.unwrap_or(model_sequence_len);
        if sequence_length > model_sequence_len {
            return Err(format!("sequence_length {sequence_length} exceeds model capacity {model_sequence_len}"));
        }
        let config = self.build(sequence_length);
        config.validate().map_err(str::to_string)?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        self.build(self.sequence_length.unwrap_or(1)).validate().map_err(str::to_string)
    }

    fn build(&self, sequence_length: usize) -> TrainingConfig {
        TrainingConfig { num_iterations: self.num_iterations, warmup_steps: self.warmup_steps,
            warmdown_ratio: self.warmdown_ratio, final_lr_frac: self.final_lr_frac,
            learning_rate: self.learning_rate, weight_decay: self.weight_decay,
            device_batch_size: self.device_batch_size, sequence_length,
            total_batch_size: self.total_batch_size,
        }
    }
}

impl RlConfig {
    fn validate(&self) -> Result<(), String> {
        validate_path(&self.dataset_path, "rl dataset_path")?;
        if self.num_steps == 0 { return Err("rl num_steps must be positive".into()); }
        if self.batch_size == 0 { return Err("rl batch_size must be positive".into()); }
        if self.num_samples == 0 { return Err("rl num_samples must be positive".into()); }
        if self.max_generation_tokens == 0 {
            return Err("rl max_generation_tokens must be positive".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate < 0.0 {
            return Err("rl learning_rate must be finite and non-negative".into());
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("rl temperature must be finite and non-negative".into());
        }
        if self.top_k == Some(0) { return Err("rl top_k must be positive when set".into()); }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err("rl repetition_penalty must be finite and positive".into());
        }
        Ok(())
    }
}

impl EvalConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_generation_tokens == 0 {
            return Err("eval max_generation_tokens must be positive".into());
        }
        if self.tasks.is_empty() { return Err("eval tasks must not be empty".into()); }
        for (index, task) in self.tasks.iter().enumerate() {
            if task.name.trim().is_empty() {
                return Err(format!("eval task {index} name must not be empty"));
            }
            validate_path(&task.path, &format!("eval task {index} path"))?;
        }
        Ok(())
    }
}

fn validate_path(path: &Path, name: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() { Err(format!("{name} must not be empty")) } else { Ok(()) }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_default_experiment_config_roundtrip() {
        let source = include_str!("../configs/mini.toml");
        let config: ExperimentConfig = toml::from_str(source).unwrap();
        config.validate().unwrap();
        let encoded = toml::to_string(&config).unwrap();
        let decoded: ExperimentConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(decoded, config);

        let path = std::env::temp_dir().join(format!(
            "nanochat-experiment-test-{}.toml", std::process::id()));
        config.save(&path).unwrap();
        assert_eq!(ExperimentConfig::load(&path).unwrap(), config);
        std::fs::remove_file(path).ok();
    }

    #[test] fn test_tiny_recipe_config() {
        let config: ExperimentConfig =
            toml::from_str(include_str!("../configs/tiny.toml")).unwrap();
        config.validate().unwrap();
        assert!(matches!(config.pretrain.corpus,
            PretrainCorpus::Text { repeats: 12, .. }));
        assert_eq!(config.eval.tasks.len(), 1);
        assert_eq!(config.eval.tasks[0].kind, EvalTaskKind::Categorical);
    }

    #[test] fn test_experiment_config_rejects_unknown_fields() {
        let source = format!("{}\nunknown = true\n", include_str!("../configs/mini.toml"));
        assert!(toml::from_str::<ExperimentConfig>(&source).is_err());
    }

    #[test] fn test_experiment_config_rejects_inconsistent_shared_values() {
        let mut config: ExperimentConfig =
            toml::from_str(include_str!("../configs/mini.toml")).unwrap();
        config.pretrain.training.sequence_length = Some(config.pretrain.model.sequence_len + 1);
        assert_eq!(config.validate().unwrap_err(), concat!(
            "invalid pretrain training: sequence_length 257 exceeds model capacity 256"));
        config.pretrain.training.sequence_length = None;
        config.sft.training.sequence_length = Some(128);
        assert_eq!(config.sft.training.resolve(config.pretrain.model.sequence_len)
            .unwrap().sequence_length, 128);
        config.artifacts.sft = PathBuf::new();
        assert_eq!(config.validate().unwrap_err(), "sft artifact path must not be empty");
    }
}
