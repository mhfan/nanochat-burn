
use std::{path::{Path, PathBuf}, time::Instant};
use burn::tensor::backend::AutodiffBackend;

use crate::{artifact::{MetricRecord, TrainingStage, append_metric,
        copy_metrics_through, load_artifact, load_resume_state, reset_metrics, save_artifact,
        save_experiment_config, save_resume_state},
    dataloader::{DistributedDataLoader, DistributedDataLoaderConfig},
    dataset::pretokenize_text_to_bin,
    engine::TrainingEngine,
    experiment::{ExperimentConfig, PretrainConfig, PretrainCorpus}, gpt::Gpt,
    tokenizer::BpeTokenizer,
};

const SYNTHETIC_PRETRAIN_CORPUS: &[&str] = &[
    "Rust is a high-performance system programming language designed for memory safety, \
     concurrency, and speed.",
    "Nanochat-burn uses the powerful Burn deep learning framework to run hardware-accelerated \
     GPT models.",
    "Autodiff backends in Burn automatically track operations to construct dynamic \
     computational graphs for backpropagation.",
    "Optimizers like Muon use orthogonalized parameter updates to accelerate parameter \
     convergence in deep transformer networks.",
    "Language models are trained using causal self-attention layers to autoregressively \
     predict the next token ID.",
    "Packed sequence datasets maximize GPU execution efficiency by batching multiple \
     conversations without padding.",
    "Reinforcement learning uses rollback policies and policy gradient updates to align base \
     pretrained models with human feedback.",
    "Evaluation harnesses execute generated programs in bounded subprocesses and record their \
     outputs for scoring.",
    "Pretraining establishes the core foundational language patterns, grammar, and generic \
     semantic knowledge inside LLM weights.",
    "Evaluation harnesses measure performance quantitatively using categorical argmax \
     selections and generative pass metrics.",
];

fn load_pretrain_corpus(config: &PretrainConfig) -> Result<Vec<String>, String> {
    let corpus = match &config.corpus {
        PretrainCorpus::Synthetic { .. } =>
            SYNTHETIC_PRETRAIN_CORPUS.iter().map(|text| (*text).to_string()).collect(),
        PretrainCorpus::Text { path, .. } => vec![std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read pretrain corpus {path:?}: {error}"))?],
    };
    if corpus.iter().all(|document| document.trim().is_empty()) {
        Err("pretrain corpus must contain text".into())
    } else { Ok(corpus) }
}

pub fn generate_pretrain_dataset(tokenizer: &BpeTokenizer, config: &PretrainConfig,
    corpus: &[String]) -> PathBuf {
    let (txt_path, bin_path) = (&config.text_path, &config.token_path);

    tracing::info!("Creating pretraining text dataset...");
    let full_text = format!("{}\n", corpus.join("\n")).repeat(config.corpus.repeats());

    if let Some(parent) = txt_path.parent() && !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).expect("Failed to create pretraining data directory");
    }
    if let Some(parent) = bin_path.parent() && !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).expect("Failed to create pretraining token directory");
    }
    std::fs::write(txt_path, &full_text).expect("Failed to write pretrain text");
    pretokenize_text_to_bin(txt_path, bin_path, tokenizer).expect("Failed to pretokenize dataset");
    tracing::info!("Pretraining dataset pretokenized to: {:?}", bin_path);

    bin_path.to_path_buf()
}

pub async fn run_pretraining<B: AutodiffBackend>(device: &B::Device,
    experiment: &ExperimentConfig) {
    let resume = std::env::var_os("NANOCHAT_RESUME_ARTIFACT").map(PathBuf::from);
    let output = std::env::var_os("NANOCHAT_OUTPUT_ARTIFACT").map(PathBuf::from)
        .or_else(|| resume.clone()).unwrap_or_else(|| experiment.artifacts.pretrain.clone());
    run_pretraining_at::<B>(device, experiment, resume.as_deref(), &output).await;
}

pub(crate) async fn run_pretraining_at<B: AutodiffBackend>(device: &B::Device,
    experiment: &ExperimentConfig, resume: Option<&Path>, output: &Path) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Foundational Pretraining        ");
    tracing::info!("=============================================");

    experiment.validate().unwrap_or_else(|error| panic!("invalid experiment config: {error}"));
    let config = &experiment.pretrain;
    let corpus = load_pretrain_corpus(config)
        .unwrap_or_else(|error| panic!("failed to load pretrain corpus: {error}"));
    let configured_training = config.training.resolve(config.model.sequence_len)
        .unwrap_or_else(|error| panic!("invalid pretrain training config: {error}"));

    let (tokenizer, mut engine) = if let Some(path) = resume {
        let artifact = load_artifact::<B>(path, device)
            .unwrap_or_else(|error| panic!("failed to load pretrain artifact: {error}"));
        assert_eq!(artifact.manifest.stage, TrainingStage::Pretrain,
            "pretraining can only resume a pretrain artifact");
        let training_config = artifact.config.training.clone()
            .expect("pretrain artifact does not contain a training config");
        assert_eq!(artifact.config.model, config.model,
            "resume artifact model config differs from experiment config");
        assert_eq!(training_config, configured_training,
            "resume artifact training config differs from experiment config");
        let (optimizer, trainer) =
            load_resume_state::<B>(path, artifact.config.model.n_layer, device)
                .unwrap_or_else(|error| panic!("failed to load pretrain state: {error}"));
        let engine = TrainingEngine::from_state(artifact.model, optimizer, training_config,
            &artifact.tokenizer, trainer);
        copy_metrics_through(path, output, engine.step)
            .unwrap_or_else(|error| panic!("failed to restore pretrain metrics: {error}"));
        tracing::info!("Resuming pretraining from {:?} at iteration {}", path, engine.step);
        (artifact.tokenizer, engine)
    } else {
        B::seed(device, experiment.seed);
        let tokenizer = BpeTokenizer::train_from_iterator(&corpus, config.model.vocab_size);
        assert_eq!(tokenizer.get_vocab_size(), config.model.vocab_size,
            "trained tokenizer vocabulary differs from model config");
        let model = Gpt::<B>::new(config.model.clone(), device);
        reset_metrics(output).unwrap_or_else(|error| panic!("failed to reset metrics: {error}"));
        let engine = TrainingEngine::new(model, configured_training, &tokenizer);
        (tokenizer, engine)
    };

    let bin_path = generate_pretrain_dataset(&tokenizer, config, &corpus);
    let training_config = engine.config.clone();
    let mut loader_config = DistributedDataLoaderConfig::single_process(
        training_config.device_batch_size, training_config.sequence_length);
    if let Some(position) = engine.dataloader_position {
        loader_config = loader_config.with_position(position);
    }
    let mut loader = DistributedDataLoader::new(vec![bin_path], loader_config);

    tracing::info!("Starting pretraining optimization iterations...");
    let start_time = Instant::now();
    let elapsed_before_resume = engine.total_training_time_secs;
    let checkpoint_interval = checkpoint_interval(config.checkpoint_interval);

    while engine.step < training_config.num_iterations {
        let i = engine.step + 1;
        let loss = engine.train_step(&mut loader, device).await;
        tracing::info!("Iteration {:02}/{:02} | Loss: {:.6}", i,
            training_config.num_iterations, loss);
        append_metric(output, &MetricRecord { stage: TrainingStage::Pretrain, step: i, loss,
            smoothed_loss: Some(loss), learning_rate: None, reward: None,
            elapsed_secs: elapsed_before_resume + start_time.elapsed().as_secs_f64(),
        }).unwrap_or_else(|error| panic!("failed to append pretrain metric: {error}"));
        if checkpoint_interval > 0 && i % checkpoint_interval == 0 &&
            i < training_config.num_iterations {
            save_pretrain_checkpoint(output, &engine, &tokenizer, experiment);
            tracing::info!("Saved pretraining checkpoint at iteration {}", i);
        }
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   Pretraining Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    save_pretrain_checkpoint(output, &engine, &tokenizer, experiment);
    tracing::info!("Pretraining artifact saved to {:?}", output);
}

fn checkpoint_interval(default: usize) -> usize {
    std::env::var("NANOCHAT_CHECKPOINT_INTERVAL").map_or(default, |value| value.parse()
        .unwrap_or_else(|_| panic!("NANOCHAT_CHECKPOINT_INTERVAL must be a non-negative integer")))
}

fn save_pretrain_checkpoint<B: AutodiffBackend>(output: &Path, engine: &TrainingEngine<B>,
    tokenizer: &BpeTokenizer, experiment: &ExperimentConfig) {
    save_artifact(output, TrainingStage::Pretrain, &engine.model, tokenizer, Some(&engine.config))
        .unwrap_or_else(|error| panic!("failed to save pretrain artifact: {error}"));
    save_resume_state(output, &engine.optimizer, &engine.state())
        .unwrap_or_else(|error| panic!("failed to save pretrain state: {error}"));
    save_experiment_config(output, experiment)
        .unwrap_or_else(|error| panic!("failed to save experiment config: {error}"));
}
