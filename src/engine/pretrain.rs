
use std::{path::{Path, PathBuf}, time::Instant};
use burn::tensor::backend::AutodiffBackend;

use crate::{artifact::{MetricRecord, PRETRAIN_ARTIFACT, TrainingStage, append_metric,
        copy_metrics_through, load_artifact, load_resume_state, reset_metrics, save_artifact,
        save_resume_state},
    dataloader::{DistributedDataLoader, DistributedDataLoaderConfig},
    dataset::pretokenize_text_to_bin,
    engine::{TrainingConfig, TrainingEngine}, gpt::{Gpt, GptConfig}, tokenizer::BpeTokenizer,
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

pub fn generate_pretrain_dataset(tokenizer: &BpeTokenizer) -> PathBuf {
    let txt_path = Path::new("data/pretrain.txt");
    let bin_path = Path::new("data/pretrain.bin");

    tracing::info!("Creating synthetic pretraining text dataset...");
    let full_text = format!("{} ", SYNTHETIC_PRETRAIN_CORPUS.join(" ")).repeat(10);

    std::fs::create_dir_all("data").expect("Failed to create data directory");
    std::fs::write(txt_path, &full_text).expect("Failed to write synthetic pretrain text");
    pretokenize_text_to_bin(txt_path, bin_path, tokenizer).expect("Failed to pretokenize dataset");
    tracing::info!("Synthetic pretraining dataset pretokenized to: {:?}", bin_path);

    bin_path.to_path_buf()
}

pub async fn run_pretraining<B: AutodiffBackend>(device: &B::Device) {
    tracing::info!("=============================================");
    tracing::info!("   Starting Foundational Pretraining        ");
    tracing::info!("=============================================");

    let resume = std::env::var_os("NANOCHAT_RESUME_ARTIFACT").map(PathBuf::from);
    let output = std::env::var_os("NANOCHAT_OUTPUT_ARTIFACT").map(PathBuf::from)
        .or_else(|| resume.clone()).unwrap_or_else(|| PathBuf::from(PRETRAIN_ARTIFACT));

    let (tokenizer, mut engine) = if let Some(path) = &resume {
        let artifact = load_artifact::<B>(path, device)
            .unwrap_or_else(|error| panic!("failed to load pretrain artifact: {error}"));
        assert_eq!(artifact.manifest.stage, TrainingStage::Pretrain,
            "pretraining can only resume a pretrain artifact");
        let training_config = artifact.config.training.clone()
            .expect("pretrain artifact does not contain a training config");
        let (optimizer, trainer) =
            load_resume_state::<B>(path, artifact.config.model.n_layer, device)
                .unwrap_or_else(|error| panic!("failed to load pretrain state: {error}"));
        let engine = TrainingEngine::from_state(artifact.model, optimizer, training_config,
            &artifact.tokenizer, trainer);
        copy_metrics_through(path, &output, engine.step)
            .unwrap_or_else(|error| panic!("failed to restore pretrain metrics: {error}"));
        tracing::info!("Resuming pretraining from {:?} at iteration {}", path, engine.step);
        (artifact.tokenizer, engine)
    } else {
        let tokenizer = BpeTokenizer::train_from_iterator(
            SYNTHETIC_PRETRAIN_CORPUS.iter().copied(), 512);
        let config = GptConfig { sequence_len: 256, vocab_size: tokenizer.get_vocab_size(),
            n_layer: 2, n_head: 2, n_kv_head: 1, n_embd: 16,
            window_pattern: "L".to_string(), quantization: None,
        };
        let training_config = TrainingConfig {
            num_iterations: 15, warmup_steps: 3, warmdown_ratio: 0.3, final_lr_frac: 0.1,
            learning_rate: 1e-3, weight_decay: 0.1, device_batch_size: 2,
            sequence_length: config.sequence_len, total_batch_size: 4,
        };
        let model = Gpt::<B>::new(config, device);
        reset_metrics(&output).unwrap_or_else(|error| panic!("failed to reset metrics: {error}"));
        let engine = TrainingEngine::new(model, training_config, &tokenizer);
        (tokenizer, engine)
    };

    let bin_path = generate_pretrain_dataset(&tokenizer);
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
    let checkpoint_interval = checkpoint_interval();

    while engine.step < training_config.num_iterations {
        let i = engine.step + 1;
        let loss = engine.train_step(&mut loader, device).await;
        tracing::info!("Iteration {:02}/{:02} | Loss: {:.6}", i,
            training_config.num_iterations, loss);
        append_metric(&output, &MetricRecord { stage: TrainingStage::Pretrain, step: i, loss,
            smoothed_loss: Some(loss), learning_rate: None, reward: None,
            elapsed_secs: elapsed_before_resume + start_time.elapsed().as_secs_f64(),
        }).unwrap_or_else(|error| panic!("failed to append pretrain metric: {error}"));
        if checkpoint_interval > 0 && i % checkpoint_interval == 0 &&
            i < training_config.num_iterations {
            save_pretrain_checkpoint(&output, &engine, &tokenizer);
            tracing::info!("Saved pretraining checkpoint at iteration {}", i);
        }
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   Pretraining Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");

    save_pretrain_checkpoint(&output, &engine, &tokenizer);
    tracing::info!("Pretraining artifact saved to {:?}", output);
}

fn checkpoint_interval() -> usize {
    std::env::var("NANOCHAT_CHECKPOINT_INTERVAL").map_or(5, |value| value.parse()
        .unwrap_or_else(|_| panic!("NANOCHAT_CHECKPOINT_INTERVAL must be a non-negative integer")))
}

fn save_pretrain_checkpoint<B: AutodiffBackend>(output: &Path, engine: &TrainingEngine<B>,
    tokenizer: &BpeTokenizer) {
    save_artifact(output, TrainingStage::Pretrain, &engine.model, tokenizer, Some(&engine.config))
        .unwrap_or_else(|error| panic!("failed to save pretrain artifact: {error}"));
    save_resume_state(output, &engine.optimizer, &engine.state())
        .unwrap_or_else(|error| panic!("failed to save pretrain state: {error}"));
}
