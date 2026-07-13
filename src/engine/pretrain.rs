
use std::{path::{Path, PathBuf}, time::Instant};
use burn::tensor::backend::AutodiffBackend;

use crate::{dataloader::{DistributedDataLoader, DistributedDataLoaderConfig},
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

    // 1. Train tokenizer
    let corpus = vec![
        "BOS user assistant python output system pretraining deep learning transformer model \
         optimizer",
    ];
    let tokenizer = BpeTokenizer::train_from_iterator(corpus, 512);

    // 2. Generate and pretokenize data
    let bin_path = generate_pretrain_dataset(&tokenizer);

    // 3. Configure training
    let config = GptConfig { sequence_len: 16, vocab_size: tokenizer.get_vocab_size(),
        n_layer: 2, n_head: 2, n_kv_head: 1, n_embd: 16,
        window_pattern: "L".to_string(), quantization: None,
    };

    let training_config = TrainingConfig {
        num_iterations: 15, warmup_steps: 3, warmdown_ratio: 0.3, final_lr_frac: 0.1,
        learning_rate: 1e-3, weight_decay: 0.1, device_batch_size: 2,
        sequence_length: config.sequence_len, total_batch_size: 4,
    };

    let model = Gpt::<B>::new(config, device);
    let mut engine = TrainingEngine::new(model, training_config.clone(), &tokenizer);

    // 4. Initialize DataLoader
    let loader_config = DistributedDataLoaderConfig::single_process(
        training_config.device_batch_size, training_config.sequence_length);
    let mut loader = DistributedDataLoader::new(vec![bin_path], loader_config);

    tracing::info!("Starting pretraining optimization iterations...");
    let start_time = Instant::now();

    for i in 1..=training_config.num_iterations {
        let loss = engine.train_step(&mut loader, device).await;
        tracing::info!("Iteration {:02}/{:02} | Loss: {:.6}", i,
            training_config.num_iterations, loss);
    }

    let elapsed = start_time.elapsed();
    tracing::info!("=============================================");
    tracing::info!("   Pretraining Completed in {:.2?}!   ", elapsed);
    tracing::info!("=============================================");
}
