
use std::time::Instant;
use burn::tensor::{DType, Int, Tensor, TensorData, backend::{AutodiffBackend, Backend}};
use serde::{Deserialize, Serialize};

use crate::{common::{int_tensor_2d, scalar_to_f32},
    dataloader::{DataLoaderPosition, DistributedDataLoader}, gpt::Gpt,
    optim::{MuonAdamW, OptimizerKind},
    tokenizer::BpeTokenizer,
};

pub mod calculator;
pub mod eval;
pub mod inference;
pub mod pretrain;
pub mod recipe;
pub mod rl;
pub mod sandbox;
pub mod scheduler;
pub mod serving;
pub mod sft;
pub mod speculative;

/// Training configuration hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingConfig {
    #[serde(default)]
    pub optimizer: OptimizerKind,
    pub num_iterations: usize,
    pub warmup_steps: usize,
    pub warmdown_ratio: f32,
    pub final_lr_frac: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub device_batch_size: usize,
    pub sequence_length: usize,
    pub total_batch_size: usize,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_iterations == 0 { return Err("num_iterations must be greater than zero"); }
        if self.device_batch_size == 0 { return Err("device_batch_size must be greater than zero"); }
        if self.sequence_length == 0 { return Err("sequence_length must be greater than zero"); }
        if self.total_batch_size < self.device_batch_size ||
            self.total_batch_size % self.device_batch_size != 0 {
            return Err("total_batch_size must be a multiple of device_batch_size");
        }
        if self.warmup_steps > self.num_iterations { return Err("warmup exceeds training length"); }
        if !(0.0..=1.0).contains(&self.warmdown_ratio) {
            return Err("warmdown_ratio must be between zero and one");
        }
        if !(0.0..=1.0).contains(&self.final_lr_frac) {
            return Err("final_lr_frac must be between zero and one");
        }
        if !self.learning_rate.is_finite() || self.learning_rate < 0.0 {
            return Err("learning_rate must be finite and non-negative");
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err("weight_decay must be finite and non-negative");
        }
        Ok(())
    }

    pub fn gradient_accumulation_steps(&self) -> usize {
        assert!(self.device_batch_size > 0, "device_batch_size must be greater than zero");
        assert!(self.total_batch_size >= self.device_batch_size &&
            self.total_batch_size % self.device_batch_size == 0,
            "total_batch_size must be a multiple of device_batch_size");
        self.total_batch_size / self.device_batch_size
    }
}

/// Compute the learning rate multiplier using a linear warmup followed by a constant phase
/// and a linear warmdown to a final fraction of the initial learning rate.
pub fn get_lr_multiplier(step: usize, num_iterations: usize, warmup_steps: usize,
    warmdown_ratio: f32, final_lr_frac: f32) -> f32 {
    let step = step.min(num_iterations);
    if step < warmup_steps { (step + 1) as f32 / warmup_steps.max(1) as f32 } else {
        let warmdown_iters = (warmdown_ratio * num_iterations as f32).round() as usize;
        if step <= num_iterations.saturating_sub(warmdown_iters) { 1.0 } else {
            let remaining = (num_iterations - step) as f32 / warmdown_iters.max(1) as f32;
            final_lr_frac + remaining * (1.0 - final_lr_frac)
        }
    }
}

/// Compute the momentum value for the Muon optimizer dynamically over the training horizon.
pub fn get_muon_momentum(step: usize, num_iterations: usize, warmdown_ratio: f32) -> f32 {
    let warmdown_iters = (warmdown_ratio * num_iterations as f32).round() as usize;
    let warmdown_start = num_iterations.saturating_sub(warmdown_iters);
    if step < 400 {
        let frac = step as f32 / 400.0;
        (1.0 - frac) * 0.85 + frac * 0.97
    } else if step >= warmdown_start {
        let progress = ((step - warmdown_start) as f32 / warmdown_iters.max(1) as f32).min(1.0);
        0.97 * (1.0 - progress) + 0.90 * progress
    } else {
        0.97
    }
}

/// Compute the weight decay value dynamically over the training horizon.
pub fn get_weight_decay(step: usize, num_iterations: usize, weight_decay: f32) -> f32 {
    let step = step.min(num_iterations);
    weight_decay * 0.5 *
        (1.0 + ((std::f32::consts::PI * step as f32) / num_iterations.max(1) as f32).cos())
}

/// Extract the byte length of each token in the BpeTokenizer vocabulary,
/// setting special tokens to 0 bytes so they are ignored in the BPB denominator.
pub fn get_token_bytes(tokenizer: &BpeTokenizer) -> Vec<usize> {
    let vocab_size = tokenizer.get_vocab_size();
    let mut token_bytes = vec![0; vocab_size];

    // Normal BPE mergeable ranks
    for (bytes, &id) in &tokenizer.mergeable_ranks {
        if id < vocab_size { token_bytes[id] = bytes.len(); }
    }

    // Single byte fallbacks
    token_bytes.iter_mut().take(256).for_each(|bytes| *bytes = 1);

    token_bytes
}

fn bpb_totals<B: Backend>(losses: Tensor<B, 1>, targets: Tensor<B, 1, Int>,
    token_bytes: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let vocab_size = token_bytes.shape().dims::<1>()[0];
    let in_range = targets.clone().greater_equal_elem(0).float().cast(DType::F32) *
        targets.clone().lower_elem(vocab_size as i32).float().cast(DType::F32);
    let bytes = token_bytes.gather(0, targets.clamp(0, vocab_size as i32 - 1));
    let counted = in_range * bytes.clone().greater_elem(0.0).float().cast(DType::F32);
    ((losses.cast(DType::F32) * counted.clone()).sum(), (bytes * counted).sum())
}

/// Evaluate validation Bits Per Byte (BPB) on a given DataLoader.
pub async fn evaluate_bpb<B: Backend>(model: &Gpt<B>, loader: &mut DistributedDataLoader,
    steps: usize, token_bytes: &[usize], device: &B::Device) -> f32 {
    assert!(!token_bytes.is_empty(), "token byte table must not be empty");
    let token_bytes = Tensor::<B, 1>::from_data(TensorData::new(
        token_bytes.iter().map(|&bytes| bytes as f32).collect(), [token_bytes.len()]), device)
        .cast(DType::F32);
    let (mut total_nats, mut total_bytes) = (Tensor::<B, 1>::zeros([1], device)
        .cast(DType::F32), Tensor::<B, 1>::zeros([1], device).cast(DType::F32));

    for _ in 0..steps {
        let Some(batch) = loader.next_batch().await else { break; };
        let t = model.config.sequence_len;
        assert_eq!(batch.x.len(), batch.y.len(), "evaluation input/target size mismatch");
        assert_eq!(batch.x.len() % t, 0, "evaluation batch is not sequence-aligned");
        let b = batch.x.len() / t;
        let x_tensor = int_tensor_2d(batch.x, [b, t], device);
        let y_tensor = int_tensor_2d(batch.y, [b, t], device);

        let logits = model.forward(x_tensor);
        let (nats, bytes) = bpb_totals(
            model.compute_unreduced_loss(logits, y_tensor.clone()), y_tensor.reshape([-1]),
            token_bytes.clone());
        total_nats = total_nats + nats;
        total_bytes = total_bytes + bytes;
    }

    let total_bytes = scalar_to_f32(total_bytes.into_scalar());
    if total_bytes == 0.0 { f32::INFINITY } else {
        scalar_to_f32(total_nats.into_scalar()) / (2.0f32.ln() * total_bytes)
    }
}

/// Pretraining engine orchestrator
pub struct TrainingEngine<B: AutodiffBackend> {
    pub model: Gpt<B>,
    pub optimizer: MuonAdamW<B>,
    pub config: TrainingConfig,
    pub token_bytes: Vec<usize>,
    pub step: usize,
    pub smooth_train_loss: f32,
    pub total_training_time_secs: f64,
    pub dataloader_position: Option<DataLoaderPosition>,
    pub last_tokens_per_second: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainerState {
    pub step: usize,
    pub smooth_train_loss: f32,
    pub total_training_time_secs: f64,
    pub dataloader_position: Option<DataLoaderPosition>,
    #[serde(default)]
    pub rng_state: Option<u64>,
}

impl<B: AutodiffBackend> TrainingEngine<B> {
    pub fn new(model: Gpt<B>, config: TrainingConfig, tokenizer: &BpeTokenizer) -> Self {
        config.validate().unwrap_or_else(|message| panic!("invalid training config: {message}"));
        assert!(config.sequence_length <= model.config.sequence_len,
            "training sequence length exceeds model capacity");
        let optimizer = MuonAdamW::new(model.config.n_layer, config.optimizer);
        let token_bytes = get_token_bytes(tokenizer);
        Self { model, optimizer, config, token_bytes, step: 0,
            smooth_train_loss: 0.0, total_training_time_secs: 0.0, dataloader_position: None,
            last_tokens_per_second: 0.0,
        }
    }

    pub fn from_state(model: Gpt<B>, optimizer: MuonAdamW<B>, config: TrainingConfig,
        tokenizer: &BpeTokenizer, state: TrainerState) -> Self {
        config.validate().unwrap_or_else(|message| panic!("invalid training config: {message}"));
        assert!(config.sequence_length <= model.config.sequence_len,
            "training sequence length exceeds model capacity");
        assert!(state.step <= config.num_iterations,
            "trainer step exceeds configured training iterations");
        assert!(state.step == 0 || state.dataloader_position.is_some(),
            "resumed trainer state is missing its dataloader position");
        assert!(state.smooth_train_loss.is_finite() && state.smooth_train_loss >= 0.0,
            "smoothed training loss must be finite and non-negative");
        assert!(state.total_training_time_secs.is_finite() &&
            state.total_training_time_secs >= 0.0,
            "total training time must be finite and non-negative");
        assert_eq!(optimizer.kind, config.optimizer,
            "resumed optimizer kind differs from training config");
        Self { model, optimizer, config, token_bytes: get_token_bytes(tokenizer), step: state.step,
            smooth_train_loss: state.smooth_train_loss,
            total_training_time_secs: state.total_training_time_secs,
            dataloader_position: state.dataloader_position,
            last_tokens_per_second: 0.0,
        }
    }

    pub fn state(&self) -> TrainerState {
        TrainerState { step: self.step, smooth_train_loss: self.smooth_train_loss,
            total_training_time_secs: self.total_training_time_secs,
            dataloader_position: self.dataloader_position, rng_state: None,
        }
    }

    /// Perform a single optimization step (with optional gradient accumulation steps)
    pub async fn train_step(&mut self, loader: &mut DistributedDataLoader,
        device: &B::Device) -> f32 {
        let start_time = Instant::now(); // Single node / mock DDP
        let grad_accum_steps = self.config.gradient_accumulation_steps();

        let (mut accumulator, mut step_loss) =
            (burn::optim::GradientsAccumulator::new(), 0.0f32);

        for _ in 0..grad_accum_steps {
            let batch = loader.next_batch().await.expect("training data loader stopped unexpectedly");
            self.dataloader_position = Some(batch.next_position);
            let shape = [self.config.device_batch_size, self.config.sequence_length];
            let x_tensor = int_tensor_2d(batch.x, shape, device);
            let y_tensor = int_tensor_2d(batch.y, shape, device);

            let logits = self.model.forward(x_tensor);
            let loss = self.model.compute_loss(logits, y_tensor) / grad_accum_steps as f32;
            step_loss += scalar_to_f32(loss.clone().into_scalar());

            let step_grads = loss.backward();
            let step_grads_params =
                burn::optim::GradientsParams::from_grads(step_grads, &self.model);
            accumulator.accumulate(&self.model, step_grads_params);
        }

        // Apply optimizer update step
        let g = accumulator.grads();
        let lrm = get_lr_multiplier(self.step, self.config.num_iterations,
            self.config.warmup_steps, self.config.warmdown_ratio, self.config.final_lr_frac);
        let lr = self.config.learning_rate * lrm;
        let wd =
            get_weight_decay(self.step, self.config.num_iterations, self.config.weight_decay);

        let momentum = get_muon_momentum(
            self.step, self.config.num_iterations, self.config.warmdown_ratio);
        self.optimizer.step(&mut self.model, &g, lr, self.step + 1, wd, momentum);

        let elapsed = start_time.elapsed().as_secs_f64();
        self.last_tokens_per_second =
            (self.config.total_batch_size * self.config.sequence_length) as f32 /
            elapsed.max(f64::EPSILON) as f32;
        if self.step > 10 { self.total_training_time_secs += elapsed; }

        // Compute debiased smoothed loss
        let ema_beta = 0.9f32;
        self.smooth_train_loss =
            ema_beta * self.smooth_train_loss + (1.0 - ema_beta) * step_loss;
        let debiased_loss =
            self.smooth_train_loss / (1.0 - ema_beta.powi((self.step + 1) as i32));

        self.step += 1;
        debiased_loss
    }
}

#[cfg(test)] mod tests { use super::*;
    use std::{fs, env, process};

    #[test] fn test_schedulers() {
        let lr_mult = get_lr_multiplier(0, 100, 10, 0.5, 0.1);
        assert!(lr_mult > 0.0 && lr_mult <= 0.1);

        let lr_mult_mid = get_lr_multiplier(30, 100, 10, 0.5, 0.1);
        assert_eq!(lr_mult_mid, 1.0);

        let lr_mult_end = get_lr_multiplier(90, 100, 10, 0.5, 0.1);
        assert!((0.1..1.0).contains(&lr_mult_end));
        assert_eq!(get_lr_multiplier(101, 100, 10, 0.5, 0.1), 0.1);

        let momentum = get_muon_momentum(0, 100, 0.5);
        assert!((0.85..=0.97).contains(&momentum));

        let wd = get_weight_decay(50, 100, 0.28);
        assert!(wd > 0.0 && wd <= 0.28);

        let config = TrainingConfig { optimizer: OptimizerKind::MuonAdamW,
            num_iterations: 10, warmup_steps: 1,
            warmdown_ratio: 0.5, final_lr_frac: 0.1, learning_rate: 1e-3,
            weight_decay: 0.1, device_batch_size: 2, sequence_length: 16,
            total_batch_size: 4,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.gradient_accumulation_steps(), 2);
    }

    #[test] fn test_token_byte_lengths() {
        let corpus = vec!["Rust is extremely elegant and high-performance."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);
        let token_bytes = get_token_bytes(&tokenizer);
        assert_eq!(token_bytes.len(), tokenizer.get_vocab_size());
        assert!(token_bytes[..256].iter().all(|&bytes| bytes == 1));
        assert_eq!(token_bytes[tokenizer.get_bos_token_id()], 0);
    }

    #[test] fn test_bpb_totals_ignore_non_text_targets() {
        use burn::tensor::Tensor;
        use crate::common::InferBackend;
        let device = Default::default();
        let losses = Tensor::<InferBackend, 1>::from_data([1.0, 2.0, 4.0, 8.0], &device);
        let targets = Tensor::from_data([0, 1, -1, 9], &device);
        let token_bytes = Tensor::<InferBackend, 1>::from_data([1.0, 0.0, 2.0], &device)
            .cast(DType::F32);
        let (nats, bytes) = bpb_totals(losses, targets, token_bytes);
        assert_eq!(scalar_to_f32(nats.into_scalar()), 1.0);
        assert_eq!(scalar_to_f32(bytes.into_scalar()), 1.0);
    }

    #[tokio::test] async fn test_training_resume_equivalence() {
        use crate::{artifact::{TrainingStage, load_artifact, load_resume_state, save_artifact,
                save_resume_state}, common::{TrainBackend,
                tensor_data_to_f32_vec}, dataloader::DistributedDataLoaderConfig,
            dataset::pretokenize_text_to_bin, gpt::GptConfig,
        };
        let device = Default::default();
        let tokenizer = BpeTokenizer::train_from_iterator(
            ["resume equivalence needs enough deterministic training tokens"], 280);
        let model_config = GptConfig { sequence_len: 4, n_layer: 1, n_head: 2,
            n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(),
            vocab_size: tokenizer.get_vocab_size(), features: Default::default(),
            quantization: None,
        };
        let training_config = TrainingConfig { optimizer: OptimizerKind::MuonAdamW,
            num_iterations: 2, warmup_steps: 0,
            warmdown_ratio: 0.0, final_lr_frac: 1.0, learning_rate: 1e-3,
            weight_decay: 0.1, device_batch_size: 1, sequence_length: 4,
            total_batch_size: 1,
        };
        let root = env::temp_dir().join(format!(
            "nanochat-resume-test-{}", process::id()));
        let data_txt = root.join("train.txt");
        let data_bin = root.join("train.bin");
        let initial = root.join("initial");
        let checkpoint = root.join("checkpoint");
        fs::create_dir_all(&root).unwrap();
        fs::write(&data_txt,
            "resume equivalence needs enough deterministic training tokens ".repeat(8)).unwrap();
        pretokenize_text_to_bin(&data_txt, &data_bin, &tokenizer).unwrap();
        let model = Gpt::<TrainBackend>::new(model_config.clone(), &device);
        save_artifact(&initial, TrainingStage::Pretrain, &model, &tokenizer,
            Some(&training_config)).unwrap();

        let uninterrupted = load_artifact::<TrainBackend>(&initial, &device).unwrap();
        let mut uninterrupted =
            TrainingEngine::new(uninterrupted.model, training_config.clone(), &tokenizer);
        let loader_config = DistributedDataLoaderConfig::single_process(1, 4);
        let mut loader = DistributedDataLoader::new(vec![data_bin.clone()], loader_config);
        uninterrupted.train_step(&mut loader, &device).await;
        uninterrupted.train_step(&mut loader, &device).await;

        let interrupted = load_artifact::<TrainBackend>(&initial, &device).unwrap();
        let mut interrupted =
            TrainingEngine::new(interrupted.model, training_config.clone(), &tokenizer);
        let mut loader =
            DistributedDataLoader::new(vec![data_bin.clone()], loader_config);
        interrupted.train_step(&mut loader, &device).await;
        save_artifact(&checkpoint, TrainingStage::Pretrain, &interrupted.model, &tokenizer,
            Some(&training_config)).unwrap();
        save_resume_state(&checkpoint, &interrupted.optimizer, &interrupted.state()).unwrap();

        let artifact = load_artifact(&checkpoint, &device).unwrap();
        let (optimizer, state) =
            load_resume_state(&checkpoint, 1, &device).unwrap();
        let position = state.dataloader_position.unwrap();
        let mut resumed = TrainingEngine::from_state(artifact.model, optimizer,
            training_config, &tokenizer, state);
        let mut loader = DistributedDataLoader::new(vec![data_bin],
            loader_config.with_position(position));
        resumed.train_step(&mut loader, &device).await;

        let input = int_tensor_2d(vec![0; 4], [1, 4], &device);
        let expected = tensor_data_to_f32_vec(
            uninterrupted.model.forward(input.clone()).into_data());
        let actual =
            tensor_data_to_f32_vec(resumed.model.forward(input).into_data());
        assert_eq!(resumed.step, uninterrupted.step);
        assert_eq!(resumed.dataloader_position, uninterrupted.dataloader_position);
        assert_eq!(actual.len(), expected.len());
        let max_error = actual.into_iter().zip(expected)
            .map(|(actual, expected)| (actual - expected).abs()).fold(0.0, f32::max);
        // Unit tests use the deterministic F32 Flex backend. The optimizer roundtrip test
        // separately asserts that Adam moments remain F32 across the checkpoint boundary.
        let tolerance = 1e-6;
        assert!(max_error <= tolerance,
            "resumed logits max error {max_error} exceeds backend tolerance {tolerance}");
        fs::remove_dir_all(root).ok();
    }

    #[tokio::test] async fn test_tiny_corpus_overfit() {
        use crate::{common::TrainBackend,
            dataloader::DistributedDataLoaderConfig, dataset::pretokenize_text_to_bin,
            gpt::{GptConfig, ModelFeatures},
        };
        let device = Default::default();
        TrainBackend::seed(&device, 11);
        let text = "rust learns tiny repeated patterns ".repeat(32);
        let tokenizer = BpeTokenizer::train_from_iterator([text.as_str()], 280);
        let root = env::temp_dir().join(format!(
            "nanochat-overfit-test-{}", process::id()));
        let (text_path, token_path) = (root.join("train.txt"), root.join("train.bin"));
        fs::create_dir_all(&root).unwrap();
        fs::write(&text_path, &text).unwrap();
        pretokenize_text_to_bin(&text_path, &token_path, &tokenizer).unwrap();

        let model = Gpt::<TrainBackend>::new(GptConfig {
            sequence_len: 4, vocab_size: tokenizer.get_vocab_size(), n_layer: 1,
            n_head: 2, n_kv_head: 1, n_embd: 16, window_pattern: "L".into(),
            features: ModelFeatures::default(), quantization: None,
        }, &device);
        let config = TrainingConfig { optimizer: OptimizerKind::AdamW,
            num_iterations: 16, warmup_steps: 0, warmdown_ratio: 0.0,
            final_lr_frac: 1.0, learning_rate: 0.01, weight_decay: 0.0,
            device_batch_size: 1, sequence_length: 4, total_batch_size: 1,
        };
        let loader_config = DistributedDataLoaderConfig::single_process(1, 4);
        let mut engine = TrainingEngine::new(model, config, &tokenizer);
        let mut eval_loader =
            DistributedDataLoader::new(vec![token_path.clone()], loader_config);
        let batch = eval_loader.next_batch().await.unwrap();
        let logits = engine.model.forward(int_tensor_2d(batch.x, [1, 4], &device));
        let initial_loss = scalar_to_f32(engine.model.compute_loss(
            logits, int_tensor_2d(batch.y, [1, 4], &device)).into_scalar());
        let mut loader =
            DistributedDataLoader::new(vec![token_path.clone()], loader_config);
        for _ in 0..engine.config.num_iterations {
            engine.train_step(&mut loader, &device).await;
        }
        let mut eval_loader = DistributedDataLoader::new(vec![token_path], loader_config);
        let batch = eval_loader.next_batch().await.unwrap();
        let logits = engine.model.forward(int_tensor_2d(batch.x, [1, 4], &device));
        let final_loss = scalar_to_f32(engine.model.compute_loss(
            logits, int_tensor_2d(batch.y, [1, 4], &device)).into_scalar());
        assert!(final_loss < initial_loss,
            "tiny overfit loss did not decrease: {initial_loss} -> {final_loss}");
        fs::remove_dir_all(root).ok();
    }
}
