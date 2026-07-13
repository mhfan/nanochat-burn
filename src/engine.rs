
use std::time::Instant;
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::{common::{int_tensor_2d, scalar_to_f32}, dataloader::DistributedDataLoader,
    gpt::Gpt, optim::MuonAdamW, tokenizer::BpeTokenizer,
};

pub mod calculator;
pub mod eval;
pub mod inference;
pub mod pretrain;
pub mod quant;
pub mod rl;
pub mod sandbox;
pub mod sft;
pub mod speculative;

/// Training configuration hyperparameters
#[derive(Debug, Clone)]
pub struct TrainingConfig {
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

/// Evaluate validation Bits Per Byte (BPB) on a given DataLoader.
pub async fn evaluate_bpb<B: Backend>(model: &Gpt<B>, loader: &mut DistributedDataLoader,
    steps: usize, token_bytes: &[usize], device: &B::Device) -> f32 {
    let (mut total_nats, mut total_bytes) = (0.0f32, 0);

    for _ in 0..steps {
        let Some(batch) = loader.next_batch().await else { break; };
        let t = model.config.sequence_len;
        assert_eq!(batch.x.len(), batch.y.len(), "evaluation input/target size mismatch");
        assert_eq!(batch.x.len() % t, 0, "evaluation batch is not sequence-aligned");
        let b = batch.x.len() / t;
        let x_tensor = int_tensor_2d(batch.x, [b, t], device);
        let y_tensor = int_tensor_2d(batch.y, [b, t], device);

        let logits = model.forward(x_tensor, None);
        let unreduced_losses = model.compute_unreduced_loss(logits, y_tensor.clone());

        let targets_vec = y_tensor.into_data().to_vec::<i32>().unwrap();
        let loss_vec = crate::common::tensor_data_to_f32_vec(unreduced_losses.into_data());

        for (loss_val, target_tok) in loss_vec.into_iter().zip(targets_vec) {
            if target_tok >= 0 {
                let bytes_len = token_bytes.get(target_tok as usize).copied().unwrap_or(0);
                if bytes_len > 0 {
                    total_nats += loss_val;
                    total_bytes += bytes_len;
                }
            }
        }
    }

    if total_bytes == 0 { f32::INFINITY } else {
        total_nats / (2.0f32.ln() * total_bytes as f32)
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
}

impl<B: AutodiffBackend> TrainingEngine<B> {
    pub fn new(model: Gpt<B>, config: TrainingConfig, tokenizer: &BpeTokenizer) -> Self {
        config.validate().unwrap_or_else(|message| panic!("invalid training config: {message}"));
        assert_eq!(config.sequence_length, model.config.sequence_len,
            "training and model sequence lengths must match");
        let optimizer = MuonAdamW::new(model.config.n_layer);
        let token_bytes = get_token_bytes(tokenizer);
        Self { model, optimizer, config, token_bytes, step: 0,
            smooth_train_loss: 0.0, total_training_time_secs: 0.0,
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
            let shape = [self.config.device_batch_size, self.config.sequence_length];
            let x_tensor = int_tensor_2d(batch.x, shape, device);
            let y_tensor = int_tensor_2d(batch.y, shape, device);

            let logits = self.model.forward(x_tensor, None);
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

        self.optimizer.step(&mut self.model, &g, lr, self.step + 1, wd);

        let elapsed = start_time.elapsed().as_secs_f64();
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

        let config = TrainingConfig { num_iterations: 10, warmup_steps: 1,
            warmdown_ratio: 0.5, final_lr_frac: 0.1, learning_rate: 1e-3,
            weight_decay: 0.1, device_batch_size: 2, sequence_length: 16,
            total_batch_size: 4,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.gradient_accumulation_steps(), 2);
    }

    #[tokio::test] async fn test_bpb_and_engine_instantiation() {
        let corpus = vec!["Rust is extremely elegant and high-performance."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);
        assert_eq!(get_token_bytes(&tokenizer).len(), tokenizer.get_vocab_size());

        let config = crate::gpt::GptConfig { sequence_len: 8, n_layer: 1, n_head: 2,
            n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(),
            vocab_size: tokenizer.get_vocab_size(), quantization: None,
        };

        let device = crate::common::init_device();
        let gpt: Gpt<crate::common::ModelBackend> = Gpt::new(config.clone(), &device);
        assert_eq!(gpt.config.sequence_len, 8);
    }
}
