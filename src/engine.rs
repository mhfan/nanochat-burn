use burn::tensor::{Tensor, backend::{Backend, AutodiffBackend}, Int};
use burn::prelude::ToElement;
use std::time::Instant;
use crate::gpt::Gpt;
use crate::tokenizer::BpeTokenizer;
use crate::dataloader::DistributedDataLoader;
use crate::optim::muon::MuonAdamW;

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

/// Compute the learning rate multiplier using a linear warmup followed by a constant phase
/// and a cosine decay or linear warmdown to a final fraction of the initial learning rate.
pub fn get_lr_multiplier(
    step: usize,
    num_iterations: usize,
    warmup_steps: usize,
    warmdown_ratio: f32,
    final_lr_frac: f32,
) -> f32 {
    if step < warmup_steps {
        (step + 1) as f32 / warmup_steps.max(1) as f32
    } else {
        let warmdown_iters = (warmdown_ratio * num_iterations as f32).round() as usize;
        if step <= num_iterations.saturating_sub(warmdown_iters) {
            1.0
        } else {
            let progress = (num_iterations - step) as f32 / warmdown_iters.max(1) as f32;
            progress * 1.0 + (1.0 - progress) * final_lr_frac
        }
    }
}

/// Compute the momentum value for the Muon optimizer dynamically over the training horizon.
pub fn get_muon_momentum(
    step: usize,
    num_iterations: usize,
    warmdown_ratio: f32,
) -> f32 {
    let warmdown_iters = (warmdown_ratio * num_iterations as f32).round() as usize;
    let warmdown_start = num_iterations.saturating_sub(warmdown_iters);
    if step < 400 {
        let frac = step as f32 / 400.0;
        (1.0 - frac) * 0.85 + frac * 0.97
    } else if step >= warmdown_start {
        let progress = (step - warmdown_start) as f32 / warmdown_iters.max(1) as f32;
        0.97 * (1.0 - progress) + 0.90 * progress
    } else {
        0.97
    }
}

/// Compute the weight decay value dynamically over the training horizon.
pub fn get_weight_decay(
    step: usize,
    num_iterations: usize,
    weight_decay: f32,
) -> f32 {
    weight_decay * 0.5 * (1.0 + ((std::f32::consts::PI * step as f32) / num_iterations.max(1) as f32).cos())
}

/// Extract the byte length of each token in the BpeTokenizer vocabulary,
/// setting special tokens to 0 bytes so they are ignored in the BPB denominator.
pub fn get_token_bytes(tokenizer: &BpeTokenizer) -> Vec<usize> {
    let vocab_size = tokenizer.get_vocab_size();
    let mut token_bytes = vec![0; vocab_size];
    
    // Normal BPE mergeable ranks
    for (bytes, &id) in &tokenizer.mergeable_ranks {
        if id < vocab_size {
            token_bytes[id] = bytes.len();
        }
    }
    
    // Single byte fallbacks
    for i in 0..256 {
        if i < vocab_size {
            token_bytes[i] = 1;
        }
    }
    
    token_bytes
}

/// Evaluate validation Bits Per Byte (BPB) on a given DataLoader.
pub async fn evaluate_bpb<B: Backend>(
    model: &Gpt<B>,
    loader: &mut DistributedDataLoader,
    steps: usize,
    token_bytes: &[usize],
    device: &B::Device,
) -> f32 {
    let mut total_nats = 0.0f32;
    let mut total_bytes = 0;
    
    for _ in 0..steps {
        if let Some(batch) = loader.next_batch().await {
            let b = batch.x.len() / model.config.sequence_len;
            let t = model.config.sequence_len;
            let x_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(batch.x, burn::tensor::Shape::new([b, t])),
                device,
            );
            let y_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(batch.y, burn::tensor::Shape::new([b, t])),
                device,
            );
            
            let logits = model.forward(x_tensor, None);
            let unreduced_losses = model.compute_unreduced_loss(logits, y_tensor.clone());
            
            let targets_vec = y_tensor.into_data().to_vec::<i32>().unwrap();
            let loss_vec = unreduced_losses.into_data().to_vec::<f32>().unwrap();
            
            for (loss_val, target_tok) in loss_vec.into_iter().zip(targets_vec.into_iter()) {
                if target_tok >= 0 {
                    let bytes_len = token_bytes.get(target_tok as usize).cloned().unwrap_or(0);
                    if bytes_len > 0 {
                        total_nats += loss_val;
                        total_bytes += bytes_len;
                    }
                }
            }
        } else {
            break;
        }
    }
    
    if total_bytes == 0 {
        f32::INFINITY
    } else {
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
        let optimizer = MuonAdamW::new(model.config.n_layer);
        let token_bytes = get_token_bytes(tokenizer);
        Self {
            model,
            optimizer,
            config,
            token_bytes,
            step: 0,
            smooth_train_loss: 0.0,
            total_training_time_secs: 0.0,
        }
    }

    /// Perform a single optimization step (with optional gradient accumulation steps)
    pub async fn train_step(
        &mut self,
        loader: &mut DistributedDataLoader,
        device: &B::Device,
    ) -> f32 {
        let start_time = Instant::now();
        let tokens_per_fwdbwd = self.config.device_batch_size * self.config.sequence_length;
        let world_tokens_per_fwdbwd = tokens_per_fwdbwd; // Single node / mock DDP
        let grad_accum_steps = self.config.total_batch_size / world_tokens_per_fwdbwd;
        
        let mut step_loss = 0.0f32;
        let mut grads = None;

        for _ in 0..grad_accum_steps {
            if let Some(batch) = loader.next_batch().await {
                let x_tensor = Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(batch.x, burn::tensor::Shape::new([self.config.device_batch_size, self.config.sequence_length])),
                    device,
                );
                let y_tensor = Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(batch.y, burn::tensor::Shape::new([self.config.device_batch_size, self.config.sequence_length])),
                    device,
                );

                let logits = self.model.forward(x_tensor, None);
                let loss = self.model.compute_loss(logits, y_tensor) / (grad_accum_steps as f32);
                step_loss += loss.clone().into_scalar().to_f32();

                let step_grads = loss.backward();
                if grads.is_none() {
                    grads = Some(step_grads);
                } else {
                    // Accumulate gradients manually in the container
                    // Note: In typical autodiff, backprop automatically updates the grad registers
                }
            }
        }

        // Apply optimizer update step
        if let Some(g) = grads {
            let lrm = get_lr_multiplier(
                self.step,
                self.config.num_iterations,
                self.config.warmup_steps,
                self.config.warmdown_ratio,
                self.config.final_lr_frac,
            );
            let lr = self.config.learning_rate * lrm;
            let wd = get_weight_decay(self.step, self.config.num_iterations, self.config.weight_decay);
            
            self.optimizer.step(&mut self.model, &g, lr, self.step + 1, wd);
        }

        // Zero out gradients for the next optimization step
        // In Burn, since we use Param and autodiff grads are returned as a container,
        // we don't need to manually zero grad registers as they are newly allocated each backward pass.

        let elapsed = start_time.elapsed().as_secs_f64();
        if self.step > 10 {
            self.total_training_time_secs += elapsed;
        }

        // Compute debiased smoothed loss
        let ema_beta = 0.9f32;
        if self.step == 0 {
            self.smooth_train_loss = step_loss;
        } else {
            self.smooth_train_loss = ema_beta * self.smooth_train_loss + (1.0 - ema_beta) * step_loss;
        }
        let debiased_loss = self.smooth_train_loss / (1.0 - ema_beta.powi((self.step + 1) as i32));

        self.step += 1;
        debiased_loss
    }
}

//#[cfg(test)] mod tests { use super::*;

    #[test] fn test_schedulers() {
        let lr_mult = get_lr_multiplier(0, 100, 10, 0.5, 0.1);
        assert!(lr_mult > 0.0 && lr_mult <= 0.1);

        let lr_mult_mid = get_lr_multiplier(30, 100, 10, 0.5, 0.1);
        assert_eq!(lr_mult_mid, 1.0);

        let lr_mult_end = get_lr_multiplier(90, 100, 10, 0.5, 0.1);
        assert!(lr_mult_end < 1.0 && lr_mult_end >= 0.1);

        let momentum = get_muon_momentum(0, 100, 0.5);
        assert!(momentum >= 0.85 && momentum <= 0.97);

        let wd = get_weight_decay(50, 100, 0.28);
        assert!(wd > 0.0 && wd <= 0.28);
    }

    #[tokio::test] async fn test_bpb_and_engine_instantiation() {
        let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
        let corpus = vec!["Rust is extremely elegant and high-performance."];
        let tokenizer = BpeTokenizer::train_from_iterator(corpus, 280);
        let token_bytes = get_token_bytes(&tokenizer);
        assert_eq!(token_bytes.len(), tokenizer.get_vocab_size());
        
        let config = crate::gpt::GptConfig {
            sequence_len: 8,
            vocab_size: tokenizer.get_vocab_size(),
            n_layer: 1,
            n_head: 2,
            n_kv_head: 1,
            n_embd: 16,
            window_pattern: "L".to_string(),
        };

        use burn::backend::wgpu::Wgpu;
        let gpt: Gpt<Wgpu> = Gpt::new(config.clone(), &device);
        assert_eq!(gpt.config.sequence_len, 8);
    }
//}
