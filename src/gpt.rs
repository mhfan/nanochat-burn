
use std::marker::PhantomData;
use serde::{Serialize, Deserialize};
use crate::engine::quant::LinearOrQuantized;
use burn::{module::{Module, Param}, nn::{Embedding, Linear},
    tensor::{Tensor, TensorData, Shape, Distribution, Int, backend::Backend, activation},
};

pub trait ForwardLayer<B: Backend>: Module<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;
}

impl<B: Backend> ForwardLayer<B> for Linear<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.forward(input)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizationConfig {
    pub bits: usize,       // 8 or 4 bits
    pub block_size: usize, // e.g. 32, 64, or 0 (row-wise)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub window_pattern: String,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl GptConfig {
    pub fn compute_window_sizes(&self) -> Vec<i32> {
        let pattern = self.window_pattern.to_uppercase();
        let long_window = -1;
        let short_window = ((self.sequence_len as f32 / 4.0 / 128.0).ceil() * 128.0) as i32;
        let mut window_sizes = Vec::new();
        for i in 0..self.n_layer {
            let ch = pattern.chars().nth(i % pattern.len()).unwrap_or('L');
            window_sizes.push(if ch == 'S' { short_window } else { long_window });
        }
        if let Some(w) = window_sizes.last_mut() { *w = long_window; }
        window_sizes
    }
}

pub fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
    layer_idx % 2 == (n_layer - 1) % 2
}

fn precompute_window_mask<B: Backend>(window_size: i32, sequence_len: usize, device: &B::Device) -> Tensor<B, 4> {
    let mask_data: Vec<f32> = (0..sequence_len)
        .flat_map(|i| {
            let left_bound = if window_size < 0 { 0 } else { i.saturating_sub(window_size as usize) };
            (0..sequence_len).map(move |j| {
                if j > i || j < left_bound { -1e9f32 } else { 0.0f32 }
            })
        }).collect();
    Tensor::<B, 4>::from_data(
        TensorData::new(mask_data, Shape::new([1, 1, sequence_len, sequence_len])), device,
    )
}

pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let variance = (x.clone() * x.clone()).mean_dim(D - 1);
    let inv_std = (variance + eps).sqrt().recip();
    x * inv_std
}

pub fn apply_rotary_emb<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,) -> Tensor<B, 4> {
    let shape: [usize; 4] = x.shape().dims();
    let d = shape[3] / 2;
    let x1 = x.clone().slice([0..shape[0], 0..shape[1], 0..shape[2], 0..d]);
    let x2 = x.slice([0..shape[0], 0..shape[1], 0..shape[2], d..shape[3]]);
    let y1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let y2 = x2 * cos - x1 * sin;
    Tensor::cat(vec![y1, y2], 3)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, group_size: usize) -> Tensor<B, 4> {
    if group_size == 1 { return x; }
    let shape: [usize; 4] = x.shape().dims();
    let (b, t, n_kv_head, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let x_reshaped: Tensor<B, 5> = x.reshape([b, t, n_kv_head, 1, head_dim]);
    let x_expanded = x_reshaped.expand([b, t, n_kv_head, group_size, head_dim]);
    x_expanded.reshape([b, t, n_kv_head * group_size, head_dim])
}

#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub k_page_pool: Vec<Tensor<B, 4>>, // Layer -> [max_num_pages, page_size, n_kv_head, head_dim]
    pub v_page_pool: Vec<Tensor<B, 4>>, // Layer -> [max_num_pages, page_size, n_kv_head, head_dim]
    pub block_table: Tensor<B, 2, Int>, // [B, max_pages_per_seq]
    pub page_size: usize,
    pub max_pages_per_seq: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new(n_layer: usize, device: &B::Device) -> Self {
        let block_table = Tensor::<B, 2, Int>::zeros([1, 1], device);
        Self {
            k_page_pool: Vec::with_capacity(n_layer),
            v_page_pool: Vec::with_capacity(n_layer),
            block_table, page_size: 8, max_pages_per_seq: 1,
        }
    }

    pub fn new_allocated(n_layer: usize, batch_size: usize, max_seq_len: usize, n_kv_head: usize,
        head_dim: usize, device: &B::Device,) -> Self {
        Self::new_paged(n_layer, batch_size, max_seq_len, n_kv_head, head_dim, 8, device)
    }

    pub fn new_paged(n_layer: usize, batch_size: usize, max_seq_len: usize, n_kv_head: usize,
        head_dim: usize, page_size: usize, device: &B::Device,) -> Self {
        let max_pages_per_seq = (max_seq_len + page_size - 1) / page_size;
        let max_num_pages = batch_size * max_pages_per_seq;

        let mut k_page_pool = Vec::with_capacity(n_layer);
        let mut v_page_pool = Vec::with_capacity(n_layer);
        let pool_shape = Shape::new([max_num_pages, page_size, n_kv_head, head_dim]);

        for _ in 0..n_layer {
            k_page_pool.push(Tensor::zeros(pool_shape.clone(), device));
            v_page_pool.push(Tensor::zeros(pool_shape.clone(), device));
        }

        let data: Vec<i32> = (0..(batch_size * max_pages_per_seq) as i32).collect();
        let block_table = Tensor::<B, 2, Int>::from_data(
            TensorData::new(data, Shape::new([batch_size, max_pages_per_seq])), device,
        );

        Self { k_page_pool, v_page_pool, block_table, page_size, max_pages_per_seq, }
    }
}

// Constant values replacing magic numbers (Issue #16)
const VE_GATE_INPUT_DIM: usize = 12;
const SMEAR_GATE_INPUT_DIM: usize = 24;
const LOGIT_CLIP_VAL: f32 = 15.0;
const ROPE_BASE_FREQ: f32 = 100000.0;

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend, L = Linear<B>> {
    pub c_q: L,
    pub c_k: L,
    pub c_v: L,
    pub c_proj: L,
    pub ve_gate: Option<L>,
    pub layer_idx: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub mask: Tensor<B, 4>,
}

impl<B: Backend, L: ForwardLayer<B>> CausalSelfAttention<B, L> {
    fn prepare_qkv(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>,) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let shape: [usize; 3] = x.shape().dims();
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let q = self.c_q.forward_layer(x.clone()).reshape([b, t, self.n_head, self.head_dim]);
        let k = self.c_k.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);
        let mut v = self.c_v.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);

        if let Some(ve_tensor) = ve {
            let ve_reshaped = ve_tensor.reshape([b, t, self.n_kv_head, self.head_dim]);
            if let Some(ref ve_gate_linear) = self.ve_gate {
                let x_slice = x.slice([0..b, 0..t, 0..VE_GATE_INPUT_DIM]);
                let gate_logits = ve_gate_linear.forward_layer(x_slice);
                let gate = ((gate_logits * -1.0).exp() + 1.0).recip() * 3.0; // range (0, 3)
                let gate_unsqueezed = gate.reshape([b, t, self.n_kv_head, 1]);
                v = v + gate_unsqueezed * ve_reshaped;
            }
        }

        let q = apply_rotary_emb(q, cos.clone(), sin.clone());
        let k = apply_rotary_emb(k, cos, sin);
        let q = rms_norm(q, 1e-5) * 1.2;
        let k = rms_norm(k, 1e-5) * 1.2;

        (q, k, v)
    }

    fn compute_attention(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>,
        mask: Tensor<B, 4>,) -> Tensor<B, 3> {
        let shape: [usize; 4] = q.shape().dims();
        let (b, t, _, _) = (shape[0], shape[1], shape[2], shape[3]);

        let group_size = self.n_head / self.n_kv_head;
        let k = repeat_kv(k, group_size);
        let v = repeat_kv(v, group_size);

        let q_trans = q.swap_dims(1, 2);
        let k_trans = k.swap_dims(1, 2);
        let v_trans = v.swap_dims(1, 2);
        let k_t = k_trans.swap_dims(2, 3);
        let mut scores = q_trans.matmul(k_t) * (1.0 / (self.head_dim as f32).sqrt());

        scores = scores + mask;

        let probs = activation::softmax(scores, 3);
        let y = probs.matmul(v_trans).swap_dims(1, 2).reshape([b, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }

    pub fn forward_with_cache(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>,
        layer_idx: usize, cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        let shape: [usize; 3] = x.shape().dims();
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let (q, k, v) = self.prepare_qkv(x, ve, cos, sin);

        let (full_k, full_v) = {
            // 1. Write the new keys and values into the physical page pools (page-wise assignments)
            for b_idx in 0..b {
                let start_page = step / cache.page_size;
                let end_page = (step + t - 1) / cache.page_size;
                for p in start_page..=end_page {
                    let pos_start = std::cmp::max(step, p * cache.page_size);
                    let pos_end = std::cmp::min(step + t, (p + 1) * cache.page_size);
                    let t_start = pos_start - step;
                    let t_end = pos_end - step;
                    let offset_start = pos_start % cache.page_size;
                    let offset_end = offset_start + (t_end - t_start);
                    let physical_page_id = b_idx * cache.max_pages_per_seq + p;

                    let k_slice = k.clone().slice([b_idx..b_idx+1, t_start..t_end, 0..self.n_kv_head, 0..self.head_dim])
                        .reshape([1, t_end - t_start, self.n_kv_head, self.head_dim]);
                    let v_slice = v.clone().slice([b_idx..b_idx+1, t_start..t_end, 0..self.n_kv_head, 0..self.head_dim])
                        .reshape([1, t_end - t_start, self.n_kv_head, self.head_dim]);

                    cache.k_page_pool[layer_idx] = cache.k_page_pool[layer_idx].clone().slice_assign(
                        [physical_page_id..physical_page_id+1, offset_start..offset_end, 0..self.n_kv_head, 0..self.head_dim],
                        k_slice,
                    );
                    cache.v_page_pool[layer_idx] = cache.v_page_pool[layer_idx].clone().slice_assign(
                        [physical_page_id..physical_page_id+1, offset_start..offset_end, 0..self.n_kv_head, 0..self.head_dim],
                        v_slice,
                    );
                }
            }

            // 2. Reconstruct contiguous history for attention using contiguous slice + reshape
            let num_tokens = step + t;
            let num_pages = (num_tokens + cache.page_size - 1) / cache.page_size;

            let mut k_seqs = Vec::with_capacity(b);
            let mut v_seqs = Vec::with_capacity(b);

            for b_idx in 0..b {
                let physical_page_start = b_idx * cache.max_pages_per_seq;
                let physical_page_end = physical_page_start + num_pages;

                let k_pages = cache.k_page_pool[layer_idx].clone()
                    .slice([physical_page_start..physical_page_end, 0..cache.page_size, 0..self.n_kv_head, 0..self.head_dim])
                    .reshape([1, num_pages * cache.page_size, self.n_kv_head, self.head_dim]);
                let k_seq = k_pages.slice([0..1, 0..num_tokens, 0..self.n_kv_head, 0..self.head_dim]);

                let v_pages = cache.v_page_pool[layer_idx].clone()
                    .slice([physical_page_start..physical_page_end, 0..cache.page_size, 0..self.n_kv_head, 0..self.head_dim])
                    .reshape([1, num_pages * cache.page_size, self.n_kv_head, self.head_dim]);
                let v_seq = v_pages.slice([0..1, 0..num_tokens, 0..self.n_kv_head, 0..self.head_dim]);

                k_seqs.push(k_seq);
                v_seqs.push(v_seq);
            }

            (Tensor::cat(k_seqs, 0), Tensor::cat(v_seqs, 0))
        };

        let mask = self.mask.clone().slice([0..1, 0..1, step..step+t, 0..step+t]);
        self.compute_attention(q, full_k, full_v, mask)
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>,) -> Tensor<B, 3> {
        let shape: [usize; 3] = x.shape().dims();
        let t = shape[1];

        let (q, k, v) = self.prepare_qkv(x, ve, cos, sin);
        let mask = self.mask.clone().slice([0..1, 0..1, 0..t, 0..t]);
        self.compute_attention(q, k, v, mask)
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend, L = Linear<B>> {
    pub c_fc: L,
    pub c_proj: L,
    pub _phantom: PhantomData<B>,
}

impl<B: Backend, L: ForwardLayer<B>> MLP<B, L> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward_layer(x);
        let x = activation::relu(x);
        let x = x.clone() * x;
        self.c_proj.forward_layer(x)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend, L = Linear<B>> {
    pub attn: CausalSelfAttention<B, L>,
    pub mlp: MLP<B, L>,
}

impl<B: Backend, L: ForwardLayer<B>> Block<B, L> {
    pub fn forward_with_cache(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>,
        layer_idx: usize, cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward_with_cache(rms_norm(x.clone(), 1e-5), ve, cos, sin, layer_idx, cache, step);
        x.clone() + self.mlp.forward(rms_norm(x, 1e-5))
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>,) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(rms_norm(x.clone(), 1e-5), ve, cos, sin);
        x.clone() + self.mlp.forward(rms_norm(x, 1e-5))
    }
}

#[derive(Module, Debug)]
pub struct Gpt<B: Backend, L = Linear<B>> {
    pub wte: Embedding<B>,
    pub h: Vec<Block<B, L>>,
    pub lm_head: L,
    pub resid_lambdas: Param<Tensor<B, 1>>,
    pub x0_lambdas: Param<Tensor<B, 1>>,
    pub smear_gate: L,
    pub smear_lambda: Param<Tensor<B, 1>>,
    pub backout_lambda: Param<Tensor<B, 1>>,
    pub value_embeds: Vec<Embedding<B>>,
    pub config: GptConfig,
}

impl<B: Backend> Gpt<B, Linear<B>> {
    pub fn new(config: GptConfig, device: &B::Device) -> Self {
        let padded_vocab_size = ((config.vocab_size + 63) / 64) * 64;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;
        let kv_dim = config.n_kv_head * head_dim;

        let wte_weight = Tensor::random([padded_vocab_size, n_embd], Distribution::Normal(0.0, 0.8), device);
        let wte = Embedding { weight: Param::from_tensor(wte_weight) };

        let s = 3.0f32.sqrt() * (n_embd as f32).powf(-0.5);
        let value_embeds: Vec<_> = (0..config.n_layer)
            .filter(|&i| has_ve(i, config.n_layer))
            .map(|_| {
                let ve_weight = Tensor::random([padded_vocab_size, kv_dim], Distribution::Uniform(-s as f64, s as f64), device);
                Embedding { weight: Param::from_tensor(ve_weight) }
            }).collect();

        let window_sizes = config.compute_window_sizes();
        let h: Vec<_> = (0..config.n_layer)
            .map(|i| {
                let c_q = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
                let c_k = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_kv_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
                let c_v = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_kv_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
                let c_proj = Linear { weight: Param::from_tensor(Tensor::zeros([n_embd, n_embd], device)), bias: None };
                let ve_gate = if has_ve(i, config.n_layer) {
                    Some(Linear { weight: Param::from_tensor(Tensor::random([VE_GATE_INPUT_DIM, config.n_kv_head], Distribution::Uniform(0.0, 0.02), device)), bias: None })
                } else { None };

                let window_size = window_sizes[i];
                let mask = precompute_window_mask::<B>(window_size, config.sequence_len, device);
                let attn = CausalSelfAttention {
                    c_q, c_k, c_v, c_proj, ve_gate, layer_idx: i, n_head: config.n_head, n_kv_head: config.n_kv_head, head_dim, mask,
                };

                let c_fc = Linear { weight: Param::from_tensor(Tensor::random([n_embd, 4 * n_embd], Distribution::Uniform((-s * 0.4) as f64, (s * 0.4) as f64), device)), bias: None };
                let c_proj_mlp = Linear { weight: Param::from_tensor(Tensor::zeros([4 * n_embd, n_embd], device)), bias: None };
                let mlp = MLP { c_fc, c_proj: c_proj_mlp, _phantom: PhantomData };

                Block { attn, mlp }
            }).collect();

        let lm_head_weight = Tensor::random([n_embd, padded_vocab_size], Distribution::Normal(0.0, 0.001), device);
        let lm_head = Linear { weight: Param::from_tensor(lm_head_weight), bias: None };

        let resid_init: Vec<f32> = (0..config.n_layer)
            .map(|i| 1.15 - (0.10 * i as f32 / (config.n_layer - 1).max(1) as f32)).collect();
        let x0_init: Vec<f32> = (0..config.n_layer)
            .map(|i| 0.20 - (0.15 * i as f32 / (config.n_layer - 1).max(1) as f32)).collect();

        let resid_lambdas = Param::from_tensor(Tensor::from_data(TensorData::new(resid_init, Shape::new([config.n_layer])), device));
        let x0_lambdas = Param::from_tensor(Tensor::from_data(TensorData::new(x0_init, Shape::new([config.n_layer])), device));

        let smear_gate = Linear { weight: Param::from_tensor(Tensor::random([SMEAR_GATE_INPUT_DIM, 1], Distribution::Uniform(0.0, 0.02), device)), bias: None };
        let smear_lambda = Param::from_tensor(Tensor::zeros([1], device));
        let backout_lambda = Param::from_tensor(Tensor::from_data(TensorData::new(vec![0.2f32], Shape::new([1])), device));

        Gpt {
            wte, h, lm_head, resid_lambdas, x0_lambdas, smear_gate, smear_lambda, backout_lambda, value_embeds, config,
        }
    }
}

impl<B: Backend, L: ForwardLayer<B>> Gpt<B, L> {
    fn precompute_rotary_embeddings(&self, start_pos: usize, len: usize, head_dim: usize,
        device: &B::Device,) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let base = ROPE_BASE_FREQ;
        let inv_freq: Vec<f32> = (0..head_dim).step_by(2).map(|i| 1.0 / base.powf(i as f32 / head_dim as f32)).collect();

        let mut cos_data = Vec::with_capacity(len * (head_dim / 2));
        let mut sin_data = Vec::with_capacity(len * (head_dim / 2));

        for t in start_pos..(start_pos + len) {
            let t_f32 = t as f32;
            cos_data.extend(inv_freq.iter().map(|&freq| (t_f32 * freq).cos()));
            sin_data.extend(inv_freq.iter().map(|&freq| (t_f32 * freq).sin()));
        }

        let cos = Tensor::<B, 4>::from_data(TensorData::new(cos_data, Shape::new([1, len, 1, head_dim / 2])), device);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sin_data, Shape::new([1, len, 1, head_dim / 2])), device);
        (cos, sin)
    }

    fn forward_inner(&self, idx: Tensor<B, 2, Int>, mut cache_opt: Option<&mut KVCache<B>>, step: usize) -> Tensor<B, 3> {
        let shape: [usize; 2] = idx.shape().dims();
        let (batch_size, seq_len) = (shape[0], shape[1]);

        let head_dim = self.config.n_embd / self.config.n_head;
        let (cos, sin) = if seq_len > 1 {
            self.precompute_rotary_embeddings(0, seq_len, head_dim, &idx.device())
        } else {
            self.precompute_rotary_embeddings(step, 1, head_dim, &idx.device())
        };

        let x = self.wte.forward(idx.clone());
        let x_normed = rms_norm(x, 1e-5);

        let mut x = if seq_len > 1 {
            let x_slice = x_normed.clone().slice([0..batch_size, 1..seq_len, 0..SMEAR_GATE_INPUT_DIM]);
            let gate_logits = self.smear_gate.forward_layer(x_slice);
            let gate = ((gate_logits * -1.0).exp() + 1.0).recip() * self.smear_lambda.clone().val().reshape([1, 1, 1]);
            let x_prev = x_normed.clone().slice([0..batch_size, 0..seq_len - 1, 0..self.config.n_embd]);
            let x_cur = x_normed.clone().slice([0..batch_size, 1..seq_len, 0..self.config.n_embd]);
            let smeared = x_cur + gate * x_prev;
            let x_first = x_normed.clone().slice([0..batch_size, 0..1, 0..self.config.n_embd]);
            Tensor::cat(vec![x_first, smeared], 1)
        } else { x_normed };

        let x0 = x.clone();
        let backout_layer = self.config.n_layer / 2;
        let mut x_backout = None;
        let resid_val = self.resid_lambdas.clone().val();
        let x0_val = self.x0_lambdas.clone().val();

        let mut ve_cnt = 0;
        for i in 0..self.config.n_layer {
            let r_lambda = resid_val.clone().slice([i..i+1]).reshape([1, 1, 1]);
            let x0_lambda = x0_val.clone().slice([i..i+1]).reshape([1, 1, 1]);
            x = x * r_lambda + x0.clone() * x0_lambda;

            let ve = if has_ve(i, self.config.n_layer) {
                let ve_embed = self.value_embeds[ve_cnt].forward(idx.clone());
                ve_cnt += 1;
                Some(ve_embed)
            } else { None };

            if let Some(ref mut cache) = cache_opt {
                x = self.h[i].forward_with_cache(x, ve, cos.clone(), sin.clone(), i, cache, step);
            } else {
                x = self.h[i].forward(x, ve, cos.clone(), sin.clone());
            }
            if i == backout_layer { x_backout = Some(x.clone()); }
        }

        if let Some(xb) = x_backout {
            x = x - xb * self.backout_lambda.clone().val().reshape([1, 1, 1]);
        }
        let x = rms_norm(x, 1e-5);

        let mut logits = self.lm_head.forward_layer(x);
        logits = logits.slice([0..batch_size, 0..seq_len, 0..self.config.vocab_size]);
        (logits / LOGIT_CLIP_VAL).tanh() * LOGIT_CLIP_VAL
    }

    pub fn forward_with_cache(&self, idx: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        self.forward_inner(idx, Some(cache), step)
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>, _targets: Option<Tensor<B, 2, Int>>) -> Tensor<B, 3> {
        self.forward_inner(idx, None, 0)
    }

    pub fn compute_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let shape: [usize; 3] = logits.shape().dims();
        let (b, t) = (shape[0], shape[1]);
        let unreduced = self.compute_unreduced_loss(logits, targets.clone());

        let flat_targets = targets.reshape([b * t]);
        let mask_valid = flat_targets.not_equal_elem(-1);
        let valid_float = mask_valid.float();
        let sum_loss = unreduced.sum();
        let num_valid = valid_float.sum().clamp(1.0, 1e9);
        sum_loss / num_valid
    }

    pub fn compute_unreduced_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let shape: [usize; 3] = logits.shape().dims();
        let (b, t, v) = (shape[0], shape[1], shape[2]);
        let flat_logits = logits.reshape([b * t, v]).clamp(-50.0, 50.0);
        let flat_targets = targets.reshape([b * t]);

        let log_probs = activation::log_softmax(flat_logits, 1);
        let mask_valid = flat_targets.clone().not_equal_elem(-1);
        let clamped_targets = flat_targets.clamp(0, (v - 1) as i32);

        let one_hot = clamped_targets.one_hot(v);
        let selected_log_probs = (log_probs * one_hot.float()).sum_dim(1).reshape([b * t]);

        let valid_float = mask_valid.float();
        selected_log_probs * valid_float * -1.0
    }
}

impl<B: Backend> Gpt<B, Linear<B>> {
    fn convert_blocks<F>(self, f: F) -> Gpt<B, LinearOrQuantized<B>>
    where
        F: Fn(Linear<B>) -> LinearOrQuantized<B>,
    {
        let mut h_conv = Vec::with_capacity(self.h.len());
        for block in self.h {
            let attn = block.attn;
            let mlp = block.mlp;

            let attn_conv = CausalSelfAttention {
                c_q: f(attn.c_q),
                c_k: f(attn.c_k),
                c_v: f(attn.c_v),
                c_proj: f(attn.c_proj),
                ve_gate: attn.ve_gate.map(&f),
                layer_idx: attn.layer_idx,
                n_head: attn.n_head,
                n_kv_head: attn.n_kv_head,
                head_dim: attn.head_dim,
                mask: attn.mask,
            };

            let mlp_conv = MLP {
                c_fc: f(mlp.c_fc),
                c_proj: f(mlp.c_proj),
                _phantom: PhantomData,
            };

            h_conv.push(Block { attn: attn_conv, mlp: mlp_conv });
        }

        let lm_head_conv = f(self.lm_head);
        let smear_gate_conv = f(self.smear_gate);

        Gpt {
            wte: self.wte,
            h: h_conv,
            lm_head: lm_head_conv,
            resid_lambdas: self.resid_lambdas,
            x0_lambdas: self.x0_lambdas,
            smear_gate: smear_gate_conv,
            smear_lambda: self.smear_lambda,
            backout_lambda: self.backout_lambda,
            value_embeds: self.value_embeds,
            config: self.config,
        }
    }

    pub fn into_linear_or_quantized(self) -> Gpt<B, LinearOrQuantized<B>> {
        self.convert_blocks(LinearOrQuantized::Standard)
    }

    pub fn quantize(self, bits: usize, block_size: usize) -> Gpt<B, LinearOrQuantized<B>> {
        use crate::engine::quant::quantize_linear;
        self.convert_blocks(|lin| {
            LinearOrQuantized::Quantized(quantize_linear(lin, bits, block_size))
        })
    }
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_gpt_forward_and_loss() {
        let device = crate::common::init_device();
        let config = GptConfig { sequence_len: 256, vocab_size: 280, n_layer: 4, n_head: 4,
            n_kv_head: 2, n_embd: 64, window_pattern: "SL".to_string(), quantization: None,
        };

        use crate::common::ModelAutodiffBackend;
        let gpt: Gpt<ModelAutodiffBackend> = Gpt::new(config, &device);

        // Use identical shape as RL training batch: [8, 255]
        let idx = Tensor::<ModelAutodiffBackend, 2, Int>::zeros([8, 255], &device);
        let targets = Tensor::<ModelAutodiffBackend, 2, Int>::zeros([8, 255], &device);

        let logits = gpt.forward(idx, None);
        assert_eq!(logits.shape().dims(), [8, 255, 280]);

        let loss = gpt.compute_loss(logits, targets);
        let loss_val = loss.clone().into_scalar();
        assert!(loss_val.to_f32() >= 0.0);

        let _grads = loss.backward();
    }

    #[test] fn test_paged_attention_roundtrip() {
        let device = crate::common::init_device();
        let config = GptConfig { sequence_len: 256, vocab_size: 280, n_layer: 2, n_head: 4,
            n_kv_head: 2, n_embd: 64, window_pattern: "SL".to_string(), quantization: None,
        };

        use crate::common::ModelAutodiffBackend;
        let gpt: Gpt<ModelAutodiffBackend> = Gpt::new(config, &device);

        let prompt = vec![12, 45, 67, 89, 90];
        let prompt_len = prompt.len();
        let num_samples = 2;

        let mut idx_data = Vec::new();
        for _ in 0..num_samples { idx_data.extend(prompt.iter().map(|&x| x as i32)); }

        // Prefill index tensor
        let prefill_idx = Tensor::<ModelAutodiffBackend, 2, Int>::from_data(
            TensorData::new(idx_data, Shape::new([num_samples, prompt_len])), &device,
        );

        let head_dim = gpt.config.n_embd / gpt.config.n_head;

        // Run prefill and 5 steps of autoregressive generation for different page sizes
        let page_sizes = vec![4, 8, 16];
        let mut outputs = Vec::new();

        for &page_size in &page_sizes {
            let mut cache = KVCache::new_paged(gpt.config.n_layer, num_samples,
                gpt.config.sequence_len, gpt.config.n_kv_head, head_dim, page_size, &device,
            );

            // 1. Prefill
            let logits = gpt.forward_with_cache(prefill_idx.clone(), &mut cache, 0);
            let mut step_logits = vec![logits.clone()];

            // 2. Autoregressive steps
            let mut current_token = Tensor::<ModelAutodiffBackend, 2, Int>::from_data(
                TensorData::new(vec![90i32; num_samples], Shape::new([num_samples, 1])),
                &device,
            );

            for step_idx in 0..5 {
                let step = prompt_len + step_idx;
                let logits_step = gpt.forward_with_cache(current_token.clone(), &mut cache, step);
                step_logits.push(logits_step.clone());

                current_token = Tensor::<ModelAutodiffBackend, 2, Int>::from_data(
                    TensorData::new(vec![91i32; num_samples], Shape::new([num_samples, 1])),
                    &device,
                );
            }

            outputs.push(step_logits);
        }

        // Assert that the logits are mathematically identical across all page sizes
        for step in 0..outputs[0].len() {
            let logits_4 = &outputs[0][step];
            let logits_8 = &outputs[1][step];
            let logits_16 = &outputs[2][step];

            let diff_8 = (logits_4.clone() - logits_8.clone()).abs().max().into_scalar().to_f32();
            let diff_16 = (logits_4.clone() - logits_16.clone()).abs().max().into_scalar().to_f32();

            assert_eq!(diff_8, 0.0, "Logits differ between page_size=4 and page_size=8 at step {}", step);
            assert_eq!(diff_16, 0.0, "Logits differ between page_size=4 and page_size=16 at step {}", step);
        }
    }
//}
