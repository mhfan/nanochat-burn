
use std::marker::PhantomData;

use burn::{module::{Module, Param}, nn::{Embedding, Linear},
    tensor::{Bool, Distribution, Element, Int, Shape, Tensor, TensorData, activation,
        backend::Backend, module, ops::AttentionModuleOptions},
};
use serde::{Deserialize, Serialize};

mod cache;
pub mod quant;

pub use cache::KVCache;

use self::quant::LinearOrQuantized;

pub trait ForwardLayer<B: Backend>: Module<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D>;
}

impl<B: Backend> ForwardLayer<B> for Linear<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.forward(input)
    }
}

#[derive(Clone, Debug)]
pub struct RotaryEmbeddings<B: Backend> {
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> RotaryEmbeddings<B> {
    pub fn new(cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Self { Self { cos, sin } }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizationConfig {
    pub bits: usize,       // 8 or 4 bits
    pub block_size: usize, // e.g. 32, 64, or 0 (row-wise)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelFeatures {
    pub relu_squared: bool,
    pub qk_norm: bool,
    pub gqa: bool,
    pub swa: bool,
    pub smear: bool,
    pub backout: bool,
}

impl Default for ModelFeatures {
    fn default() -> Self {
        Self { relu_squared: true, qk_norm: true, gqa: true, swa: true,
            smear: true, backout: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GptConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub window_pattern: String,
    #[serde(default)]
    pub features: ModelFeatures,
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl GptConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.sequence_len == 0 { return Err("sequence_len must be greater than zero"); }
        if self.vocab_size == 0 { return Err("vocab_size must be greater than zero"); }
        if self.n_layer == 0 { return Err("n_layer must be greater than zero"); }
        if self.n_head == 0 { return Err("n_head must be greater than zero"); }
        if self.n_kv_head == 0 { return Err("n_kv_head must be greater than zero"); }
        if self.n_embd == 0 { return Err("n_embd must be greater than zero"); }
        if self.n_head % self.n_kv_head != 0 { return Err("n_kv_head must divide n_head"); }
        if !self.features.gqa && self.n_kv_head != self.n_head {
            return Err("disabling GQA requires n_kv_head to equal n_head");
        }
        if self.n_embd % self.n_head != 0 { return Err("n_head must divide n_embd"); }
        if (self.n_embd / self.n_head) % 2 != 0 { return Err("attention head size must be even"); }
        if let Some(quantization) = &self.quantization && !matches!(quantization.bits, 4 | 8) {
            return Err("quantization bits must be 4 or 8");
        }
        if !self.window_pattern.bytes().all(|byte| matches!(byte, b'S' | b's' | b'L' | b'l')) {
            return Err("window_pattern may only contain S and L");
        }
        Ok(())
    }

    pub fn compute_window_sizes(&self) -> Vec<i32> {
        if !self.features.swa { return vec![-1; self.n_layer]; }
        let short_window = ((self.sequence_len as f32 / 4.0 / 128.0).ceil() * 128.0) as i32;
        let pattern = self.window_pattern.to_ascii_uppercase().into_bytes();
        let long_window = -1;
        if pattern.is_empty() { return vec![long_window; self.n_layer]; }

        let mut window_sizes: Vec<_> = (0..self.n_layer).map(|i| {
            if pattern[i % pattern.len()] == b'S' { short_window } else { long_window }
        }).collect();
        if let Some(w) = window_sizes.last_mut() { *w = long_window; }
        window_sizes
    }
}

pub fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
    n_layer > 0 && layer_idx % 2 == (n_layer - 1) % 2
}

fn precompute_window_mask<B: Backend>(window_size: i32, sequence_len: usize,
    device: &B::Device) -> Tensor<B, 4, Bool> {
    let mask_data: Vec<_> = (0..sequence_len).flat_map(|i| {
            let left_bound =
                if window_size < 0 { 0 } else { i.saturating_sub(window_size as usize) };
            (0..sequence_len).map(move |j| j > i || j < left_bound)
        }).collect();
    Tensor::from_data(
        TensorData::new(mask_data, Shape::new([1, 1, sequence_len, sequence_len])), device)
}

fn precompute_rope_cache<B: Backend>(sequence_len: usize, head_dim: usize,
    device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let inv_freq: Vec<_> = (0..head_dim).step_by(2)
        .map(|i| 1.0 / ROPE_BASE_FREQ.powf(i as f32 / head_dim as f32)).collect();
    let mut cos = Vec::with_capacity(sequence_len * inv_freq.len());
    let mut sin = Vec::with_capacity(sequence_len * inv_freq.len());
    for position in 0..sequence_len {
        let position = position as f32;
        cos.extend(inv_freq.iter().map(|&frequency| (position * frequency).cos()));
        sin.extend(inv_freq.iter().map(|&frequency| (position * frequency).sin()));
    }
    let shape = Shape::new([sequence_len, head_dim / 2]);
    (Tensor::from_data(TensorData::new(cos, shape.clone()), device),
        Tensor::from_data(TensorData::new(sin, shape), device))
}

pub fn scaled_dot_product_attention_reference<B: Backend>(q: Tensor<B, 4>,
    k: Tensor<B, 4>, v: Tensor<B, 4>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 4> {
    let [batch_size, n_head, query_len, _] = q.shape().dims();
    let key_len = k.shape().dims::<4>()[2];
    let scale = 1.0 / (q.shape().dims::<4>()[3] as f32).sqrt();
    let mask = mask.expand([batch_size, n_head, query_len, key_len]);
    let scores = (q.matmul(k.swap_dims(2, 3)) * scale).mask_fill(mask, f32::NEG_INFINITY);
    activation::softmax(scores, 3).matmul(v)
}

pub fn scaled_dot_product_attention_burn<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>,
    v: Tensor<B, 4>, mask: Option<Tensor<B, 4, Bool>>, is_causal: bool) -> Tensor<B, 4> {
    let [batch_size, n_head, query_len, _] = q.shape().dims();
    let key_len = k.shape().dims::<4>()[2];
    let mask = mask.map(|mask| mask.expand([batch_size, n_head, query_len, key_len]));
    module::attention(q, k, v, mask, None,
        AttentionModuleOptions { is_causal, ..Default::default() })
}

pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f32) -> Tensor<B, D> {
    let variance = x.clone().square().mean_dim(D - 1);
    x * (variance + eps).sqrt().recip()
}

fn norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let eps = B::FloatElem::dtype().finfo().expect("float backend dtype").epsilon as f32;
    rms_norm(x, eps)
}

pub fn apply_rotary_emb<B: Backend>(x: Tensor<B, 4>,
    cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, seq_len, n_head, c] = x.shape().dims();
    let d = c / 2;
    let x1 = x.clone().slice([0..b, 0..seq_len, 0..n_head, 0..d]);
    let x2 = x.slice([0..b, 0..seq_len, 0..n_head, d..c]);
    let y1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let y2 = x2 * cos - x1 * sin;
    Tensor::cat(vec![y1, y2], 3)
}

fn repeat_kv<B: Backend>(x: Tensor<B, 4>, group_size: usize) -> Tensor<B, 4> {
    if group_size == 1 { return x; }
    let [b, t, n_kv_head, head_dim] = x.shape().dims();
    let x_reshaped: Tensor<B, 5> = x.reshape([b, t, n_kv_head, 1, head_dim]);
    let x_expanded = x_reshaped.expand([b, t, n_kv_head, group_size, head_dim]);
    x_expanded.reshape([b, t, n_kv_head * group_size, head_dim])
}

const VE_GATE_INPUT_DIM: usize = 12;
const SMEAR_GATE_INPUT_DIM: usize = 24;
const LOGIT_CLIP_VAL: f32 = 15.0;
const ROPE_BASE_FREQ: f32 = 100000.0;

fn param<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Param<Tensor<B, D>> {
    Param::from_tensor(tensor)
}

fn linear<B: Backend>(weight: Tensor<B, 2>) -> Linear<B> {
    Linear { weight: param(weight), bias: None }
}

fn random_linear<B: Backend>(in_dim: usize, out_dim: usize, dist: Distribution,
    device: &B::Device) -> Linear<B> {
    linear(Tensor::random([in_dim, out_dim], dist, device))
}

fn zero_linear<B: Backend>(in_dim: usize, out_dim: usize, device: &B::Device) -> Linear<B> {
    linear(Tensor::zeros([in_dim, out_dim], device))
}

fn embedding<B: Backend>(weight: Tensor<B, 2>) -> Embedding<B> {
    Embedding { weight: param(weight) }
}

fn tensor_param<B: Backend>(data: Vec<f32>, device: &B::Device) -> Param<Tensor<B, 1>> {
    let len = data.len();
    param(Tensor::from_data(TensorData::new(data, Shape::new([len])), device))
}

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
    pub qk_norm: bool,
    pub is_causal: bool,
    pub mask: Tensor<B, 4, Bool>,
}

impl<B: Backend, L: ForwardLayer<B>> CausalSelfAttention<B, L> {
    fn prepare_qkv(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, t, channels] = x.shape().dims();

        let q = self.c_q.forward_layer(x.clone()).reshape([b, t, self.n_head, self.head_dim]);
        let k =
            self.c_k.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);
        let mut v =
            self.c_v.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);

        if let (Some(ve_tensor), Some(ve_gate_linear)) = (ve, self.ve_gate.as_ref()) {
            let ve_reshaped = ve_tensor.reshape([b, t, self.n_kv_head, self.head_dim]);
            let x_slice = x.slice([0..b, 0..t, 0..channels.min(VE_GATE_INPUT_DIM)]);
            let gate_logits = ve_gate_linear.forward_layer(x_slice);
            let gate = activation::sigmoid(gate_logits) * 3.0; // range (0, 3)
            let gate_unsqueezed = gate.reshape([b, t, self.n_kv_head, 1]);
            v = v + gate_unsqueezed * ve_reshaped;
        }

        let q = apply_rotary_emb(q, rope.cos.clone(), rope.sin.clone());
        let k = apply_rotary_emb(k, rope.cos, rope.sin);
        let (q, k) = if self.qk_norm { (norm(q) * 1.2, norm(k) * 1.2) } else { (q, k) };

        (q, k, v)
    }

    fn compute_attention(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>,
        mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [b, t, _, _] = q.shape().dims();

        let group_size = self.n_head / self.n_kv_head;
        let (k, v) = (repeat_kv(k, group_size), repeat_kv(v, group_size));
        let (q_trans, k_trans, v_trans) =
            (q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2));
        let (mask, is_causal) = if self.is_causal { (None, true) } else { (Some(mask), false) };
        let y = scaled_dot_product_attention_burn(q_trans, k_trans, v_trans, mask, is_causal)
            .swap_dims(1, 2).reshape([b, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }

    pub fn forward_with_cache(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>, cache: &mut KVCache<B>, step: usize) -> Tensor<B, 3> {
        let [_, t, _] = x.shape().dims();
        let (q, k, v) = self.prepare_qkv(x, ve, rope);
        cache.update(self.layer_idx, k, v, step);
        let mask = self.mask.clone().slice([0..1, 0..1, step..step + t, 0..step + t]);
        let [batch_size, _, _, _] = q.shape().dims();
        let y = cache.attend(self.layer_idx, q, mask, step + t,
            (self.n_head, self.n_kv_head, self.head_dim))
            .reshape([batch_size, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }

    pub fn forward_with_cache_rows(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>, cache: &mut KVCache<B>, requests: &[usize], steps: &[usize])
        -> Tensor<B, 3> {
        let [batch_size, t, _] = x.shape().dims();
        let (q, k, v) = self.prepare_qkv(x, ve, rope);
        cache.update_rows(self.layer_idx, k, v, requests, steps);
        let y = cache.attend_rows(self.layer_idx, q, self.mask.clone(), requests, steps,
            (self.n_head, self.n_kv_head, self.head_dim))
            .reshape([batch_size, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>) -> Tensor<B, 3> {
        let shape: [usize; 3] = x.shape().dims();
        let t = shape[1];

        let (q, k, v) = self.prepare_qkv(x, ve, rope);
        let mask = self.mask.clone().slice([0..1, 0..1, 0..t, 0..t]);
        self.compute_attention(q, k, v, mask)
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend, L = Linear<B>> {
    pub c_fc: L,
    pub c_proj: L,
    pub relu_squared: bool,
    pub _phantom: PhantomData<B>,
}

impl<B: Backend, L: ForwardLayer<B>> MLP<B, L> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = activation::relu(self.c_fc.forward_layer(x));
        self.c_proj.forward_layer(if self.relu_squared { x.square() } else { x })
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend, L = Linear<B>> {
    pub attn: CausalSelfAttention<B, L>,
    pub mlp: MLP<B, L>,
}

impl<B: Backend, L: ForwardLayer<B>> Block<B, L> {
    pub fn forward_with_cache(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>, cache: &mut KVCache<B>,
        step: usize) -> Tensor<B, 3> {
        let x = x.clone() +
            self.attn.forward_with_cache(norm(x.clone()),
                ve, rope, cache, step);
        x.clone() + self.mlp.forward(norm(x))
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(norm(x.clone()), ve, rope);
        x.clone() + self.mlp.forward(norm(x))
    }

    pub fn forward_with_cache_rows(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        rope: RotaryEmbeddings<B>, cache: &mut KVCache<B>, requests: &[usize], steps: &[usize])
        -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward_with_cache_rows(
            norm(x.clone()), ve, rope, cache, requests, steps);
        x.clone() + self.mlp.forward(norm(x))
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
    pub rope_cos: Tensor<B, 2>,
    pub rope_sin: Tensor<B, 2>,
    pub config: GptConfig,
}

impl<B: Backend> Gpt<B, Linear<B>> {
    pub fn new(config: GptConfig, device: &B::Device) -> Self {
        config.validate().unwrap_or_else(|message| panic!("invalid GPT config: {message}"));
        let padded_vocab_size = config.vocab_size.div_ceil(64) * 64;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;
        let kv_dim = config.n_kv_head * head_dim;
        let (rope_cos, rope_sin) =
            precompute_rope_cache(config.sequence_len, head_dim, device);

        let wte = embedding(Tensor::random([padded_vocab_size, n_embd],
            Distribution::Normal(0.0, 0.8), device));

        let s = 3.0f32.sqrt() * (n_embd as f32).powf(-0.5);
        let init = Distribution::Uniform(-s as f64, s as f64);
        let value_embeds: Vec<_> = (0..config.n_layer)
            .filter(|&i| has_ve(i, config.n_layer)).map(|_|
                embedding(Tensor::random([padded_vocab_size, kv_dim], init, device))).collect();

        let window_sizes = config.compute_window_sizes();
        let causal_mask = precompute_window_mask(-1, config.sequence_len, device);
        let short_window = window_sizes.iter().copied().find(|&window| window >= 0);
        let short_mask = short_window.map(|window|
            precompute_window_mask(window, config.sequence_len, device));
        let h: Vec<_> = (0..config.n_layer).map(|i| {
                let c_q = random_linear(n_embd, config.n_head * head_dim, init, device);
                let c_k = random_linear(n_embd, kv_dim, init, device);
                let c_v = random_linear(n_embd, kv_dim, init, device);
                let c_proj = zero_linear(n_embd, n_embd, device);
                let ve_gate = if has_ve(i, config.n_layer) {
                    Some(random_linear(n_embd.min(VE_GATE_INPUT_DIM), config.n_kv_head,
                        Distribution::Uniform(0.0, 0.02), device))
                } else { None };

                let window_size = window_sizes[i];
                let is_causal = window_size < 0;
                let mask = if is_causal { causal_mask.clone() } else {
                    short_mask.as_ref().expect("short attention mask").clone()
                };
                let attn = CausalSelfAttention { c_q, c_k, c_v, c_proj, ve_gate, layer_idx: i,
                    n_head: config.n_head, n_kv_head: config.n_kv_head, head_dim,
                    qk_norm: config.features.qk_norm, is_causal, mask,
                };

                let mlp_init = Distribution::Uniform((-s * 0.4) as f64, (s * 0.4) as f64);
                let c_fc = random_linear(n_embd, 4 * n_embd, mlp_init, device);
                let c_proj_mlp = zero_linear(4 * n_embd, n_embd, device);
                let mlp = MLP { c_fc, c_proj: c_proj_mlp,
                    relu_squared: config.features.relu_squared, _phantom: PhantomData };

                Block { attn, mlp }
            }).collect();

        let lm_head =
            random_linear(n_embd, padded_vocab_size, Distribution::Normal(0.0, 0.001), device);

        let resid_init: Vec<_> = (0..config.n_layer)
            .map(|i| 1.15 - (0.10 * i as f32 / (config.n_layer - 1).max(1) as f32)).collect();
        let x0_init: Vec<_> = (0..config.n_layer)
            .map(|i| 0.20 - (0.15 * i as f32 / (config.n_layer - 1).max(1) as f32)).collect();

        let resid_lambdas = tensor_param(resid_init, device);
        let x0_lambdas = tensor_param(x0_init, device);

        let smear_gate = random_linear(n_embd.min(SMEAR_GATE_INPUT_DIM), 1,
                Distribution::Uniform(0.0, 0.02), device);
        let smear_lambda = param(Tensor::zeros([1], device));
        let backout_lambda = tensor_param(vec![0.2], device);

        Gpt { wte, h, lm_head, resid_lambdas, x0_lambdas, smear_gate,
            smear_lambda, backout_lambda, value_embeds, rope_cos, rope_sin, config, }
    }
}

impl<B: Backend, L: ForwardLayer<B>> Gpt<B, L> {
    fn rotary_embeddings(&self, start: usize, len: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half_dim = self.config.n_embd / self.config.n_head / 2;
        let range = [start..start + len, 0..half_dim];
        (self.rope_cos.clone().slice(range.clone()).reshape([1, len, 1, half_dim]),
            self.rope_sin.clone().slice(range).reshape([1, len, 1, half_dim]))
    }

    fn rotary_embeddings_rows(&self, positions: &[usize], len: usize)
        -> (Tensor<B, 4>, Tensor<B, 4>) {
        let indices: Vec<_> = positions.iter().flat_map(|&start|
            (start..start + len).map(|position| position as i32)).collect();
        let indices = Tensor::from_data(
            TensorData::new(indices, Shape::new([positions.len() * len])),
            &self.rope_cos.device());
        let half_dim = self.config.n_embd / self.config.n_head / 2;
        let shape = [positions.len(), len, 1, half_dim];
        (self.rope_cos.clone().select(0, indices.clone()).reshape(shape),
            self.rope_sin.clone().select(0, indices).reshape(shape))
    }

    fn smear_embeddings(&self, x: Tensor<B, 3>, previous: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let [batch_size, seq_len, n_embd] = x.shape().dims();
        let start = usize::from(previous.is_none());
        if seq_len <= start { return x; }

        let current = x.clone().slice([0..batch_size, start..seq_len, 0..n_embd]);
        let predecessors = if let Some(previous) = previous {
            if seq_len == 1 { previous } else {
                Tensor::cat(vec![previous,
                    x.clone().slice([0..batch_size, 0..seq_len - 1, 0..n_embd])], 1)
            }
        } else {    x.clone().slice([0..batch_size, 0..seq_len - 1, 0..n_embd]) };
        let gate_input = current.clone().slice([
            0..batch_size, 0..seq_len - start, 0..n_embd.min(SMEAR_GATE_INPUT_DIM)]);
        let gate = activation::sigmoid(self.smear_gate.forward_layer(gate_input)) *
            self.smear_lambda.clone().val().reshape([1, 1, 1]);
        let smeared = current + gate * predecessors;

        if start == 0 { smeared } else {
            Tensor::cat(vec![x.slice([0..batch_size, 0..1, 0..n_embd]), smeared], 1)
        }
    }

    fn smear_embeddings_with_cache(&self, idx: Tensor<B, 2, Int>, x: Tensor<B, 3>,
        cache: &mut KVCache<B>, step: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.shape().dims();
        assert_eq!(idx.shape().dims(), [batch_size, seq_len], "token embedding shape mismatch");
        assert_eq!(batch_size, cache.batch_size, "cache batch size mismatch");
        let end = step.checked_add(seq_len).expect("cache position overflow");
        assert!(end <= cache.max_seq_len, "cached sequence exceeds cache capacity");

        let history = cache.token_history.take().unwrap_or_else(|| {
            assert_eq!(step, 0, "cache has no token history before position {step}");
            Tensor::zeros([batch_size, cache.max_seq_len], &idx.device())
        });
        assert_eq!(history.shape().dims(), [batch_size, cache.max_seq_len],
            "cache token history shape mismatch");
        let previous = (step > 0).then(|| {
            let previous_idx = history.clone().slice([0..batch_size, step - 1..step]);
            norm(self.wte.forward(previous_idx))
        });
        let output = self.smear_embeddings(x, previous);
        cache.token_history = Some(history.slice_assign([0..batch_size, step..end], idx));
        output
    }

    fn smear_embeddings_with_cache_rows(&self, idx: Tensor<B, 2, Int>, x: Tensor<B, 3>,
        cache: &mut KVCache<B>, requests: &[usize], steps: &[usize]) -> Tensor<B, 3> {
        let [source_batch_size, seq_len, _] = x.shape().dims();
        assert_eq!(idx.shape().dims(), [source_batch_size, seq_len],
            "token embedding shape mismatch");
        assert_eq!(requests.len(), source_batch_size, "cache request mapping size mismatch");
        assert_eq!(steps.len(), source_batch_size, "cache position mapping size mismatch");
        let has_previous = steps.first().is_some_and(|&step| step > 0);
        assert!(steps.iter().all(|&step| (step > 0) == has_previous),
            "a cached batch cannot mix prefill and decode rows");

        let mut history = cache.token_history.take().unwrap_or_else(||
            Tensor::zeros([cache.batch_size, cache.max_seq_len], &idx.device()));
        let previous = has_previous.then(|| {
            let rows = requests.iter().zip(steps).map(|(&request, &step)|
                history.clone().slice([request..request + 1, step - 1..step])).collect();
            norm(self.wte.forward(Tensor::cat(rows, 0)))
        });
        let output = self.smear_embeddings(x, previous);
        for (source, (&request, &step)) in requests.iter().zip(steps).enumerate() {
            let end = step.checked_add(seq_len).expect("cache position overflow");
            assert!(end <= cache.max_seq_len, "cached sequence exceeds cache capacity");
            history = history.slice_assign([request..request + 1, step..end],
                idx.clone().slice([source..source + 1, 0..seq_len]));
        }
        cache.token_history = Some(history);
        output
    }

    #[allow(clippy::single_range_in_vec_init)]
    fn forward_inner(&self, idx: Tensor<B, 2, Int>, mut cache_opt: Option<&mut KVCache<B>>,
        step: usize) -> Tensor<B, 3> {
        let [batch_size, seq_len] = idx.shape().dims();
        assert!(seq_len > 0, "input must contain at least one token");
        let end = step.checked_add(seq_len).expect("sequence position overflow");
        assert!(end <= self.config.sequence_len, "input exceeds model sequence length");

        let start_pos = if cache_opt.is_some() { step } else { 0 };
        let (cos, sin) = self.rotary_embeddings(start_pos, seq_len);
        let rope = RotaryEmbeddings::new(cos, sin);

        let x_normed = norm(self.wte.forward(idx.clone()));

        let mut x = if !self.config.features.smear { x_normed } else if let Some(cache) =
            cache_opt.as_mut() {
            self.smear_embeddings_with_cache(idx.clone(), x_normed, cache, step)
        } else { self.smear_embeddings(x_normed, None) };

        let (x0, mut x_backout) = (x.clone(), None);
        let backout_layer = self.config.n_layer / 2;
        let resid_val = self.resid_lambdas.clone().val();
        let x0_val = self.x0_lambdas.clone().val();

        let mut ve_cnt = 0;
        for i in 0..self.config.n_layer {
            let r_lambda = resid_val.clone().slice([i..i + 1]).reshape([1, 1, 1]);
            let x0_lambda = x0_val.clone().slice([i..i + 1]).reshape([1, 1, 1]);
            x = x * r_lambda + x0.clone() * x0_lambda;

            let ve = if has_ve(i, self.config.n_layer) {
                let ve_embed = self.value_embeds[ve_cnt].forward(idx.clone());
                ve_cnt += 1;
                Some(ve_embed)
            } else { None };

            if let Some(ref mut cache) = cache_opt {
                x = self.h[i].forward_with_cache(x, ve, rope.clone(), cache, step);
            } else {
                x = self.h[i].forward(x, ve, rope.clone());
            }
            if self.config.features.backout && i == backout_layer {
                x_backout = Some(x.clone());
            }
        }

        if let Some(xb) = x_backout {
            x = x - xb * self.backout_lambda.clone().val().reshape([1, 1, 1]);
        }
        let x = norm(x);

        let mut logits = self.lm_head.forward_layer(x);
        logits = logits.slice([0..batch_size, 0..seq_len, 0..self.config.vocab_size]);
        (logits / LOGIT_CLIP_VAL).tanh() * LOGIT_CLIP_VAL
    }

    // Burn's rank-1 Tensor::slice API intentionally uses a one-element range array.
    #[allow(clippy::single_range_in_vec_init)]
    fn forward_inner_rows(&self, idx: Tensor<B, 2, Int>, cache: &mut KVCache<B>,
        requests: &[usize], steps: &[usize]) -> Tensor<B, 3> {
        let [batch_size, seq_len] = idx.shape().dims();
        assert!(seq_len > 0, "input must contain at least one token");
        assert_eq!(requests.len(), batch_size, "cache request mapping size mismatch");
        assert_eq!(steps.len(), batch_size, "cache position mapping size mismatch");
        assert!((0..requests.len()).all(|index|
            !requests[..index].contains(&requests[index])),
            "cache request mapping contains duplicates");
        assert!(steps.iter().all(|&step| step.checked_add(seq_len)
            .is_some_and(|end| end <= self.config.sequence_len)),
            "input exceeds model sequence length");

        let (cos, sin) = self.rotary_embeddings_rows(steps, seq_len);
        let rope = RotaryEmbeddings::new(cos, sin);
        let x_normed = norm(self.wte.forward(idx.clone()));
        let mut x = if self.config.features.smear {
            self.smear_embeddings_with_cache_rows(
                idx.clone(), x_normed, cache, requests, steps)
        } else { x_normed };

        let (x0, mut x_backout) = (x.clone(), None);
        let backout_layer = self.config.n_layer / 2;
        let resid_val = self.resid_lambdas.clone().val();
        let x0_val = self.x0_lambdas.clone().val();
        let mut ve_cnt = 0;
        for i in 0..self.config.n_layer {
            let r_lambda = resid_val.clone().slice([i..i + 1]).reshape([1, 1, 1]);
            let x0_lambda = x0_val.clone().slice([i..i + 1]).reshape([1, 1, 1]);
            x = x * r_lambda + x0.clone() * x0_lambda;
            let ve = if has_ve(i, self.config.n_layer) {
                let ve_embed = self.value_embeds[ve_cnt].forward(idx.clone());
                ve_cnt += 1;
                Some(ve_embed)
            } else { None };
            x = self.h[i].forward_with_cache_rows(
                x, ve, rope.clone(), cache, requests, steps);
            if self.config.features.backout && i == backout_layer {
                x_backout = Some(x.clone());
            }
        }
        if let Some(xb) = x_backout {
            x = x - xb * self.backout_lambda.clone().val().reshape([1, 1, 1]);
        }
        let logits = self.lm_head.forward_layer(norm(x))
            .slice([0..batch_size, 0..seq_len, 0..self.config.vocab_size]);
        (logits / LOGIT_CLIP_VAL).tanh() * LOGIT_CLIP_VAL
    }

    pub fn forward_with_cache(&self, idx: Tensor<B, 2, Int>, cache: &mut KVCache<B>,
        step: usize) -> Tensor<B, 3> {
        self.forward_inner(idx, Some(cache), step)
    }

    pub fn forward_with_cache_rows(&self, idx: Tensor<B, 2, Int>, cache: &mut KVCache<B>,
        requests: &[usize], steps: &[usize]) -> Tensor<B, 3> {
        self.forward_inner_rows(idx, cache, requests, steps)
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_inner(idx, None, 0)
    }

    pub fn compute_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>)
        -> Tensor<B, 1> {
        let unreduced = self.compute_unreduced_loss(logits, targets.clone());
        let num_valid = targets.reshape([-1]).not_equal_elem(-1).float().sum().clamp(1.0, 1e9);
        unreduced.sum() / num_valid
    }

    pub fn compute_unreduced_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>)
        -> Tensor<B, 1> {
        let shape: [usize; 3] = logits.shape().dims();
        let (b, t, v) = (shape[0], shape[1], shape[2]);
        let flat_logits = logits.reshape([b * t, v]).clamp(-50.0, 50.0);
        let flat_targets = targets.reshape([b * t]);

        let log_probs = activation::log_softmax(flat_logits, 1);
        let mask_valid = flat_targets.clone().not_equal_elem(-1);
        let clamped_targets = flat_targets.clamp(0, (v - 1) as i32);

        let selected_log_probs = log_probs
            .gather(1, clamped_targets.reshape([b * t, 1])).reshape([b * t]);

        let valid_float = mask_valid.float();
        selected_log_probs * valid_float * -1.0
    }
}

impl<B: Backend> Gpt<B, Linear<B>> {
    fn convert_blocks<F>(self, f: F) -> Gpt<B, LinearOrQuantized<B>>
        where F: Fn(Linear<B>) -> LinearOrQuantized<B> {
        let h = self.h.into_iter().map(|block| {
                let Block { attn, mlp } = block;

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
                    qk_norm: attn.qk_norm,
                    is_causal: attn.is_causal,
                    mask: attn.mask,
                };

                let mlp_conv = MLP { c_fc: f(mlp.c_fc), c_proj: f(mlp.c_proj),
                    relu_squared: mlp.relu_squared, _phantom: PhantomData };

                Block { attn: attn_conv, mlp: mlp_conv }
            }).collect();

        let lm_head_conv = f(self.lm_head);
        let smear_gate_conv = f(self.smear_gate);

        Gpt { wte: self.wte, h,
            lm_head: lm_head_conv,
            resid_lambdas: self.resid_lambdas,
            x0_lambdas: self.x0_lambdas,
            smear_gate: smear_gate_conv,
            smear_lambda: self.smear_lambda,
            backout_lambda: self.backout_lambda,
            value_embeds: self.value_embeds,
            rope_cos: self.rope_cos,
            rope_sin: self.rope_sin,
            config: self.config,
        }
    }

    pub fn into_linear_or_quantized(self) -> Gpt<B, LinearOrQuantized<B>> {
        self.convert_blocks(LinearOrQuantized::Standard)
    }

    pub fn quantize(self, bits: usize, block_size: usize) -> Gpt<B, LinearOrQuantized<B>> {
        use self::quant::quantize_linear_or_standard;
        assert!(matches!(bits, 4 | 8), "quantization bits must be 4 or 8");
        self.convert_blocks(|linear| quantize_linear_or_standard(linear, bits, block_size))
    }
}

#[cfg(test)] mod parity;
#[cfg(test)] mod tests;
