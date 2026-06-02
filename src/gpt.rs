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

pub fn has_ve(layer_idx: usize, n_layer: usize) -> bool {
    layer_idx % 2 == (n_layer - 1) % 2
}

pub fn get_window_size_for_layer(config: &GptConfig, layer_idx: usize) -> i32 {
    let pattern = config.window_pattern.to_uppercase();
    let long_window = -1;
    let short_window = ((config.sequence_len as f32 / 4.0 / 128.0).ceil() * 128.0) as i32;
    let mut window_sizes = Vec::new();
    for i in 0..config.n_layer {
        let ch = pattern.chars().nth(i % pattern.len()).unwrap_or('L');
        window_sizes.push(if ch == 'S' { short_window } else { long_window });
    }
    if let Some(w) = window_sizes.last_mut() { *w = long_window; }
    window_sizes[layer_idx]
}

fn precompute_window_mask<B: Backend>(window_size: i32, sequence_len: usize, device: &B::Device) -> Tensor<B, 4> {
    let mut mask_data = Vec::with_capacity(sequence_len * sequence_len);
    for i in 0..sequence_len {
        for j in 0..sequence_len {
            let left_bound = if window_size < 0 { 0 } else { i.saturating_sub(window_size as usize) };
            mask_data.push(if j > i || j < left_bound { -1e9f32 } else { 0.0f32 });
        }
    }
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
    let y2 = x1 * (sin * -1.0) + x2 * cos;
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
    pub k_cache: Vec<Tensor<B, 4>>, // Layer -> [B, seq_len, n_kv_head, head_dim]
    pub v_cache: Vec<Tensor<B, 4>>, // Layer -> [B, seq_len, n_kv_head, head_dim]
}

impl<B: Backend> KVCache<B> {
    pub fn new(n_layer: usize) -> Self {
        Self {
            k_cache: Vec::with_capacity(n_layer),
            v_cache: Vec::with_capacity(n_layer),
        }
    }

    pub fn new_allocated(n_layer: usize, batch_size: usize, max_seq_len: usize, n_kv_head: usize,
        head_dim: usize, device: &B::Device,) -> Self {
        let mut k_cache = Vec::with_capacity(n_layer);
        let mut v_cache = Vec::with_capacity(n_layer);
        let shape = Shape::new([batch_size, max_seq_len, n_kv_head, head_dim]);
        for _ in 0..n_layer {
            k_cache.push(Tensor::zeros(shape.clone(), device));
            v_cache.push(Tensor::zeros(shape.clone(), device));
        }
        Self { k_cache, v_cache }
    }
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
    pub mask: Tensor<B, 4>,
}

impl<B: Backend, L: ForwardLayer<B>> CausalSelfAttention<B, L> {
    pub fn forward_with_cache(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>, _window_size: i32,
        layer_idx: usize, cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        let shape: [usize; 3] = x.shape().dims();
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let mut q = self.c_q.forward_layer(x.clone()).reshape([b, t, self.n_head, self.head_dim]);
        let mut k = self.c_k.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);
        let mut v = self.c_v.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);

        if let Some(ve_tensor) = ve {
            let ve_reshaped = ve_tensor.reshape([b, t, self.n_kv_head, self.head_dim]);
            if let Some(ref ve_gate_linear) = self.ve_gate {
                let x_slice = x.clone().slice([0..b, 0..t, 0..12]);
                let gate_logits = ve_gate_linear.forward_layer(x_slice);
                let gate = ((gate_logits * -1.0).exp() + 1.0).recip() * 3.0; // range (0, 3)
                let gate_unsqueezed = gate.reshape([b, t, self.n_kv_head, 1]);
                v = v + gate_unsqueezed * ve_reshaped;
            }
        }

        q = apply_rotary_emb(q, cos.clone(), sin.clone());
        k = apply_rotary_emb(k, cos, sin);
        q = rms_norm(q, 1e-5) * 1.2;
        k = rms_norm(k, 1e-5) * 1.2;

        let mask_dims: [usize; 4] = self.mask.shape().dims();
        let max_seq_len = mask_dims[3];

        let (mut full_k, mut full_v) = if step == 0 {
            let full_k = cache.k_cache[layer_idx].clone().slice_assign(
                [0..b, 0..t, 0..self.n_kv_head, 0..self.head_dim], k,);
            let full_v = cache.v_cache[layer_idx].clone().slice_assign(
                [0..b, 0..t, 0..self.n_kv_head, 0..self.head_dim], v,);
            cache.k_cache[layer_idx] = full_k.clone();
            cache.v_cache[layer_idx] = full_v.clone();
            (full_k, full_v)
        } else {
            let p = step;
            let full_k = cache.k_cache[layer_idx].clone().slice_assign(
                [0..b, p..p+t, 0..self.n_kv_head, 0..self.head_dim], k,);
            let full_v = cache.v_cache[layer_idx].clone().slice_assign(
                [0..b, p..p+t, 0..self.n_kv_head, 0..self.head_dim], v,);
            cache.k_cache[layer_idx] = full_k.clone();
            cache.v_cache[layer_idx] = full_v.clone();
            (full_k, full_v)
        };

        let group_size = self.n_head / self.n_kv_head;
        full_k = repeat_kv(full_k, group_size);
        full_v = repeat_kv(full_v, group_size);

        let q_trans = q.swap_dims(1, 2);
        let k_trans = full_k.swap_dims(1, 2);
        let v_trans = full_v.swap_dims(1, 2);
        let k_t = k_trans.swap_dims(2, 3);
        let mut scores = q_trans.matmul(k_t) * (1.0 / (self.head_dim as f32).sqrt());

        let mask = if step == 0 {
            self.mask.clone().slice([0..1, 0..1, 0..t, 0..max_seq_len])
        } else {
            let p = step;
            self.mask.clone().slice([0..1, 0..1, p..p+1, 0..max_seq_len])
        };

        scores = scores + mask;

        let probs = activation::softmax(scores, 3);
        let y = probs.matmul(v_trans).swap_dims(1, 2).reshape([b, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>, _window_size: i32,) -> Tensor<B, 3> {
        let shape: [usize; 3] = x.shape().dims();
        let (b, t, _) = (shape[0], shape[1], shape[2]);

        let mut q = self.c_q.forward_layer(x.clone()).reshape([b, t, self.n_head, self.head_dim]);
        let mut k = self.c_k.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);
        let mut v = self.c_v.forward_layer(x.clone()).reshape([b, t, self.n_kv_head, self.head_dim]);

        if let Some(ve_tensor) = ve {
            let ve_reshaped = ve_tensor.reshape([b, t, self.n_kv_head, self.head_dim]);
            if let Some(ref ve_gate_linear) = self.ve_gate {
                let x_slice = x.clone().slice([0..b, 0..t, 0..12]);
                let gate_logits = ve_gate_linear.forward_layer(x_slice);
                let gate = ((gate_logits * -1.0).exp() + 1.0).recip() * 3.0; // range (0, 3)
                let gate_unsqueezed = gate.reshape([b, t, self.n_kv_head, 1]);
                v = v + gate_unsqueezed * ve_reshaped;
            }
        }

        q = apply_rotary_emb(q, cos.clone(), sin.clone());
        k = apply_rotary_emb(k, cos, sin);
        q = rms_norm(q, 1e-5) * 1.2;
        k = rms_norm(k, 1e-5) * 1.2;

        let group_size = self.n_head / self.n_kv_head;
        k = repeat_kv(k, group_size);
        v = repeat_kv(v, group_size);

        let q_trans = q.swap_dims(1, 2);
        let k_trans = k.swap_dims(1, 2);
        let v_trans = v.swap_dims(1, 2);
        let k_t = k_trans.swap_dims(2, 3);
        let mut scores = q_trans.matmul(k_t) * (1.0 / (self.head_dim as f32).sqrt());

        let mask = self.mask.clone().slice([0..1, 0..1, 0..t, 0..t]);
        scores = scores + mask;

        let probs = activation::softmax(scores, 3);
        let y = probs.matmul(v_trans).swap_dims(1, 2).reshape([b, t, self.n_head * self.head_dim]);
        self.c_proj.forward_layer(y)
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend, L = Linear<B>> {
    pub c_fc: L,
    pub c_proj: L,
    pub _phantom: std::marker::PhantomData<B>,
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
        cos: Tensor<B, 4>, sin: Tensor<B, 4>, window_size: i32,
        layer_idx: usize, cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward_with_cache(rms_norm(x.clone(), 1e-5), ve, cos, sin, window_size, layer_idx, cache, step);
        x.clone() + self.mlp.forward(rms_norm(x, 1e-5))
    }

    pub fn forward(&self, x: Tensor<B, 3>, ve: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>, sin: Tensor<B, 4>, window_size: i32,) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(rms_norm(x.clone(), 1e-5), ve, cos, sin, window_size);
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

        let mut value_embeds = Vec::new();
        let s = 3.0f32.sqrt() * (n_embd as f32).powf(-0.5);
        for i in 0..config.n_layer {
            if has_ve(i, config.n_layer) {
                let ve_weight = Tensor::random([padded_vocab_size, kv_dim], Distribution::Uniform(-s as f64, s as f64), device);
                value_embeds.push(Embedding { weight: Param::from_tensor(ve_weight) });
            }
        }

        let mut h = Vec::new();
        for i in 0..config.n_layer {
            let c_q = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
            let c_k = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_kv_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
            let c_v = Linear { weight: Param::from_tensor(Tensor::random([n_embd, config.n_kv_head * head_dim], Distribution::Uniform(-s as f64, s as f64), device)), bias: None };
            let c_proj = Linear { weight: Param::from_tensor(Tensor::zeros([n_embd, n_embd], device)), bias: None };
            let ve_gate = if has_ve(i, config.n_layer) {
                Some(Linear { weight: Param::from_tensor(Tensor::random([12, config.n_kv_head], Distribution::Uniform(0.0, 0.02), device)), bias: None })
            } else { None };

            let window_size = get_window_size_for_layer(&config, i);
            let mask = precompute_window_mask::<B>(window_size, config.sequence_len, device);
            let attn = CausalSelfAttention {
                c_q, c_k, c_v, c_proj, ve_gate, layer_idx: i, n_head: config.n_head, n_kv_head: config.n_kv_head, head_dim, mask,
            };

            let c_fc = Linear { weight: Param::from_tensor(Tensor::random([n_embd, 4 * n_embd], Distribution::Uniform((-s * 0.4) as f64, (s * 0.4) as f64), device)), bias: None };
            let c_proj_mlp = Linear { weight: Param::from_tensor(Tensor::zeros([4 * n_embd, n_embd], device)), bias: None };
            let mlp = MLP { c_fc, c_proj: c_proj_mlp, _phantom: std::marker::PhantomData };

            h.push(Block { attn, mlp });
        }

        let lm_head_weight = Tensor::random([n_embd, padded_vocab_size], Distribution::Normal(0.0, 0.001), device);
        let lm_head = Linear { weight: Param::from_tensor(lm_head_weight), bias: None };

        let mut resid_init = vec![0.0f32; config.n_layer];
        let mut x0_init = vec![0.0f32; config.n_layer];
        for i in 0..config.n_layer {
            resid_init[i] = 1.15 - (0.10 * i as f32 / (config.n_layer - 1).max(1) as f32);
            x0_init[i] = 0.20 - (0.15 * i as f32 / (config.n_layer - 1).max(1) as f32);
        }

        let resid_lambdas = Param::from_tensor(Tensor::from_data(TensorData::new(resid_init, Shape::new([config.n_layer])), device));
        let x0_lambdas = Param::from_tensor(Tensor::from_data(TensorData::new(x0_init, Shape::new([config.n_layer])), device));

        let smear_gate = Linear { weight: Param::from_tensor(Tensor::random([24, 1], Distribution::Uniform(0.0, 0.02), device)), bias: None };
        let smear_lambda = Param::from_tensor(Tensor::zeros([1], device));
        let backout_lambda = Param::from_tensor(Tensor::from_data(TensorData::new(vec![0.2f32], Shape::new([1])), device));

        Gpt {
            wte, h, lm_head, resid_lambdas, x0_lambdas, smear_gate, smear_lambda, backout_lambda, value_embeds, config,
        }
    }
}

impl<B: Backend, L: ForwardLayer<B>> Gpt<B, L> {
    fn precompute_rotary_embeddings(&self, seq_len: usize, head_dim: usize,
        device: &B::Device,) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let base = 100000.0f32;
        let mut inv_freq = Vec::with_capacity(head_dim / 2);
        for i in (0..head_dim).step_by(2) { inv_freq.push(1.0 / base.powf(i as f32 / head_dim as f32)); }

        let mut cos_data = Vec::with_capacity(seq_len * (head_dim / 2));
        let mut sin_data = Vec::with_capacity(seq_len * (head_dim / 2));

        for t in 0..seq_len {
            for &freq in &inv_freq {
                let angle = t as f32 * freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }

        let cos = Tensor::<B, 4>::from_data(TensorData::new(cos_data, Shape::new([1, seq_len, 1, head_dim / 2])), device);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sin_data, Shape::new([1, seq_len, 1, head_dim / 2])), device);
        (cos, sin)
    }

    fn precompute_rotary_embeddings_at_step(&self, step: usize, head_dim: usize,
        device: &B::Device,) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let base = 100000.0f32;
        let mut inv_freq = Vec::with_capacity(head_dim / 2);
        for i in (0..head_dim).step_by(2) { inv_freq.push(1.0 / base.powf(i as f32 / head_dim as f32)); }

        let mut cos_data = Vec::with_capacity(head_dim / 2);
        let mut sin_data = Vec::with_capacity(head_dim / 2);

        for &freq in &inv_freq {
            let angle = step as f32 * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }

        let cos = Tensor::<B, 4>::from_data(TensorData::new(cos_data, Shape::new([1, 1, 1, head_dim / 2])), device);
        let sin = Tensor::<B, 4>::from_data(TensorData::new(sin_data, Shape::new([1, 1, 1, head_dim / 2])), device);
        (cos, sin)
    }


    fn compute_window_sizes(&self) -> Vec<i32> {
        let pattern = self.config.window_pattern.to_uppercase();
        let long_window = -1;
        let short_window = ((self.config.sequence_len as f32 / 4.0 / 128.0).ceil() * 128.0) as i32;
        let mut window_sizes = Vec::new();
        for i in 0..self.config.n_layer {
            let ch = pattern.chars().nth(i % pattern.len()).unwrap_or('L');
            window_sizes.push(if ch == 'S' { short_window } else { long_window });
        }
        if let Some(w) = window_sizes.last_mut() { *w = long_window; }
        window_sizes
    }

    pub fn forward_with_cache(&self, idx: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>, step: usize,) -> Tensor<B, 3> {
        let shape: [usize; 2] = idx.shape().dims();
        let (batch_size, seq_len) = (shape[0], shape[1]);

        let head_dim = self.config.n_embd / self.config.n_head;
        let (cos, sin) = if seq_len > 1 {
            self.precompute_rotary_embeddings(seq_len, head_dim, &idx.device())
        } else {
            self.precompute_rotary_embeddings_at_step(step, head_dim, &idx.device())
        };

        let x = self.wte.forward(idx.clone());
        let x_normed = rms_norm(x, 1e-5);

        let mut x = if seq_len > 1 {
            let x_slice = x_normed.clone().slice([0..batch_size, 1..seq_len, 0..24]);
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
        let window_sizes = self.compute_window_sizes();

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

            x = self.h[i].forward_with_cache(x, ve, cos.clone(), sin.clone(), window_sizes[i], i, cache, step);
            if i == backout_layer { x_backout = Some(x.clone()); }
        }

        if let Some(xb) = x_backout {
            x = x - xb * self.backout_lambda.clone().val().reshape([1, 1, 1]);
        }
        let x = rms_norm(x, 1e-5);

        let mut logits = self.lm_head.forward_layer(x);
        logits = logits.slice([0..batch_size, 0..seq_len, 0..self.config.vocab_size]);
        (logits / 15.0).tanh() * 15.0
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>, _targets: Option<Tensor<B, 2, Int>>) -> Tensor<B, 3> {
        let shape: [usize; 2] = idx.shape().dims();
        let (batch_size, seq_len) = (shape[0], shape[1]);

        let head_dim = self.config.n_embd / self.config.n_head;
        let (cos, sin) = self.precompute_rotary_embeddings(seq_len, head_dim, &idx.device());

        let x = self.wte.forward(idx.clone());
        let x_normed = rms_norm(x, 1e-5);

        let mut x = if seq_len > 1 {
            let x_slice = x_normed.clone().slice([0..batch_size, 1..seq_len, 0..24]);
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
        let window_sizes = self.compute_window_sizes();

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

            x = self.h[i].forward(x, ve, cos.clone(), sin.clone(), window_sizes[i]);
            if i == backout_layer { x_backout = Some(x.clone()); }
        }

        if let Some(xb) = x_backout {
            x = x - xb * self.backout_lambda.clone().val().reshape([1, 1, 1]);
        }
        let x = rms_norm(x, 1e-5);

        let mut logits = self.lm_head.forward_layer(x);
        logits = logits.slice([0..batch_size, 0..seq_len, 0..self.config.vocab_size]);
        (logits / 15.0).tanh() * 15.0
    }

    pub fn compute_loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
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
        let sum_loss = (selected_log_probs * valid_float.clone() * -1.0).sum();
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
    pub fn into_linear_or_quantized(self) -> Gpt<B, LinearOrQuantized<B>> {
        let mut h_conv = Vec::with_capacity(self.h.len());
        for block in self.h {
            let attn = block.attn;
            let mlp = block.mlp;

            let attn_conv = CausalSelfAttention {
                c_q: LinearOrQuantized::Standard(attn.c_q),
                c_k: LinearOrQuantized::Standard(attn.c_k),
                c_v: LinearOrQuantized::Standard(attn.c_v),
                c_proj: LinearOrQuantized::Standard(attn.c_proj),
                ve_gate: attn.ve_gate.map(|ve| LinearOrQuantized::Standard(ve)),
                layer_idx: attn.layer_idx,
                n_head: attn.n_head,
                n_kv_head: attn.n_kv_head,
                head_dim: attn.head_dim,
                mask: attn.mask,
            };

            let mlp_conv = MLP {
                c_fc: LinearOrQuantized::Standard(mlp.c_fc),
                c_proj: LinearOrQuantized::Standard(mlp.c_proj),
                _phantom: std::marker::PhantomData,
            };

            h_conv.push(Block { attn: attn_conv, mlp: mlp_conv });
        }

        let lm_head_conv = LinearOrQuantized::Standard(self.lm_head);
        let smear_gate_conv = LinearOrQuantized::Standard(self.smear_gate);

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

    pub fn quantize(self, bits: usize, block_size: usize) -> Gpt<B, LinearOrQuantized<B>> {
        use crate::engine::quant::quantize_linear;

        let mut h_quant = Vec::with_capacity(self.h.len());
        for block in self.h {
            let attn = block.attn;
            let mlp = block.mlp;

            let attn_quant = CausalSelfAttention {
                c_q: LinearOrQuantized::Quantized(quantize_linear(attn.c_q, bits, block_size)),
                c_k: LinearOrQuantized::Quantized(quantize_linear(attn.c_k, bits, block_size)),
                c_v: LinearOrQuantized::Quantized(quantize_linear(attn.c_v, bits, block_size)),
                c_proj: LinearOrQuantized::Quantized(quantize_linear(attn.c_proj, bits, block_size)),
                ve_gate: attn.ve_gate.map(|ve| LinearOrQuantized::Quantized(quantize_linear(ve, bits, block_size))),
                layer_idx: attn.layer_idx,
                n_head: attn.n_head,
                n_kv_head: attn.n_kv_head,
                head_dim: attn.head_dim,
                mask: attn.mask,
            };

            let mlp_quant = MLP {
                c_fc: LinearOrQuantized::Quantized(quantize_linear(mlp.c_fc, bits, block_size)),
                c_proj: LinearOrQuantized::Quantized(quantize_linear(mlp.c_proj, bits, block_size)),
                _phantom: std::marker::PhantomData,
            };

            h_quant.push(Block { attn: attn_quant, mlp: mlp_quant });
        }

        let lm_head_quant = LinearOrQuantized::Quantized(quantize_linear(self.lm_head, bits, block_size));
        let smear_gate_quant = LinearOrQuantized::Quantized(quantize_linear(self.smear_gate, bits, block_size));

        Gpt {
            wte: self.wte,
            h: h_quant,
            lm_head: lm_head_quant,
            resid_lambdas: self.resid_lambdas,
            x0_lambdas: self.x0_lambdas,
            smear_gate: smear_gate_quant,
            smear_lambda: self.smear_lambda,
            backout_lambda: self.backout_lambda,
            value_embeds: self.value_embeds,
            config: self.config,
        }
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
//}
