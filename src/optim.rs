
use burn::tensor::{Tensor, backend::{Backend, AutodiffBackend}};
use burn::optim::GradientsParams;
use crate::gpt::{Gpt, has_ve};

const ADAMW_LR_SCALE: f32 = 0.2;
const EMBEDDING_LR_SCALE: f32 = 10.0;
const VALUE_EMBED_LR_SCALE: f32 = 0.5;
const SCALAR_LR_SCALE: f32 = 25.0;
const SMEAR_LR_SCALE: f32 = 10.0;

pub struct AdamWState<B: Backend, const D: usize> {
    pub exp_avg: Tensor<B, D>,
    pub exp_avg_sq: Tensor<B, D>,
}

pub struct MuonState<B: Backend, const D: usize> {
    pub momentum_buffer: Tensor<B, D>,
    pub second_momentum_buffer: Tensor<B, D>,
}

pub struct BlockMuonState<B: Backend> {
    pub c_q: Option<MuonState<B, 2>>,
    pub c_k: Option<MuonState<B, 2>>,
    pub c_v: Option<MuonState<B, 2>>,
    pub c_proj: Option<MuonState<B, 2>>,
    pub ve_gate: Option<MuonState<B, 2>>,
    pub c_fc: Option<MuonState<B, 2>>,
    pub c_proj_mlp: Option<MuonState<B, 2>>,
}

pub struct MuonAdamW<B: AutodiffBackend> {
    pub wte: Option<AdamWState<B::InnerBackend, 2>>,
    pub lm_head: Option<AdamWState<B::InnerBackend, 2>>,
    pub value_embeds: Vec<Option<AdamWState<B::InnerBackend, 2>>>,
    pub resid_lambdas: Option<AdamWState<B::InnerBackend, 1>>,
    pub x0_lambdas: Option<AdamWState<B::InnerBackend, 1>>,
    pub smear_gate: Option<AdamWState<B::InnerBackend, 2>>,
    pub smear_lambda: Option<AdamWState<B::InnerBackend, 1>>,
    pub backout_lambda: Option<AdamWState<B::InnerBackend, 1>>,
    pub h: Vec<BlockMuonState<B::InnerBackend>>,
}

impl<B: AutodiffBackend> MuonAdamW<B> {
    pub fn new(n_layer: usize) -> Self {
        let mut value_embeds = Vec::with_capacity(n_layer);
        let mut h = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            value_embeds.push(None);
            h.push(BlockMuonState { c_q: None, c_k: None, c_v: None,
                c_proj: None, ve_gate: None, c_fc: None, c_proj_mlp: None,
            });
        }
        Self { wte: None, lm_head: None, value_embeds, resid_lambdas: None,
            x0_lambdas: None, smear_gate: None, smear_lambda: None, backout_lambda: None, h,
        }
    }

    pub fn step(&mut self, gpt: &mut Gpt<B>, grads: &GradientsParams,
        lr: f32, step: usize, weight_decay: f32,) {
        let model_dim = gpt.config.n_embd as f32;
        let dmodel_lr_scale = (model_dim / 768.0).powf(-0.5);

        let adamw_lr = lr * dmodel_lr_scale * ADAMW_LR_SCALE;
        let embedding_lr = lr * dmodel_lr_scale * EMBEDDING_LR_SCALE;
        let value_embed_lr = embedding_lr * VALUE_EMBED_LR_SCALE;
        let scalar_lr = lr * SCALAR_LR_SCALE;

        use burn::module::Param;
        // 1. Embeddings, lm_head, and scalars go into AdamW
        if let Some(grad) = grads.get::<B::InnerBackend, 2>(gpt.wte.weight.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.wte.weight.val().inner(), grad, &mut self.wte, embedding_lr, 0.001, 0.8, 0.995, 1e-4, step));
            gpt.wte.weight = Param::from_tensor(new_w);
        }

        if let Some(grad) = grads.get::<B::InnerBackend, 2>(gpt.lm_head.weight.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.lm_head.weight.val().inner(), grad, &mut self.lm_head, adamw_lr, 0.01, 0.8, 0.96, 1e-4, step));
            gpt.lm_head.weight = Param::from_tensor(new_w);
        }

        let mut ve_cnt = 0;
        for i in 0..gpt.config.n_layer {
            if has_ve(i, gpt.config.n_layer) {
                if let Some(grad) = grads.get::<B::InnerBackend, 2>(gpt.value_embeds[ve_cnt].weight.id) {
                    let new_w = Tensor::from_inner(adamw_step(gpt.value_embeds[ve_cnt].weight.val().inner(), grad, &mut self.value_embeds[ve_cnt], value_embed_lr, 0.01, 0.8, 0.995, 1e-4, step));
                    gpt.value_embeds[ve_cnt].weight = Param::from_tensor(new_w);
                }
                ve_cnt += 1;
            }
        }

        if let Some(grad) = grads.get::<B::InnerBackend, 1>(gpt.resid_lambdas.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.resid_lambdas.val().inner(), grad, &mut self.resid_lambdas, scalar_lr * 0.01, 0.05, 0.8, 0.95, 1e-4, step));
            gpt.resid_lambdas = Param::from_tensor(new_w);
        }

        if let Some(grad) = grads.get::<B::InnerBackend, 1>(gpt.x0_lambdas.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.x0_lambdas.val().inner(), grad, &mut self.x0_lambdas, scalar_lr, 0.0, 0.96, 0.95, 1e-4, step));
            gpt.x0_lambdas = Param::from_tensor(new_w);
        }

        let smear_lr = lr * SMEAR_LR_SCALE;

        if let Some(grad) = grads.get::<B::InnerBackend, 2>(gpt.smear_gate.weight.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.smear_gate.weight.val().inner(), grad, &mut self.smear_gate, smear_lr, 0.0, 0.8, 0.95, 1e-4, step));
            gpt.smear_gate.weight = Param::from_tensor(new_w);
        }

        if let Some(grad) = grads.get::<B::InnerBackend, 1>(gpt.smear_lambda.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.smear_lambda.val().inner(), grad, &mut self.smear_lambda, smear_lr, 0.0, 0.8, 0.95, 1e-4, step));
            gpt.smear_lambda = Param::from_tensor(new_w);
        }

        if let Some(grad) = grads.get::<B::InnerBackend, 1>(gpt.backout_lambda.id) {
            let new_w = Tensor::from_inner(adamw_step(gpt.backout_lambda.val().inner(), grad, &mut self.backout_lambda, smear_lr, 0.0, 0.8, 0.95, 1e-4, step));
            gpt.backout_lambda = Param::from_tensor(new_w);
        }

        // 2. Transformer Block matrices go into Muon
        let update_muon = |param: &mut Param<Tensor<B, 2>>, state_opt: &mut Option<MuonState<B::InnerBackend, 2>>| {
            if let Some(grad) = grads.get::<B::InnerBackend, 2>(param.id) {
                let new_w = Tensor::from_inner(muon_step(
                    param.val().inner(), grad, state_opt, lr, weight_decay, 0.95, 0.9, 5,
                ));
                *param = Param::from_tensor(new_w);
            }
        };

        for i in 0..gpt.config.n_layer {
            let block = &mut gpt.h[i];
            let state = &mut self.h[i];

            update_muon(&mut block.attn.c_q.weight, &mut state.c_q);
            update_muon(&mut block.attn.c_k.weight, &mut state.c_k);
            update_muon(&mut block.attn.c_v.weight, &mut state.c_v);
            update_muon(&mut block.attn.c_proj.weight, &mut state.c_proj);

            if has_ve(i, gpt.config.n_layer) {
                if let Some(ref mut gate_linear) = block.attn.ve_gate {
                    update_muon(&mut gate_linear.weight, &mut state.ve_gate);
                }
            }

            update_muon(&mut block.mlp.c_fc.weight, &mut state.c_fc);
            update_muon(&mut block.mlp.c_proj.weight, &mut state.c_proj_mlp);
        }
    }
}

fn adamw_step<B: Backend, const D: usize>(p: Tensor<B, D>, grad: Tensor<B, D>,
    state: &mut Option<AdamWState<B, D>>, lr: f32, wd: f32, beta1: f32, beta2: f32,
    eps: f32, step: usize,) -> Tensor<B, D> {
    let s = state.get_or_insert_with(|| {
        AdamWState {
            exp_avg: Tensor::zeros(p.shape(), &p.device()),
            exp_avg_sq: Tensor::zeros(p.shape(), &p.device()),
        }
    });

    s.exp_avg = s.exp_avg.clone().mul_scalar(beta1) + grad.clone().mul_scalar(1.0 - beta1);
    s.exp_avg_sq = s.exp_avg_sq.clone().mul_scalar(beta2) + grad.powf_scalar(2.0).mul_scalar(1.0 - beta2);

    let bias1 = 1.0 - beta1.powi(step as i32);
    let bias2 = 1.0 - beta2.powi(step as i32);

    let denom = (s.exp_avg_sq.clone() / bias2).clamp(0.0, 1e10).sqrt().add_scalar(eps);
    let step_size = lr / bias1;

    let p_decayed = p.mul_scalar(1.0 - lr * wd);
    p_decayed - (s.exp_avg.clone() / denom).mul_scalar(step_size)
}

#[allow(non_snake_case)] fn muon_step<B: Backend>(p: Tensor<B, 2>, grad: Tensor<B, 2>,
    state: &mut Option<MuonState<B, 2>>, lr: f32, wd: f32, momentum: f32,
    beta2: f32, ns_steps: usize,) -> Tensor<B, 2> {
    let shape: [usize; 2] = p.shape().dims();
    let (rows, cols) = (shape[0], shape[1]);
    let red_dim = if rows >= cols { 1 } else { 0 };

    let s = state.get_or_insert_with(|| {
        let state_shape = if rows >= cols { [rows, 1] } else { [1, cols] };
        MuonState {
            momentum_buffer: Tensor::zeros(p.shape(), &p.device()),
            second_momentum_buffer: Tensor::zeros(state_shape, &p.device()),
        }
    });

    s.momentum_buffer = s.momentum_buffer.clone().mul_scalar(momentum) + grad.mul_scalar(1.0 - momentum);
    let g = s.momentum_buffer.clone();

    let g_scaled = g.clone().mul_scalar(10000.0);
    let norm = g_scaled.powf_scalar(2.0).sum().clamp(0.0, 1e10).sqrt().mul_scalar(0.0001);
    let norm_scaled = norm.mul_scalar(1.01).add_scalar(1e-6).reshape([1, 1]);
    let mut X = g / norm_scaled;

    let polar_express_coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ];

    let steps = ns_steps.min(5);
    let is_transposed = rows > cols;
    for i in 0..steps {
        let (a, b, c) = polar_express_coeffs[i];
        let A = if is_transposed {
            X.clone().transpose().matmul(X.clone())
        } else {
            X.clone().matmul(X.clone().transpose())
        };
        let A_sq = A.clone().matmul(A.clone());
        let B = A.mul_scalar(b) + A_sq.mul_scalar(c);
        X = if is_transposed {
            X.clone().mul_scalar(a) + X.matmul(B)
        } else {
            X.clone().mul_scalar(a) + B.matmul(X)
        };
    }
    let mut g_ortho = X;

    let v_mean = g_ortho.clone().powf_scalar(2.0).mean_dim(red_dim).clamp(0.0, 1e10);
    let red_dim_size = shape[red_dim] as f32;
    let v_norm = (v_mean.clone().sum() * red_dim_size).clamp(0.0, 1e10).sqrt();

    s.second_momentum_buffer = s.second_momentum_buffer.clone().mul_scalar(beta2) + v_mean.clone().mul_scalar(1.0 - beta2);
    let step_size = (s.second_momentum_buffer.clone().clamp(1e-4, 1e4)).recip().sqrt();

    let scaled_sq_sum = (v_mean * red_dim_size) * step_size.clone().powf_scalar(2.0);
    let v_norm_new = scaled_sq_sum.sum().clamp(0.0, 1e10).sqrt();

    let ratio = (v_norm / v_norm_new.clamp(1e-4, 1e4)).reshape([1, 1]);
    let scale = step_size * ratio;
    g_ortho = g_ortho * scale;

    let lr_scaled = lr * ((rows as f32 / cols as f32).max(1.0)).sqrt();

    let prod = g_ortho.clone() * p.clone();
    let mask = prod.greater_equal_elem(0.0).float();
    let update = g_ortho.mul_scalar(lr_scaled) + p.clone().mul_scalar(lr_scaled * wd) * mask;
    p - update
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_muon_orthogonalization() {
        use crate::common::ModelBackend;
        let device = crate::common::init_device();
        let p = Tensor::<ModelBackend, 2>::from_data([[2.0, 0.0], [0.0, 3.0]], &device);
        let grad = Tensor::<ModelBackend, 2>::from_data([[0.1, 0.2], [0.3, 0.4]], &device);
        let mut state = None;

        let new_p = muon_step(p, grad, &mut state, 0.02, 0.0, 0.95, 0.9, 5);
        let shape: [usize; 2] = new_p.shape().dims();
        assert_eq!(shape, [2, 2]);
    }
//}
