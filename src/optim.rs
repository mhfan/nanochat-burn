
use burn::{module::Param, optim::GradientsParams,
    tensor::{Tensor, backend::{AutodiffBackend, Backend}},
};

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

impl<B: Backend> Default for BlockMuonState<B> {
    fn default() -> Self {
        Self { c_q: None, c_k: None, c_v: None, c_proj: None,
            ve_gate: None, c_fc: None, c_proj_mlp: None,
        }
    }
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
        let value_embeds = (0..n_layer).filter(|&i| has_ve(i, n_layer)).map(|_| None).collect();
        let h = (0..n_layer).map(|_| BlockMuonState::default()).collect();
        Self {
            wte: None,
            lm_head: None,
            value_embeds,
            resid_lambdas: None,
            x0_lambdas: None,
            smear_gate: None,
            smear_lambda: None,
            backout_lambda: None,
            h,
        }
    }

    pub fn step(&mut self, gpt: &mut Gpt<B>, grads: &GradientsParams, lr: f32,
        step: usize, weight_decay: f32) {
        assert!(step > 0, "optimizer step must be one-based");
        assert!(lr.is_finite() && lr >= 0.0, "learning rate must be finite and non-negative");
        assert!(weight_decay.is_finite() && weight_decay >= 0.0,
            "weight decay must be finite and non-negative");
        assert_eq!(self.h.len(), gpt.h.len(), "optimizer/model layer count mismatch");
        assert_eq!(self.value_embeds.len(), gpt.value_embeds.len(),
            "optimizer/model value embedding count mismatch");
        let model_dim = gpt.config.n_embd as f32;
        let dmodel_lr_scale = (model_dim / 768.0).powf(-0.5);

        let adamw_lr = lr * dmodel_lr_scale * ADAMW_LR_SCALE;
        let embedding_lr = lr * dmodel_lr_scale * EMBEDDING_LR_SCALE;
        let value_embed_lr = embedding_lr * VALUE_EMBED_LR_SCALE;
        let scalar_lr = lr * SCALAR_LR_SCALE;

        // 1. Embeddings, lm_head, and scalars go into AdamW
        update_adamw_param(&mut gpt.wte.weight, grads, &mut self.wte,
            AdamWHyper::new(embedding_lr, 0.001, 0.8, 0.995, step));
        update_adamw_param(&mut gpt.lm_head.weight, grads, &mut self.lm_head,
            AdamWHyper::new(adamw_lr, 0.01, 0.8, 0.96, step));

        for (ve_cnt, _) in
            (0..gpt.config.n_layer).filter(|&i| has_ve(i, gpt.config.n_layer)).enumerate() {
            update_adamw_param(&mut gpt.value_embeds[ve_cnt].weight, grads,
                &mut self.value_embeds[ve_cnt],
                AdamWHyper::new(value_embed_lr, 0.01, 0.8, 0.995, step));
        }

        update_adamw_param(&mut gpt.resid_lambdas, grads, &mut self.resid_lambdas,
            AdamWHyper::new(scalar_lr * 0.01, 0.05, 0.8, 0.95, step));
        update_adamw_param(&mut gpt.x0_lambdas, grads, &mut self.x0_lambdas,
            AdamWHyper::new(scalar_lr, 0.0, 0.96, 0.95, step));

        let smear_lr = lr * SMEAR_LR_SCALE;
        let smear_hyper = AdamWHyper::new(smear_lr, 0.0, 0.8, 0.95, step);
        update_adamw_param(&mut gpt.smear_gate.weight, grads, &mut self.smear_gate,
            smear_hyper);
        update_adamw_param(&mut gpt.smear_lambda, grads, &mut self.smear_lambda, smear_hyper);
        update_adamw_param(&mut gpt.backout_lambda, grads, &mut self.backout_lambda,
            smear_hyper);

        // 2. Transformer Block matrices go into Muon
        let muon_hyper =
            MuonHyper { lr, weight_decay, momentum: 0.95, beta2: 0.9, ns_steps: 5 };

        for i in 0..gpt.config.n_layer {
            let (block, state) = (&mut gpt.h[i], &mut self.h[i]);

            update_muon_param(&mut block.attn.c_q.weight, grads, &mut state.c_q, muon_hyper);
            update_muon_param(&mut block.attn.c_k.weight, grads, &mut state.c_k, muon_hyper);
            update_muon_param(&mut block.attn.c_v.weight, grads, &mut state.c_v, muon_hyper);
            update_muon_param(&mut block.attn.c_proj.weight, grads, &mut state.c_proj,
                muon_hyper);

            if has_ve(i, gpt.config.n_layer) &&
                let Some(ref mut gate_linear) = block.attn.ve_gate {
                update_muon_param(&mut gate_linear.weight, grads, &mut state.ve_gate,
                    muon_hyper);
            }

            update_muon_param(&mut block.mlp.c_fc.weight, grads, &mut state.c_fc, muon_hyper);
            update_muon_param(&mut block.mlp.c_proj.weight, grads, &mut state.c_proj_mlp,
                muon_hyper);
        }
    }
}

#[derive(Clone, Copy)]
struct AdamWHyper {
    lr: f32,
    wd: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
}

impl AdamWHyper {
    fn new(lr: f32, wd: f32, beta1: f32, beta2: f32, step: usize) -> Self {
        Self { lr, wd, beta1, beta2, eps: 1e-4, step }
    }
}

#[derive(Clone, Copy)]
struct MuonHyper {
    lr: f32,
    weight_decay: f32,
    momentum: f32,
    beta2: f32,
    ns_steps: usize,
}

fn update_adamw_param<B: AutodiffBackend, const D: usize>(
    param: &mut Param<Tensor<B, D>>, grads: &GradientsParams,
    state: &mut Option<AdamWState<B::InnerBackend, D>>, hyper: AdamWHyper,
) {
    if let Some(grad) = grads.get::<B::InnerBackend, D>(param.id) {
        *param = Param::from_tensor(Tensor::from_inner(adamw_step(
            param.val().inner(), grad, state, hyper,
        )));
    }
}

fn update_muon_param<B: AutodiffBackend>(param: &mut Param<Tensor<B, 2>>,
    grads: &GradientsParams, state: &mut Option<MuonState<B::InnerBackend, 2>>,
    hyper: MuonHyper) {
    if let Some(grad) = grads.get::<B::InnerBackend, 2>(param.id) {
        *param = Param::from_tensor(Tensor::from_inner(muon_step(
            param.val().inner(), grad, state, hyper,
        )));
    }
}

fn adamw_step<B: Backend, const D: usize>(p: Tensor<B, D>, grad: Tensor<B, D>,
    state: &mut Option<AdamWState<B, D>>, hyper: AdamWHyper) -> Tensor<B, D> {
    let s = state.get_or_insert_with(|| AdamWState {
        exp_avg: Tensor::zeros(p.shape(), &p.device()),
        exp_avg_sq: Tensor::zeros(p.shape(), &p.device()),
    });

    s.exp_avg =
        s.exp_avg.clone().mul_scalar(hyper.beta1) + grad.clone().mul_scalar(1.0 - hyper.beta1);
    s.exp_avg_sq = s.exp_avg_sq.clone().mul_scalar(hyper.beta2) +
        grad.powf_scalar(2.0).mul_scalar(1.0 - hyper.beta2);

    let bias1 = 1.0 - hyper.beta1.powi(hyper.step as i32);
    let bias2 = 1.0 - hyper.beta2.powi(hyper.step as i32);
    let denom = (s.exp_avg_sq.clone() / bias2).clamp(0.0, 1e10).sqrt().add_scalar(hyper.eps);

    p.mul_scalar(1.0 - hyper.lr * hyper.wd) -
        (s.exp_avg.clone() / denom).mul_scalar(hyper.lr / bias1)
}

fn muon_step<B: Backend>(p: Tensor<B, 2>, grad: Tensor<B, 2>,
    state: &mut Option<MuonState<B, 2>>, hyper: MuonHyper) -> Tensor<B, 2> {
    let [rows, cols] = p.shape().dims();
    let shape = [rows, cols];
    let red_dim = if rows >= cols { 1 } else { 0 };

    let s = state.get_or_insert_with(|| {
        let state_shape = if rows >= cols { [rows, 1] } else { [1, cols] };
        MuonState {
            momentum_buffer: Tensor::zeros(p.shape(), &p.device()),
            second_momentum_buffer: Tensor::zeros(state_shape, &p.device()),
        }
    });

    s.momentum_buffer = s.momentum_buffer.clone().mul_scalar(hyper.momentum) +
        grad.mul_scalar(1.0 - hyper.momentum);

    let g = s.momentum_buffer.clone();
    let g_scaled = g.clone().mul_scalar(10000.0);
    let norm = g_scaled.powf_scalar(2.0).sum().clamp(0.0, 1e10).sqrt().mul_scalar(0.0001);
    let mut x = g / norm.mul_scalar(1.01).add_scalar(1e-6).reshape([1, 1]);

    let polar_express_coeffs = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ];

    let (steps, is_transposed) = (hyper.ns_steps.min(5), rows > cols);
    for i in 0..steps {
        let (a, b, c) = polar_express_coeffs[i];
        let a_mat = if is_transposed {
            x.clone().transpose().matmul(x.clone())
        } else {
            x.clone().matmul(x.clone().transpose())
        };
        let a_sq = a_mat.clone().matmul(a_mat.clone());
        let poly = a_mat.mul_scalar(b) + a_sq.mul_scalar(c);
        x = x.clone().mul_scalar(a) +
            if is_transposed { x.matmul(poly) } else { poly.matmul(x) };
    }
    let mut g_ortho = x;

    let v_mean = g_ortho.clone().powf_scalar(2.0).mean_dim(red_dim).clamp(0.0, 1e10);
    let red_dim_size = shape[red_dim] as f32;
    let v_norm = (v_mean.clone().sum() * red_dim_size).clamp(0.0, 1e10).sqrt();

    s.second_momentum_buffer = s.second_momentum_buffer.clone().mul_scalar(hyper.beta2) +
        v_mean.clone().mul_scalar(1.0 - hyper.beta2);
    let step_size = (s.second_momentum_buffer.clone().clamp(1e-4, 1e4)).recip().sqrt();

    let scaled_sq_sum = (v_mean * red_dim_size) * step_size.clone().powf_scalar(2.0);
    let v_norm_new = scaled_sq_sum.sum().clamp(0.0, 1e10).sqrt();

    let ratio = (v_norm / v_norm_new.clamp(1e-4, 1e4)).reshape([1, 1]);
    g_ortho = g_ortho * step_size * ratio;

    let lr_scaled = hyper.lr * ((rows as f32 / cols as f32).max(1.0)).sqrt();

    let prod = g_ortho.clone() * p.clone();
    let mask = prod.greater_equal_elem(0.0).float();
    let update = g_ortho.mul_scalar(lr_scaled) +
        p.clone().mul_scalar(lr_scaled * hyper.weight_decay) * mask;
    p - update
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_muon_orthogonalization() {
        use crate::common::ModelBackend;
        let device = crate::common::init_device();
        let p = Tensor::<ModelBackend, 2>::from_data([[2.0, 0.0], [0.0, 3.0]], &device);
        let grad = Tensor::<ModelBackend, 2>::from_data([[0.1, 0.2], [0.3, 0.4]], &device);
        let mut state = None;

        let hyper =
            MuonHyper { lr: 0.02, weight_decay: 0.0, momentum: 0.95, beta2: 0.9, ns_steps: 5 };
        let new_p = muon_step(p, grad, &mut state, hyper);
        let shape: [usize; 2] = new_p.shape().dims();
        assert_eq!(shape, [2, 2]);
    }
}
