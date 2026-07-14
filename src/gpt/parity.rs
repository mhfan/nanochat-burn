use super::*;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use crate::common::{ModelAutodiffBackend, ModelBackend, ModelDevice};

type ParityGradients = <ModelAutodiffBackend as AutodiffBackend>::Gradients;

#[derive(Deserialize)]
struct ModuleParityFixture {
    schema_version: u32,
    source: ModuleFixtureSource,
    config: ModuleFixtureConfig,
    rms_norm: UnaryFixture,
    rope: RopeFixture,
    mlp: MlpFixture,
    attention: AttentionFixture,
}

#[derive(Deserialize)]
struct ModuleFixtureSource {
    implementation: String,
    torch: String,
    dtype: String,
    linear_weight_layout: String,
}

#[derive(Deserialize)]
struct ModuleFixtureConfig {
    sequence_len: usize,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
}

#[derive(Deserialize)]
struct TensorFixture { shape: Vec<usize>, values: Vec<f32> }

#[derive(Deserialize)]
struct UnaryFixture { input: TensorFixture, output: TensorFixture }

#[derive(Deserialize)]
struct RopeFixture {
    input: TensorFixture,
    cos: TensorFixture,
    sin: TensorFixture,
    output: TensorFixture,
}

#[derive(Deserialize)]
struct MlpFixture {
    input: TensorFixture,
    c_fc_weight: TensorFixture,
    c_proj_weight: TensorFixture,
    output: TensorFixture,
}

#[derive(Deserialize)]
struct AttentionFixture {
    input: TensorFixture,
    value_embedding: TensorFixture,
    cos: TensorFixture,
    sin: TensorFixture,
    c_q_weight: TensorFixture,
    c_k_weight: TensorFixture,
    c_v_weight: TensorFixture,
    c_proj_weight: TensorFixture,
    ve_gate_weight: TensorFixture,
    output: TensorFixture,
}

fn python_module_fixture() -> ModuleParityFixture {
    let fixture: ModuleParityFixture = serde_json::from_str(
        include_str!("../../data/fixtures/parity/modules.json")).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.implementation, "nanochat.gpt");
    assert_eq!(fixture.source.torch, "2.9.1");
    assert_eq!(fixture.source.dtype, "float32");
    assert_eq!(fixture.source.linear_weight_layout, "out_in");
    fixture
}

fn fixture_tensor<B: Backend, const D: usize>(fixture: &TensorFixture,
    device: &B::Device) -> Tensor<B, D> {
    let shape: [usize; D] = fixture.shape.as_slice().try_into().unwrap();
    Tensor::from_data(TensorData::new(fixture.values.clone(), Shape::new(shape)), device)
}

fn fixture_linear<B: Backend>(fixture: &TensorFixture, device: &B::Device) -> Linear<B> {
    assert_eq!(fixture.shape.len(), 2);
    linear(fixture_tensor::<B, 2>(fixture, device).swap_dims(0, 1))
}

fn assert_fixture_close<B: Backend, const D: usize>(actual: Tensor<B, D>,
    expected: &TensorFixture, tolerance: f32, label: &str) {
    assert_eq!(actual.shape().dims::<D>().as_slice(), expected.shape, "{label} shape mismatch");
    let actual = crate::common::tensor_data_to_f32_vec(actual.into_data());
    assert_eq!(actual.len(), expected.values.len(), "{label} length mismatch");
    let (index, max_error) = actual.iter().zip(&expected.values).enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    assert!(max_error <= tolerance,
        "{label} max error {max_error} at index {index} exceeds {tolerance}: {} != {}",
        actual[index], expected.values[index]);
}

#[test] fn test_python_rms_norm_and_rope_parity() {
    let (fixture, device) = (python_module_fixture(), crate::common::init_device());
    let actual = norm(fixture_tensor::<ModelBackend, 3>(&fixture.rms_norm.input, &device));
    assert_fixture_close(actual, &fixture.rms_norm.output, 2e-6, "RMSNorm");

    let rope = &fixture.rope;
    let actual = apply_rotary_emb(fixture_tensor::<ModelBackend, 4>(&rope.input, &device),
        fixture_tensor::<ModelBackend, 4>(&rope.cos, &device),
        fixture_tensor::<ModelBackend, 4>(&rope.sin, &device));
    assert_fixture_close(actual, &rope.output, 2e-6, "RoPE");
}

#[test] fn test_python_mlp_parity() {
    let (fixture, device) = (python_module_fixture(), crate::common::init_device());
    let mlp = &fixture.mlp;
    let module = MLP {
        c_fc: fixture_linear(&mlp.c_fc_weight, &device),
        c_proj: fixture_linear(&mlp.c_proj_weight, &device),
        _phantom: PhantomData,
    };
    assert_fixture_close(module.forward(fixture_tensor::<ModelBackend, 3>(&mlp.input, &device)),
        &mlp.output, 2e-5, "MLP");
}

#[test] fn test_python_attention_parity() {
    let (fixture, device) = (python_module_fixture(), crate::common::init_device());
    let (config, attention) = (&fixture.config, &fixture.attention);
    let module = CausalSelfAttention {
        c_q: fixture_linear(&attention.c_q_weight, &device),
        c_k: fixture_linear(&attention.c_k_weight, &device),
        c_v: fixture_linear(&attention.c_v_weight, &device),
        c_proj: fixture_linear(&attention.c_proj_weight, &device),
        ve_gate: Some(fixture_linear(&attention.ve_gate_weight, &device)),
        layer_idx: 0,
        n_head: config.n_head,
        n_kv_head: config.n_kv_head,
        head_dim: config.n_embd / config.n_head,
        mask: precompute_window_mask(-1, config.sequence_len, &device),
    };
    let actual = module.forward(fixture_tensor::<ModelBackend, 3>(&attention.input, &device),
        Some(fixture_tensor::<ModelBackend, 3>(&attention.value_embedding, &device)),
        fixture_tensor::<ModelBackend, 4>(&attention.cos, &device),
        fixture_tensor::<ModelBackend, 4>(&attention.sin, &device));
    assert_fixture_close(actual, &attention.output, 2e-5, "attention");
}

#[derive(Deserialize)]
struct ModelParityFixture {
    schema_version: u32,
    source: ModelFixtureSource,
    config: GptConfig,
    input_ids: Vec<Vec<i32>>,
    targets: Vec<Vec<i32>>,
    parameters: HashMap<String, TensorFixture>,
    logits: TensorFixture,
    loss: f32,
    gradients: HashMap<String, TensorFixture>,
}

#[derive(Deserialize)]
struct ModelFixtureSource {
    implementation: String,
    torch: String,
    dtype: String,
    linear_weight_layout: String,
}

fn model_fixture() -> ModelParityFixture {
    let fixture: ModelParityFixture = serde_json::from_str(
        include_str!("../../data/fixtures/parity/model.json")).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.implementation, "nanochat.gpt.GPT");
    assert_eq!(fixture.source.torch, "2.9.1");
    assert_eq!(fixture.source.dtype, "float32");
    assert_eq!(fixture.source.linear_weight_layout, "out_in");
    fixture.config.validate().unwrap();
    fixture
}

fn named<'a>(tensors: &'a HashMap<String, TensorFixture>, name: &str) -> &'a TensorFixture {
    tensors.get(name).unwrap_or_else(|| panic!("missing fixture tensor {name}"))
}

fn fixture_ids(values: &[Vec<i32>], device: &ModelDevice)
    -> Tensor<ModelAutodiffBackend, 2, Int> {
    let rows = values.len();
    let columns = values.first().map_or(0, Vec::len);
    assert!(rows > 0 && columns > 0 && values.iter().all(|row| row.len() == columns));
    let values = values.iter().flatten().copied().collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, Shape::new([rows, columns])), device)
}

fn model_from_fixture(fixture: &ModelParityFixture, device: &ModelDevice)
    -> Gpt<ModelAutodiffBackend> {
    let parameters = &fixture.parameters;
    let mut model = Gpt::new(fixture.config.clone(), device);
    model.wte = embedding(fixture_tensor::<ModelAutodiffBackend, 2>(
        named(parameters, "transformer.wte.weight"), device));
    model.lm_head = fixture_linear(named(parameters, "lm_head.weight"), device);
    model.resid_lambdas = param(fixture_tensor::<ModelAutodiffBackend, 1>(
        named(parameters, "resid_lambdas"), device));
    model.x0_lambdas = param(fixture_tensor::<ModelAutodiffBackend, 1>(
        named(parameters, "x0_lambdas"), device));
    model.smear_gate = fixture_linear(named(parameters, "smear_gate.weight"), device);
    model.smear_lambda = param(fixture_tensor::<ModelAutodiffBackend, 1>(
        named(parameters, "smear_lambda"), device));
    model.backout_lambda = param(fixture_tensor::<ModelAutodiffBackend, 1>(
        named(parameters, "backout_lambda"), device));

    for (layer, block) in model.h.iter_mut().enumerate() {
        let prefix = format!("transformer.h.{layer}");
        block.attn.c_q = fixture_linear(named(parameters, &format!("{prefix}.attn.c_q.weight")),
            device);
        block.attn.c_k = fixture_linear(named(parameters, &format!("{prefix}.attn.c_k.weight")),
            device);
        block.attn.c_v = fixture_linear(named(parameters, &format!("{prefix}.attn.c_v.weight")),
            device);
        block.attn.c_proj = fixture_linear(
            named(parameters, &format!("{prefix}.attn.c_proj.weight")), device);
        if block.attn.ve_gate.is_some() {
            block.attn.ve_gate = Some(fixture_linear(
                named(parameters, &format!("{prefix}.attn.ve_gate.weight")), device));
        }
        block.mlp.c_fc = fixture_linear(named(parameters, &format!("{prefix}.mlp.c_fc.weight")),
            device);
        block.mlp.c_proj = fixture_linear(
            named(parameters, &format!("{prefix}.mlp.c_proj.weight")), device);
    }

    let mut value_index = 0;
    for layer in 0..fixture.config.n_layer {
        if has_ve(layer, fixture.config.n_layer) {
            model.value_embeds[value_index] = embedding(fixture_tensor::<ModelAutodiffBackend, 2>(
                named(parameters, &format!("value_embeds.{layer}.weight")), device));
            value_index += 1;
        }
    }
    assert_eq!(value_index, model.value_embeds.len());
    model
}

fn assert_gradient<const D: usize>(tensor: Tensor<ModelAutodiffBackend, D>,
    gradients: &ParityGradients, expected: &TensorFixture, label: &str) {
    let actual = tensor.grad(gradients).unwrap_or_else(|| panic!("missing gradient for {label}"));
    assert_fixture_close(actual, expected, 5e-5, label);
}

fn assert_linear_gradient(linear: &Linear<ModelAutodiffBackend>, gradients: &ParityGradients,
    expected: &TensorFixture, label: &str) {
    let actual = linear.weight.val().grad(gradients)
        .unwrap_or_else(|| panic!("missing gradient for {label}")).swap_dims(0, 1);
    assert_fixture_close(actual, expected, 5e-5, label);
}

#[test] fn test_python_full_model_logits_loss_and_gradients_parity() {
    let (fixture, device) = (model_fixture(), crate::common::init_device());
    let model = model_from_fixture(&fixture, &device);
    let input = fixture_ids(&fixture.input_ids, &device);
    let targets = fixture_ids(&fixture.targets, &device);
    let logits = model.forward(input, None);
    assert_fixture_close(logits.clone(), &fixture.logits, 5e-5, "full-model logits");

    let loss = model.compute_loss(logits, targets);
    let actual_loss = crate::common::scalar_to_f32(loss.clone().into_scalar());
    assert!((actual_loss - fixture.loss).abs() <= 3e-6,
        "full-model loss mismatch: {actual_loss} != {}", fixture.loss);
    let gradients = loss.backward();
    let expected = &fixture.gradients;

    assert_gradient(model.wte.weight.val(), &gradients,
        named(expected, "transformer.wte.weight"), "wte gradient");
    assert_linear_gradient(&model.h[0].attn.c_q, &gradients,
        named(expected, "transformer.h.0.attn.c_q.weight"), "layer 0 Q gradient");
    assert_linear_gradient(&model.h[0].attn.c_proj, &gradients,
        named(expected, "transformer.h.0.attn.c_proj.weight"), "layer 0 attention projection");
    assert_linear_gradient(model.h[1].attn.ve_gate.as_ref().unwrap(), &gradients,
        named(expected, "transformer.h.1.attn.ve_gate.weight"), "layer 1 VE gate gradient");
    assert_linear_gradient(&model.h[1].mlp.c_fc, &gradients,
        named(expected, "transformer.h.1.mlp.c_fc.weight"), "layer 1 MLP input gradient");
    assert_linear_gradient(&model.h[1].mlp.c_proj, &gradients,
        named(expected, "transformer.h.1.mlp.c_proj.weight"), "layer 1 MLP projection gradient");
    assert_linear_gradient(&model.lm_head, &gradients,
        named(expected, "lm_head.weight"), "lm head gradient");
    assert_gradient(model.resid_lambdas.val(), &gradients,
        named(expected, "resid_lambdas"), "residual lambda gradient");
    assert_gradient(model.x0_lambdas.val(), &gradients,
        named(expected, "x0_lambdas"), "x0 lambda gradient");
    assert_linear_gradient(&model.smear_gate, &gradients,
        named(expected, "smear_gate.weight"), "smear gate gradient");
    assert_gradient(model.smear_lambda.val(), &gradients,
        named(expected, "smear_lambda"), "smear lambda gradient");
    assert_gradient(model.value_embeds[0].weight.val(), &gradients,
        named(expected, "value_embeds.1.weight"), "value embedding gradient");
}

fn cached_forward(model: &Gpt<ModelAutodiffBackend>, input: Tensor<ModelAutodiffBackend, 2, Int>,
    chunks: &[usize], page_size: usize, device: &ModelDevice) -> Tensor<ModelAutodiffBackend, 3> {
    let [batch_size, sequence_len] = input.shape().dims();
    assert!(chunks.iter().all(|&len| len > 0));
    assert_eq!(chunks.iter().sum::<usize>(), sequence_len);
    let head_dim = model.config.n_embd / model.config.n_head;
    let mut cache = KVCache::new_paged(model.config.n_layer, batch_size,
        model.config.sequence_len, model.config.n_kv_head, head_dim, page_size, device);
    let (mut step, mut outputs) = (0, Vec::with_capacity(chunks.len()));
    for &len in chunks {
        let end = step + len;
        outputs.push(model.forward_with_cache(
            input.clone().slice([0..batch_size, step..end]), &mut cache, step));
        step = end;
    }
    Tensor::cat(outputs, 1)
}

fn assert_logits_close(actual: Tensor<ModelAutodiffBackend, 3>,
    expected: Tensor<ModelAutodiffBackend, 3>, label: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{label} shape mismatch");
    let max_error = crate::common::scalar_to_f32((actual - expected).abs().max().into_scalar());
    assert!(max_error <= 5e-5, "{label} max error {max_error} exceeds 0.00005");
}

#[test] fn test_full_chunked_and_token_cache_parity() {
    let (fixture, device) = (model_fixture(), crate::common::init_device());
    let model = model_from_fixture(&fixture, &device);
    let input = fixture_ids(&fixture.input_ids, &device);
    let full = model.forward(input.clone(), None);
    let chunked = cached_forward(&model, input.clone(), &[2, 2], 2, &device);
    let uneven = cached_forward(&model, input.clone(), &[3, 1], 3, &device);
    let tokenwise = cached_forward(&model, input, &[1, 1, 1, 1], 1, &device);

    assert_logits_close(chunked.clone(), full.clone(), "chunked/full logits");
    assert_logits_close(uneven, full.clone(), "uneven/full logits");
    assert_logits_close(tokenwise.clone(), full, "token/full logits");
    assert_logits_close(chunked, tokenwise, "chunked/token logits");
}
