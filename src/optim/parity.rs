use super::*;
use crate::common::{InferBackend, ModelDevice};

#[derive(serde::Deserialize)]
struct OptimizerParityFixture {
    schema_version: u32,
    source: FixtureSource,
    adamw: AdamWFixture,
    muon_tall: MuonFixture,
    muon_wide: MuonFixture,
}

#[derive(serde::Deserialize)]
struct FixtureSource { implementation: String, torch: String, dtype: String }

#[derive(serde::Deserialize)]
struct TensorFixture { shape: Vec<usize>, values: Vec<f32> }

#[derive(serde::Deserialize)]
struct AdamWFixture {
    parameter: TensorFixture,
    gradient: TensorFixture,
    hyper: AdamWHyperFixture,
    output: TensorFixture,
    exp_avg: TensorFixture,
    exp_avg_sq: TensorFixture,
}

#[derive(serde::Deserialize)]
struct AdamWHyperFixture { lr: f32, betas: [f32; 2], eps: f32, weight_decay: f32 }

#[derive(serde::Deserialize)]
struct MuonFixture {
    parameter: TensorFixture,
    gradient: TensorFixture,
    hyper: MuonHyperFixture,
    output: TensorFixture,
    momentum_buffer: TensorFixture,
    second_momentum_buffer: TensorFixture,
}

#[derive(serde::Deserialize)]
struct MuonHyperFixture {
    lr: f32,
    momentum: f32,
    ns_steps: usize,
    beta2: f32,
    weight_decay: f32,
}

fn optimizer_fixture() -> OptimizerParityFixture {
    let fixture: OptimizerParityFixture = serde_json::from_str(
        include_str!("../../data/fixtures/parity/optimizer.json")).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.implementation, "nanochat.optim.MuonAdamW");
    assert_eq!(fixture.source.torch, "2.9.1");
    assert_eq!(fixture.source.dtype, "float32");
    fixture
}

fn fixture_tensor<const D: usize>(fixture: &TensorFixture,
    device: &ModelDevice) -> Tensor<InferBackend, D> {
    let shape: [usize; D] = fixture.shape.as_slice().try_into().unwrap();
    Tensor::from_data(TensorData::new(fixture.values.clone(), Shape::new(shape)), device)
}

fn assert_fixture_close<const D: usize>(actual: Tensor<InferBackend, D>,
    expected: &TensorFixture, tolerance: f32, label: &str) {
    assert_eq!(actual.shape().dims::<D>().as_slice(), expected.shape, "{label} shape mismatch");
    let actual = crate::common::tensor_data_to_f32_vec(actual.into_data());
    let max_error = actual.iter().zip(&expected.values)
        .map(|(actual, expected)| (actual - expected).abs()).fold(0.0, f32::max);
    assert!(max_error <= tolerance, "{label} max error {max_error} exceeds {tolerance}");
}

#[test] fn test_python_adamw_single_step_parity() {
    let (fixture, device) = (optimizer_fixture(), Default::default());
    let case = fixture.adamw;
    let hyper = AdamWHyper::new(case.hyper.lr, case.hyper.weight_decay,
        case.hyper.betas[0], case.hyper.betas[1], 1);
    assert_eq!(hyper.eps, case.hyper.eps);
    let mut state = None;
    let output = adamw_step(fixture_tensor::<2>(&case.parameter, &device),
        fixture_tensor::<2>(&case.gradient, &device), &mut state, hyper);
    let state = state.unwrap();
    assert_fixture_close(output, &case.output, 2e-6, "AdamW parameter");
    assert_fixture_close(state.exp_avg, &case.exp_avg, 2e-7, "AdamW first moment");
    assert_fixture_close(state.exp_avg_sq, &case.exp_avg_sq, 2e-7, "AdamW second moment");
}

fn assert_muon_case(case: MuonFixture, device: &ModelDevice, label: &str) {
    let hyper = MuonHyper { lr: case.hyper.lr, weight_decay: case.hyper.weight_decay,
        momentum: case.hyper.momentum, beta2: case.hyper.beta2,
        ns_steps: case.hyper.ns_steps };
    let mut state = None;
    let output = muon_step(fixture_tensor(&case.parameter, device),
        fixture_tensor(&case.gradient, device), &mut state, hyper);
    let state = state.unwrap();
    assert_fixture_close(output, &case.output, 3e-5, &format!("{label} parameter"));
    assert_fixture_close(state.momentum_buffer, &case.momentum_buffer, 2e-7,
        &format!("{label} momentum"));
    assert_fixture_close(state.second_momentum_buffer, &case.second_momentum_buffer, 3e-5,
        &format!("{label} second moment"));
}

#[test] fn test_python_muon_single_step_parity() {
    let (fixture, device) = (optimizer_fixture(), Default::default());
    assert_muon_case(fixture.muon_tall, &device, "tall Muon");
    assert_muon_case(fixture.muon_wide, &device, "wide Muon");
}
