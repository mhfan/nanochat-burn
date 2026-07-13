
use std::{fs::File, io::{self, BufRead, BufReader}, path::Path};

#[cfg(feature = "ndarray")] use burn::backend::ndarray::NdArray;
#[cfg(not(feature = "ndarray"))] use burn::backend::wgpu::Wgpu;
use burn::{prelude::ToElement, backend::autodiff::Autodiff,
    tensor::{DType, Int, Tensor, TensorData, backend::{Backend, BackendTypes}, f16},
};
use serde::de::DeserializeOwned;

/// 定义默认的 GPU 后端与自动微分包装
//pub type ModelBackend = Wgpu;
#[cfg(feature = "ndarray")]
pub type ModelBackend = NdArray<f32, i32>;
#[cfg(not(feature = "ndarray"))]
pub type ModelBackend = Wgpu<f16, i32>;
pub type ModelDevice = <ModelBackend as BackendTypes>::Device;
pub type ModelAutodiffBackend = Autodiff<ModelBackend>;

/// 初始化系统计算设备：优先使用 WGPU GPU；
/// 如果环境变量 BURN_DEVICE=cpu，则强行使用 CPU 运行以规避 GPU JIT 编译开销
pub fn init_device() -> ModelDevice {
    #[cfg(feature = "ndarray")] { Default::default() }

    #[cfg(not(feature = "ndarray"))] {
        let use_cpu =
            std::env::var("BURN_DEVICE").is_ok_and(|value| value.eq_ignore_ascii_case("cpu"));
        if use_cpu { burn::backend::wgpu::WgpuDevice::Cpu } else { Default::default() }
    }
}

pub fn extract_answer(text: &str) -> Option<i32> {
    let marker = "#### ";
    text.rfind(marker).and_then(|idx| {
        let candidate = text[idx + marker.len()..].split_whitespace().next()?;
        let clean_num: String = candidate.chars()
            .filter(|c| c.is_ascii_digit() || *c == '-').collect();
        clean_num.parse::<i32>().ok()
    })
}

pub fn tensor_data_to_f32_vec(data: TensorData) -> Vec<f32> {
    match data.dtype {
        DType::F32 => data.to_vec::<f32>().unwrap(),
        DType::F16 => data.to_vec::<f16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
        _ => data.to_vec::<f32>().unwrap_or_else(|_| {
            data.to_vec::<f16>().unwrap().into_iter().map(|v| v.to_f32()).collect()
        }),
    }
}

pub fn scalar_to_f32<E: ToElement>(value: E) -> f32 { value.to_f32() }

pub(crate) fn int_tensor_2d<B: Backend>(data: Vec<i32>, shape: [usize; 2],
    device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

pub(crate) fn read_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> io::Result<Vec<T>> {
    BufReader::new(File::open(path)?).lines()
        .filter_map(|line| match line {
            Ok(line) if line.trim().is_empty() => None,
            other => Some(other.and_then(|line| {
                serde_json::from_str(line.trim()).map_err(io::Error::other)
            })),
        }).collect()
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn verify_autodiff_pipeline() {
        // 初始化测试日志，以便在测试失败时观察底层驱动输出
        //tracing_subscriber::fmt().with_env_filter("info").init();

        let device = init_device();

        // 1. 创建前向张量（使用 Autodiff 包装 of Wgpu 后端）
        // 注意：Burn 中可以直接使用 Tensor::from_data 传入多维数组
        let x: Tensor<ModelAutodiffBackend, 2> =
            Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let w: Tensor<ModelAutodiffBackend, 2> =
            Tensor::from_data([[2.0f32, 0.0], [0.0, 2.0]], &device);

        // 2. 显式要求追踪这两个张量的梯度
        let (x, w) = (x.require_grad(), w.require_grad());

        // 3. 执行前向计算：y = x * w (矩阵乘法)
        let y = x.clone().matmul(w.clone());

        // 4. 将输出规约为标量 Loss：loss = sum(y)
        let (loss, epsilon) = (y.sum(), f32::EPSILON); // 1e-4

        // Burn 中转换为标量的写法非常直接
        let loss_val = loss.clone().into_scalar();
        tracing::debug!("Forward Pass Loss: {}", loss_val);
        assert!((scalar_to_f32(loss_val) - 20.0).abs() < epsilon);

        // 5. 触发反向传播
        let grads = loss.backward();

        // 6. 提取输入和权重的梯度
        let x_grad = x.grad(&grads).expect("Failed to compute x gradient");
        let w_grad = w.grad(&grads).expect("Failed to compute w gradient");

        // 打印计算出来的梯度，验证 GPU 端与 CPU 端的数值闭环，预期数学结果：
        //   y = [[2, 4], [6, 8]], loss = 20
        // dy/dx = w^T => x_grad 应该为 [[2.0, 2.0], [2.0, 2.0]]
        // dy/dw = x^T => w_grad 应该为 [[4.0, 4.0], [6.0, 6.0]]
        tracing::debug!("Gradient of x (dy/dx): \n{}", x_grad);
        tracing::debug!("Gradient of w (dy/dw): \n{}", w_grad);

        fn assert_close_slice(actual: &[f32], expected: &[f32], epsilon: f32, label: &str) {
            for (act, exp) in actual.iter().zip(expected) {
                assert!((act - exp).abs() < epsilon, "{} mismatch: {actual:?}", label);
            }
        }

        assert_close_slice(&tensor_data_to_f32_vec(x_grad.into_data()),
            &[2.0, 2.0, 2.0, 2.0], epsilon, "x gradient");
        assert_close_slice(&tensor_data_to_f32_vec(w_grad.into_data()),
            &[4.0, 4.0, 6.0, 6.0], epsilon, "w gradient");
    }
}
