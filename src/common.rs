
use std::{fs::File, io::{self, BufRead, BufReader}, path::Path};
use burn::{prelude::ToElement, backend::autodiff::Autodiff,
    tensor::{DType, Int, Tensor, TensorData, backend::{Backend, BackendTypes}, f16},
};
use serde::de::DeserializeOwned;

/// Selected inference and training backends. NdArray is the portable default; WGPU is selected
/// explicitly with `--no-default-features --features wgpu`.
#[cfg(feature = "wgpu")]
pub type WgpuBackend = burn::backend::wgpu::Wgpu<f16, i32>;
#[cfg(feature = "ndarray")]
pub type InferBackend = burn::backend::ndarray::NdArray<f32, i32>;
#[cfg(all(not(feature = "ndarray"), feature = "wgpu"))]
pub type InferBackend = WgpuBackend;
#[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
compile_error!("enable either the `wgpu` or `ndarray` feature");

pub type ModelDevice = <InferBackend as BackendTypes>::Device;
pub type TrainBackend = Autodiff<InferBackend>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryUsage {
    pub bytes_reserved: u64,
    pub bytes_in_use: u64,
}

/// 初始化当前 feature 选择的计算设备；WGPU-only 构建可用 `BURN_DEVICE=cpu` 强制选择 CPU。
pub fn init_device() -> ModelDevice {
    #[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
    if std::env::var("BURN_DEVICE").is_ok_and(|value| value.eq_ignore_ascii_case("cpu")) {
        return burn::backend::wgpu::WgpuDevice::Cpu;
    }
    Default::default()
}

#[cfg(all(feature = "wgpu", any(test, not(feature = "ndarray"))))]
fn wgpu_memory_usage(device: &burn::backend::wgpu::WgpuDevice) -> Option<DeviceMemoryUsage> {
    use cubecl_runtime::runtime::Runtime;
    let usage = burn::backend::wgpu::WgpuRuntime::client(device).memory_usage().ok()?;
    Some(DeviceMemoryUsage {
        bytes_in_use: usage.bytes_in_use, bytes_reserved: usage.bytes_reserved,
    })
}

#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
pub fn device_memory_usage(device: &ModelDevice) -> Option<DeviceMemoryUsage> {
    wgpu_memory_usage(device)
}

#[cfg(feature = "ndarray")]
pub fn device_memory_usage(_device: &ModelDevice) -> Option<DeviceMemoryUsage> { None }

pub fn process_memory_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    system.process(pid).map(sysinfo::Process::memory)
}

/// Extracts the final GSM8K-style answer following a `#### ` marker.
///
/// ```
/// use nanochat_burn::common::extract_answer;
///
/// assert_eq!(extract_answer("The answer is #### 12,345"), Some(12_345));
/// assert_eq!(extract_answer("#### -7 apples and 3 pears"), Some(-7));
/// assert_eq!(extract_answer("No answer here"), None);
/// ```
pub fn extract_answer(text: &str) -> Option<i32> {
    let marker = "#### ";
    text.rfind(marker).and_then(|idx| {
        let candidate = text[idx + marker.len()..].split_whitespace().next()?;
        let clean_num: String = candidate.chars()
            .filter(|c| c.is_ascii_digit() || *c == '-').collect();
        clean_num.parse().ok()
    })
}

pub fn tensor_data_to_f32_vec(data: TensorData) -> Vec<f32> {
    match data.dtype {
        DType::F32 => data.to_vec().unwrap(),
        DType::F16 => data.to_vec::<f16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
        _ => data.to_vec().unwrap_or_else(|_| {
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
    #[cfg(feature = "wgpu")]
    #[test] fn verify_device_memory_usage() {
        let device = Default::default();
        let tensor = Tensor::<WgpuBackend, 1>::zeros([1024], &device);
        let _ = tensor.clone().into_data();
        let usage = wgpu_memory_usage(&device).expect("WGPU allocator memory report");
        assert!(usage.bytes_in_use > 0);
        assert!(usage.bytes_reserved >= usage.bytes_in_use);
    }

    #[test] fn verify_autodiff_pipeline() {
        // 初始化测试日志，以便在测试失败时观察底层驱动输出
        //tracing_subscriber::fmt().with_env_filter("info").init();

        let device = Default::default();

        // 1. 创建前向张量（使用默认后端的 Autodiff 包装）
        // 注意：Burn 中可以直接使用 Tensor::from_data 传入多维数组
        let x: Tensor<TrainBackend, 2> =
            Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let w = Tensor::from_data([[2.0, 0.0], [0.0, 2.0]], &device);

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
