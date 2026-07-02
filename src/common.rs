
use burn::{backend::autodiff::Autodiff,
    tensor::{DType, backend::BackendTypes, f16},
};
#[cfg(feature = "ndarray")]
use burn::backend::ndarray::NdArray;
#[cfg(not(feature = "ndarray"))]
use burn::backend::wgpu::Wgpu;

/// 定义默认的 GPU 后端与自动微分包装
//pub type ModelBackend = Wgpu;
#[cfg(feature = "ndarray")]
pub type ModelBackend = NdArray<f32, i32>;
#[cfg(not(feature = "ndarray"))]
pub type ModelBackend = Wgpu<f16, i32>;
pub type ModelAutodiffBackend = Autodiff<ModelBackend>;
pub type ModelDevice = <ModelBackend as BackendTypes>::Device;

/// 初始化系统计算设备：优先使用 WGPU GPU；
/// 如果环境变量 BURN_DEVICE=cpu，则强行使用 CPU 运行以规避 GPU JIT 编译开销
pub fn init_device() -> ModelDevice {
    #[cfg(feature = "ndarray")] { Default::default() }

    #[cfg(not(feature = "ndarray"))] {
    let device = if std::env::var("BURN_DEVICE").unwrap_or_default().to_lowercase() == "cpu" {
        burn::backend::wgpu::WgpuDevice::Cpu
    } else { Default::default() };
    //tracing::info!("Initializing computational device: {:?}", device);
    device
    }
}

pub fn extract_answer(text: &str) -> Option<i32> {
    let marker = "#### ";
    if let Some(idx) = text.rfind(marker) {
        let num_part = text[idx + marker.len()..].trim();
        let clean_num: String = num_part.chars()
            .filter(|c| c.is_ascii_digit() || *c == '-').collect();
        clean_num.parse::<i32>().ok()
    } else { None }
}

pub fn tensor_data_to_f32_vec(data: burn::tensor::TensorData) -> Vec<f32> {
    match data.dtype {
        DType::F32 => data.to_vec::<f32>().unwrap(),
        DType::F16 => data.to_vec::<f16>().unwrap().into_iter().map(|v| v.to_f32()).collect(),
        _ => data.to_vec::<f32>().unwrap_or_else(|_| {
            data.to_vec::<f16>().unwrap().into_iter().map(|v| v.to_f32()).collect()
        }),
    }
}

//#[cfg(test)] mod tests { use super::*;
#[cfg(all(test, feature = "ndarray"))]
use burn::prelude::ToElement;

    /// 执行数值校验与反向传播管道验证
    #[test] pub fn verify_autodiff_pipeline() {
        // 初始化测试日志，以便在测试失败时观察底层驱动输出
        //tracing_subscriber::fmt().with_env_filter("info").init();

        use burn::tensor::Tensor;
        let device = init_device();

        // 1. 创建前向张量（使用 Autodiff 包装 of Wgpu 后端）
        // 注意：Burn 中可以直接使用 Tensor::from_data 传入多维数组
        let x: Tensor<ModelAutodiffBackend, 2> =
            Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let w: Tensor<ModelAutodiffBackend, 2> =
            Tensor::from_data([[2.0f32, 0.0], [0.0, 2.0]], &device);

        // 2. 显式要求追踪这两个张量的梯度
        let x = x.require_grad();
        let w = w.require_grad();

        // 3. 执行前向计算：y = x * w (矩阵乘法)
        let y = x.clone().matmul(w.clone());

        // 4. 将输出规约为标量 Loss：loss = sum(y)
        let loss = y.sum();

        let epsilon = f32::EPSILON; // 1e-4

        // Burn 中转换为标量的写法非常直接
        let loss_val = loss.clone().into_scalar();
        tracing::debug!("Forward Pass Loss: {}", loss_val);
        assert!((loss_val.to_f32() - 20.0).abs() < epsilon);

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

        let expected_x_grad = [2.0f32, 2.0, 2.0, 2.0];
        for (act, exp) in x_grad.clone().into_data()
            .iter().map(|v: f16| v.to_f32()).zip(expected_x_grad.iter()) {
            assert!((act - exp).abs() < epsilon, "x 梯度不匹配! 实际: {:?}", x_grad);
        }

        let expected_w_grad = [4.0f32, 4.0, 6.0, 6.0];
        for (act, exp) in w_grad.clone().into_data()
            .iter().map(|v: f16| v.to_f32()).zip(expected_w_grad.iter()) {
            assert!((act - exp).abs() < epsilon, "w 梯度不匹配! 实际: {:?}", w_grad);
        }
    }
//}
