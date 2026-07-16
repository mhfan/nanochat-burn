
use burn::{module::{Module, Param}, nn::Linear,
    tensor::{ElementConversion, Int, Tensor, backend::Backend,
        quantization::{QuantLevel, QuantScheme, QuantStore, QuantValue},
    },
};

use super::ForwardLayer;

#[derive(Module, Debug)]
pub enum LinearOrQuantized<B: Backend> {
    Standard(Linear<B>),
    Quantized(QuantizedLinear<B>),
}

impl<B: Backend> LinearOrQuantized<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Standard(lin) => lin.forward(input),
            Self::Quantized(quant) => quant.forward(input),
        }
    }
}

impl<B: Backend> ForwardLayer<B> for LinearOrQuantized<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.forward(input)
    }
}

#[derive(Module, Debug)]
pub struct QuantizedLinear<B: Backend> {
    weights: QuantizedWeights<B>,
    pub bias: Option<Param<Tensor<B, 1>>>, // [O]
    pub bits: usize,                       // 8 or 4
    pub block_size: usize,                 // Block size (e.g. 32, 64, or 0 for row-wise)
}

#[derive(Module, Debug)]
enum QuantizedWeights<B: Backend> {
    Native(Tensor<B, 2>), // [O, I], preserving contiguous quantization blocks
    Packed(PackedWeights<B>),
}

#[derive(Module, Debug)]
struct PackedWeights<B: Backend> {
    values: Tensor<B, 2, Int>, // [I / pack_factor, O]
    scales: Tensor<B, 3>,      // [I_blocks, 1, O]
}

impl<B: Backend> ForwardLayer<B> for QuantizedLinear<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.forward(input)
    }
}

impl<B: Backend> QuantizedLinear<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let shape = input.shape().dims::<D>();
        let in_features = shape[D - 1];
        let flattened = input.reshape([-1, in_features as i32]);
        let (mut output, out_features) = match &self.weights {
            QuantizedWeights::Native(weight) => {
                (flattened.matmul(weight.clone().swap_dims(0, 1)), weight.shape().dims::<2>()[0])
            }
            QuantizedWeights::Packed(weight) => {
                let weight = weight.dequantize(self.bits, self.block_size);
                let out_features = weight.shape().dims::<2>()[1];
                (flattened.matmul(weight), out_features)
            }
        };

        if let Some(bias) = &self.bias {
            output = output + bias.val().unsqueeze_dim(0);
        }

        let mut out_shape = shape;
        out_shape[D - 1] = out_features;
        output.reshape(out_shape)
    }

    /// Dequantize the stored weights to `[in_features, out_features]`.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        match &self.weights {
            QuantizedWeights::Native(weight) => weight.clone().dequantize().swap_dims(0, 1),
            QuantizedWeights::Packed(weight) => weight.dequantize(self.bits, self.block_size),
        }
    }

    /// Whether forward uses Burn's packed quantized matmul directly.
    pub fn uses_native_weights(&self) -> bool {
        matches!(self.weights, QuantizedWeights::Native(_))
    }
}

impl<B: Backend> PackedWeights<B> {
    fn dequantize(&self, bits: usize, block_size: usize) -> Tensor<B, 2> {
        let [i_packed, o] = self.values.shape().dims();
        let (pack_factor, mask, offset) = quant_layout(bits);
        let mut q_floats = Vec::<Tensor<B, 3>>::with_capacity(pack_factor);
        for shift in (0..32).step_by(bits) {
            let values = self.values.clone().bitwise_right_shift_scalar(shift.elem())
                .bitwise_and_scalar(mask.elem());
            q_floats.push((values.float() - offset).unsqueeze_dim(2));
        }

        let q_unpacked =
            Tensor::cat(q_floats, 2).swap_dims(1, 2).reshape([i_packed * pack_factor, o]);
        let i = q_unpacked.shape().dims::<2>()[0];

        if block_size > 0 {
            let i_blocks = self.scales.shape().dims::<3>()[0];
            (q_unpacked.reshape([i_blocks, block_size, o]) * self.scales.clone()).reshape([i, o])
        } else {
            q_unpacked * self.scales.clone().reshape([1, o])
        }
    }
}

/// Dynamically quantize a standard floating-point Linear layer
/// into a QuantizedLinear layer in-place
pub fn quantize_linear<B: Backend>(linear: Linear<B>, bits: usize, block_size: usize)
    -> QuantizedLinear<B> {
    assert!(matches!(bits, 4 | 8), "quantization bits must be 4 or 8");

    let weight = linear.weight.val();
    let [i, _] = weight.shape().dims();
    let (pack_factor, ..) = quant_layout(bits);
    assert!(i % pack_factor == 0,
        "input dimension {} must be divisible by pack factor {} for W{} quantization",
        i, pack_factor, bits);

    let block_size = effective_block_size(bits, block_size);
    if block_size > 0 {
        assert!(i % block_size == 0,
            "input dimension {} must be divisible by quantization block size {}", i, block_size);
    }

    let weights = if native_quantization_supported(i, block_size) {
        QuantizedWeights::Native(quantize_native(weight, bits, block_size))
    } else {
        QuantizedWeights::Packed(quantize_packed(weight, bits, block_size))
    };

    QuantizedLinear { weights, bias: linear.bias, bits, block_size }
}

fn quantize_native<B: Backend>(weight: Tensor<B, 2>, bits: usize,
    block_size: usize) -> Tensor<B, 2> {
    let block_size = if block_size == 0 { weight.shape().dims::<2>()[0] } else { block_size };
    let scheme = QuantScheme::default()
        .with_value(if bits == 8 { QuantValue::Q8S } else { QuantValue::Q4S })
        .with_level(QuantLevel::block([block_size as u8]))
        .with_store(QuantStore::PackedU32(0));

    weight.swap_dims(0, 1).quantize_dynamic(&scheme)
}

fn quantize_packed<B: Backend>(weight: Tensor<B, 2>, bits: usize,
    block_size: usize) -> PackedWeights<B> {
    let [i, o] = weight.shape().dims();
    let (pack_factor, mask, offset) = quant_layout(bits);
    let max_val = offset - 1.0;

    #[allow(clippy::manual_checked_ops)]
    let (q_weight, scales) = if block_size > 0 {
        let num_blocks = i / block_size;
        let reshaped = weight.reshape([num_blocks, block_size, o]);
        let scales = (reshaped.clone().abs().max_dim(1) / max_val).clamp(1e-6, 1e6);
        let quantized = (reshaped / scales.clone()).round() + offset;
        (quantized.clamp(0.0, mask as f32).reshape([i, o]), scales)
    } else {
        let scales = (weight.clone().abs().max_dim(0) / max_val).clamp(1e-6, 1e6);
        let quantized = (weight / scales.clone()).round() + offset;
        (quantized.clamp(0.0, mask as f32), scales.reshape([1, 1, o]))
    };

    let num_packed = i / pack_factor;
    let q_reshaped = q_weight.reshape([num_packed, pack_factor, o]).int();
    let mut values =
        q_reshaped.clone().slice([0..num_packed, 0..1, 0..o]).reshape([num_packed, o]);
    for k in 1..pack_factor {
        let mut slice =
            q_reshaped.clone().slice([0..num_packed, k..k + 1, 0..o]).reshape([num_packed, o]);
        if k + 1 == pack_factor {
            // Encode the highest lane as two's complement before shifting into the sign bits.
            let sign = slice.clone().greater_equal_elem(1_i32 << (bits - 1)).int();
            slice = slice - sign.mul_scalar(mask + 1);
        }
        values = values.bitwise_or(
            slice.bitwise_left_shift_scalar(((k * bits) as i32).elem()));
    }

    PackedWeights { values, scales }
}

pub fn quantize_linear_or_standard<B: Backend>(linear: Linear<B>, bits: usize,
    block_size: usize) -> LinearOrQuantized<B> {
    let input_dim = linear.weight.val().shape().dims::<2>()[0];
    let (pack_factor, ..) = quant_layout(bits);
    let block_size = effective_block_size(bits, block_size);
    let supported = input_dim % pack_factor == 0 &&
        (block_size == 0 || input_dim % block_size == 0);

    if supported {
        LinearOrQuantized::Quantized(quantize_linear(linear, bits, block_size))
    } else {
        tracing::debug!(input_dim, bits, block_size,
            "keeping layer in floating point because its input shape cannot be packed");
        LinearOrQuantized::Standard(linear)
    }
}

fn quant_layout(bits: usize) -> (usize, i32, f32) {
    match bits {
        8 => (4, 0xff, 128.0),
        4 => (8, 0x0f, 8.0),
        _ => panic!("quantization bits must be 4 or 8"),
    }
}

fn native_quantization_supported(input_dim: usize, block_size: usize) -> bool {
    let native_block_size = if block_size == 0 { input_dim } else { block_size };
    cfg!(not(feature = "ndarray")) && native_block_size <= u8::MAX as usize
}

fn effective_block_size(bits: usize, block_size: usize) -> usize {
    if bits == 4 && block_size == 0 { 64 } else { block_size }
}

#[cfg(test)] mod tests { use super::*;
    use burn::tensor::Distribution;
    use crate::common::ModelBackend;

    #[test] fn test_quantization_w8_rowwise() {
        let device = crate::common::init_device();

        // Shape [64, 128] is divisible by 4 and 8
        let weight = Tensor::<ModelBackend, 2>::random([64, 128],
            Distribution::Normal(0.0, 1.0), &device);
        let linear = Linear { weight: Param::from_tensor(weight.clone()), bias: None };

        // Quantize to W8 row-wise (block_size = 0)
        let q_linear = quantize_linear(linear, 8, 0);

        assert_eq!(q_linear.bits, 8);
        assert_eq!(q_linear.block_size, 0);
        assert_eq!(q_linear.uses_native_weights(), cfg!(not(feature = "ndarray")));

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff =
            crate::common::scalar_to_f32((dequantized - weight).abs().mean().into_scalar());
        // Standard normal distribution values quantized to 256 levels should have very low
        // error (< 0.02)
        assert!(diff < 0.02, "Error too high: {}", diff);
    }

    #[test] fn test_quantization_w4_blockwise() {
        let device = crate::common::init_device();

        // Shape [64, 128]
        let weight = Tensor::<ModelBackend, 2>::random([64, 128],
            Distribution::Normal(0.0, 1.0), &device);
        let linear = Linear { weight: Param::from_tensor(weight.clone()), bias: None };

        // Quantize to W4 block-wise (block_size = 32)
        let q_linear = quantize_linear(linear, 4, 32);

        assert_eq!(q_linear.bits, 4);
        assert_eq!(q_linear.block_size, 32);
        assert_eq!(q_linear.uses_native_weights(), cfg!(not(feature = "ndarray")));

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff =
            crate::common::scalar_to_f32((dequantized - weight).abs().mean().into_scalar());
        // Standard normal values quantized to 16 levels block-wise
        // should have acceptable error (< 0.25)
        assert!(diff < 0.25, "Error too high: {}", diff);
    }

    #[test] fn test_quantized_linear_forward() {
        let device = crate::common::init_device();

        let weight = Tensor::<ModelBackend, 2>::random([64, 128],
            Distribution::Normal(0.0, 0.5), &device);
        let bias = Tensor::random([128], Distribution::Normal(0.0, 0.1), &device);
        let linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: Some(Param::from_tensor(bias.clone())),
        };

        // Input shape [2, 8, 64]
        let input = Tensor::<ModelBackend, 3>::random([2, 8, 64],
            Distribution::Normal(0.0, 1.0), &device);
        let out_std = linear.forward(input.clone());

        for (bits, tolerance) in [(8, 0.05), (4, 0.5)] {
            let out_quant = quantize_linear(linear.clone(), bits, 32).forward(input.clone());
            assert_eq!(out_quant.shape().dims(), [2, 8, 128]);
            let diff = crate::common::scalar_to_f32(
                (out_std.clone() - out_quant).abs().mean().into_scalar());
            assert!(diff < tolerance, "W{} forward difference too high: {}", bits, diff);
        }
    }

    #[test] fn test_large_rowwise_quantization_uses_portable_fallback() {
        let device = crate::common::init_device();
        let weight = Tensor::<ModelBackend, 2>::random([256, 32],
            Distribution::Normal(0.0, 1.0), &device);
        let linear = Linear { weight: Param::from_tensor(weight.clone()), bias: None };

        let q_linear = quantize_linear(linear, 8, 0);

        assert!(!q_linear.uses_native_weights());
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [256, 32]);
        let diff = crate::common::scalar_to_f32(
            (dequantized - weight).abs().mean().into_scalar());
        assert!(diff < 0.02, "Error too high: {}", diff);
    }
}
