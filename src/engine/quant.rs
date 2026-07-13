
use burn::{module::{Module, Param}, nn::Linear, tensor::{Int, Tensor, backend::Backend}};

use crate::gpt::ForwardLayer;

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
    pub packed_weights: Tensor<B, 2, Int>, // [I_packed, O]
    pub scales: Tensor<B, 3>,              // [I_blocks, 1, O]
    pub bias: Option<Param<Tensor<B, 1>>>, // [O]
    pub bits: usize,                       // 8 or 4
    pub block_size: usize,                 // Block size (e.g. 32, 64, or 0 for row-wise)
}

impl<B: Backend> ForwardLayer<B> for QuantizedLinear<B> {
    fn forward_layer<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.forward(input)
    }
}

impl<B: Backend> QuantizedLinear<B> {
    /// Perform the forward pass with dynamic in-place de-quantization
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let shape: [usize; D] = input.shape().dims::<D>();
        let in_features = shape[D - 1];

        // 1. Reshape D-dimensional input to 2D
        let flattened = input.reshape([-1, in_features as i32]);

        // 2. De-quantize packed weights
        let w_float = self.dequantize();
        let out_features = w_float.shape().dims::<2>()[1];

        // 3. Matrix multiplication: Y = X * W
        // input: [B, I], w_float: [I, O] -> Y: [B, O]
        let mut output = flattened.matmul(w_float);

        // 4. Add optional bias
        if let Some(ref bias) = self.bias {
            output = output + bias.val().unsqueeze_dim(0);
        }

        // 5. Reshape back to D-dimensional
        let mut out_shape = shape;
        out_shape[D - 1] = out_features;
        output.reshape(out_shape)
    }

    /// De-quantize packed integer weights to float weights of shape [I, O]
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let packed = self.packed_weights.clone();
        let [i_packed, o] = packed.shape().dims::<2>();
        let (pack_factor, base, offset) = quant_layout(self.bits);
        let mut cur = packed;
        let mut q_floats: Vec<Tensor<B, 3>> = Vec::with_capacity(pack_factor);
        for _ in 0..pack_factor {
            let cur_q = cur.clone() / base;
            let next_cur =
                cur_q.clone() - (cur.clone() - cur_q.clone() * base).lower_elem(0).int();
            let rem = cur.clone() - next_cur.clone() * base;
            q_floats.push((rem.float() - offset).unsqueeze_dim(2));
            cur = next_cur;
        }

        let q_unpacked =
            Tensor::cat(q_floats, 2).swap_dims(1, 2).reshape([i_packed * pack_factor, o]);

        // 3. Apply Scales & Shift
        let i = q_unpacked.shape().dims::<2>()[0];
        let scales_val = self.scales.clone();
        let i_blocks = scales_val.shape().dims::<3>()[0];

        if self.block_size > 0 {
            // Block-wise scaling
            let reshaped = q_unpacked.reshape([i_blocks, self.block_size, o]);
            let scaled = reshaped * scales_val;
            scaled.reshape([i, o])
        } else {
            // Row-wise scaling
            q_unpacked * scales_val.reshape([1, o])
        }
    }
}

/// Dynamically quantize a standard floating-point Linear layer
/// into a QuantizedLinear layer in-place
pub fn quantize_linear<B: Backend>(linear: Linear<B>, bits: usize, block_size: usize)
    -> QuantizedLinear<B> {
    assert!(bits == 4 || bits == 8, "quantization bits must be 4 or 8");

    // Weight has shape [I, O] in Burn's Linear layer
    let weight = linear.weight.val();
    let shape = weight.shape().dims::<2>();
    let (i, o) = (shape[0], shape[1]);
    let (pack_factor, base, _offset) = quant_layout(bits);
    assert!(i % pack_factor == 0,
        "input dimension {} must be divisible by pack factor {} for W{} quantization",
        i, pack_factor, bits);

    let block_size_actual = if bits == 4 && block_size == 0 {
        64 // Default block size of 64 for W4
    } else { block_size };

    let (max_val, offset, max_q) =
        if bits == 8 { (127.0, 128.0, 255.0) } else { (7.0, 8.0, 15.0) };

    let (q_weight, scales) = if block_size_actual > 0 {
        assert!(i % block_size_actual == 0,
            "input dimension {} must be divisible by quantization block size {}",
            i, block_size_actual);
        let num_blocks = i / block_size_actual;
        // Block-wise quantization
        let reshaped = weight.reshape([num_blocks, block_size_actual, o]);

        // Compute block max absolute value along block dimension (dimension 1)
        // Note: max_dim returns [num_blocks, 1, o]
        let max_abs = reshaped.clone().abs().max_dim(1);

        // Prevent division by zero
        let block_scales = (max_abs / max_val).clamp(1e-6, 1e6);

        // Quantize
        let q_shifted = (reshaped / block_scales.clone()).round() + offset;
        let q_flat = q_shifted.clamp(0.0, max_q).reshape([i, o]);

        (q_flat, block_scales)
    } else {
        // Row-wise quantization (scales has shape [1, 1, O])
        let max_abs = weight.clone().abs().max_dim(0); // [1, o]
        let channel_scales = (max_abs / max_val).clamp(1e-6, 1e6);

        let reshaped_scales = channel_scales.clone().reshape([1, o]);
        let q_shifted = (weight / reshaped_scales).round() + offset;
        let clamped = q_shifted.clamp(0.0, max_q);

        (clamped, channel_scales.reshape([1, 1, o]))
    };

    // Pack the quantized floats into packed i32 integers
    let (num_packed, mut coeff) = (i / pack_factor, base);
    let q_reshaped = q_weight.reshape([num_packed, pack_factor, o]).int();
    let mut packed_weights =
        q_reshaped.clone().slice([0..num_packed, 0..1, 0..o]).reshape([num_packed, o]);
    for k in 1..pack_factor {
        let slice =
            q_reshaped.clone().slice([0..num_packed, k..k + 1, 0..o]).reshape([num_packed, o]);
        packed_weights = packed_weights + slice.mul_scalar(coeff);
        if k + 1 < pack_factor { coeff *= base; }
    }

    QuantizedLinear { packed_weights, scales, bias: linear.bias, bits,
        block_size: block_size_actual,
    }
}

pub fn quantize_linear_or_standard<B: Backend>(linear: Linear<B>, bits: usize,
    block_size: usize) -> LinearOrQuantized<B> {
    let input_dim = linear.weight.val().shape().dims::<2>()[0];
    let (pack_factor, ..) = quant_layout(bits);
    let block_size = if bits == 4 && block_size == 0 { 64 } else { block_size };
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
        8 => (4, 256, 128.0),
        4 => (8, 16, 8.0),
        _ => panic!("quantization bits must be 4 or 8"),
    }
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
        assert_eq!(q_linear.packed_weights.shape().dims(), [16, 128]); // 64 / 4 = 16

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff =
            crate::common::scalar_to_f32((dequantized - weight).abs().mean().into_scalar());
        println!("W8 Row-wise Quantization Mean Absolute Error: {}", diff);
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
        assert_eq!(q_linear.packed_weights.shape().dims(), [8, 128]); // 64 / 8 = 8

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff =
            crate::common::scalar_to_f32((dequantized - weight).abs().mean().into_scalar());
        println!("W4 Block-wise (32) Quantization Mean Absolute Error: {}", diff);
        // Standard normal values quantized to 16 levels block-wise
        // should have acceptable error (< 0.25)
        assert!(diff < 0.25, "Error too high: {}", diff);
    }

    #[test] fn test_quantized_linear_forward() {
        let device = crate::common::init_device();

        let weight = Tensor::<ModelBackend, 2>::random([64, 128],
            Distribution::Normal(0.0, 0.5), &device);
        let bias =
            Tensor::<ModelBackend, 1>::random([128], Distribution::Normal(0.0, 0.1), &device);
        let linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: Some(Param::from_tensor(bias.clone())),
        };

        let q_linear = quantize_linear(linear.clone(), 8, 32);

        // Input shape [2, 8, 64]
        let input = Tensor::<ModelBackend, 3>::random([2, 8, 64],
            Distribution::Normal(0.0, 1.0), &device);

        let out_std = linear.forward(input.clone());
        let out_quant = q_linear.forward(input);

        assert_eq!(out_std.shape().dims(), [2, 8, 128]);
        assert_eq!(out_quant.shape().dims(), [2, 8, 128]);

        let diff =
            crate::common::scalar_to_f32((out_std - out_quant).abs().mean().into_scalar());
        println!("W8 Block-wise Forward Mean Absolute Difference: {}", diff);
        assert!(diff < 0.05, "Forward difference too high: {}", diff);
    }
}
