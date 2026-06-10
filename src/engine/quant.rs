
use burn::{nn::Linear, module::{Module, Param}, tensor::{backend::Backend, Int, Tensor}};
use crate::gpt::ForwardLayer;

#[derive(Module, Debug)]
pub enum LinearOrQuantized<B: Backend> {
    Standard(Linear<B>),
    Quantized(QuantizedLinear<B>),
}

impl<B: Backend> LinearOrQuantized<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Standard(ref lin) => lin.forward(input),
            Self::Quantized(ref quant) => quant.forward(input),
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
    pub packed_weights: Tensor<B, 2, Int>,       // [I_packed, O]
    pub scales: Tensor<B, 3>,                    // [I_blocks, 1, O]
    pub bias: Option<Param<Tensor<B, 1>>>,        // [O]
    pub bits: usize,                              // 8 or 4
    pub block_size: usize,                        // Block size (e.g. 32, 64, or 0 for row-wise)
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

        // 3. Matrix multiplication: Y = X * W
        // input: [B, I], w_float: [I, O] -> Y: [B, O]
        let mut output = flattened.matmul(w_float.clone());

        // 4. Add optional bias
        if let Some(ref bias) = self.bias {
            output = output + bias.val().unsqueeze_dim(0);
        }

        // 5. Reshape back to D-dimensional
        let mut out_shape = shape;
        out_shape[D - 1] = w_float.shape().dims::<2>()[1]; // out_features
        output.reshape(out_shape)
    }

    /// De-quantize packed integer weights to float weights of shape [I, O]
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let packed = self.packed_weights.clone();
        let packed_shape = packed.shape().dims::<2>();
        let i_packed = packed_shape[0];
        let o = packed_shape[1];

        let q_unpacked = if self.bits == 8 {
            // W8A16 Unpacking (Pack Factor = 4)
            // packed: [I / 4, O]
            let mut q_floats = Vec::with_capacity(4);
            let mut cur = packed;
            for _ in 0..4 {
                let cur_q = cur.clone() / 256;
                let next_cur = cur_q.clone() - (cur.clone() - cur_q.clone() * 256).lower_elem(0).int();
                let rem = cur.clone() - next_cur.clone() * 256;

                q_floats.push((rem.float() - 128.0).unsqueeze_dim(2)); // Shift to [-128, 127]
                cur = next_cur;
            }

            // Stack along dimension 2 and reshape to [I, O]
            let stacked: Tensor<B, 3> = Tensor::cat(q_floats, 2); // [I / 4, O, 4]
            let permuted: Tensor<B, 3> = stacked.swap_dims(1, 2); // [I / 4, 4, O]
            permuted.reshape([i_packed * 4, o])
        } else {
            // W4A16 Unpacking (Pack Factor = 8)
            // packed: [I / 8, O]
            let mut q_floats: Vec<Tensor<B, 3>> = Vec::with_capacity(8);
            let mut cur = packed;
            for _ in 0..8 {
                let cur_q = cur.clone() / 16;
                let next_cur = cur_q.clone() - (cur.clone() - cur_q.clone() * 16).lower_elem(0).int();
                let rem = cur.clone() - next_cur.clone() * 16;

                q_floats.push((rem.float() - 8.0).unsqueeze_dim(2)); // Shift to [-8, 7]
                cur = next_cur;
            }

            // Stack along dimension 2 and reshape to [I, O]
            let stacked: Tensor<B, 3> = Tensor::cat(q_floats, 2); // [I / 8, O, 8]
            let permuted: Tensor<B, 3> = stacked.swap_dims(1, 2); // [I / 8, 8, O]
            permuted.reshape([i_packed * 8, o])
        };

        // 3. Apply Scales & Shift
        let i = q_unpacked.shape().dims::<2>()[0];
        let scales_val = self.scales.clone();
        let i_blocks = scales_val.shape().dims::<3>()[0];

        if self.block_size > 0 {
            // Block-wise scaling
            let block_len = self.block_size;
            let reshaped = q_unpacked.reshape([i_blocks, block_len, o]);
            let scaled = reshaped * scales_val;
            scaled.reshape([i, o])
        } else {
            // Row-wise scaling
            q_unpacked * scales_val.reshape([1, o])
        }
    }
}

/// Dynamically quantize a standard floating-point Linear layer into a QuantizedLinear layer in-place
pub fn quantize_linear<B: Backend>(linear: Linear<B>, bits: usize,
    block_size: usize,) -> QuantizedLinear<B> {
    let _device = linear.weight.device();

    // Weight has shape [I, O] in Burn's Linear layer
    let weight = linear.weight.val();
    let shape = weight.shape().dims::<2>();
    let (i, o) = (shape[0], shape[1]);

    let block_size_actual = if bits == 4 && block_size == 0 {
        64 // Default block size of 64 for W4
    } else {
        block_size
    };

    let (max_val, offset, max_q) = if bits == 8 {
        (127.0, 128.0, 255.0)
    } else {
        (7.0, 8.0, 15.0)
    };

    let (q_weight, scales) = if block_size_actual > 0 {
        // Block-wise quantization
        let num_blocks = i / block_size_actual;
        let reshaped = weight.reshape([num_blocks, block_size_actual, o]);

        // Compute block max absolute value along block dimension (dimension 1)
        // Note: max_dim returns [num_blocks, 1, o]
        let max_abs = reshaped.clone().abs().max_dim(1);

        // Prevent division by zero
        let block_scales = max_abs.clone() / max_val;

        // Quantize
        let q_shifted = (reshaped / block_scales.clone()).round() + offset;
        let clamped = q_shifted.clamp(0.0, max_q);
        let q_flat = clamped.reshape([i, o]);

        (q_flat, block_scales)
    } else {
        // Row-wise quantization (scales has shape [1, 1, O])
        let max_abs = weight.clone().abs().max_dim(0); // [1, o]
        let channel_scales = max_abs.clone() / max_val;

        let reshaped_scales = channel_scales.clone().reshape([1, o]);
        let q_shifted = (weight / reshaped_scales).round() + offset;
        let clamped = q_shifted.clamp(0.0, max_q);

        (clamped, channel_scales.reshape([1, 1, o]))
    };

    // Pack the quantized floats into packed i32 integers
    let packed_weights = if bits == 8 {
        // INT8: Pack factor 4
        let num_packed = i / 4;
        let q_reshaped = q_weight.reshape([num_packed, 4, o]).int();

        let mut packed = q_reshaped.clone().slice([0..num_packed, 0..1, 0..o]).reshape([num_packed, o]);
        let coeffs = [256, 65536, 16777216];
        for k in 0..3 {
            let slice = q_reshaped.clone().slice([0..num_packed, (k+1)..(k+2), 0..o]).reshape([num_packed, o]);
            packed = packed + slice.mul_scalar(coeffs[k]);
        }
        packed
    } else {
        // INT4: Pack factor 8
        let num_packed = i / 8;
        let q_reshaped = q_weight.reshape([num_packed, 8, o]).int();

        let mut packed = q_reshaped.clone().slice([0..num_packed, 0..1, 0..o]).reshape([num_packed, o]);
        let coeffs = [16, 256, 4096, 65536, 1048576, 16777216, 268435456];
        for k in 0..7 {
            let slice = q_reshaped.clone().slice([0..num_packed, (k+1)..(k+2), 0..o]).reshape([num_packed, o]);
            packed = packed + slice.mul_scalar(coeffs[k]);
        }
        packed
    };

    QuantizedLinear { packed_weights, scales, bias: linear.bias,
        bits, block_size: block_size_actual,
    }
}

#[cfg(test)] mod tests { use super::*;
    use burn::tensor::Distribution;

    #[test] fn test_quantization_w8_rowwise() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        // Shape [64, 128] is divisible by 4 and 8
        let weight = Tensor::<ModelBackend, 2>::random([64, 128], Distribution::Normal(0.0, 1.0), &device);
        let linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: None,
        };

        // Quantize to W8 row-wise (block_size = 0)
        let q_linear = quantize_linear(linear, 8, 0);

        assert_eq!(q_linear.bits, 8);
        assert_eq!(q_linear.block_size, 0);
        assert_eq!(q_linear.packed_weights.shape().dims(), [16, 128]); // 64 / 4 = 16

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff = (dequantized - weight).abs().mean().into_scalar().to_f32();
        println!("W8 Row-wise Quantization Mean Absolute Error: {}", diff);
        // Standard normal distribution values quantized to 256 levels should have very low error (< 0.02)
        assert!(diff < 0.02, "Error too high: {}", diff);
    }

    #[test] fn test_quantization_w4_blockwise() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        // Shape [64, 128]
        let weight = Tensor::<ModelBackend, 2>::random([64, 128], Distribution::Normal(0.0, 1.0), &device);
        let linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: None,
        };

        // Quantize to W4 block-wise (block_size = 32)
        let q_linear = quantize_linear(linear, 4, 32);

        assert_eq!(q_linear.bits, 4);
        assert_eq!(q_linear.block_size, 32);
        assert_eq!(q_linear.packed_weights.shape().dims(), [8, 128]); // 64 / 8 = 8

        // Dequantize and check difference
        let dequantized = q_linear.dequantize();
        assert_eq!(dequantized.shape().dims(), [64, 128]);

        let diff = (dequantized - weight).abs().mean().into_scalar().to_f32();
        println!("W4 Block-wise (32) Quantization Mean Absolute Error: {}", diff);
        // Standard normal values quantized to 16 levels block-wise should have acceptable error (< 0.25)
        assert!(diff < 0.25, "Error too high: {}", diff);
    }

    #[test] fn test_quantized_linear_forward() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        let weight = Tensor::<ModelBackend, 2>::random([64, 128], Distribution::Normal(0.0, 0.5), &device);
        let bias = Tensor::<ModelBackend, 1>::random([128], Distribution::Normal(0.0, 0.1), &device);
        let linear = Linear {
            weight: Param::from_tensor(weight.clone()),
            bias: Some(Param::from_tensor(bias.clone())),
        };

        let q_linear = quantize_linear(linear.clone(), 8, 32);

        // Input shape [2, 8, 64]
        let input = Tensor::<ModelBackend, 3>::random([2, 8, 64], Distribution::Normal(0.0, 1.0), &device);

        let out_std = linear.forward(input.clone());
        let out_quant = q_linear.forward(input);

        assert_eq!(out_std.shape().dims(), [2, 8, 128]);
        assert_eq!(out_quant.shape().dims(), [2, 8, 128]);

        let diff = (out_std - out_quant).abs().mean().into_scalar().to_f32();
        println!("W8 Block-wise Forward Mean Absolute Difference: {}", diff);
        assert!(diff < 0.05, "Forward difference too high: {}", diff);
    }
}
