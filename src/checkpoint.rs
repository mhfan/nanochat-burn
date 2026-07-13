
use std::{collections::BTreeMap, path::Path};

use burn::{module::Param, tensor::{Shape, Tensor, TensorData, backend::Backend}};
use safetensors::SafeTensors;

use crate::gpt::{Gpt, has_ve};

fn block_weight_name(layer: usize, path: &str) -> String {
    format!("transformer.h.{layer}.{path}.weight")
}

fn value_embed_weight_name(layer: usize) -> String { format!("value_embeds.{layer}.weight") }

pub fn load_safetensors_to_gpt<B: Backend>(gpt: &mut Gpt<B>, path: &Path, device: &B::Device)
    -> Result<(), String> {
    let file_data =
        std::fs::read(path).map_err(|e| format!("Failed to read safetensors file: {:?}", e))?;
    let tensors = SafeTensors::deserialize(&file_data)
        .map_err(|e| format!("Failed to parse safetensors: {:?}", e))?;

    let get_f32_data = |name: &str, expected_rank: usize| -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = tensors.tensor(name)
            .map_err(|e| format!("Tensor '{}' not found in safetensors: {:?}", name, e))?;
        if view.dtype() != safetensors::tensor::Dtype::F32 {
            return Err(format!("Tensor '{name}' must use F32 storage, got {:?}", view.dtype()));
        }
        let shape: Vec<usize> = view.shape().to_vec();
        if  shape.len() != expected_rank {
            return Err(format!("Tensor '{name}' must have rank {expected_rank}, got {}",
                shape.len()));
        }
        let data = view.data();
        let expected_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        if  data.len() != expected_bytes {
            return Err(format!("Tensor '{name}' has {} bytes, expected {expected_bytes}",
                data.len()));
        }
        let f32_data = match bytemuck::try_cast_slice::<u8, f32>(data) {
            Ok(slice) => slice.to_vec(),
            Err(_) => data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect(),
        };
        Ok((f32_data, shape))
    };

    let load_param = |name: &str| -> Result<Tensor<B, 2>, String> {
        let (data, shape) = get_f32_data(name, 2)?;
        Ok(Tensor::<B, 2>::from_data(TensorData::new(data, Shape::from(shape)), device))
    };

    let load_transposed_param =
        |name: &str| -> Result<Tensor<B, 2>, String> { Ok(load_param(name)?.transpose()) };

    let load_vector = |name: &str| -> Result<Tensor<B, 1>, String> {
        let (data, shape) = get_f32_data(name, 1)?;
        Ok(Tensor::<B, 1>::from_data(TensorData::new(data, Shape::from(shape)), device))
    };

    gpt.wte.weight = Param::from_tensor(load_param("transformer.wte.weight")?);
    gpt.lm_head.weight = Param::from_tensor(load_transposed_param("lm_head.weight")?);

    for i in 0..gpt.config.n_layer {
        let block = &mut gpt.h[i];

        block.attn.c_q.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "attn.c_q"))?);
        block.attn.c_k.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "attn.c_k"))?);
        block.attn.c_v.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "attn.c_v"))?);
        block.attn.c_proj.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "attn.c_proj"))?);

        if has_ve(i, gpt.config.n_layer) &&
            let Some(ref mut gate_linear) = block.attn.ve_gate {
            gate_linear.weight = Param::from_tensor(
                load_transposed_param(&block_weight_name(i, "attn.ve_gate"))?);
        }

        block.mlp.c_fc.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "mlp.c_fc"))?);
        block.mlp.c_proj.weight =
            Param::from_tensor(load_transposed_param(&block_weight_name(i, "mlp.c_proj"))?);
    }

    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            gpt.value_embeds[ve_cnt].weight =
                Param::from_tensor(load_param(&value_embed_weight_name(i))?);
            ve_cnt += 1;
        }
    }

    gpt.resid_lambdas = Param::from_tensor(load_vector("resid_lambdas")?);
    gpt.x0_lambdas = Param::from_tensor(load_vector("x0_lambdas")?);
    gpt.smear_gate.weight = Param::from_tensor(load_transposed_param("smear_gate.weight")?);
    gpt.smear_lambda = Param::from_tensor(load_vector("smear_lambda")?);
    gpt.backout_lambda = Param::from_tensor(load_vector("backout_lambda")?);

    Ok(())
}

struct SavedTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
}

#[derive(Default)]
struct ParamSaver { tensors: BTreeMap<String, SavedTensor> }

impl ParamSaver {
    fn save_tensor<B: Backend, const D: usize>(&mut self, name: &str, tensor: Tensor<B, D>) {
        let shape = tensor.shape().dims::<D>().to_vec();
        let data = crate::common::tensor_data_to_f32_vec(tensor.into_data());
        let bytes = bytemuck::cast_slice::<f32, u8>(&data).to_vec();
        self.tensors.insert(name.to_string(), SavedTensor { bytes, shape });
    }

    fn save_param<B: Backend>(&mut self, name: &str, tensor: Tensor<B, 2>) {
        self.save_tensor(name, tensor);
    }

    fn save_transposed_param<B: Backend>(&mut self, name: &str, tensor: Tensor<B, 2>) {
        self.save_tensor(name, tensor.transpose());
    }

    fn save_vector<B: Backend>(&mut self, name: &str, tensor: Tensor<B, 1>) {
        self.save_tensor(name, tensor);
    }
}

pub fn save_gpt_to_safetensors<B: Backend>(gpt: &Gpt<B>, path: &Path) -> Result<(), String> {
    let mut saver = ParamSaver::default();

    saver.save_param("transformer.wte.weight", gpt.wte.weight.val());
    saver.save_transposed_param("lm_head.weight", gpt.lm_head.weight.val());

    for i in 0..gpt.config.n_layer {
        let block = &gpt.h[i];

        saver.save_transposed_param(
            &block_weight_name(i, "attn.c_q"),
            block.attn.c_q.weight.val(),
        );
        saver.save_transposed_param(
            &block_weight_name(i, "attn.c_k"),
            block.attn.c_k.weight.val(),
        );
        saver.save_transposed_param(
            &block_weight_name(i, "attn.c_v"),
            block.attn.c_v.weight.val(),
        );
        saver.save_transposed_param(
            &block_weight_name(i, "attn.c_proj"),
            block.attn.c_proj.weight.val(),
        );

        if has_ve(i, gpt.config.n_layer) &&
            let Some(ref gate_linear) = block.attn.ve_gate {
            saver.save_transposed_param(
                &block_weight_name(i, "attn.ve_gate"),
                gate_linear.weight.val(),
            );
        }

        saver.save_transposed_param(
            &block_weight_name(i, "mlp.c_fc"),
            block.mlp.c_fc.weight.val(),
        );
        saver.save_transposed_param(
            &block_weight_name(i, "mlp.c_proj"),
            block.mlp.c_proj.weight.val(),
        );
    }

    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            saver.save_param(&value_embed_weight_name(i),
                gpt.value_embeds[ve_cnt].weight.val());
            ve_cnt += 1;
        }
    }

    saver.save_vector("resid_lambdas", gpt.resid_lambdas.val());
    saver.save_vector("x0_lambdas", gpt.x0_lambdas.val());
    saver.save_transposed_param("smear_gate.weight", gpt.smear_gate.weight.val());
    saver.save_vector("smear_lambda", gpt.smear_lambda.val());
    saver.save_vector("backout_lambda", gpt.backout_lambda.val());

    // 6. Serialize to BTreeMap of TensorViews and write to file
    let mut tensors_map = BTreeMap::new();
    for (name, tensor) in &saver.tensors {
        let view = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32, tensor.shape.clone(), &tensor.bytes,
        ).map_err(|e| format!("Failed to create TensorView for '{}': {:?}", name, e))?;
        tensors_map.insert(name.clone(), view);
    }

    safetensors::tensor::serialize_to_file(&tensors_map, None, path)
        .map_err(|e| format!("Failed to serialize safetensors: {:?}", e))?;

    Ok(())
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_safetensors_roundtrip() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        let config = crate::gpt::GptConfig { sequence_len: 16, vocab_size: 32, n_layer: 2,
            n_head: 2, n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(),
            quantization: None,
        };

        // 1. Create a dummy source model
        let model_src = Gpt::<ModelBackend>::new(config.clone(), &device);

        // 2. Save it to a temporary safetensors file
        let path = Path::new("test_model.safetensors");
        save_gpt_to_safetensors(&model_src, path).expect("Failed to save safetensors");

        // 3. Create a target model and load the saved safetensors back
        let mut model_dst = Gpt::<ModelBackend>::new(config, &device);
        load_safetensors_to_gpt(&mut model_dst, path, &device).expect("Failed to load safetensors");

        fn assert_tensor_exact_eq<B: Backend, const D: usize>(
            left: Tensor<B, D>, right: Tensor<B, D>, message: &str) {
            let diff = crate::common::scalar_to_f32((left - right).abs().sum().into_scalar());
            assert_eq!(diff, 0.0, "{}", message);
        }

        // 4. Assert that all weights are identical
        assert_tensor_exact_eq(
            model_src.wte.weight.val(),
            model_dst.wte.weight.val(),
            "wte weight mismatch",
        );
        assert_tensor_exact_eq(
            model_src.lm_head.weight.val(),
            model_dst.lm_head.weight.val(),
            "lm_head weight mismatch",
        );

        for i in 0..model_src.config.n_layer {
            let (src_block, dst_block) = (&model_src.h[i], &model_dst.h[i]);

            assert_tensor_exact_eq(
                src_block.attn.c_q.weight.val(),
                dst_block.attn.c_q.weight.val(),
                &format!("c_q weight mismatch at layer {}", i),
            );
            assert_tensor_exact_eq(
                src_block.attn.c_k.weight.val(),
                dst_block.attn.c_k.weight.val(),
                &format!("c_k weight mismatch at layer {}", i),
            );
            assert_tensor_exact_eq(
                src_block.attn.c_v.weight.val(),
                dst_block.attn.c_v.weight.val(),
                &format!("c_v weight mismatch at layer {}", i),
            );
            assert_tensor_exact_eq(
                src_block.attn.c_proj.weight.val(),
                dst_block.attn.c_proj.weight.val(),
                &format!("c_proj weight mismatch at layer {}", i),
            );
            assert_tensor_exact_eq(
                src_block.mlp.c_fc.weight.val(),
                dst_block.mlp.c_fc.weight.val(),
                &format!("c_fc weight mismatch at layer {}", i),
            );
            assert_tensor_exact_eq(
                src_block.mlp.c_proj.weight.val(),
                dst_block.mlp.c_proj.weight.val(),
                &format!("mlp c_proj weight mismatch at layer {}", i),
            );
        }

        // Clean up temporary file
        std::fs::remove_file(path).ok();
    }
//}
