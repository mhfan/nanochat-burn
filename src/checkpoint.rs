
use std::{collections::HashMap, marker::PhantomData, path::Path};

use burn::{module::Param, tensor::{Shape, Tensor, TensorData, backend::Backend}};
use safetensors::SafeTensors;

use crate::gpt::{Gpt, has_ve};

pub fn load_safetensors_to_gpt<B: Backend>(gpt: &mut Gpt<B>, path: &Path,
    device: &B::Device) -> Result<(), String> {
    let file_data =
        std::fs::read(path).map_err(|e| format!("Failed to read safetensors file: {:?}", e))?;
    let tensors = SafeTensors::deserialize(&file_data)
        .map_err(|e| format!("Failed to parse safetensors: {:?}", e))?;

    let get_f32_data = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("Tensor '{}' not found in safetensors: {:?}", name, e))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let f32_data = match bytemuck::try_cast_slice::<u8, f32>(data) {
            Ok(slice) => slice.to_vec(),
            Err(_) => data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect(),
        };
        Ok((f32_data, shape))
    };

    let load_param = |name: &str| -> Result<Tensor<B, 2>, String> {
        let (data, shape) = get_f32_data(name)?;
        Ok(Tensor::<B, 2>::from_data(TensorData::new(data, Shape::from(shape)), device))
    };

    let load_transposed_param =
        |name: &str| -> Result<Tensor<B, 2>, String> { Ok(load_param(name)?.transpose()) };

    let load_vector = |name: &str| -> Result<Tensor<B, 1>, String> {
        let (data, shape) = get_f32_data(name)?;
        Ok(Tensor::<B, 1>::from_data(TensorData::new(data, Shape::from(shape)), device))
    };

    gpt.wte.weight = Param::from_tensor(load_param("transformer.wte.weight")?);
    gpt.lm_head.weight = Param::from_tensor(load_transposed_param("lm_head.weight")?);

    for i in 0..gpt.config.n_layer {
        let block = &mut gpt.h[i];

        block.attn.c_q.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.attn.c_q.weight", i
        ))?);
        block.attn.c_k.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.attn.c_k.weight", i
        ))?);
        block.attn.c_v.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.attn.c_v.weight", i
        ))?);
        block.attn.c_proj.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.attn.c_proj.weight", i
        ))?);

        if has_ve(i, gpt.config.n_layer) &&
            let Some(ref mut gate_linear) = block.attn.ve_gate {
                gate_linear.weight = Param::from_tensor(load_transposed_param(&format!(
                    "transformer.h.{}.attn.ve_gate.weight", i
                ))?);
            }

        block.mlp.c_fc.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.mlp.c_fc.weight", i
        ))?);
        block.mlp.c_proj.weight = Param::from_tensor(load_transposed_param(&format!(
            "transformer.h.{}.mlp.c_proj.weight", i
        ))?);
    }

    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            gpt.value_embeds[ve_cnt].weight =
                Param::from_tensor(load_param(&format!("value_embeds.{}.weight", i))?);
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

struct ParamSaver<B: Backend> {
    buffers: HashMap<String, Vec<u8>>,
    shapes: HashMap<String, Vec<usize>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> ParamSaver<B> {
    fn new() -> Self {
        Self { buffers: HashMap::new(), shapes: HashMap::new(), _phantom: PhantomData }
    }

    fn save_tensor<const D: usize>(&mut self, name: &str, tensor: Tensor<B, D>) {
        let f32_data = crate::common::tensor_data_to_f32_vec(tensor.clone().into_data());
        let shape = tensor.shape().dims::<D>().to_vec();
        let u8_data = bytemuck::cast_slice::<f32, u8>(&f32_data).to_vec();
        self.buffers.insert(name.to_string(), u8_data);
        self.shapes.insert(name.to_string(), shape);
    }

    fn save_param(&mut self, name: &str, tensor: Tensor<B, 2>) {
        self.save_tensor(name, tensor);
    }

    fn save_transposed_param(&mut self, name: &str, tensor: Tensor<B, 2>) {
        self.save_tensor(name, tensor.transpose());
    }

    fn save_vector(&mut self, name: &str, tensor: Tensor<B, 1>) {
        self.save_tensor(name, tensor);
    }
}

pub fn save_gpt_to_safetensors<B: Backend>(gpt: &Gpt<B>, path: &Path) -> Result<(), String> {
    let mut saver = ParamSaver::new();

    saver.save_param("transformer.wte.weight", gpt.wte.weight.val());
    saver.save_transposed_param("lm_head.weight", gpt.lm_head.weight.val());

    for i in 0..gpt.config.n_layer {
        let block = &gpt.h[i];

        saver.save_transposed_param(&format!("transformer.h.{}.attn.c_q.weight", i),
            block.attn.c_q.weight.val());
        saver.save_transposed_param(&format!("transformer.h.{}.attn.c_k.weight", i),
            block.attn.c_k.weight.val());
        saver.save_transposed_param(&format!("transformer.h.{}.attn.c_v.weight", i),
            block.attn.c_v.weight.val());
        saver.save_transposed_param(&format!("transformer.h.{}.attn.c_proj.weight", i),
            block.attn.c_proj.weight.val());

        if has_ve(i, gpt.config.n_layer) &&
            let Some(ref gate_linear) = block.attn.ve_gate {
                saver.save_transposed_param(
                    &format!("transformer.h.{}.attn.ve_gate.weight", i),
                    gate_linear.weight.val());
            }

        saver.save_transposed_param(&format!("transformer.h.{}.mlp.c_fc.weight", i),
            block.mlp.c_fc.weight.val());
        saver.save_transposed_param(&format!("transformer.h.{}.mlp.c_proj.weight", i),
            block.mlp.c_proj.weight.val());
    }

    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            saver.save_param(&format!("value_embeds.{}.weight", i),
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
    let mut tensors_map = std::collections::BTreeMap::new();
    for (name, buffer) in &saver.buffers {
        let shape = &saver.shapes[name];
        let view = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32,
            shape.clone(), buffer,
        ).map_err(|e| format!("Failed to create TensorView for '{}': {:?}", name, e))?;
        tensors_map.insert(name.clone(), view);
    }

    safetensors::tensor::serialize_to_file(&tensors_map, None, path)
        .map_err(|e| format!("Failed to serialize safetensors: {:?}", e))?;

    Ok(())
}

//#[cfg(test)] mod tests { use super::*;
#[cfg(all(test, feature = "ndarray"))]
use burn::prelude::ToElement;

    #[test] fn test_safetensors_roundtrip() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        let config = crate::gpt::GptConfig { sequence_len: 16, vocab_size: 32,
            n_layer: 2, n_head: 2, n_kv_head: 1, n_embd: 16,
            window_pattern: "L".to_string(), quantization: None,
        };

        // 1. Create a dummy source model
        let model_src = Gpt::<ModelBackend>::new(config.clone(), &device);

        // 2. Save it to a temporary safetensors file
        let path = Path::new("test_model.safetensors");
        save_gpt_to_safetensors(&model_src, path).expect("Failed to save safetensors");

        // 3. Create a target model and load the saved safetensors back
        let mut model_dst = Gpt::<ModelBackend>::new(config, &device);
        load_safetensors_to_gpt(&mut model_dst, path, &device)
            .expect("Failed to load safetensors");

        // 4. Assert that all weights are identical
        let diff_wte = (model_src.wte.weight.val() - model_dst.wte.weight.val())
            .abs().sum().into_scalar().to_f32();
        assert_eq!(diff_wte, 0.0, "wte weight mismatch");

        let diff_lm = (model_src.lm_head.weight.val() - model_dst.lm_head.weight.val())
            .abs().sum().into_scalar().to_f32();
        assert_eq!(diff_lm, 0.0, "lm_head weight mismatch");

        for i in 0..model_src.config.n_layer {
            let (src_block, dst_block) = (&model_src.h[i], &model_dst.h[i]);

            let diff_q = (src_block.attn.c_q.weight.val() - dst_block.attn.c_q.weight.val())
                .abs().sum().into_scalar().to_f32();
            assert_eq!(diff_q, 0.0, "c_q weight mismatch at layer {}", i);

            let diff_k = (src_block.attn.c_k.weight.val() - dst_block.attn.c_k.weight.val())
                .abs().sum().into_scalar().to_f32();
            assert_eq!(diff_k, 0.0, "c_k weight mismatch at layer {}", i);

            let diff_v = (src_block.attn.c_v.weight.val() - dst_block.attn.c_v.weight.val())
                .abs().sum().into_scalar().to_f32();
            assert_eq!(diff_v, 0.0, "c_v weight mismatch at layer {}", i);

            let diff_proj = (src_block.attn.c_proj.weight.val() -
                    dst_block.attn.c_proj.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_proj, 0.0, "c_proj weight mismatch at layer {}", i);

            let diff_fc = (src_block.mlp.c_fc.weight.val() - dst_block.mlp.c_fc.weight.val())
                .abs().sum().into_scalar().to_f32();
            assert_eq!(diff_fc, 0.0, "c_fc weight mismatch at layer {}", i);

            let diff_p = (src_block.mlp.c_proj.weight.val() - dst_block.mlp.c_proj.weight.val())
                .abs().sum().into_scalar().to_f32();
            assert_eq!(diff_p, 0.0, "c_proj weight mismatch at layer {}", i);
        }

        // Clean up temporary file
        std::fs::remove_file(path).ok();
    }
//}
