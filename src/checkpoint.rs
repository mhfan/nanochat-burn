
use safetensors::SafeTensors;
use std::{path::Path, collections::HashMap};
use burn::{module::Param, tensor::{Tensor, backend::Backend, TensorData, Shape}};
use crate::gpt::{Gpt, has_ve};

pub fn load_safetensors_to_gpt<B: Backend>(gpt: &mut Gpt<B>,
    path: &Path, device: &B::Device,) -> Result<(), String> {
    let file_data = std::fs::read(path).map_err(|e| format!("Failed to read safetensors file: {:?}", e))?;
    let tensors = SafeTensors::deserialize(&file_data).map_err(|e| format!("Failed to parse safetensors: {:?}", e))?;

    let get_f32_data = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
        let view = tensors.tensor(name).map_err(|e| format!("Tensor '{}' not found in safetensors: {:?}", name, e))?;
        let shape: Vec<usize> = view.shape().iter().map(|&x| x as usize).collect();
        let data = view.data();
        let f32_data = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
        Ok((f32_data, shape))
    };

    let (wte_data, wte_shape) = get_f32_data("transformer.wte.weight")?;
    let wte_tensor = Tensor::<B, 2>::from_data(TensorData::new(wte_data, Shape::from(wte_shape)), device);
    gpt.wte.weight = Param::from_tensor(wte_tensor);

    let (lm_data, lm_shape) = get_f32_data("lm_head.weight")?;
    let lm_tensor = Tensor::<B, 2>::from_data(TensorData::new(lm_data, Shape::from(lm_shape)), device).transpose();
    gpt.lm_head.weight = Param::from_tensor(lm_tensor);

    for i in 0..gpt.config.n_layer {
        let block = &mut gpt.h[i];

        let (q_data, q_shape) = get_f32_data(&format!("transformer.h.{}.attn.c_q.weight", i))?;
        block.attn.c_q.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(q_data, Shape::from(q_shape)), device).transpose());

        let (k_data, k_shape) = get_f32_data(&format!("transformer.h.{}.attn.c_k.weight", i))?;
        block.attn.c_k.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(k_data, Shape::from(k_shape)), device).transpose());

        let (v_data, v_shape) = get_f32_data(&format!("transformer.h.{}.attn.c_v.weight", i))?;
        block.attn.c_v.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(v_data, Shape::from(v_shape)), device).transpose());

        let (proj_data, proj_shape) = get_f32_data(&format!("transformer.h.{}.attn.c_proj.weight", i))?;
        block.attn.c_proj.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(proj_data, Shape::from(proj_shape)), device).transpose());

        if has_ve(i, gpt.config.n_layer) {
            let (gate_data, gate_shape) = get_f32_data(&format!("transformer.h.{}.attn.ve_gate.weight", i))?;
            if let Some(ref mut gate_linear) = block.attn.ve_gate {
                gate_linear.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(gate_data, Shape::from(gate_shape)), device).transpose());
            }
        }

        let (fc_data, fc_shape) = get_f32_data(&format!("transformer.h.{}.mlp.c_fc.weight", i))?;
        block.mlp.c_fc.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(fc_data, Shape::from(fc_shape)), device).transpose());

        let (p_data, p_shape) = get_f32_data(&format!("transformer.h.{}.mlp.c_proj.weight", i))?;
        block.mlp.c_proj.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(p_data, Shape::from(p_shape)), device).transpose());
    }

    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            let (ve_data, ve_shape) = get_f32_data(&format!("value_embeds.{}.weight", i))?;
            gpt.value_embeds[ve_cnt].weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(ve_data, Shape::from(ve_shape)), device));
            ve_cnt += 1;
        }
    }

    let (res_data, res_shape) = get_f32_data("resid_lambdas")?;
    gpt.resid_lambdas = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(res_data, Shape::from(res_shape)), device));

    let (x0_data, x0_shape) = get_f32_data("x0_lambdas")?;
    gpt.x0_lambdas = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(x0_data, Shape::from(x0_shape)), device));

    let (smear_gate_data, smear_gate_shape) = get_f32_data("smear_gate.weight")?;
    gpt.smear_gate.weight = Param::from_tensor(Tensor::<B, 2>::from_data(TensorData::new(smear_gate_data, Shape::from(smear_gate_shape)), device).transpose());

    let (smear_lam_data, smear_lam_shape) = get_f32_data("smear_lambda")?;
    gpt.smear_lambda = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(smear_lam_data, Shape::from(smear_lam_shape)), device));

    let (back_data, back_shape) = get_f32_data("backout_lambda")?;
    gpt.backout_lambda = Param::from_tensor(Tensor::<B, 1>::from_data(TensorData::new(back_data, Shape::from(back_shape)), device));

    Ok(())
}

fn save_tensor<B: Backend, const D: usize>(name: &str, tensor: Tensor<B, D>,
    buffers: &mut HashMap<String, Vec<u8>>, shapes: &mut HashMap<String, Vec<usize>>,) {
    let f32_data = crate::common::tensor_data_to_f32_vec(tensor.clone().into_data());
    let shape = tensor.shape().dims::<D>().to_vec();
    let mut u8_data = Vec::with_capacity(f32_data.len() * 4);
    for &val in &f32_data {
        u8_data.extend_from_slice(&val.to_le_bytes());
    }
    buffers.insert(name.to_string(), u8_data);
    shapes.insert(name.to_string(), shape);
}

pub fn save_gpt_to_safetensors<B: Backend>(gpt: &Gpt<B>, path: &Path,) -> Result<(), String> {
    let (mut buffers, mut shapes) = (HashMap::new(), HashMap::new());

    // 1. Embedding weight
    save_tensor("transformer.wte.weight", gpt.wte.weight.val(), &mut buffers, &mut shapes);

    // 2. LM Head (Transposed back to [O, I])
    save_tensor("lm_head.weight", gpt.lm_head.weight.val().transpose(), &mut buffers, &mut shapes);

    // 3. Blocks
    for i in 0..gpt.config.n_layer {
        let block = &gpt.h[i];

        save_tensor(&format!("transformer.h.{}.attn.c_q.weight", i), block.attn.c_q.weight.val().transpose(), &mut buffers, &mut shapes);
        save_tensor(&format!("transformer.h.{}.attn.c_k.weight", i), block.attn.c_k.weight.val().transpose(), &mut buffers, &mut shapes);
        save_tensor(&format!("transformer.h.{}.attn.c_v.weight", i), block.attn.c_v.weight.val().transpose(), &mut buffers, &mut shapes);
        save_tensor(&format!("transformer.h.{}.attn.c_proj.weight", i), block.attn.c_proj.weight.val().transpose(), &mut buffers, &mut shapes);

        if has_ve(i, gpt.config.n_layer) {
            if let Some(ref gate_linear) = block.attn.ve_gate {
                save_tensor(&format!("transformer.h.{}.attn.ve_gate.weight", i), gate_linear.weight.val().transpose(), &mut buffers, &mut shapes);
            }
        }

        save_tensor(&format!("transformer.h.{}.mlp.c_fc.weight", i), block.mlp.c_fc.weight.val().transpose(), &mut buffers, &mut shapes);
        save_tensor(&format!("transformer.h.{}.mlp.c_proj.weight", i), block.mlp.c_proj.weight.val().transpose(), &mut buffers, &mut shapes);
    }

    // 4. Value Embeddings
    let mut ve_cnt = 0;
    for i in 0..gpt.config.n_layer {
        if has_ve(i, gpt.config.n_layer) {
            save_tensor(&format!("value_embeds.{}.weight", i), gpt.value_embeds[ve_cnt].weight.val(), &mut buffers, &mut shapes);
            ve_cnt += 1;
        }
    }

    // 5. Global parameters
    save_tensor("resid_lambdas", gpt.resid_lambdas.val(), &mut buffers, &mut shapes);
    save_tensor("x0_lambdas", gpt.x0_lambdas.val(), &mut buffers, &mut shapes);
    save_tensor("smear_gate.weight", gpt.smear_gate.weight.val().transpose(), &mut buffers, &mut shapes);
    save_tensor("smear_lambda", gpt.smear_lambda.val(), &mut buffers, &mut shapes);
    save_tensor("backout_lambda", gpt.backout_lambda.val(), &mut buffers, &mut shapes);

    // 6. Serialize to BTreeMap of TensorViews and write to file
    let mut tensors_map = std::collections::BTreeMap::new();
    for (name, buffer) in &buffers {
        let shape = &shapes[name];
        let view = safetensors::tensor::TensorView::new(
            safetensors::tensor::Dtype::F32, shape.clone(), buffer,
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

        let config = crate::gpt::GptConfig { sequence_len: 16, vocab_size: 32, n_layer: 2, n_head: 2,
            n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(), quantization: None,
        };

        // 1. Create a dummy source model
        let model_src = Gpt::<ModelBackend>::new(config.clone(), &device);

        // 2. Save it to a temporary safetensors file
        let path = Path::new("test_model.safetensors");
        save_gpt_to_safetensors(&model_src, &path).expect("Failed to save safetensors");

        // 3. Create a target model and load the saved safetensors back
        let mut model_dst = Gpt::<ModelBackend>::new(config, &device);
        load_safetensors_to_gpt(&mut model_dst, &path, &device).expect("Failed to load safetensors");

        // 4. Assert that all weights are identical
        let diff_wte = (model_src.wte.weight.val() - model_dst.wte.weight.val()).abs().sum().into_scalar().to_f32();
        assert_eq!(diff_wte, 0.0, "wte weight mismatch");

        let diff_lm = (model_src.lm_head.weight.val() - model_dst.lm_head.weight.val()).abs().sum().into_scalar().to_f32();
        assert_eq!(diff_lm, 0.0, "lm_head weight mismatch");

        for i in 0..model_src.config.n_layer {
            let (src_block, dst_block) = (&model_src.h[i], &model_dst.h[i]);

            let diff_q = (src_block.attn.c_q.weight.val() - dst_block.attn.c_q.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_q, 0.0, "c_q weight mismatch at layer {}", i);

            let diff_k = (src_block.attn.c_k.weight.val() - dst_block.attn.c_k.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_k, 0.0, "c_k weight mismatch at layer {}", i);

            let diff_v = (src_block.attn.c_v.weight.val() - dst_block.attn.c_v.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_v, 0.0, "c_v weight mismatch at layer {}", i);

            let diff_proj = (src_block.attn.c_proj.weight.val() - dst_block.attn.c_proj.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_proj, 0.0, "c_proj weight mismatch at layer {}", i);

            let diff_fc = (src_block.mlp.c_fc.weight.val() - dst_block.mlp.c_fc.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_fc, 0.0, "c_fc weight mismatch at layer {}", i);

            let diff_p = (src_block.mlp.c_proj.weight.val() - dst_block.mlp.c_proj.weight.val()).abs().sum().into_scalar().to_f32();
            assert_eq!(diff_p, 0.0, "c_proj weight mismatch at layer {}", i);
        }

        // Clean up temporary file
        std::fs::remove_file(path).ok();
    }
//}
