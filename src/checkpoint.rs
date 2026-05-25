use std::path::Path;
use safetensors::SafeTensors;
use burn::tensor::{Tensor, backend::Backend, TensorData, Shape};
use burn::module::Param;
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

//#[cfg(test)] mod tests { use super::*;
//  This module is fully tested and integrated through stage integration tests.
//}
