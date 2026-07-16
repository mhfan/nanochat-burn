use std::{path::Path, rc::Rc};

use burn::{store::{BurnToPyTorchAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot,
        PyTorchToBurnAdapter, SafetensorsStore, TensorSnapshot},
    tensor::{DType, Element, backend::Backend}};

use crate::gpt::{Gpt, has_ve};

#[derive(Clone)]
struct FloatPrecisionAdapter(DType);

impl ModuleAdapter for FloatPrecisionAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype == self.0 ||
            !matches!(snapshot.dtype, DType::F64 | DType::F32 | DType::Flex32 |
                DType::F16 | DType::BF16) {
            return snapshot.clone();
        }
        let data = snapshot.clone_data_fn();
        let dtype = self.0;
        TensorSnapshot::from_closure(Rc::new(move ||
                data().map(|data| data.convert_dtype(dtype))),
            dtype, snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default())
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> { Box::new(self.clone()) }
}

fn load_remapper() -> KeyRemapper {
    KeyRemapper::new()
        .add_pattern(r"^transformer\.wte\.", "wte.").expect("valid checkpoint key pattern")
        .add_pattern(r"^transformer\.h\.", "h.").expect("valid checkpoint key pattern")
}

fn save_remapper(n_layer: usize) -> KeyRemapper {
    let mut remapper = KeyRemapper::new()
        .add_pattern(r"^wte\.", "transformer.wte.").expect("valid checkpoint key pattern")
        .add_pattern(r"^h\.", "transformer.h.").expect("valid checkpoint key pattern");
    for (value_index, layer) in (0..n_layer).filter(|&layer| has_ve(layer, n_layer)).enumerate() {
        remapper = remapper.add_pattern(
            format!(r"^value_embeds\.{value_index}\."), format!("__value_embeds.{layer}.")
        ).expect("valid value embedding key pattern");
    }
    remapper.add_pattern(r"^__value_embeds\.", "value_embeds.")
        .expect("valid checkpoint key pattern")
}

pub fn load_safetensors_to_gpt<B: Backend>(gpt: &mut Gpt<B>, path: &Path, _device: &B::Device)
    -> Result<(), String> {
    let mut store = SafetensorsStore::from_file(path)
        .remap(load_remapper()).map_indices_contiguous(true)
        .with_from_adapter(PyTorchToBurnAdapter.chain(
            FloatPrecisionAdapter(B::FloatElem::dtype())))
        .skip_enum_variants(true);
    let mut loaded = gpt.clone();
    loaded.load_from(&mut store).map_err(|error| error.to_string())?;
    *gpt = loaded;
    Ok(())
}

pub fn save_gpt_to_safetensors<B: Backend>(gpt: &Gpt<B>, path: &Path) -> Result<(), String> {
    let mut store = SafetensorsStore::from_file(path)
        .remap(save_remapper(gpt.config.n_layer))
        .with_to_adapter(BurnToPyTorchAdapter.chain(FloatPrecisionAdapter(DType::F32)))
        .skip_enum_variants(true)
        .clear_metadata().overwrite(true);
    gpt.save_into(&mut store).map_err(|error| error.to_string())
}

#[cfg(test)] mod tests { use super::*;
    use burn::tensor::{Int, Tensor};

    #[test] fn test_safetensors_roundtrip() {
        let device = crate::common::init_device();
        use crate::common::ModelBackend;

        let config = crate::gpt::GptConfig { sequence_len: 16, vocab_size: 32, n_layer: 2,
            n_head: 2, n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(),
            features: Default::default(), quantization: None,
        };
        let source = Gpt::new(config.clone(), &device);
        let path = std::env::temp_dir().join(format!(
            "nanochat-test-model-{}.safetensors", std::process::id()));
        save_gpt_to_safetensors(&source, &path).unwrap();

        let tensors = std::fs::read(&path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&tensors).unwrap();
        assert_eq!(tensors.tensor("transformer.h.0.attn.c_q.weight").unwrap().dtype(),
            safetensors::Dtype::F32);
        assert!(tensors.tensor("value_embeds.1.weight").is_ok());

        let mut restored = Gpt::new(config, &device);
        load_safetensors_to_gpt(&mut restored, &path, &device).unwrap();
        assert_eq!(crate::common::tensor_data_to_f32_vec(source.wte.weight.val().into_data()),
            crate::common::tensor_data_to_f32_vec(restored.wte.weight.val().into_data()));
        assert_eq!(crate::common::tensor_data_to_f32_vec(
                source.h[0].attn.c_q.weight.val().into_data()),
            crate::common::tensor_data_to_f32_vec(
                restored.h[0].attn.c_q.weight.val().into_data()));
        assert_eq!(crate::common::tensor_data_to_f32_vec(
                source.value_embeds[0].weight.val().into_data()),
            crate::common::tensor_data_to_f32_vec(
                restored.value_embeds[0].weight.val().into_data()));
        let tokens = Tensor::<ModelBackend, 2, Int>::from_data([[1, 2, 3, 4]], &device);
        let error = crate::common::scalar_to_f32(
            (source.forward(tokens.clone(), None) - restored.forward(tokens, None))
                .abs().max().into_scalar());
        assert!(error <= 5e-5, "checkpoint roundtrip max logit error: {error}");
        std::fs::remove_file(path).ok();
    }
}
