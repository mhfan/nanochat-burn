use super::*;

#[test] fn test_gpt_forward_and_loss() {
    let device = crate::common::init_device();
    let config = GptConfig { sequence_len: 32, vocab_size: 280, n_layer: 1, n_head: 2,
        n_kv_head: 1, n_embd: 16, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };

    use crate::common::ModelAutodiffBackend;
    let gpt: Gpt<ModelAutodiffBackend> = Gpt::new(config, &device);

    let idx = Tensor::<ModelAutodiffBackend, 2, Int>::zeros([2, 16], &device);
    let targets = Tensor::<ModelAutodiffBackend, 2, Int>::zeros([2, 16], &device);

    let logits = gpt.forward(idx, None);
    assert_eq!(logits.shape().dims(), [2, 16, 280]);

    let loss = gpt.compute_loss(logits, targets);
    let loss_val = loss.clone().into_scalar();
    assert!(crate::common::scalar_to_f32(loss_val) >= 0.0);

    let _grads = loss.backward();
}

use crate::common::ModelBackend;
type Int2DModelTensor = Tensor<ModelBackend, 2, Int>;

#[test] fn test_burn_attention_paths_match_reference() {
    let device = crate::common::init_device();
    let (batch_size, n_head, sequence_len, head_dim) = (2, 4, 32, 16);
    let q = Tensor::<ModelBackend, 4>::random([batch_size, n_head, sequence_len, head_dim],
        Distribution::Normal(0.0, 1.0), &device);
    let k = Tensor::<ModelBackend, 4>::random([batch_size, n_head, sequence_len, head_dim],
        Distribution::Normal(0.0, 1.0), &device);
    let v = Tensor::<ModelBackend, 4>::random([batch_size, n_head, sequence_len, head_dim],
        Distribution::Normal(0.0, 1.0), &device);
    let mask = precompute_window_mask::<ModelBackend>(-1, sequence_len, &device);
    let reference = scaled_dot_product_attention_reference(
        q.clone(), k.clone(), v.clone(), mask.clone());
    let masked = scaled_dot_product_attention_burn(
        q.clone(), k.clone(), v.clone(), Some(mask), false);
    let causal = scaled_dot_product_attention_burn(q, k, v, None, true);
    let masked_error = crate::common::scalar_to_f32(
        (reference.clone() - masked).abs().max().into_scalar());
    let causal_error = crate::common::scalar_to_f32(
        (reference - causal).abs().max().into_scalar());
    let tolerance = if cfg!(feature = "ndarray") { 5e-5 } else { 5e-3 };
    assert!(masked_error <= tolerance,
        "masked Burn attention max error {masked_error} exceeds {tolerance}");
    assert!(causal_error <= tolerance,
        "causal Burn attention max error {causal_error} exceeds {tolerance}");
}

#[test] fn test_gpt_config_validation() {
    let mut config = GptConfig { sequence_len: 16, vocab_size: 280, n_layer: 1, n_head: 4,
        n_kv_head: 1, n_embd: 32, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };
    assert!(config.validate().is_ok());
    config.n_kv_head = 3;
    assert_eq!(config.validate(), Err("n_kv_head must divide n_head"));

    config.n_kv_head = 1;
    config.features.gqa = false;
    assert_eq!(config.validate(),
        Err("disabling GQA requires n_kv_head to equal n_head"));
    config.n_kv_head = config.n_head;
    assert!(config.validate().is_ok());
    config.features.swa = false;
    assert_eq!(config.compute_window_sizes(), vec![-1]);
}

#[test] fn test_relu_squared_ablation_changes_mlp_output() {
    let device = crate::common::init_device();
    let fc = linear(Tensor::<ModelBackend, 2>::from_data(
        [[1.0, 0.0], [0.0, 1.0]], &device));
    let projection = linear(Tensor::<ModelBackend, 2>::from_data(
        [[1.0, 0.0], [0.0, 1.0]], &device));
    let input = Tensor::<ModelBackend, 3>::from_data([[[2.0, 3.0]]], &device);
    let relu = MLP { c_fc: fc.clone(), c_proj: projection.clone(),
        relu_squared: false, _phantom: PhantomData };
    let relu_squared = MLP { c_fc: fc, c_proj: projection,
        relu_squared: true, _phantom: PhantomData };

    assert_eq!(crate::common::tensor_data_to_f32_vec(relu.forward(input.clone()).into_data()),
        vec![2.0, 3.0]);
    assert_eq!(crate::common::tensor_data_to_f32_vec(
        relu_squared.forward(input).into_data()), vec![4.0, 9.0]);
}

#[test] fn test_cached_forward_matches_full_and_incremental_forward() {
    let device = crate::common::init_device();
    let config = GptConfig { sequence_len: 16, vocab_size: 280, n_layer: 1, n_head: 4,
        n_kv_head: 1, n_embd: 32, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };

    let mut gpt: Gpt<ModelBackend> = Gpt::new(config, &device);
    gpt.h[0].attn.c_proj = random_linear(32, 32, Distribution::Uniform(-0.1, 0.1), &device);
    gpt.smear_lambda = tensor_param(vec![1.0], &device);

    let head_dim = gpt.config.n_embd / gpt.config.n_head;
    let mut chunk_cache = KVCache::new_paged(
        1, 1, gpt.config.sequence_len, gpt.config.n_kv_head, head_dim, 2, &device);
    let mut incremental_cache = KVCache::new_paged(
        1, 1, gpt.config.sequence_len, gpt.config.n_kv_head, head_dim, 2, &device);

    let full_logits = gpt.forward(
            Int2DModelTensor::from_data([[12, 45, 67, 68, 69]], &device), None)
        .slice([0..1, 3..5, 0..gpt.config.vocab_size]);
    let prompt = Int2DModelTensor::from_data([[12, 45, 67]], &device);
    gpt.forward_with_cache(prompt.clone(), &mut chunk_cache, 0);
    gpt.forward_with_cache(prompt, &mut incremental_cache, 0);

    let chunk = Int2DModelTensor::from_data([[68, 69]], &device);
    let chunk_logits = gpt.forward_with_cache(chunk, &mut chunk_cache, 3);
    let first = gpt.forward_with_cache(Int2DModelTensor::from_data([[68]], &device),
        &mut incremental_cache, 3);
    let second = gpt.forward_with_cache(Int2DModelTensor::from_data([[69]], &device),
        &mut incremental_cache, 4);
    let incremental_logits = Tensor::cat(vec![first, second], 1);

    let incremental_diff = crate::common::scalar_to_f32(
        (chunk_logits - incremental_logits.clone()).abs().max().into_scalar());
    assert!(incremental_diff < 5e-5,
        "chunked cache logits differ from incremental logits by {incremental_diff}");

    let full_diff = crate::common::scalar_to_f32(
        (full_logits - incremental_logits).abs().max().into_scalar());
    assert!(full_diff < 5e-5,
        "cached logits differ from full forward logits by {full_diff}");
}

#[test] fn test_row_mapped_cache_batches_different_positions() {
    let device = crate::common::init_device();
    let config = GptConfig { sequence_len: 16, vocab_size: 280, n_layer: 1, n_head: 4,
        n_kv_head: 1, n_embd: 32, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };
    let mut gpt: Gpt<ModelBackend> = Gpt::new(config, &device);
    gpt.h[0].attn.c_proj = random_linear(32, 32, Distribution::Uniform(-0.1, 0.1), &device);
    gpt.smear_lambda = tensor_param(vec![1.0], &device);
    let head_dim = gpt.config.n_embd / gpt.config.n_head;
    let mut cache = KVCache::new_paged(1, 2, 16, 1, head_dim, 2, &device);

    gpt.forward_with_cache_rows(
        Int2DModelTensor::from_data([[12, 45, 67]], &device), &mut cache, &[0], &[0]);
    gpt.forward_with_cache_rows(
        Int2DModelTensor::from_data([[12, 45]], &device), &mut cache, &[1], &[0]);
    let batched = gpt.forward_with_cache_rows(
        Int2DModelTensor::from_data([[68], [67]], &device),
        &mut cache, &[0, 1], &[3, 2]);

    let expected_first = gpt.forward(
        Int2DModelTensor::from_data([[12, 45, 67, 68]], &device), None)
        .slice([0..1, 3..4, 0..gpt.config.vocab_size]);
    let expected_second = gpt.forward(
        Int2DModelTensor::from_data([[12, 45, 67]], &device), None)
        .slice([0..1, 2..3, 0..gpt.config.vocab_size]);
    let expected = Tensor::cat(vec![expected_first, expected_second], 0);
    let diff = crate::common::scalar_to_f32((batched - expected).abs().max().into_scalar());
    assert!(diff < 5e-5, "row-mapped cache logits differ from full forward by {diff}");
}

#[test] fn test_w4_quantization_keeps_unsupported_gate_layers_float() {
    let device = crate::common::init_device();
    let config = GptConfig { sequence_len: 8, vocab_size: 280, n_layer: 1, n_head: 4,
        n_kv_head: 1, n_embd: 64, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };

    let gpt = Gpt::<ModelBackend>::new(config, &device).quantize(4, 0);
    assert!(matches!(&gpt.h[0].attn.c_q, LinearOrQuantized::Quantized(_)));
    assert!(matches!(gpt.h[0].attn.ve_gate.as_ref(), Some(LinearOrQuantized::Standard(_))));
    assert!(matches!(&gpt.smear_gate, LinearOrQuantized::Standard(_)));

    let logits = gpt.forward(Int2DModelTensor::zeros([1, 4], &device), None);
    assert_eq!(logits.shape().dims(), [1, 4, 280]);
}

#[test] fn test_paged_attention_roundtrip() {
    let device = crate::common::init_device();
    let config = GptConfig { sequence_len: 16, vocab_size: 280, n_layer: 1, n_head: 4,
        n_kv_head: 1, n_embd: 32, window_pattern: "L".to_string(), quantization: None,
        features: Default::default(),
    };

    let gpt: Gpt<ModelBackend> = Gpt::new(config, &device);

    let prompt = [12, 45, 67];
    let (prompt_len, num_samples) = (prompt.len(), 1);

    let idx_data: Vec<_> = std::iter::repeat_n(prompt, num_samples).flatten().collect();

    // Prefill index tensor
    let prefill_idx = Int2DModelTensor::from_data(
        TensorData::new(idx_data, Shape::new([num_samples, prompt_len])), &device);

    let head_dim = gpt.config.n_embd / gpt.config.n_head;

    // Run prefill and a couple of autoregressive steps across page sizes.
    let (page_sizes, mut outputs) = (vec![2, 4], Vec::new());

    for &page_size in &page_sizes {
        let mut cache = KVCache::new_paged(
            gpt.config.n_layer, num_samples, gpt.config.sequence_len, gpt.config.n_kv_head,
            head_dim, page_size, &device,
        );

        // 1. Prefill
        let logits = gpt.forward_with_cache(prefill_idx.clone(), &mut cache, 0);
        let mut step_logits = vec![logits.clone()];

        // 2. Autoregressive steps
        let mut current_token = Int2DModelTensor::from_data(
            TensorData::new(vec![68i32; num_samples], Shape::new([num_samples, 1])), &device);

        for step_idx in 0..2 {
            let step = prompt_len + step_idx;
            let logits_step = gpt.forward_with_cache(current_token.clone(), &mut cache, step);
            step_logits.push(logits_step.clone());

            current_token = Int2DModelTensor::from_data(
                TensorData::new(vec![69i32; num_samples], Shape::new([num_samples, 1])),
                &device,
            );
        }

        outputs.push(step_logits);
    }

    // Different page layouts may change floating-point reduction order on GPU.
    for step in 0..outputs[0].len() {
        let (logits_2, logits_4) = (&outputs[0][step], &outputs[1][step]);

        let max_error = crate::common::scalar_to_f32(
            (logits_2.clone() - logits_4.clone()).abs().max().into_scalar(),
        );

        assert!(max_error <= 5e-5,
            "page_size=2/4 logits differ by {max_error} at step {step}");
    }
}

#[test] fn test_page_allocator_reuses_released_pages() {
    let mut allocator = PageAllocator::new(3);
    let (first, second) = (allocator.allocate().unwrap(), allocator.allocate().unwrap());
    assert_eq!(allocator.available(), 1);
    allocator.release(first);
    assert_eq!(allocator.allocate(), Some(first));
    allocator.release(second);
    assert_eq!(allocator.available(), 2);
}

#[test] fn test_attention_sink_eviction_preserves_logical_pages() {
    let device = crate::common::init_device();
    let mut cache = KVCache::<ModelBackend>::new_paged(1, 1, 16, 1, 8, 2, &device);
    cache.ensure_pages(0, 7);
    cache.len = 16;
    cache.request_lens[0] = 16;
    cache.evict_for_attention_sinks(0, 2, 4);
    let table = cache.block_table(0);
    assert!(table[0].is_some());
    assert!(table[1..6].iter().all(Option::is_none));
    assert!(table[6..8].iter().all(Option::is_some));
    assert_eq!(cache.allocator.available(), 5);
}
