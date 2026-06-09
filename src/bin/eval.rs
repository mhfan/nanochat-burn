
fn main() {
    // Initialize tracing subscriber
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    use nanochat_burn::common::{ModelBackend, init_device};
    use nanochat_burn::tokenizer::BpeTokenizer;
    use nanochat_burn::{gpt::{Gpt, GptConfig}, dataset::SftDataset};

    let device = init_device();

    let sft_dataset_path = "data/sft_train.jsonl";
    let dataset = SftDataset::new(sft_dataset_path).expect("Failed to load SFT dataset");

    let corpus = dataset.get_corpus();
    let mut tokenizer = BpeTokenizer::train_from_iterator(corpus, 1024);
    tokenizer.build_inverse_mappings();

    let config = GptConfig { sequence_len: 256, n_layer: 4, n_head: 4,
        n_kv_head: 2, n_embd: 64, window_pattern: "L".to_string(),
        vocab_size: tokenizer.get_vocab_size(), quantization: None,
    };

    let model = Gpt::<ModelBackend>::new(config, &device);

    nanochat_burn::engine::eval::run_all_evaluations(&model, &tokenizer, &device);
}
