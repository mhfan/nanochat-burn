
use std::{env, io::{self, Write}, time::Instant};

use nanochat_burn::{common::{ModelBackend, init_device},
    engine::inference::{InferenceEngine, SamplingConfig},
    gpt::{Gpt, GptConfig, QuantizationConfig},
    tokenizer::{BpeTokenizer, Conversation, ConversationMessage, MessageContent},
};

fn main() {
    use tracing_subscriber::EnvFilter; // Initialize logging
    let _ = tracing_subscriber::fmt().with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        ).try_init();

    println!("==================================================================");
    println!("   🔥 NanoChat Burn CLI Chat Client (WGPU Accelerated f16) 🔥    ");
    println!("==================================================================");
    println!("* Supporting multi-turn chat history");
    println!("* Built-in secure calculator tool-use state machine");
    println!("* Press Ctrl+C or type 'quit' / 'exit' to exit.");
    println!("==================================================================\n");

    // 1. Train a miniature tokenizer for mock CLI interaction
    // In a real SFT deployment, a pre-saved tokenizer JSON is loaded.
    let corpus = vec![
        "Hello! How can I help you today?",
        "The planets of the solar system are: \
            Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        "The capital of France is Paris.",
        "If 5*x + 3 = 13, then x is <|python_start|>(13 - 3) / \
            5<|python_end|><|output_start|>2<|output_end|>.",
        "System programming in Rust is extremely safe, concurrent, and high-performance.",
    ];
    let tokenizer = BpeTokenizer::train_from_iterator(corpus, 320);
    let vocab_size = tokenizer.get_vocab_size();
    println!("Vocabulary size: {}", vocab_size);

    // Parse CLI parameters and Environment Variables for quantization
    let args: Vec<String> = env::args().collect();
    let mut quantize_bits =
        env::var("NANOCHAT_QUANTIZE").ok().and_then(|v| v.parse::<usize>().ok());
    let mut quantize_block = env::var("NANOCHAT_QUANTIZE_BLOCK").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);

    if let Some(pos) = args.iter().position(|arg| arg == "--quantize") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse::<usize>().ok()) {
        quantize_bits = Some(val);
    }
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize-block") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse::<usize>().ok()) {
        quantize_block = val;
    }

    // 2. Initialize a small Gpt model on WGPU
    let mut config = GptConfig { sequence_len: 512, vocab_size, n_layer: 4, n_head: 4,
        n_kv_head: 2, n_embd: 128, window_pattern: "SSL".to_string(), quantization: None,
    };

    if let Some(bits) = quantize_bits {
        config.quantization = Some(QuantizationConfig { bits, block_size: quantize_block });
    }

    let device = init_device();
    println!("Constructing WGPU Transformer Model...");
    let gpt_fp: Gpt<ModelBackend> = Gpt::new(config.clone(), &device);

    let gpt = if let Some(q_config) = config.quantization {
        println!("Dynamically quantizing model to INT{} (block_size = {})...",
            q_config.bits, q_config.block_size);
        gpt_fp.quantize(q_config.bits, q_config.block_size)
    } else {
        gpt_fp.into_linear_or_quantized()
    };

    let engine = InferenceEngine::new(gpt, tokenizer.clone());
    let sampling = SamplingConfig {
        temperature: 0.7, top_k: Some(50), repetition_penalty: 1.2,
    };

    // Initialize conversation state
    let mut conversation = Conversation { messages: vec![] };
    let special_tokens = tokenizer.special_token_ids();

    loop {
        print!("\n\x1b[32m\x1b[1mUser >\x1b[0m ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let trimmed = user_input.trim();
        if  trimmed.eq_ignore_ascii_case("quit") ||
            trimmed.eq_ignore_ascii_case("exit") { break; }
        if  trimmed.is_empty() { continue; }

        // Add user message to conversation
        conversation.messages.push(ConversationMessage {
            content: MessageContent::Simple(trimmed.to_string()),
            role: "user".to_string(),
        });

        // Add dummy assistant response start
        conversation.messages.push(ConversationMessage {
            content: MessageContent::Simple(String::new()),
            role: "assistant".to_string(),
        });

        // Render multi-turn conversation into tokens
        let (prompt_tokens, _) = tokenizer.render_conversation(&conversation, 500);

        // Remove the trailing assistant end-of-text or bos tokens
        // so we generate from the prompt end
        let mut clean_prompt = prompt_tokens;
        if clean_prompt.last().is_some_and(|&last| {
            last == special_tokens.assistant_end || last == special_tokens.bos
        }) {
            clean_prompt.pop();
        }

        print!("\x1b[35m\x1b[1mAssistant >\x1b[0m ");
        io::stdout().flush().unwrap();

        let start_time = Instant::now();
        let (mut state, mut cur_logits) = engine.prefill(&clean_prompt, 1, &device);

        let (mut first_token, mut token_count) = (true, 0);
        let mut tft = start_time.elapsed().as_secs_f64();
        let mut assistant_response_tokens = Vec::new();

        for _ in 0..256 {
            if state.completed[0] || state.step >= engine.model.config.sequence_len { break; }
            let (next_tokens, _, next_logits) =
                engine.step_generation(&mut state, cur_logits, sampling, &device);
            cur_logits = next_logits;

            let token = next_tokens[0];
            assistant_response_tokens.push(token);
            token_count += 1;

            if first_token {
                tft = start_time.elapsed().as_secs_f64();
                first_token = false;
            }

            let text = tokenizer.decode(&[token]);
            print!("{}", text);
            io::stdout().flush().unwrap();
        }
        println!();

        let total_time = start_time.elapsed().as_secs_f64();
        let tok_per_sec = token_count as f64 / total_time;
        println!("\x1b[90m[Benchmark: TFT: {:.2}ms | Speed: {:.2} tok/sec | \
            Total generated: {} tokens]\x1b[0m", tft * 1000.0, tok_per_sec, token_count);

        // Save generated tokens back to conversation
        let response_text = tokenizer.decode(&assistant_response_tokens);
        if let Some(msg) = conversation.messages.last_mut() {
            msg.content = MessageContent::Simple(response_text);
        }
    }
}
