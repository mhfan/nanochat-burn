
use std::{env, io::{self, Write}, time::Instant};

use nanochat_burn::{artifact::{inference_artifact_path, load_artifact},
    common::{ModelBackend, init_device},
    engine::inference::{InferenceEngine, SamplingConfig},
    tokenizer::{Conversation, ConversationMessage, MessageContent},
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

    // Parse CLI parameters and environment variables for quantization.
    let args: Vec<_> = env::args().collect();
    let mut quantize_bits =
        env::var("NANOCHAT_QUANTIZE").ok().and_then(|v| v.parse().ok());
    let mut quantize_block = env::var("NANOCHAT_QUANTIZE_BLOCK").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0);

    if let Some(pos) = args.iter().position(|arg| arg == "--quantize") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
        quantize_bits = Some(val);
    }
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize-block") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
        quantize_block = val;
    }

    let device = init_device();
    let artifact_path = inference_artifact_path();
    let artifact = load_artifact::<ModelBackend>(&artifact_path, &device)
        .unwrap_or_else(|error| panic!("failed to load artifact {artifact_path:?}: {error}"));
    println!("Loaded {:?} artifact from {:?} (vocab {})", artifact.manifest.stage,
        artifact_path, artifact.tokenizer.get_vocab_size());
    let tokenizer = artifact.tokenizer;
    let gpt = if let Some(bits) = quantize_bits {
        println!("Dynamically quantizing model to INT{} (block_size = {})...",
            bits, quantize_block);
        artifact.model.quantize(bits, quantize_block)
    } else {
        artifact.model.into_linear_or_quantized()
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
        if io::stdin().read_line(&mut user_input).unwrap() == 0 { break; }
        let trimmed = user_input.trim();
        if trimmed.eq_ignore_ascii_case("quit") ||
            trimmed.eq_ignore_ascii_case("exit") { break; }
        if trimmed.is_empty() { continue; }

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
        let generation_budget = 64.min(engine.model.config.sequence_len.saturating_sub(1));
        let prompt_limit = engine.model.config.sequence_len - generation_budget;
        let (prompt_tokens, _) = tokenizer.render_conversation(&conversation, prompt_limit);

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

        for _ in 0..generation_budget {
            if state.completed[0] || state.step >= engine.model.config.sequence_len { break; }
            let (next_tokens, _, next_logits) =
                engine.step_generation(&mut state, cur_logits, sampling, &device);
            cur_logits = next_logits;

            let token = next_tokens[0];
            if token == special_tokens.assistant_end || token == special_tokens.bos { break; }
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
