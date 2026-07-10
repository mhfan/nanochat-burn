
use std::{convert::Infallible, env, sync::Arc};

use axum::{Json, Router, routing::{get, post},
    response::{Html, IntoResponse, sse::{Event, KeepAlive, Sse}},
};
use nanochat_burn::{
    common::{ModelBackend, ModelDevice, init_device},
    engine::{inference::InferenceEngine, quant::LinearOrQuantized},
    gpt::{Gpt, GptConfig, QuantizationConfig},
    tokenizer::{BpeTokenizer, Conversation, ConversationMessage, MessageContent},
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ConversationMessage>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    device: String,
}

// Custom Stream wrapper to avoid tokio-stream dependency
struct SseStream {
    rx: mpsc::Receiver<Result<Event, Infallible>>,
}

impl futures_core::Stream for SseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>)
        -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

// Shared engine state wrapper
struct AppState {
    device: ModelDevice,
    engine: InferenceEngine<ModelBackend, LinearOrQuantized<ModelBackend>>,
}

#[tokio::main] async fn main() {
    // Initialize logging
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt().with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        ).try_init();

    // 1. Train BpeTokenizer
    let corpus = vec![
        "Hello! How can I help you today?",
        "The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        "The capital of France is Paris.",
        "If 5*x + 3 = 13, then x is <|python_start|>(13 - 3) / 5<|python_end|><|output_start|>2<|output_end|>.",
        "System programming in Rust is extremely safe, concurrent, and high-performance.",
    ];
    let tokenizer = BpeTokenizer::train_from_iterator(corpus, 320);
    let vocab_size = tokenizer.get_vocab_size();

    // Parse CLI parameters and Environment Variables for quantization
    let mut quantize_bits =
        env::var("NANOCHAT_QUANTIZE").ok().and_then(|v| v.parse::<usize>().ok());
    let mut quantize_block = env::var("NANOCHAT_QUANTIZE_BLOCK").ok()
        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);

    let args: Vec<String> = env::args().collect();
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize") {
        if let Some(val) = args.get(pos + 1).and_then(|s| s.parse::<usize>().ok()) {
            quantize_bits = Some(val);
        }
    }
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize-block") {
        if let Some(val) = args.get(pos + 1).and_then(|s| s.parse::<usize>().ok()) {
            quantize_block = val;
        }
    }

    // 2. Initialize GPT on WGPU
    let mut config = GptConfig { sequence_len: 512, vocab_size, n_layer: 4, n_head: 4,
        n_kv_head: 2, n_embd: 128, window_pattern: "SSL".to_string(), quantization: None,
    };

    if let Some(bits) = quantize_bits {
        config.quantization = Some(QuantizationConfig { bits, block_size: quantize_block });
    }

    let device = init_device();
    let gpt_fp: Gpt<ModelBackend> = Gpt::new(config.clone(), &device);

    let gpt = if let Some(q_config) = config.quantization {
        println!("Dynamically quantizing web server model to INT{} (block = {})...",
            q_config.bits, q_config.block_size);
        gpt_fp.quantize(q_config.bits, q_config.block_size)
    } else {
        gpt_fp.into_linear_or_quantized()
    };

    let engine = InferenceEngine::new(gpt, tokenizer);
    let shared_state = Arc::new(AppState { engine, device });

    // 3. Build Axum Router
    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/health", get(health_check))
        .route("/chat/completions", post(chat_completions))
        .layer(axum::Extension(shared_state));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await.unwrap();
    println!("============================================================");
    println!("  🚀 Server running at: http://127.0.0.1:8080 🚀            ");
    println!("============================================================");
    axum::serve(listener, app).await.unwrap();
}

async fn serve_ui() -> impl IntoResponse { Html(include_str!("../../../nanochat/ui.html")) }

async fn health_check(axum::Extension(state): axum::Extension<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse { device: format!("{:?}", state.device), status: "ok" })
}

async fn chat_completions(axum::Extension(state): axum::Extension<Arc<AppState>>,
    Json(payload): Json<ChatRequest>) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel(100);
    let engine_ref = state.clone();

    tokio::spawn(async move {
        let mut conversation = Conversation { messages: payload.messages };

        // Add dummy assistant response to construct correct SFT prompt
        conversation.messages.push(ConversationMessage {
            content: MessageContent::Simple(String::new()),
            role: "assistant".to_string(),
        });

        let tokenizer = &engine_ref.engine.tokenizer;
        let (prompt_tokens, _) = tokenizer.render_conversation(&conversation, 500);

        let special_tokens = tokenizer.special_token_ids();
        let mut clean_prompt = prompt_tokens;
        if clean_prompt.last().is_some_and(|&last| {
            last == special_tokens.assistant_end || last == special_tokens.bos
        }) { clean_prompt.pop(); }

        let top_k = payload.top_k.or(Some(50));
        let temp = payload.temperature.unwrap_or(0.7);
        let max_tok = payload.max_tokens.unwrap_or(256);

        let (mut gen_state, mut cur_logits) =
            engine_ref.engine.prefill(&clean_prompt, 1, &engine_ref.device);

        for _ in 0..max_tok {
            if gen_state.completed[0] { break; }
            let (next_tokens, _, next_logits) = engine_ref.engine.step_generation(
                &mut gen_state, cur_logits, temp, top_k, 1.2, &engine_ref.device,
            );
            cur_logits = next_logits;
            let text = tokenizer.decode(&[next_tokens[0]]);

            let event_data = serde_json::json!({ "token": text });
            let event = Event::default().json_data(event_data).unwrap();

            if tx.send(Ok(event)).await.is_err() { break; }
        }
    });

    Sse::new(SseStream { rx }).keep_alive(KeepAlive::default())
}
