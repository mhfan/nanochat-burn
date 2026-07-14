
use std::{convert::Infallible, env, sync::Arc};

use axum::{Json, Router, http::StatusCode, routing::{get, post},
    response::{Html, IntoResponse, sse::{Event, KeepAlive, Sse}},
};
use nanochat_burn::{
    artifact::{inference_artifact_path, load_artifact},
    common::{ModelBackend, ModelDevice, init_device},
    engine::{inference::{InferenceEngine, SamplingConfig}, quant::LinearOrQuantized},
    tokenizer::{Conversation, ConversationMessage, MessageContent},
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

fn validate_request(request: &ChatRequest) -> Result<(), String> {
    if request.messages.is_empty() { return Err("messages must not be empty".to_string()); }
    let offset = usize::from(request.messages[0].role == "system");
    if offset == request.messages.len() {
        return Err("system message must be followed by a user message".to_string());
    }
    if offset == 1 && matches!(request.messages[0].content, MessageContent::Parts(_)) {
        return Err("system message cannot use multipart content".to_string());
    }
    for (index, message) in request.messages.iter().enumerate().skip(offset) {
        let expected = if (index - offset) % 2 == 0 { "user" } else { "assistant" };
        if message.role != expected {
            return Err(format!("message {index} must have role '{expected}'"));
        }
        if message.role != "assistant" && matches!(message.content, MessageContent::Parts(_)) {
            return Err(format!("message {index} cannot use multipart content"));
        }
        if let MessageContent::Parts(parts) = &message.content &&
            parts.iter().any(|part| !matches!(part.part_type.as_str(),
                "text" | "python" | "python_output")) {
            return Err(format!("message {index} contains an unknown part type"));
        }
    }
    if request.messages.last().is_none_or(|message| message.role != "user") {
        return Err("conversation must end with a user message".to_string());
    }
    if request.temperature.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err("temperature must be finite and non-negative".to_string());
    }
    if request.top_k == Some(0) { return Err("top_k must be greater than zero".to_string()); }
    Ok(())
}

// Custom Stream wrapper to avoid tokio-stream dependency
struct SseStream { rx: mpsc::Receiver<Result<Event, Infallible>>, }

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

    let device = init_device();
    let artifact_path = inference_artifact_path();
    let artifact = load_artifact::<ModelBackend>(&artifact_path, &device)
        .unwrap_or_else(|error| panic!("failed to load artifact {artifact_path:?}: {error}"));
    println!("Loaded {:?} artifact from {:?}", artifact.manifest.stage, artifact_path);
    let tokenizer = artifact.tokenizer;
    let gpt = if let Some(bits) = quantize_bits {
        println!("Dynamically quantizing web server model to INT{} (block = {})...",
            bits, quantize_block);
        artifact.model.quantize(bits, quantize_block)
    } else {
        artifact.model.into_linear_or_quantized()
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
    Json(payload): Json<ChatRequest>)
    -> Result<impl IntoResponse, (StatusCode, String)> {
    validate_request(&payload).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
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
        let max_tok = payload.max_tokens.unwrap_or(256)
            .min(engine_ref.engine.model.config.sequence_len.saturating_sub(1));
        let prompt_limit = engine_ref.engine.model.config.sequence_len - max_tok;
        let (prompt_tokens, _) = tokenizer.render_conversation(&conversation, prompt_limit);

        let special_tokens = tokenizer.special_token_ids();
        let mut clean_prompt = prompt_tokens;
        if clean_prompt.last().is_some_and(|&last| {
            last == special_tokens.assistant_end || last == special_tokens.bos
        }) { clean_prompt.pop(); }

        let top_k = payload.top_k.or(Some(50));
        let temp = payload.temperature.unwrap_or(0.7);
        let sampling = SamplingConfig { temperature: temp, top_k, repetition_penalty: 1.2, };

        let (mut gen_state, mut cur_logits) =
            engine_ref.engine.prefill(&clean_prompt, 1, &engine_ref.device);

        for _ in 0..max_tok {
            if gen_state.completed[0] ||
                gen_state.step >= engine_ref.engine.model.config.sequence_len { break; }
            let (next_tokens, _, next_logits) = engine_ref.engine.step_generation(&mut gen_state,
                cur_logits, sampling, &engine_ref.device);
            cur_logits = next_logits;
            let token = next_tokens[0];
            if token == special_tokens.assistant_end || token == special_tokens.bos { break; }
            let text = tokenizer.decode(&[token]);

            let event_data = serde_json::json!({ "token": text });
            let event = Event::default().json_data(event_data).unwrap();

            if tx.send(Ok(event)).await.is_err() { break; }
        }
    });

    Ok(Sse::new(SseStream { rx }).keep_alive(KeepAlive::default()))
}
