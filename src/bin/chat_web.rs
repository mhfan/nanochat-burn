
use std::{collections::BTreeMap, convert::Infallible, env, sync::Arc};

use axum::{Json, Router, http::StatusCode, routing::{get, post},
    response::{Html, IntoResponse, sse::{Event, KeepAlive, Sse}},
};
use nanochat_burn::{
    artifact::{inference_artifact_path, load_artifact},
    common::{ModelBackend, init_device},
    engine::{inference::{GenerationConfig, InferenceEngine, SamplingConfig},
        scheduler::RequestId, serving::DynamicGenerationEngine},
    experiment::ArtifactPaths,
    gpt::quant::LinearOrQuantized,
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

struct AppState {
    device: String,
    tokenizer: BpeTokenizer,
    sequence_len: usize,
    jobs: mpsc::Sender<GenerationJob>,
}

struct GenerationJob {
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    events: mpsc::Sender<Result<Event, Infallible>>,
}

type WebGenerationEngine =
    DynamicGenerationEngine<ModelBackend, LinearOrQuantized<ModelBackend>>;

#[tokio::main] async fn main() {
    // Initialize logging
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt().with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        ).try_init();

    // Parse CLI parameters and Environment Variables for quantization
    let mut quantize_bits =
        env::var("NANOCHAT_QUANTIZE").ok().and_then(|v| v.parse().ok());
    let mut quantize_block = env::var("NANOCHAT_QUANTIZE_BLOCK").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0);

    let args: Vec<_> = env::args().collect();
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
        quantize_bits = Some(val);
    }
    if let Some(pos) = args.iter().position(|arg| arg == "--quantize-block") &&
        let Some(val) = args.get(pos + 1).and_then(|s| s.parse().ok()) {
        quantize_block = val;
    }

    let device = init_device();
    let artifact_path = inference_artifact_path(&ArtifactPaths::default());
    let artifact = load_artifact(&artifact_path, &device)
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

    let engine = InferenceEngine::new(gpt, tokenizer.clone());
    let sequence_len = engine.model.config.sequence_len;
    let max_batch: usize = env::var("NANOCHAT_MAX_BATCH").ok()
        .and_then(|value| value.parse().ok()).filter(|&capacity| capacity > 0).unwrap_or(8);
    let (jobs, job_rx) = mpsc::channel(max_batch.saturating_mul(4));
    let device_name = format!("{device:?}");
    let generation = DynamicGenerationEngine::new(engine, device, max_batch);
    tokio::spawn(generation_worker(generation, job_rx));
    let shared_state = Arc::new(AppState {
        device: device_name, tokenizer, sequence_len, jobs,
    });

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

async fn serve_ui() -> impl IntoResponse { Html(include_str!("../../data/assets/ui.html")) }

async fn health_check(axum::Extension(state): axum::Extension<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse { device: state.device.clone(), status: "ok" })
}

async fn generation_worker(mut generation: WebGenerationEngine,
    mut jobs: mpsc::Receiver<GenerationJob>) {
    let mut outputs = BTreeMap::new();
    loop {
        if generation.active_len() == 0 && generation.waiting_len() == 0 {
            let Some(job) = jobs.recv().await else { break; };
            submit_job(&mut generation, &mut outputs, job);
        }
        while let Ok(job) = jobs.try_recv() {
            submit_job(&mut generation, &mut outputs, job);
        }

        let cancelled: Vec<_> = outputs.iter()
            .filter_map(|(&id, sender)| sender.is_closed().then_some(id)).collect();
        for id in cancelled {
            generation.cancel(id);
            outputs.remove(&id);
        }

        for step in generation.step() {
            let sender = outputs.get(&step.request_id).cloned();
            let send_failed = if let (Some(token), Some(sender)) = (step.token, sender) {
                let text = generation.tokenizer().decode(&[token]);
                let event = Event::default()
                    .json_data(serde_json::json!({ "token": text })).unwrap();
                sender.try_send(Ok(event)).is_err()
            } else { false };
            if send_failed {
                generation.cancel(step.request_id);
                outputs.remove(&step.request_id);
            } else if step.finish_reason.is_some() {
                outputs.remove(&step.request_id);
            }
        }
        tokio::task::yield_now().await;
    }
}

fn submit_job(generation: &mut WebGenerationEngine,
    outputs: &mut BTreeMap<RequestId, mpsc::Sender<Result<Event, Infallible>>>, job: GenerationJob) {
    let id = generation.submit(job.prompt_tokens, job.config);
    outputs.insert(id, job.events);
}

async fn chat_completions(axum::Extension(state): axum::Extension<Arc<AppState>>,
    Json(payload): Json<ChatRequest>)
    -> Result<impl IntoResponse, (StatusCode, String)> {
    validate_request(&payload).map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let (tx, rx) = mpsc::channel(100);
    let mut conversation = Conversation { messages: payload.messages };
    conversation.messages.push(ConversationMessage {
        content: MessageContent::Simple(String::new()), role: "assistant".to_string(),
    });

    let max_tokens = payload.max_tokens.unwrap_or(256)
        .min(state.sequence_len.saturating_sub(1));
    let (prompt_tokens, _) = state.tokenizer.render_conversation(
        &conversation, state.sequence_len - max_tokens);
    let special = state.tokenizer.special_token_ids();
    let mut prompt_tokens = prompt_tokens;
    if prompt_tokens.last().is_some_and(|&token| {
        token == special.assistant_end || token == special.bos
    }) { prompt_tokens.pop(); }
    if prompt_tokens.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "rendered prompt is empty".to_string()));
    }

    let sampling = SamplingConfig { temperature: payload.temperature.unwrap_or(0.7),
        top_k: payload.top_k.or(Some(50)), repetition_penalty: 1.2 };
    let config = GenerationConfig { max_tokens, sampling, seed: 42 };
    state.jobs.send(GenerationJob { prompt_tokens, config, events: tx }).await
        .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE,
            "generation worker is unavailable".to_string()))?;

    Ok(Sse::new(SseStream { rx }).keep_alive(KeepAlive::default()))
}
