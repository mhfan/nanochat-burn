
fn main() {
    // Initialize tracing subscriber
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    use nanochat_burn::{artifact::{inference_artifact_path, load_artifact},
        common::{ModelBackend, init_device},
    };

    let device = init_device();
    let artifact_path = inference_artifact_path();
    let artifact = load_artifact::<ModelBackend>(&artifact_path, &device)
        .unwrap_or_else(|error| panic!("failed to load artifact {artifact_path:?}: {error}"));
    tracing::info!("Evaluating {:?} artifact from {:?}", artifact.manifest.stage, artifact_path);
    nanochat_burn::engine::eval::run_all_evaluations(
        &artifact.model, &artifact.tokenizer, &device);
}
