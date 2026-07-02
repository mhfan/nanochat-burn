
use nanochat_burn::{common::{ModelAutodiffBackend, init_device},
    engine::{pretrain::run_pretraining, rl::run_rl_training, sft::run_sft_training},
};

#[tokio::main] async fn main() {
    // Initialize logging subscriber
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 {
        args[1].trim_start_matches('-').to_lowercase()
    } else { "pretrain".to_string() };
    let device = init_device();

    tracing::info!("=============================================");
    tracing::info!("   Initializing nanochat-burn Training      ");
    tracing::info!("   Mode: {}                                 ", mode);
    tracing::info!("=============================================");

    match mode.as_str() {
        "pretrain" => {
            tracing::info!("Starting Foundational Pretraining...");
            run_pretraining::<ModelAutodiffBackend>(&device).await;
        }
        "sft" => {
            tracing::info!("Starting Supervised Fine-Tuning (SFT)...");
            run_sft_training::<ModelAutodiffBackend>(&device);
        }
        "rl" => {
            tracing::info!("Starting Reinforcement Learning (RL)...");
            run_rl_training::<ModelAutodiffBackend>(&device);
        }
        _ => {
            tracing::error!("Unknown training mode: {}", mode);
            tracing::error!("Available modes: pretrain, sft, rl");
            std::process::exit(1);
        }
    }
}
