use std::path::PathBuf;

use nanochat_burn::{common::{TrainBackend, init_device},
    engine::{pretrain::run_pretraining, recipe::run_recipe, rl::run_rl_training,
        sft::run_sft_training},
    experiment::{DEFAULT_EXPERIMENT_CONFIG, ExperimentConfig, RlAlgorithm},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingMode { Pretrain, Sft, Rl, Recipe }

impl TrainingMode {
    fn parse(value: &str) -> Option<Self> {
        match value.trim_start_matches('-').to_ascii_lowercase().as_str() {
            "pretrain" => Some(Self::Pretrain),
            "sft" => Some(Self::Sft),
            "rl" => Some(Self::Rl),
            "recipe" => Some(Self::Recipe),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self { Self::Pretrain => "pretrain", Self::Sft => "sft", Self::Rl => "rl",
            Self::Recipe => "recipe" }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct TrainArgs {
    mode: TrainingMode,
    config: PathBuf,
    rl_algorithm: Option<RlAlgorithm>,
}

fn parse_args(args: impl IntoIterator<Item = String>, env_config: Option<PathBuf>)
    -> Result<TrainArgs, String> {
    let (mut mode, mut config, mut rl_algorithm) =
        (None, env_config.unwrap_or_else(|| PathBuf::from(DEFAULT_EXPERIMENT_CONFIG)), None);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--config" {
            config = args.next().map(PathBuf::from)
                .ok_or_else(|| "--config requires a path".to_string())?;
        } else if arg == "--rl-algorithm" {
            rl_algorithm = Some(match args.next().as_deref() {
                Some("group_normalized_reinforce") => RlAlgorithm::GroupNormalizedReinforce,
                Some("grpo") => RlAlgorithm::Grpo,
                Some(value) => return Err(format!("unsupported RL algorithm: {value}")),
                None => return Err("--rl-algorithm requires a value".into()),
            });
        } else if let Some(parsed) = TrainingMode::parse(&arg) {
            if mode.replace(parsed).is_some() {
                return Err("training mode may only be specified once".into());
            }
        } else {
            return Err(format!("unknown training argument: {arg}"));
        }
    }
    let mode = mode.unwrap_or(TrainingMode::Pretrain);
    if rl_algorithm.is_some() && mode != TrainingMode::Rl {
        return Err("--rl-algorithm requires --rl".into());
    }
    Ok(TrainArgs { mode, config, rl_algorithm })
}

#[tokio::main] async fn main() {
    // Initialize logging subscriber
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    let args = parse_args(std::env::args().skip(1),
        std::env::var_os("NANOCHAT_CONFIG").map(PathBuf::from))
        .unwrap_or_else(|error| panic!("{error}"));
    let mut config = ExperimentConfig::load(&args.config)
        .unwrap_or_else(|error| panic!("{error}"));
    if let Some(algorithm) = args.rl_algorithm { config.rl.algorithm = algorithm; }
    let device = init_device();

    tracing::info!("=============================================");
    tracing::info!("   Initializing nanochat-burn Training      ");
    tracing::info!("   Mode: {}                                 ", args.mode.name());
    tracing::info!("   Config: {:?}                             ", args.config);
    tracing::info!("=============================================");

    match args.mode {
        TrainingMode::Pretrain => {
            tracing::info!("Starting Foundational Pretraining...");
            run_pretraining::<TrainBackend>(&device, &config).await;
        }
        TrainingMode::Sft => {
            tracing::info!("Starting Supervised Fine-Tuning (SFT)...");
            run_sft_training::<TrainBackend>(&device, &config);
        }
        TrainingMode::Rl => {
            tracing::info!("Starting Reinforcement Learning (RL)...");
            run_rl_training::<TrainBackend>(&device, &config);
        }
        TrainingMode::Recipe => {
            tracing::info!("Starting end-to-end training recipe...");
            run_recipe::<TrainBackend>(&device, &config).await;
        }
    }
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_parse_training_args() {
        let args = parse_args(["--sft".into(), "--config".into(), "custom.toml".into()], None)
            .unwrap();
        assert_eq!(args, TrainArgs { mode: TrainingMode::Sft,
            config: PathBuf::from("custom.toml"), rl_algorithm: None });
        let args = parse_args(Vec::new(), Some(PathBuf::from("env.toml"))).unwrap();
        assert_eq!(args, TrainArgs { mode: TrainingMode::Pretrain,
            config: PathBuf::from("env.toml"), rl_algorithm: None });
        assert!(parse_args(["--rl".into(), "--pretrain".into()], None).is_err());
        assert!(parse_args(["--config".into()], None).is_err());
        assert_eq!(parse_args(["--recipe".into()], None).unwrap().mode, TrainingMode::Recipe);
        assert_eq!(parse_args(["--rl".into(), "--rl-algorithm".into(), "grpo".into()], None)
            .unwrap().rl_algorithm, Some(RlAlgorithm::Grpo));
        assert!(parse_args(["--rl-algorithm".into(), "grpo".into()], None).is_err());
    }
}
