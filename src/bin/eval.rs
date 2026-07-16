
use std::path::PathBuf;

use nanochat_burn::{artifact::{inference_artifact_path, load_artifact},
    common::{ModelBackend, init_device}, engine::eval::run_evaluations,
    experiment::{DEFAULT_EXPERIMENT_CONFIG, ExperimentConfig},
};

fn config_path(args: impl IntoIterator<Item = String>, env_config: Option<PathBuf>)
    -> Result<PathBuf, String> {
    let mut path = env_config.unwrap_or_else(|| PathBuf::from(DEFAULT_EXPERIMENT_CONFIG));
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg != "--config" { return Err(format!("unknown eval argument: {arg}")); }
        path = args.next().map(PathBuf::from)
            .ok_or_else(|| "--config requires a path".to_string())?;
    }
    Ok(path)
}

fn main() {
    // Initialize tracing subscriber
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO).finish();
    tracing::subscriber::set_global_default(subscriber).ok();

    let config_path = config_path(std::env::args().skip(1),
        std::env::var_os("NANOCHAT_CONFIG").map(PathBuf::from))
        .unwrap_or_else(|error| panic!("{error}"));
    let config = ExperimentConfig::load(&config_path)
        .unwrap_or_else(|error| panic!("{error}"));
    let device = init_device();
    let artifact_path = inference_artifact_path(&config.artifacts);
    let artifact = load_artifact::<ModelBackend>(&artifact_path, &device)
        .unwrap_or_else(|error| panic!("failed to load artifact {artifact_path:?}: {error}"));
    tracing::info!("Evaluating {:?} artifact from {:?}", artifact.manifest.stage, artifact_path);
    let report = run_evaluations(&artifact.model, &artifact.tokenizer, &config.eval, &device);
    let output = artifact_path.join("eval.json");
    let encoded = serde_json::to_vec_pretty(&report)
        .unwrap_or_else(|error| panic!("failed to serialize evaluation report: {error}"));
    std::fs::write(&output, encoded)
        .unwrap_or_else(|error| panic!("failed to write evaluation report {output:?}: {error}"));
    tracing::info!("Evaluation report saved to {:?}", output);
}

#[cfg(test)] mod tests { use super::*;
    #[test] fn test_eval_config_path() {
        assert_eq!(config_path(Vec::new(), Some(PathBuf::from("env.toml"))).unwrap(),
            PathBuf::from("env.toml"));
        assert_eq!(config_path(["--config".into(), "custom.toml".into()], None).unwrap(),
            PathBuf::from("custom.toml"));
        assert!(config_path(["--config".into()], None).is_err());
        assert!(config_path(["--unknown".into()], None).is_err());
    }
}
