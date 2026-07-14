use std::path::Path;

use burn::tensor::backend::AutodiffBackend;

use crate::{artifact::{TrainingStage, load_artifact}, engine::{eval::{EvalReport,
        run_evaluations}, pretrain::run_pretraining_at, sft::run_sft_training},
    experiment::ExperimentConfig,
};

fn artifact_matches(path: &Path, experiment: &ExperimentConfig) -> bool {
    ExperimentConfig::load(path.join("experiment.toml"))
        .is_ok_and(|saved| saved == *experiment)
}

pub async fn run_recipe<B: AutodiffBackend>(device: &B::Device,
    experiment: &ExperimentConfig) -> EvalReport {
    for variable in ["NANOCHAT_RESUME_ARTIFACT", "NANOCHAT_INPUT_ARTIFACT",
        "NANOCHAT_OUTPUT_ARTIFACT"] {
        assert!(std::env::var_os(variable).is_none(),
            "{variable} is a stage-level override and cannot be used with --recipe");
    }

    let (pretrain, sft) = (&experiment.artifacts.pretrain, &experiment.artifacts.sft);
    if artifact_matches(sft, experiment) {
        tracing::info!("Reusing completed SFT artifact at {:?}", sft);
    } else {
        let resume = artifact_matches(pretrain, experiment).then_some(pretrain.as_path());
        if resume.is_some() {
            tracing::info!("Resuming recipe pretraining artifact at {:?}", pretrain);
        }
        run_pretraining_at::<B>(device, experiment, resume, pretrain).await;
        run_sft_training::<B>(device, experiment);
    }

    let path = sft;
    let artifact = load_artifact::<B>(path, device)
        .unwrap_or_else(|error| panic!("failed to load recipe SFT artifact {path:?}: {error}"));
    assert_eq!(artifact.manifest.stage, TrainingStage::Sft,
        "recipe evaluation requires an SFT artifact");
    let report = run_evaluations(&artifact.model, &artifact.tokenizer, &experiment.eval, device);
    assert!(report.aggregate.is_some(), "recipe did not evaluate any configured task");
    report
}
