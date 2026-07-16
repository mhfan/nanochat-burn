
use std::path::Path;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::{common::{extract_answer, int_tensor_2d, read_jsonl}, gpt::Gpt,
    engine::{inference::{GenerationConfig, InferenceEngine, SamplingConfig},
        sandbox::{ExecutionConfig, execute_code}},
    experiment::{EvalConfig, EvalTaskKind},
    tokenizer::{BpeTokenizer, Conversation, ConversationMessage},
};

#[derive(Debug, Clone, Deserialize)]
pub struct EvalItem {
    pub messages: Vec<ConversationMessage>,
    pub letters: Option<Vec<String>>, // For categorical
    pub entry_point: Option<String>,  // For HumanEval
    pub test: Option<String>,         // For HumanEval
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalScore { pub name: String, pub score: f32 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalReport { pub scores: Vec<EvalScore>, pub aggregate: Option<f32> }

pub fn load_eval_dataset<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<EvalItem>> {
    read_jsonl(path)
}

fn prompt_tokens(tokenizer: &BpeTokenizer, item: &EvalItem) -> Vec<usize> {
    tokenizer.render_for_completion(&Conversation { messages: item.messages.clone() })
}

fn fit_prompt(mut tokens: Vec<usize>, max_len: usize) -> Vec<usize> {
    assert!(max_len > 0, "model context length must be greater than zero");
    if tokens.len() > max_len { tokens.drain(..tokens.len() - max_len); }
    tokens
}

fn generate_completion<B: Backend>(engine: &InferenceEngine<B>, tokenizer: &BpeTokenizer,
    prompt_tokens: &[usize], max_tokens: usize, device: &B::Device) -> String {
    let config = GenerationConfig { max_tokens, sampling: SamplingConfig::greedy(), seed: 42 };
    let (rollouts, _) = engine.generate_batch(prompt_tokens, 1, config, device);
    tokenizer.decode(&rollouts[0][prompt_tokens.len()..])
}

fn accuracy(correct: usize, total: usize) -> f32 {
    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn evaluate_categorical<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);

    for item in items {
        let prompt_tokens = fit_prompt(prompt_tokens(tokenizer, item), model.config.sequence_len);
        let prompt_len = prompt_tokens.len();

        let inputs = int_tensor_2d(
            prompt_tokens.iter().map(|&token| token as i32).collect(), [1, prompt_len], device);

        let (logits, vocab_size) = (model.forward(inputs), model.config.vocab_size);
        let last_logits = logits
            .slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab_size]).reshape([vocab_size]);
        let ground_truth =
            item.messages.last().unwrap().content.to_string_content().trim().to_string();
        let logits_vec = crate::common::tensor_data_to_f32_vec(last_logits.into_data());

        let letters = item.letters.as_ref().cloned().unwrap_or_else(|| {
            vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]
        });

        let best_letter = letters.iter().filter_map(|letter| {
                tokenizer.encode_ordinary(letter).first().copied()
                    .and_then(|token_id| logits_vec.get(token_id).map(|&logit| (letter, logit)))
            }).max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(letter, _)| letter.clone()).unwrap_or_default();

        if best_letter == ground_truth { correct += 1; }
        total += 1;
    }

    accuracy(correct, total)
}

pub fn evaluate_generative<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], max_tokens: usize, device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let prompt = fit_prompt(prompt_tokens(tokenizer, item),
            model.config.sequence_len.saturating_sub(1).max(1));
        let generated_text =
            generate_completion(&inference_engine, tokenizer, &prompt, max_tokens, device);

        let ground_truth_text = item.messages.last().unwrap().content.to_string_content();
        let (pred_ans, gt_ans) =
            (extract_answer(&generated_text), extract_answer(&ground_truth_text));

        if gt_ans.is_some() && pred_ans == gt_ans { correct += 1; }
        total += 1;
    }

    accuracy(correct, total)
}

pub fn evaluate_humaneval<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], max_tokens: usize, device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let prompt = fit_prompt(prompt_tokens(tokenizer, item),
            model.config.sequence_len.saturating_sub(1).max(1));
        let generated_completion =
            generate_completion(&inference_engine, tokenizer, &prompt, max_tokens, device);

        let prompt_pure_code = item.messages[0].content.to_string_content();
        let full_code = format!("{}{}", prompt_pure_code, generated_completion);

        if let (Some(entry_point), Some(test)) = (&item.entry_point, &item.test) {
            let runnable_code = format!("{}\n\n{}\n\ncheck({})", full_code, test, entry_point);
            if execute_code(&runnable_code, ExecutionConfig::default()).success { correct += 1; }
        }
        total += 1;
    }

    accuracy(correct, total)
}

pub fn run_evaluations<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    config: &EvalConfig, device: &B::Device) -> EvalReport {
    tracing::info!("=============================================");
    tracing::info!("   Starting ChatCORE Evaluation Harness      ");
    tracing::info!("=============================================");

    let mut scores = Vec::new();

    for task in &config.tasks {
        if !task.path.exists() {
            tracing::warn!("Task {} dataset not found at {:?}, skipping.", task.name, task.path);
            continue;
        }

        let items = match load_eval_dataset(&task.path) {
            Ok(items) => items,
            Err(error) => {
                tracing::warn!("Failed to load task {} from {:?}: {error}", task.name, task.path);
                continue;
            }
        };
        if items.is_empty() {
            tracing::warn!("Task {} loaded 0 items, skipping.", task.name);
            continue;
        }

        tracing::info!("Evaluating {} ({} items)...", task.name, items.len());
        let score = match task.kind {
            EvalTaskKind::Categorical => evaluate_categorical(model, tokenizer, &items, device),
            EvalTaskKind::Generative => evaluate_generative(model, tokenizer, &items,
                config.max_generation_tokens, device),
            EvalTaskKind::HumanEval => evaluate_humaneval(model, tokenizer, &items,
                config.max_generation_tokens, device),
        };

        tracing::info!("> {} Score: {:.2}%", task.name, score * 100.0);
        scores.push(EvalScore { name: task.name.clone(), score });
    }

    if scores.is_empty() {
        tracing::error!("No tasks were evaluated!");
        return EvalReport { scores, aggregate: None };
    }

    let chatcore = scores.iter().map(|score| score.score).sum::<f32>() / scores.len() as f32;

    tracing::info!("=============================================");
    tracing::info!("   EVALUATION REPORT SUMMARY                 ");
    tracing::info!("=============================================");
    for score in &scores {
        tracing::info!(" - {:<15}: {:.2}%", score.name, score.score * 100.0);
    }
    tracing::info!("---------------------------------------------");
    tracing::info!("   ChatCORE Metric score: {:.2}%             ", chatcore * 100.0);
    tracing::info!("=============================================");
    EvalReport { scores, aggregate: Some(chatcore) }
}
