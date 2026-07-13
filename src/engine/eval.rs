
use std::path::Path;

use burn::tensor::backend::Backend;
use serde::Deserialize;

use crate::{common::{extract_answer, int_tensor_2d, read_jsonl}, gpt::Gpt,
    engine::{inference::{GenerationConfig, InferenceEngine, SamplingConfig},
        sandbox::execute_code},
    tokenizer::{BpeTokenizer, Conversation, ConversationMessage},
};

#[derive(Debug, Clone, Deserialize)]
pub struct EvalItem {
    pub messages: Vec<ConversationMessage>,
    pub letters: Option<Vec<String>>, // For categorical
    pub entry_point: Option<String>,  // For HumanEval
    pub test: Option<String>,         // For HumanEval
}

pub fn load_eval_dataset<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<EvalItem>> {
    read_jsonl(path)
}

fn prompt_tokens(tokenizer: &BpeTokenizer, item: &EvalItem) -> Vec<usize> {
    tokenizer.render_for_completion(&Conversation { messages: item.messages.clone() })
}

fn generate_completion<B: Backend>(engine: &InferenceEngine<B>, tokenizer: &BpeTokenizer,
    prompt_tokens: &[usize], device: &B::Device) -> String {
    let config = GenerationConfig { max_tokens: 128, sampling: SamplingConfig::greedy() };
    let (rollouts, _) = engine.generate_batch(prompt_tokens, 1, config, device);
    tokenizer.decode(&rollouts[0][prompt_tokens.len()..])
}

pub fn evaluate_categorical<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);

    for item in items {
        let prompt_tokens = prompt_tokens(tokenizer, item);
        let prompt_len = prompt_tokens.len();

        let inputs = int_tensor_2d(
            prompt_tokens.iter().map(|&token| token as i32).collect(), [1, prompt_len], device);

        let (logits, vocab_size) = (model.forward(inputs, None), model.config.vocab_size);
        let last_logits = logits
            .slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab_size]).reshape([vocab_size]);
        let ground_truth =
            item.messages.last().unwrap().content.to_string_content().trim().to_string();
        let logits_vec = crate::common::tensor_data_to_f32_vec(last_logits.into_data());

        let letters = item.letters.as_ref().cloned().unwrap_or_else(|| {
            vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]
        });

        let best_letter = letters .iter().filter_map(|letter| {
                tokenizer.encode_ordinary(letter).first().copied().and_then(|token_id| {
                    logits_vec.get(token_id).map(|&logit| (letter, logit))
                })
            }).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(letter, _)| letter.clone()).unwrap_or_default();

        if best_letter == ground_truth { correct += 1; }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn evaluate_generative<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let generated_text = generate_completion(&inference_engine, tokenizer,
            &prompt_tokens(tokenizer, item), device);

        let ground_truth_text = item.messages.last().unwrap().content.to_string_content();
        let (pred_ans, gt_ans) =
            (extract_answer(&generated_text), extract_answer(&ground_truth_text));

        if gt_ans.is_some() && pred_ans == gt_ans { correct += 1; }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn evaluate_humaneval<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let generated_completion = generate_completion(&inference_engine,
            tokenizer, &prompt_tokens(tokenizer, item), device);

        let prompt_pure_code = item.messages[0].content.to_string_content();
        let full_code = format!("{}{}", prompt_pure_code, generated_completion);

        if let (Some(entry_point), Some(test)) = (&item.entry_point, &item.test) {
            let runnable_code = format!("{}\n\n{}\n\ncheck({})", full_code, test, entry_point);
            if execute_code(&runnable_code, 5).success { correct += 1; }
        }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn run_all_evaluations<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    device: &B::Device) {
    tracing::info!("=============================================");
    tracing::info!("   Starting ChatCORE Evaluation Harness      ");
    tracing::info!("=============================================");

    let tasks = [
        ("ARC-Easy", "data/eval/arc_easy.jsonl", true),
        ("ARC-Challenge", "data/eval/arc_challenge.jsonl", true),
        ("MMLU", "data/eval/mmlu.jsonl", true),
        ("GSM8K", "data/eval/gsm8k.jsonl", false),
        ("SpellingBee", "data/eval/spellingbee.jsonl", false),
    ];

    let mut scores = Vec::new();

    for (name, path, is_cat) in tasks {
        if !Path::new(path).exists() {
            tracing::warn!("Task {} dataset not found at {}, skipping.", name, path);
            continue;
        }

        let items = load_eval_dataset(path).unwrap_or_default();
        if items.is_empty() {
            tracing::warn!("Task {} loaded 0 items, skipping.", name);
            continue;
        }

        tracing::info!("Evaluating {} ({} items)...", name, items.len());
        let score = if is_cat {
            evaluate_categorical(model, tokenizer, &items, device)
        } else {
            evaluate_generative(model, tokenizer, &items, device)
        };

        tracing::info!("> {} Score: {:.2}%", name, score * 100.0);
        scores.push((name.to_string(), score));
    }

    let humaneval_path = "data/eval/humaneval.jsonl";
    if Path::new(humaneval_path).exists() {
        let items = load_eval_dataset(humaneval_path).unwrap_or_default();
        if !items.is_empty() {
            tracing::info!("Evaluating HumanEval ({} items)...", items.len());
            let score = evaluate_humaneval(model, tokenizer, &items, device);
            tracing::info!("> HumanEval Score: {:.2}%", score * 100.0);
            scores.push(("HumanEval".to_string(), score));
        }
    }

    if scores.is_empty() {
        tracing::error!("No tasks were evaluated!");
        return;
    }

    let chatcore: f32 = scores.iter().map(|(_, s)| s).sum::<f32>() / (scores.len() as f32);

    tracing::info!("=============================================");
    tracing::info!("   EVALUATION REPORT SUMMARY                 ");
    tracing::info!("=============================================");
    for (name, score) in &scores {
        tracing::info!(" - {:<15}: {:.2}%", name, score * 100.0);
    }
    tracing::info!("---------------------------------------------");
    tracing::info!("   ChatCORE Metric score: {:.2}%             ", chatcore * 100.0);
    tracing::info!("=============================================");
}

//#[cfg(test)] mod tests { use super::*;
    #[test] fn test_extract_answer() {
        assert_eq!(extract_answer("The answer is #### 42"), Some(42));
        assert_eq!(extract_answer("#### -7"), Some(-7));
        assert_eq!(extract_answer("No answer here"), None);
        assert_eq!(extract_answer("#### 12,345"), Some(12345));
    }
//}
