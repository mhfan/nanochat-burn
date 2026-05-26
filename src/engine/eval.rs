
use std::{fs::File, io::{BufRead, BufReader}, path::Path};
use burn::tensor::backend::Backend;
use serde::Deserialize;
use crate::{gpt::Gpt,
    tokenizer::{BpeTokenizer, Conversation, ConversationMessage},
    engine::{inference::InferenceEngine, sandbox::execute_code},
};

#[derive(Debug, Clone, Deserialize)]
pub struct EvalItem {
    pub messages: Vec<ConversationMessage>,
    pub letters: Option<Vec<String>>,     // For categorical
    pub entry_point: Option<String>,      // For HumanEval
    pub test: Option<String>,             // For HumanEval
}

pub fn load_eval_dataset<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<EvalItem>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    for line in reader.lines() {
        let line_str = line?;
        if line_str.trim().is_empty() { continue; }
        let item: EvalItem = serde_json::from_str(&line_str)?;
        items.push(item);
    }
    Ok(items)
}

fn extract_answer(text: &str) -> Option<i32> {
    let marker = "#### ";
    if let Some(idx) = text.rfind(marker) {
        let num_part = text[idx + marker.len()..].trim();
        let clean_num: String = num_part.chars().filter(|c| c.is_digit(10) || *c == '-').collect();
        clean_num.parse::<i32>().ok()
    } else { None }
}

pub fn evaluate_categorical<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device,) -> f32 {
    let (mut correct, mut total) = (0, 0);

    for item in items {
        let conv = Conversation { messages: item.messages.clone() };
        let prompt_tokens = tokenizer.render_for_completion(&conv);
        let prompt_len = prompt_tokens.len();

        let inputs = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
            prompt_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>().as_slice(), device,
        ).reshape([1, prompt_len]);

        let logits = model.forward(inputs, None);
        let vocab_size = model.config.vocab_size;
        let last_logits = logits.slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab_size]).reshape([vocab_size]);
        let logits_vec = last_logits.into_data().to_vec::<f32>().unwrap();

        let last_msg = item.messages.last().unwrap();
        let ground_truth = match &last_msg.content {
            crate::tokenizer::MessageContent::Simple(s) => s.trim().to_string(),
            crate::tokenizer::MessageContent::Parts(parts) => {
                parts.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join("").trim().to_string()
            }
        };

        let letters = item.letters.as_ref().cloned().unwrap_or_else(|| {
            vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]
        });

        let mut best_letter = String::new();
        let mut best_logit = f32::NEG_INFINITY;
        for letter in &letters {
            let letter_tokens = tokenizer.encode_ordinary(letter);
            if !letter_tokens.is_empty() {
                let token_id = letter_tokens[0];
                if token_id < logits_vec.len() {
                    let logit = logits_vec[token_id];
                    if logit > best_logit {
                        best_logit = logit;
                        best_letter = letter.clone();
                    }
                }
            }
        }

        if best_letter == ground_truth { correct += 1; }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn evaluate_generative<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device,) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let conv = Conversation { messages: item.messages.clone() };
        let prompt_tokens = tokenizer.render_for_completion(&conv);
        let prompt_len = prompt_tokens.len();

        let rollouts = inference_engine.generate_batch(&prompt_tokens, 1, 128, 0.0, None, 1.0, device,);

        let rollout = &rollouts[0];
        let generated_tokens = &rollout[prompt_len..];
        let generated_text = tokenizer.decode(generated_tokens);

        let last_msg = item.messages.last().unwrap();
        let ground_truth_text = match &last_msg.content {
            crate::tokenizer::MessageContent::Simple(s) => s.clone(),
            crate::tokenizer::MessageContent::Parts(parts) => {
                parts.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join("")
            }
        };

        let pred_ans = extract_answer(&generated_text);
        let gt_ans = extract_answer(&ground_truth_text);

        if gt_ans.is_some() && pred_ans == gt_ans { correct += 1; }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn evaluate_humaneval<B: Backend>(model: &Gpt<B>, tokenizer: &BpeTokenizer,
    items: &[EvalItem], device: &B::Device,) -> f32 {
    let (mut correct, mut total) = (0, 0);
    let inference_engine = InferenceEngine::new(model.clone(), tokenizer.clone());

    for item in items {
        let conv = Conversation { messages: item.messages.clone() };
        let prompt_tokens = tokenizer.render_for_completion(&conv);
        let prompt_len = prompt_tokens.len();

        let rollouts = inference_engine.generate_batch(&prompt_tokens, 1, 128, 0.0, None, 1.0, device,);

        let rollout = &rollouts[0];
        let generated_tokens = &rollout[prompt_len..];
        let generated_completion = tokenizer.decode(generated_tokens);

        let prompt_pure_code = match &item.messages[0].content {
            crate::tokenizer::MessageContent::Simple(s) => s.clone(),
            crate::tokenizer::MessageContent::Parts(parts) => {
                parts.iter().map(|p| p.text.clone()).collect::<Vec<_>>().join("")
            }
        };

        let full_code = format!("{}{}", prompt_pure_code, generated_completion);

        if let (Some(ref entry_point), Some(ref test)) = (&item.entry_point, &item.test) {
            let runnable_code = format!(
                "{}\n\n{}\n\ncheck({})",
                full_code, test, entry_point
            );

            let result = execute_code(&runnable_code, 5);
            if result.success { correct += 1; }
        }
        total += 1;
    }

    if total == 0 { 0.0 } else { correct as f32 / total as f32 }
}

pub fn run_all_evaluations<B: Backend>(model: &Gpt<B>,
    tokenizer: &BpeTokenizer, device: &B::Device,) {
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
