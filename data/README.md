# 📊 NanoChat Burn Dataset Repository

This directory contains repository-owned fixtures, the embedded Web UI, and scripts that generate
local SFT and evaluation datasets. Generated datasets and download caches are ignored by Git.

---

## 📂 Directory Layout

```text
    data/
    ├── assets/ui.html                 # Web Chat UI embedded by chat_web
    ├── fixtures/parity/               # Deterministic Python/Rust parity fixtures
    ├── fixtures/tiny/                 # Offline end-to-end recipe inputs
    ├── README.md                      # This documentation file
    ├── sft_train.jsonl                # Unified packed SFT training mixture
    ├── download_datasets.py           # Download, parse and export all public datasets
    ├── generate_synthetic_datasets.py # Script to generate custom synthetic spelling datasets
    └── eval/                          # Multi-task capability evaluation suite
        ├── arc_easy.jsonl             # Science multiple-choice reasoning (Easy split)
        ├── arc_challenge.jsonl        # Science multiple-choice reasoning (Challenge split)
        ├── mmlu.jsonl                 # Massive Multitask Language Understanding benchmark
        ├── gsm8k.jsonl                # Grade School Math 8K reasoning benchmark
        ├── spellingbee.jsonl          # Character count and spelling reasoning benchmark
        └── humaneval.jsonl            # OpenAI HumanEval code generation benchmark
```

---

## 🛠️ Data Generation & Acquisition Scripts

All output paths are derived from this repository; the clone directory does not need to be named
`burn` or live inside the Python nanochat checkout.

1. **`download_datasets.py`**:
   - **Purpose**: The single acquisition/export entry point. It downloads public raw files for
     MMLU, GSM8K, ARC, HumanEval, spelling tasks and Karpathy's identity dataset, then generates
     `sft_train.jsonl` and the six `eval/*.jsonl` datasets. Downloads are cached under
     repository-level `.cache/downloads/`.
   - **Dependencies**: Python standard library only. It deliberately does not import private modules
     from a sibling Python nanochat checkout, so upstream task refactors cannot break this pipeline.

2. **`generate_synthetic_datasets.py`**:
   - **Purpose**: Natively generates high-quality synthetic data for our `spellingbee` benchmarks. It programmatically formats spelling and letter-counting puzzles using templates to test the LLM's character-level reasoning abilities.

---

## 📝 Conversation Data Structure

All training and evaluation data files are formatted as JSON Lines (`.jsonl`), where each line represents a single structured `Conversation`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How many 'r's are in the word 'strawberry'?"
    },
    {
      "role": "assistant",
      "content": "The word 'strawberry' contains 3 'r's."
    }
  ]
}
```

For specialized evaluation tasks (like `spellingbee` or `humaneval`), additional metadata fields (such as `letters`, `entry_point`, or `test` assertions) are packed alongside the standard fields to facilitate programmatic verification during sandboxed evaluation.

---

## 🔄 Reproduction Guide

To re-export or regenerate all training and evaluation datasets:

```bash
uv run --no-project data/download_datasets.py
```
This updates `sft_train.jsonl` and the benchmarks in `eval/`. It downloads public source data and
therefore requires network access. For a tiny fully synthetic dataset, run
`uv run --no-project data/generate_synthetic_datasets.py` instead.
