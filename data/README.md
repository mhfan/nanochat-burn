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
    ├── export_datasets.py             # Script to export/mix datasets using HF Hub
    ├── download_raw_datasets.py       # Standalone script to download and parse raw datasets (no HF dependencies)
    ├── generate_synthetic_datasets.py # Script to generate custom synthetic spelling datasets
    ├── downloads/                     # Download cache folder
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

1. **`export_datasets.py`**:
   - **Purpose**: Generates the unified `sft_train.jsonl` mixture and the six `eval/*.jsonl` benchmark datasets.
   - **Source Data**: Downloads datasets from Hugging Face (`SmolTalk`, `MMLU`, `GSM8K`, `ARC`, `HumanEval`) and Karpathy's S3 storage (`identity_conversations.jsonl`), formatting them into the unified `Conversation` format.
   - **Dependencies**: Requires `huggingface_hub`, `datasets`, `tqdm`, and an explicitly selected
     Python nanochat checkout containing `tasks/`.

2. **`download_raw_datasets.py`**:
   - **Purpose**: A standalone dataset downloader and parser that does **not** depend on the Hugging Face `datasets` library. It uses Python's standard `urllib` to retrieve raw text files (e.g. GSM8K, ARC, MMLU, HumanEval) and formats them into our conversation structures directly.

3. **`generate_synthetic_datasets.py`**:
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
# Use the upstream Python task implementations explicitly:
uv run data/export_datasets.py --nanochat-root /path/to/python-nanochat

# Or use the environment for repeated commands:
NANOCHAT_ROOT=/path/to/python-nanochat uv run data/export_datasets.py
```
This updates `sft_train.jsonl` and the benchmarks in `eval/`. For a repository-local synthetic
dataset with no Python nanochat checkout, run `uv run data/generate_synthetic_datasets.py`.
