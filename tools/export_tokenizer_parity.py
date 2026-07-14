#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "rustbpe==0.1.0",
#   "tiktoken==0.11.0",
#   "tokenizers==0.22.0",
# ]
# ///
"""Export deterministic tokenizer parity data from Python nanochat."""

import argparse
import copy
import importlib.metadata
import json
import os
import sys
from pathlib import Path


VOCAB_SIZE = 320
CORPUS = [
    "Rust makes systems programming explicit, fast, and memory safe.",
    "A tokenizer turns ordinary text into token identifiers for a language model.",
    "Pretraining predicts the next token; fine-tuning learns from conversations.",
    "Numbers: 12 34 5678. Contractions: I'm, you're, we'll, they'd.",
    "Tools can run Python, observe output, and then explain the final result.",
    "Unicode remains byte-safe: \u4f60\u597d\u4e16\u754c \U0001f30d.",
]
ENCODING_CASES = [
    "Hello, nanochat!",
    "Numbers: 123, 4567, 89",
    "I'm testing\nnew lines and  spaces.",
    "Unicode: \u4f60\u597d\u4e16\u754c \U0001f30d",
]
CONVERSATION_CASES = [
    {
        "name": "simple",
        "max_tokens": 128,
        "conversation": {"messages": [
            {"role": "user", "content": "What does ownership protect?"},
            {"role": "assistant", "content": "It helps protect memory safety."},
        ]},
    },
    {
        "name": "system_and_tool",
        "max_tokens": 128,
        "conversation": {"messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Compute 6 * 7."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I will calculate it."},
                {"type": "python", "text": "6 * 7"},
                {"type": "python_output", "text": "42"},
                {"type": "text", "text": "The answer is 42."},
            ]},
        ]},
    },
    {
        "name": "truncated",
        "max_tokens": 12,
        "conversation": {"messages": [
            {"role": "user", "content": "Explain deterministic tokenizer parity carefully."},
            {"role": "assistant", "content": "Use fixed inputs and compare every token ID."},
        ]},
    },
]


def export_fixture(output: Path, tokenizer_type: type) -> None:
    tokenizer = tokenizer_type.train_from_iterator(CORPUS, VOCAB_SIZE)
    encoding_cases = [
        {"text": text, "ids": tokenizer.encode(text)} for text in ENCODING_CASES
    ]
    conversation_cases = []
    for case in CONVERSATION_CASES:
        conversation = copy.deepcopy(case["conversation"])
        ids, mask = tokenizer.render_conversation(conversation, case["max_tokens"])
        completion_ids = tokenizer.render_for_completion(copy.deepcopy(conversation))
        conversation_cases.append({**case, "ids": ids, "mask": mask,
            "completion_ids": completion_ids})

    fixture = {
        "schema_version": 1,
        "source": {
            "implementation": "nanochat.tokenizer.RustBPETokenizer",
            "rustbpe": importlib.metadata.version("rustbpe"),
            "tiktoken": importlib.metadata.version("tiktoken"),
        },
        "vocab_size": VOCAB_SIZE,
        "corpus": CORPUS,
        "mergeable_ranks": [
            {"bytes": list(token), "id": token_id}
            for token, token_id in sorted(tokenizer.enc._mergeable_ranks.items(),
                key=lambda item: item[1])
        ],
        "special_tokens": dict(sorted(tokenizer.enc._special_tokens.items())),
        "encoding_cases": encoding_cases,
        "conversation_cases": conversation_cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote tokenizer parity fixture to {output}")


def main() -> None:
    burn_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanochat-root", type=Path, default=os.environ.get("NANOCHAT_ROOT"),
        help="Python nanochat repository root (or set NANOCHAT_ROOT)")
    parser.add_argument("--output", type=Path,
        default=burn_root / "data/fixtures/parity/tokenizer.json")
    args = parser.parse_args()
    if args.nanochat_root is None:
        parser.error("--nanochat-root or NANOCHAT_ROOT is required")
    root = args.nanochat_root.expanduser().resolve()
    if not (root / "nanochat/tokenizer.py").is_file():
        parser.error(f"{root} does not contain nanochat/tokenizer.py")
    sys.path.insert(0, str(root))
    from nanochat.tokenizer import RustBPETokenizer

    export_fixture(args.output, RustBPETokenizer)


if __name__ == "__main__":
    main()
