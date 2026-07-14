import argparse
import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

os.environ["NANOCHAT_BASE_DIR"] = os.path.join(project_root, ".cache/nanochat")
os.environ["HF_HOME"] = os.path.join(project_root, ".cache/huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(project_root, ".cache/huggingface/datasets")
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(project_root, ".cache/tokenizer")

import json

def load_reference(root: Path) -> None:
    global tqdm, TaskMixture, GSM8K, MMLU, SmolTalk, CustomJSON
    global SimpleSpelling, SpellingBee, ARC, HumanEval
    if not (root / "tasks/common.py").is_file():
        raise ValueError(f"{root} does not contain tasks/common.py")
    sys.path.insert(0, str(root))
    from tqdm import tqdm as progress
    from tasks.common import TaskMixture as Mixture
    from tasks.gsm8k import GSM8K as Gsm
    from tasks.mmlu import MMLU as Mmlu
    from tasks.smoltalk import SmolTalk as Talk
    from tasks.customjson import CustomJSON as Custom
    from tasks.spellingbee import SimpleSpelling as Simple, SpellingBee as Bee
    from tasks.arc import ARC as Arc
    from tasks.humaneval import HumanEval as Human
    tqdm, TaskMixture, GSM8K, MMLU, SmolTalk, CustomJSON = (
        progress, Mixture, Gsm, Mmlu, Talk, Custom)
    SimpleSpelling, SpellingBee, ARC, HumanEval = Simple, Bee, Arc, Human

def ensure_dirs():
    os.makedirs(os.path.join(script_dir, "eval"), exist_ok=True)

def export_sft_train():
    print("Preparing SFT training mixture...")
    downloads = os.path.join(script_dir, "downloads")
    os.makedirs(downloads, exist_ok=True)
    identity_path = os.path.join(downloads, "identity_conversations.jsonl")
    
    # Download custom identity dataset if not present
    if not os.path.exists(identity_path):
        import urllib.request
        print("Downloading identity conversations...")
        urllib.request.urlretrieve(
            "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
            identity_path
        )

    # Initialize SFT task components with capped sizes for efficient export and testing
    tasks = [
        SmolTalk(split="train", start=0, stop=3000),
        CustomJSON(filepath=identity_path),
        CustomJSON(filepath=identity_path), # 2 epochs of this
        MMLU(subset="all", split="auxiliary_train", start=0, stop=2000),
        GSM8K(subset="main", split="train", start=0, stop=2000),
        SimpleSpelling(size=2000, split="train"),
        SpellingBee(size=2000, split="train")
    ]
    
    mixture = TaskMixture(tasks)
    print(f"Total SFT training rows: {len(mixture)}")
    
    output_path = os.path.join(script_dir, "sft_train.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(mixture)), desc="Exporting SFT Train"):
            conv = mixture[i]
            # Convert any non-serializable fields (like letters tuples) to list
            if "letters" in conv:
                conv["letters"] = list(conv["letters"])
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    print(f"Successfully exported SFT train dataset to {output_path}")

def export_eval_datasets():
    print("Preparing evaluation datasets...")
    
    eval_configs = [
        ("arc_easy", lambda: ARC(subset="ARC-Easy", split="test", start=0, stop=400)),
        ("arc_challenge", lambda: ARC(subset="ARC-Challenge", split="test", start=0, stop=400)),
        ("mmlu", lambda: MMLU(subset="all", split="test", start=0, stop=500)),
        ("gsm8k", lambda: GSM8K(subset="main", split="test", start=0, stop=200)),
        ("spellingbee", lambda: SpellingBee(size=256, split="test")),
        ("humaneval", lambda: HumanEval())
    ]
    
    for name, init_fn in eval_configs:
        task = init_fn()
        output_path = os.path.join(script_dir, f"eval/{name}.jsonl")
        print(f"Exporting {name} ({len(task)} rows)...")
        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(len(task)):
                conv = task[i]
                if "letters" in conv:
                    conv["letters"] = list(conv["letters"])
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"Successfully exported {name} to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanochat-root", type=Path, default=os.environ.get("NANOCHAT_ROOT"),
        help="Python nanochat repository root (or set NANOCHAT_ROOT)")
    args = parser.parse_args()
    if args.nanochat_root is None:
        parser.error("--nanochat-root or NANOCHAT_ROOT is required")
    try:
        load_reference(args.nanochat_root.expanduser().resolve())
    except ValueError as error:
        parser.error(str(error))
    ensure_dirs()
    export_sft_train()
    export_eval_datasets()


if __name__ == "__main__":
    main()
