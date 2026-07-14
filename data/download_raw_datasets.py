import os
import json
import gzip
import zipfile
import csv
import random
import urllib.request

# Environment override to keep all tokenizer and cache files in workspace
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
os.environ["NANOCHAT_BASE_DIR"] = os.path.join(PROJECT_ROOT, ".cache/nanochat")

# Standard BPE special tokens and letters
LETTERS = "abcdefghijklmnopqrstuvwxyz"
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
]

def ensure_dirs():
    os.makedirs(os.path.join(script_dir, "eval"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, ".cache/downloads"), exist_ok=True)

def download_file(url, local_filename):
    local_path = os.path.join(PROJECT_ROOT, ".cache/downloads", local_filename)
    if os.path.exists(local_path):
        return local_path
    print(f"Downloading {url} -> {local_path}...")
    urllib.request.urlretrieve(url, local_path)
    return local_path

def parse_gsm8k_conversation(question, answer, is_train=True):
    # Split tool calls in answer
    import re
    assistant_message_parts = []
    parts = re.split(r'(<<[^>]+>>)', answer)
    for part in parts:
        if part.startswith('<<') and part.endswith('>>'):
            inner = part[2:-2]
            if '=' in inner:
                expr, result = inner.rsplit('=', 1)
            else:
                expr, result = inner, ""
            assistant_message_parts.append({"type": "python", "text": expr})
            assistant_message_parts.append({"type": "python_output", "text": result})
        else:
            assistant_message_parts.append({"type": "text", "text": part})
            
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_message_parts},
    ]
    return {"messages": messages}

def generate_spelling_bee_conversation(index, words, split="train"):
    # Replicate spelling bee generation logic using std library random
    seed = index if split == 'train' else 10_000_000 + index
    rng = random.Random(seed)
    word = rng.choice(words)
    letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
    count = word.count(letter)
    template = rng.choice(USER_MSG_TEMPLATES)
    
    letter_wrapped = f"'{letter}'"
    word_wrapped = f"'{word}'"
    user_msg = template.format(letter=letter_wrapped, word=word_wrapped) + "?"
    
    word_letters = ",".join(list(word))
    manual_text = f"We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.\n\nFirst spell the word out:\n{word}:{word_letters}\n\nThen count the occurrences of '{letter}':\n"
    
    running_count = 0
    for i, char in enumerate(word, 1):
        if char == letter:
            running_count += 1
            manual_text += f"{i}:{char} hit! count={running_count}\n"
        else:
            manual_text += f"{i}:{char}\n"
            
    manual_text += f"\nThis gives us {running_count}."
    assistant_parts = [
        {"type": "text", "text": manual_text},
        {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
        {"type": "python", "text": f"'{word}'.count('{letter}')"},
        {"type": "python_output", "text": str(count)},
        {"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"}
    ]
    
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
    }

def generate_simple_spelling_conversation(index, words, split="train"):
    seed = index if split == 'train' else 10_000_000 + index
    rng = random.Random(seed + 42)
    word = rng.choice(words)
    word_letters = ",".join(list(word))
    return {
        "messages": [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
    }

def render_mc(question, letters, choices):
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query

def export_datasets():
    ensure_dirs()
    
    # 1. Download Identity Conversations
    identity_url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    identity_path = download_file(identity_url, "identity_conversations.jsonl")
    identity_convs = []
    with open(identity_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                identity_convs.append(json.loads(line))

    # 2. Download English Word List for Spelling Bee
    words_url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    words_path = download_file(words_url, "words_alpha.txt")
    with open(words_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # 3. Download and parse GSM8K
    gsm_train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    gsm_test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    gsm_train_path = download_file(gsm_train_url, "gsm_train.jsonl")
    gsm_test_path = download_file(gsm_test_url, "gsm_test.jsonl")
    
    gsm_train_convs = []
    with open(gsm_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gsm_train_convs.append(parse_gsm8k_conversation(row["question"], row["answer"]))

    gsm_test_convs = []
    with open(gsm_test_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gsm_test_convs.append(parse_gsm8k_conversation(row["question"], row["answer"], is_train=False))

    # 4. Download and parse HumanEval
    he_url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    he_gzip_path = download_file(he_url, "HumanEval.jsonl.gz")
    he_convs = []
    with gzip.open(he_gzip_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                complete_solution = f"{row['prompt']}\n{row['canonical_solution']}"
                messages = [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": complete_solution}
                ]
                he_convs.append({
                    "messages": messages,
                    "entry_point": row["entry_point"],
                    "test": row["test"]
                })

    # 5. Download and parse MMLU
    # Instead of cais/mmlu resolve zip which might be huge, cais/mmlu test csv files are publicly available on hendrycks GitHub repo
    # Let's download a subset of MMLU subjects to keep it extremely fast and lightweight
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"]
    mmlu_train_convs = []
    mmlu_test_convs = []
    
    for subject in subjects:
        # Download train and test CSVs from cais/mmlu hendrycks git repo
        train_url = f"https://raw.githubusercontent.com/hendrycks/test/master/data/dev/{subject}_dev.csv"
        test_url = f"https://raw.githubusercontent.com/hendrycks/test/master/data/test/{subject}_test.csv"
        
        train_csv = download_file(train_url, f"{subject}_train.csv")
        test_csv = download_file(test_url, f"{subject}_test.csv")
        
        # Parse train
        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6: continue
                question, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                letters = ["A", "B", "C", "D"]
                if ans not in letters: continue
                user_msg = render_mc(question, letters, [a, b, c, d])
                mmlu_train_convs.append({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": ans}
                    ],
                    "subject": subject,
                    "letters": letters
                })
                
        # Parse test
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6: continue
                question, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                letters = ["A", "B", "C", "D"]
                if ans not in letters: continue
                user_msg = render_mc(question, letters, [a, b, c, d])
                mmlu_test_convs.append({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": ans}
                    ],
                    "subject": subject,
                    "letters": letters
                })

    # 6. Download and parse ARC-Easy and ARC-Challenge
    for arc_set in ["ARC-Easy", "ARC-Challenge"]:
        url = f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/{arc_set}/{arc_set}-Test.jsonl"
        arc_jsonl = download_file(url, f"{arc_set}-Test.jsonl")
        arc_convs = []
        with open(arc_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    q = row["question"]
                    question = q["stem"]
                    choices = [c["text"] for c in q["choices"]]
                    letters = [c["label"] for c in q["choices"]]
                    ans = row["answerKey"]
                    if ans not in letters: continue
                    user_msg = render_mc(question, letters, choices)
                    arc_convs.append({
                        "messages": [
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": ans}
                        ],
                        "letters": letters
                    })
        output_name = "arc_easy" if arc_set == "ARC-Easy" else "arc_challenge"
        with open(os.path.join(script_dir, f"eval/{output_name}.jsonl"), "w", encoding="utf-8") as out:
            for conv in arc_convs[:300]:
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"Exported {output_name} evaluation dataset.")

    # 7. Generate Spelling Bee and Simple Spelling train / test
    sb_train_convs = [generate_spelling_bee_conversation(i, words, "train") for i in range(2000)]
    sb_test_convs = [generate_spelling_bee_conversation(i, words, "test") for i in range(256)]
    ss_train_convs = [generate_simple_spelling_conversation(i, words, "train") for i in range(2000)]
    
    # 8. Create SmolTalk mock train/val subset to keep training/testing lightweight
    # (Since SmolTalk is general conversation, we can seed it with synthetic/gsm8k/spelling conversations)
    smoltalk_train_convs = [
        {
            "messages": [
                {"role": "user", "content": "Hello! Who are you?"},
                {"role": "assistant", "content": "Hello! I am nanochat, a clean and idiomatic Rust port of nanochat using the Burn framework."}
            ]
        }
    ] * 1000

    # 9. Build SFT Training Mixture
    sft_train_tasks = []
    sft_train_tasks.extend(smoltalk_train_convs[:1000])
    sft_train_tasks.extend(identity_convs)
    sft_train_tasks.extend(identity_convs) # 2 epochs
    sft_train_tasks.extend(mmlu_train_convs[:1000])
    sft_train_tasks.extend(gsm_train_convs[:1000])
    sft_train_tasks.extend(ss_train_convs)
    sft_train_tasks.extend(sb_train_convs)
    
    # Shuffle training mixture deterministically
    rng = random.Random(42)
    rng.shuffle(sft_train_tasks)
    
    # Write SFT Train
    sft_path = os.path.join(script_dir, "sft_train.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for conv in sft_train_tasks:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Exported SFT training mixture to {sft_path} ({len(sft_train_tasks)} rows).")

    # 10. Write Evaluation Datasets
    with open(os.path.join(script_dir, "eval/mmlu.jsonl"), "w", encoding="utf-8") as f:
        for conv in mmlu_test_convs[:300]:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    with open(os.path.join(script_dir, "eval/gsm8k.jsonl"), "w", encoding="utf-8") as f:
        for conv in gsm_test_convs[:200]:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    with open(os.path.join(script_dir, "eval/spellingbee.jsonl"), "w", encoding="utf-8") as f:
        for conv in sb_test_convs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    with open(os.path.join(script_dir, "eval/humaneval.jsonl"), "w", encoding="utf-8") as f:
        for conv in he_convs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            
    print("All datasets downloaded and exported successfully!")

if __name__ == "__main__":
    export_datasets()
