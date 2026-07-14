import os
import json
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "eval"), exist_ok=True)

WORDS = ["strawberry", "apple", "banana", "programming", "rust", "framework", "intelligence", "computer", "system", "hardware"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"

def gen_identity():
    convs = []
    queries = [
        ("Who are you?", "I am nanochat, a clean, idiomatic Rust port of nanochat using the Burn framework."),
        ("What is your name?", "My name is nanochat."),
        ("Explain your architecture.", "I am built on the Burn framework in Rust. I use a custom GPT block with RoPE, Untied Weights, ReLU² activation, QK Norm, and Sliding Window Attention."),
        ("Who developed you?", "I was developed in Rust to explore high-performance LLM systems engineering.")
    ]
    for q, a in queries:
        convs.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    return convs

def gen_smoltalk():
    convs = []
    chats = [
        ("Hello!", "Hello! How can I help you today?"),
        ("What is system programming?", "System programming is the activity of computer programming, where the software being written provides services to the computer hardware."),
        ("Why use Rust?", "Rust provides memory safety without garbage collection, modern developer tooling, and high performance, making it ideal for systems programming."),
        ("What is deep learning?", "Deep learning is a subset of machine learning based on artificial neural networks with representation learning.")
    ]
    for q, a in chats:
        convs.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })
    return convs * 10  # 40 examples

def gen_mmlu_arc(name):
    convs = []
    questions = [
        ("Which language is compiled to native machine code?", "A", ["Python", "Rust", "JavaScript", "HTML"]),
        ("What is the primary CPU cache level that is fastest?", "C", ["L3", "L2", "L1", "RAM"]),
        ("Which activation function has a squared formulation in nanochat?", "B", ["GELU", "ReLU²", "SwiGLU", "Sigmoid"]),
        ("What is the complexity of standard matrix multiplication?", "D", ["O(N)", "O(N log N)", "O(N²)", "O(N³)"]),
        ("Which framework is used for deep learning in nanochat?", "A", ["Burn", "PyTorch", "TensorFlow", "Keras"])
    ]
    for q, ans, choices in questions:
        letters = ["A", "B", "C", "D"]
        query = f"Multiple Choice question: {q}\n"
        query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
        query += "\nRespond only with the letter of the correct answer."
        convs.append({
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": ans}
            ],
            "letters": letters,
            "subject": name
        })
    return convs * 10  # 50 examples

def gen_gsm8k():
    convs = []
    problems = [
        ("If John has 5 apples and buys 3 more, how many does he have?", "John starts with 5. He buys 3. <<5+3=8>> John has 8 apples. #### 8"),
        ("A box holds 12 blocks. If you remove 4 blocks, how many remain?", "Start with 12 blocks. Remove 4. <<12-4=8>> There are 8 blocks remaining. #### 8"),
        ("If John earns $15 per hour and works 2 hours, how much does he earn?", "Earns 15 per hour. Works 2 hours. <<15*2=30>> John earns $30. #### 30"),
        ("A bag has 20 marbles. If you share them equally among 4 friends, how many does each get?", "Total marbles is 20. Divide by 4. <<20/4=5>> Each gets 5. #### 5"),
    ]
    import re
    for q, ans in problems:
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', ans)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                expr, result = inner.rsplit('=', 1) if '=' in inner else (inner, "")
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})
        convs.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant_message_parts}
            ]
        })
    return convs * 10  # 40 examples

def gen_spelling_bee():
    convs = []
    for i in range(50):
        rng = random.Random(i)
        word = rng.choice(WORDS)
        letter = rng.choice(word)
        count = word.count(letter)
        
        user_msg = f"How many {letter} are in the word {word}?"
        word_letters = ",".join(list(word))
        manual_text = f"We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.\n\nFirst spell the word out:\n{word}:{word_letters}\n\nThen count the occurrences of '{letter}':\n"
        
        running_count = 0
        for idx, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{idx}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{idx}:{char}\n"
                
        manual_text += f"\nThis gives us {running_count}."
        assistant_parts = [
            {"type": "text", "text": manual_text},
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"},
            {"type": "python", "text": f"'{word}'.count('{letter}')"},
            {"type": "python_output", "text": str(count)},
            {"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"}
        ]
        
        convs.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts}
            ]
        })
    return convs

def gen_simple_spelling():
    convs = []
    for i in range(50):
        rng = random.Random(i + 100)
        word = rng.choice(WORDS)
        word_letters = ",".join(list(word))
        convs.append({
            "messages": [
                {"role": "user", "content": f"Spell the word: {word}"},
                {"role": "assistant", "content": f"{word}:{word_letters}"}
            ]
        })
    return convs

def gen_humaneval():
    convs = []
    problems = [
        {
            "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b\"\"\"\n",
            "solution": "    return a + b",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"
        },
        {
            "prompt": "def multiply(a, b):\n    \"\"\"Return the product of a and b\"\"\"\n",
            "solution": "    return a * b",
            "entry_point": "multiply",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 6\n    assert candidate(0, 5) == 0\n"
        },
        {
            "prompt": "def is_even(n):\n    \"\"\"Return True if n is even, False otherwise\"\"\"\n",
            "solution": "    return n % 2 == 0",
            "entry_point": "is_even",
            "test": "def check(candidate):\n    assert candidate(2) is True\n    assert candidate(3) is False\n"
        }
    ]
    for p in problems:
        messages = [
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": f"{p['prompt']}{p['solution']}"}
        ]
        convs.append({
            "messages": messages,
            "entry_point": p["entry_point"],
            "test": p["test"]
        })
    return convs * 10 # 30 examples

def main():
    print("Generating local SFT training dataset...")
    sft_train = []
    sft_train.extend(gen_smoltalk())
    sft_train.extend(gen_identity())
    sft_train.extend(gen_identity())  # 2 epochs
    sft_train.extend(gen_mmlu_arc("mmlu"))
    sft_train.extend(gen_gsm8k())
    sft_train.extend(gen_simple_spelling())
    sft_train.extend(gen_spelling_bee())
    
    random.Random(42).shuffle(sft_train)
    
    sft_path = os.path.join(script_dir, "sft_train.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for conv in sft_train:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Exported SFT train to {sft_path} ({len(sft_train)} rows)")

    # Export eval sets
    evals = {
        "arc_easy": gen_mmlu_arc("arc_easy"),
        "arc_challenge": gen_mmlu_arc("arc_challenge"),
        "mmlu": gen_mmlu_arc("mmlu"),
        "gsm8k": gen_gsm8k(),
        "spellingbee": gen_spelling_bee(),
        "humaneval": gen_humaneval()
    }
    
    for name, data in evals.items():
        eval_path = os.path.join(script_dir, f"eval/{name}.jsonl")
        with open(eval_path, "w", encoding="utf-8") as f:
            for conv in data:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"Exported eval {name} to {eval_path} ({len(data)} rows)")

    print("All synthetic datasets generated perfectly!")

if __name__ == "__main__":
    main()
