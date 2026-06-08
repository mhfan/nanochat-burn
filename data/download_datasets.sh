#!/bin/bash
set -e

echo "Creating data directories..."
mkdir -p data/eval
mkdir -p data/downloads

echo "Downloading Identity Conversations..."
curl -L -o data/downloads/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

echo "Downloading English Word List..."
curl -L -o data/downloads/words_alpha.txt https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt

echo "Downloading GSM8K Train & Test..."
curl -L -o data/downloads/gsm_train.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
curl -L -o data/downloads/gsm_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

echo "Downloading HumanEval..."
curl -L -o data/downloads/HumanEval.jsonl.gz https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz

echo "Downloading ARC Easy & Challenge..."
curl -L -o data/downloads/ARC-Easy-Test.jsonl https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/ARC-Easy-Test.jsonl
curl -L -o data/downloads/ARC-Challenge-Test.jsonl https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/ARC-Challenge-Test.jsonl

echo "Downloading MMLU Train & Test subjects..."
MMLU_SUBJECTS=("abstract_algebra" "anatomy" "astronomy" "business_ethics" "clinical_knowledge")
for sub in "${MMLU_SUBJECTS[@]}"; do
    echo "  Subject: $sub..."
    curl -L -o "data/downloads/${sub}_train.csv" "https://raw.githubusercontent.com/hendrycks/test/master/data/dev/${sub}_dev.csv"
    curl -L -o "data/downloads/${sub}_test.csv" "https://raw.githubusercontent.com/hendrycks/test/master/data/test/${sub}_test.csv"
done

echo "All raw downloads completed successfully!"
