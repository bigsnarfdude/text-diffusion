#!/bin/bash
# Quick test of GPT-2 generative classifier
# Tests with 100 samples per class, 1 epoch

set -e

cd ~/text-diffusion
source venv/bin/activate

echo "========================================"
echo "GPT-2 GENERATIVE CLASSIFIER - QUICK TEST"
echo "========================================"
echo ""

# Train on IMDB with minimal data
python src/train_gpt2_generative.py \
    --data-dir data/imdb-classifier \
    --output-dir results-gpt2-test \
    --model-size gpt2 \
    --epochs 1 \
    --batch-size 8 \
    --max-samples 100 \
    --fp16

echo ""
echo "Training complete! Now evaluating..."
echo ""

# Evaluate
python src/eval_gpt2_generative.py \
    --model-dir results-gpt2-test \
    --data-dir data/imdb-classifier \
    --max-test 100 \
    --output results-gpt2-test/eval_results.json

echo ""
echo "========================================"
echo "QUICK TEST COMPLETE"
echo "========================================"
