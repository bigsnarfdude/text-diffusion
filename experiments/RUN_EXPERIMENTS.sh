#!/bin/bash
#
# Run Comprehensive Classification Experiments
#
# This script runs all experiments comparing 4 classification approaches:
# 1. GPT-2 Zero-Shot (no training)
# 2. GPT-2 Native (discriminative fine-tuning)
# 3. Diffusion Baseline (untrained models)
# 4. Diffusion Trained (per-class fine-tuning)
#
# Usage:
#   ./experiments/RUN_EXPERIMENTS.sh [quick|full]
#
# Options:
#   quick - Fast test on 100 samples (~15 minutes)
#   full  - Full comparison on 1000 samples (~6 hours)
#

set -e  # Exit on error

MODE=${1:-quick}

echo "======================================================================"
echo "COMPREHENSIVE CLASSIFICATION EXPERIMENTS"
echo "======================================================================"
echo ""
echo "Mode: $MODE"
echo ""

# Check if data is prepared
if [ ! -d "data/imdb-classifier" ]; then
    echo "⚠️  Data not found. Preparing IMDB dataset..."
    python scripts/prepare_imdb.py --max-samples 5000 --max-test 1000
    echo ""
fi

# Check if diffusion models are trained
if [ ! -d "results-generative-classifier" ]; then
    echo "======================================================================"
    echo "STEP 1: Training Diffusion Models (Per-Class Fine-tuning)"
    echo "======================================================================"
    echo ""
    echo "This trains separate RoBERTa models for each class."
    echo "Expected time: 2-3 hours"
    echo ""

    python src/train_generative_classifier.py \
        --data-dir data/imdb-classifier \
        --output-dir results-generative-classifier \
        --epochs 3 \
        --batch-size 8 \
        --learning-rate 5e-5

    echo ""
    echo "✅ Diffusion models trained!"
    echo ""
else
    echo "✅ Diffusion models already trained (results-generative-classifier/)"
    echo ""
fi

# Run comparison experiments
echo "======================================================================"
echo "STEP 2: Running Comparison Experiments"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    echo "Running QUICK comparison (100 test samples)..."
    echo "Expected time: 10-15 minutes"
    echo ""

    python experiments/compare_all_approaches.py \
        --dataset imdb \
        --data-dir data/imdb-classifier \
        --diffusion-model-dir results-generative-classifier \
        --output results/comparison \
        --quick

elif [ "$MODE" = "full" ]; then
    echo "Running FULL comparison (1000 test samples)..."
    echo "Expected time: 2-3 hours"
    echo ""

    python experiments/compare_all_approaches.py \
        --dataset imdb \
        --data-dir data/imdb-classifier \
        --diffusion-model-dir results-generative-classifier \
        --output results/comparison \
        --gpt2-epochs 3 \
        --num-likelihood-samples 5 \
        --mask-prob 0.15

else
    echo "❌ Invalid mode: $MODE"
    echo "Usage: $0 [quick|full]"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ EXPERIMENTS COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved to: results/comparison/"
echo ""
echo "To view results:"
echo "  cat results/comparison/comparison_imdb_*.json | jq '.approaches[] | {approach, accuracy: .metrics.accuracy}'"
echo ""
echo "Next steps:"
echo "  1. Review comparison table in output above"
echo "  2. Check statistical significance results"
echo "  3. Analyze results to answer key questions:"
echo "     - Does diffusion-trained beat diffusion-baseline significantly?"
echo "     - Is diffusion-trained competitive with GPT-2 native?"
echo "     - What are the efficiency trade-offs?"
echo ""
