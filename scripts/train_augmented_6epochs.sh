#!/bin/bash
# Train GPT-2 on augmented dataset (IMDB + Synthetic) with 6 epochs
# Run this AFTER synthetic generation completes

set -e  # Exit on error

cd ~/text-diffusion

echo "=================================================="
echo "GPT-2 AUGMENTED TRAINING (6 EPOCHS)"
echo "=================================================="
echo ""

# Check if synthetic data exists
if [ ! -d "data/synthetic-imdb" ]; then
    echo "❌ Synthetic data not found at data/synthetic-imdb/"
    echo "   Run merge first: python scripts/merge_datasets.py"
    exit 1
fi

# Check if augmented dataset exists
if [ ! -d "data/imdb-augmented" ]; then
    echo "Step 1: Merging datasets..."
    echo "  - IMDB: 25,000 samples (12,500 per class)"
    echo "  - Synthetic: 10,000 samples (5,000 per class)"
    echo "  - Total: 35,000 samples (17,500 per class)"
    echo ""

    python scripts/merge_datasets.py \
        --datasets data/imdb-classifier data/synthetic-imdb \
        --output data/imdb-augmented

    echo ""
    echo "✅ Dataset merge complete"
    echo ""
fi

# Train with 6 epochs
echo "Step 2: Training GPT-2 on augmented dataset..."
echo "  - Dataset: data/imdb-augmented"
echo "  - Total samples: 35,000 (17,500 per class)"
echo "  - Epochs: 6 (vs 3 previously)"
echo "  - Expected time: ~30-40 minutes"
echo "  - Output: results-gpt2-augmented/"
echo ""
echo "Starting training..."
echo ""

nohup python src/train_gpt2_generative.py \
    --data-dir data/imdb-augmented \
    --output-dir results-gpt2-augmented \
    --model-size gpt2 \
    --epochs 6 \
    --batch-size 8 \
    --fp16 > ~/gpt2-augmented-training.log 2>&1 &

TRAIN_PID=$!
echo "Training started (PID: $TRAIN_PID)"
echo ""
echo "Monitor progress with:"
echo "  tail -f ~/gpt2-augmented-training.log"
echo ""
echo "Or check GPU usage:"
echo "  nvidia-smi"
echo ""

# Wait a moment for training to start
sleep 3

# Show initial output
echo "Initial output:"
tail -20 ~/gpt2-augmented-training.log
