#!/bin/bash
# Fixed MDLM Training Pipeline with Normalized Log Probs and Extended Training
# Addresses key issues:
# - Normalized log probabilities by sequence length
# - Bayesian classification with proper priors
# - Increased training steps from 20k to 50k
# - Uses pretrained weights for faster convergence

set -e  # Exit on error

OUTPUT_DIR="results-mdlm-fixed"
LOG_FILE="mdlm_fixed_training.log"

echo "========================================"
echo "MDLM Fixed Training Pipeline"
echo "========================================"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

# Activate virtual environment
source venv/bin/activate

# Clear previous results
echo "Cleaning previous results..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Phase 1: Train Class 0 (Negative Reviews)
echo "========================================"
echo "Phase 1: Training Class 0 (Negative)"
echo "========================================"
echo "Expected time: ~2-3 hours (50k steps)"
echo ""

python train_mdlm_classifier.py \
    --class_id=0 \
    --data_file=data/imdb-combined/train_class_0.json \
    --output_dir="$OUTPUT_DIR/class_0" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "✅ Class 0 training complete"
echo ""

# Phase 2: Train Class 1 (Positive Reviews)
echo "========================================"
echo "Phase 2: Training Class 1 (Positive)"
echo "========================================"
echo "Expected time: ~2-3 hours (50k steps)"
echo ""

python train_mdlm_classifier.py \
    --class_id=1 \
    --data_file=data/imdb-combined/train_class_1.json \
    --output_dir="$OUTPUT_DIR/class_1" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "✅ Class 1 training complete"
echo ""

# Phase 3: Evaluate with Fixed Classifier
echo "========================================"
echo "Phase 3: Evaluation (Fixed Classifier)"
echo "========================================"
echo ""

# Find best checkpoints
CLASS_0_CKPT=$(find "$OUTPUT_DIR/class_0/checkpoints" -name "*.ckpt" -type f | grep -v "last.ckpt" | sort -V | tail -1)
CLASS_1_CKPT=$(find "$OUTPUT_DIR/class_1/checkpoints" -name "*.ckpt" -type f | grep -v "last.ckpt" | sort -V | tail -1)

if [ -z "$CLASS_0_CKPT" ]; then
    CLASS_0_CKPT="$OUTPUT_DIR/class_0/checkpoints/last.ckpt"
fi

if [ -z "$CLASS_1_CKPT" ]; then
    CLASS_1_CKPT="$OUTPUT_DIR/class_1/checkpoints/last.ckpt"
fi

echo "Using checkpoints:"
echo "  Class 0: $CLASS_0_CKPT"
echo "  Class 1: $CLASS_1_CKPT"
echo ""

python eval_mdlm_classifier.py \
    --model_0="$CLASS_0_CKPT" \
    --model_1="$CLASS_1_CKPT" \
    --test_file=data/imdb-combined/test_1000.json \
    --output_file="$OUTPUT_DIR/evaluation_results.json" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================"
echo "✅ MDLM Fixed Pipeline Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Class 0 model: $OUTPUT_DIR/class_0/"
echo "  - Class 1 model: $OUTPUT_DIR/class_1/"
echo "  - Evaluation: $OUTPUT_DIR/evaluation_results.json"
echo "  - Log: $LOG_FILE"
echo ""
echo "Key improvements:"
echo "  ✅ Normalized log probabilities by sequence length"
echo "  ✅ Bayesian classification with proper priors"
echo "  ✅ 50k training steps (vs 20k before) for better convergence"
echo "  ✅ Pretrained MDLM weights for faster learning"
echo ""
echo "Target: 85-90% accuracy (comparable to GPT-2 baseline at 90.1%)"
echo "Previous: 57.8% accuracy (broken due to unnormalized log probs)"
echo ""
