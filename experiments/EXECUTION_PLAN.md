# Execution Plan: Comprehensive Classification Experiments

## Status: Ready to Execute

✅ Framework implemented and tested
✅ Data prepared (5000 train samples per class, 1000 test samples)
✅ Quick test passed
⏳ Awaiting execution of full experiments

## What We Built

### 1. Comparison Framework (`compare_all_approaches.py`)

A comprehensive experiment harness that compares **4 different classification approaches**:

1. **GPT-2 Zero-Shot**: Perplexity-based classification (no training)
2. **GPT-2 Native**: Standard discriminative fine-tuning
3. **Diffusion Baseline**: Likelihood-based with untrained models
4. **Diffusion Trained**: Likelihood-based with per-class trained models

### 2. Statistical Testing

- McNemar's test for pairwise significance
- Comparison tables with accuracy/precision/recall/F1
- Runtime measurements
- Full JSON output for reproducibility

### 3. Validation

Quick test passed on 10 examples:
- GPT-2 zero-shot: 90% accuracy (on tiny sample)
- Statistical tests working correctly
- All components functional

## Execution Steps

### Step 1: Train Diffusion Models (Required for Approach 4)

```bash
# Train per-class diffusion models
python src/train_generative_classifier.py \
    --data-dir data/imdb-classifier \
    --output-dir results-generative-classifier \
    --epochs 3 \
    --mask-prob 0.15 \
    --batch-size 8 \
    --learning-rate 5e-5
```

**Expected time**: ~2-3 hours (5000 samples × 2 classes × 3 epochs)
**Output**: `results-generative-classifier/class_0/` and `class_1/` with trained models

### Step 2: Run Quick Comparison (100 test samples)

```bash
# Quick test on 100 samples to verify everything works
python experiments/compare_all_approaches.py \
    --dataset imdb \
    --data-dir data/imdb-classifier \
    --diffusion-model-dir results-generative-classifier \
    --output results/comparison \
    --quick
```

**Expected time**: ~10-15 minutes
**Purpose**: Verify all 4 approaches work end-to-end

### Step 3: Run Full Comparison (1000 test samples)

```bash
# Full comparison on all test data
python experiments/compare_all_approaches.py \
    --dataset imdb \
    --data-dir data/imdb-classifier \
    --diffusion-model-dir results-generative-classifier \
    --output results/comparison \
    --gpt2-epochs 3 \
    --num-likelihood-samples 5 \
    --mask-prob 0.15
```

**Expected time**: ~2-3 hours
**Output**: JSON file in `results/comparison/comparison_imdb_TIMESTAMP.json`

### Step 4: Analyze Results

Check output for:

1. **Accuracy comparison**:
   - Which approach performs best?
   - Gap between baseline and trained diffusion?

2. **Statistical significance**:
   - Is diffusion-trained significantly better than diffusion-baseline?
   - How does it compare to GPT-2 native?

3. **Efficiency**:
   - Runtime per approach
   - Accuracy/compute trade-offs

## Expected Hypotheses Validation

### H1: GPT-2 zero-shot > random but < fine-tuned
**Prediction**: 0.6 < accuracy < 0.8
**Rationale**: Pretrained model understands sentiment but not optimized

### H2: GPT-2 native strong baseline
**Prediction**: accuracy > 0.85
**Rationale**: Standard fine-tuning works well on IMDB

### H3: Diffusion baseline ≈ random
**Prediction**: 0.5 < accuracy < 0.6
**Rationale**: Same model for all classes can't discriminate

### H4: Diffusion trained >> diffusion baseline
**Prediction**: accuracy gap > 0.2, p < 0.05
**Rationale**: **Core claim** - per-class training enables discrimination

### H5: Diffusion trained ≈ GPT-2 native
**Prediction**: accuracy within 0.05
**Rationale**: If true, validates generative approach

## Key Questions to Answer

### Question 1: Does our approach work at all?

**Test**: Compare diffusion-baseline vs diffusion-trained
**Success criteria**: Significant improvement (p < 0.05) with accuracy gap > 0.15
**Why it matters**: This validates our core hypothesis

### Question 2: Is our approach competitive?

**Test**: Compare diffusion-trained vs GPT-2 native
**Success criteria**: Accuracy within 0.05 or better
**Why it matters**: Determines if generative approach is viable alternative

### Question 3: What's the efficiency trade-off?

**Test**: Compare runtime and accuracy
**Analysis**: Does diffusion-trained justify its computational cost?
**Why it matters**: Practical deployment considerations

## Potential Outcomes & Next Steps

### Outcome A: Diffusion-trained >> baseline, competitive with GPT-2
**Interpretation**: Success! Approach validated.
**Next steps**:
- Scale to larger datasets
- Try more complex tasks (multi-class, long-form text)
- Optimize inference speed
- Write paper

### Outcome B: Diffusion-trained > baseline, but << GPT-2
**Interpretation**: Approach works but underperforms.
**Next steps**:
- Investigate why: likelihood estimation? training?
- Try different architectures (GPT instead of RoBERTa?)
- Tune hyperparameters (more epochs, better masking?)

### Outcome C: Diffusion-trained ≈ baseline (both poor)
**Interpretation**: Core hypothesis false - likelihood-based classification doesn't work.
**Next steps**:
- Analyze failure modes
- Check if models are actually learning different distributions
- Consider fundamental changes (different generative model?)

### Outcome D: Unexpected results (e.g., baseline > trained)
**Interpretation**: Bug or implementation error.
**Next steps**:
- Debug thoroughly
- Check model loading/training
- Verify likelihood computation

## Files Created

```
experiments/
├── README.md                      # Experiment design documentation
├── EXECUTION_PLAN.md             # This file - step-by-step guide
├── compare_all_approaches.py     # Main comparison script
└── quick_test.py                 # Validation test (passed ✅)

data/
└── imdb-classifier/              # Prepared dataset
    ├── metadata.json
    ├── train_class_0.json        # 5000 negative examples
    ├── train_class_1.json        # 5000 positive examples
    └── test.json                 # 1000 test examples
```

## Timeline Estimate

| Step | Task | Time | Status |
|------|------|------|--------|
| 1 | Train diffusion models | 2-3 hours | ⏳ Pending |
| 2 | Quick comparison test | 10-15 min | ⏳ Pending |
| 3 | Full comparison | 2-3 hours | ⏳ Pending |
| 4 | Analysis & documentation | 1 hour | ⏳ Pending |
| **Total** | | **~6 hours** | |

## Notes

- All experiments use **identical test sets** for fair comparison
- Results include **full configuration** for reproducibility
- Statistical tests account for **paired predictions**
- Framework is **extensible** - easy to add new approaches

## Ready to Execute

Everything is set up and validated. You can now run:

```bash
# Step 1: Train models
python src/train_generative_classifier.py --data-dir data/imdb-classifier --epochs 3

# Step 2: Quick test
python experiments/compare_all_approaches.py --quick

# Step 3: Full comparison
python experiments/compare_all_approaches.py --dataset imdb

# Step 4: Review results
cat results/comparison/comparison_imdb_*.json | jq '.approaches[] | {approach: .approach, accuracy: .metrics.accuracy}'
```

The framework will definitively answer whether our diffusion-based generative classification approach works compared to strong baselines.
