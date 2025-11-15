# Synthetic Data Generation for Classifier Training

## Overview

Using Ollama gpt-oss:20b (locally running on nigel.birs.ca) to generate synthetic Amazon product reviews for data augmentation.

## Current Generation in Progress

**Status**: 🔄 Generating 10,000 synthetic reviews
- Model: Ollama gpt-oss:20b (20 billion parameter reasoning model)
- Speed: ~14 seconds per review
- Total time estimate: ~38 hours for 10,000 samples
- Output: `data/synthetic-amazon/`

**Progress monitoring:**
```bash
ssh vincent@nigel.birs.ca "tail -f ~/synthetic-generation.log"
```

## Quality Assessment

**Sample Negative Review:**
> "I bought the universal car seat belt installer, but it fell apart after just a week. The plastic brackets snapped off and the device didn't fit my SUV's seat frame, leaving the belt still unsecured. I wasted money and now have to look for a replacement that actually works."

**Sample Positive Review:**
> "I've been using the SmartTreat Dog Treat Dispenser for the past month, and my pup is absolutely obsessed! The adjustable difficulty settings keep her mentally stimulated, and the high‑quality, all‑natural ingredients give her a healthy boost of energy. Plus, the durable silicone construction has withstood countless chewing sessions—total game changer!"

**Quality Features:**
- ✅ Realistic product reviews (specific products, details, emotions)
- ✅ Appropriate length (2-4 sentences, 50-100 words)
- ✅ Clear sentiment (positive vs negative distinction)
- ✅ Product variety (electronics, pet supplies, automotive, home goods)
- ✅ Natural language (contractions, emphasis, conversational tone)

## Generation Timeline

| Stage | Samples | Time Estimate | Status |
|-------|---------|---------------|--------|
| Test (10 samples) | 10 | 2-3 minutes | ✅ Complete |
| Full generation | 10,000 | ~38 hours | 🔄 In Progress |
| Negative reviews | 5,000 | ~19 hours | 🔄 In Progress |
| Positive reviews | 5,000 | ~19 hours | ⏳ Pending |

**Speed Calculation:**
- 14 seconds/review × 10,000 reviews = 140,000 seconds = 38.9 hours

## Dataset Augmentation Plan

### Phase 1: Generate Synthetic Data (Current)
```bash
# Generate 10,000 synthetic Amazon reviews
python scripts/generate_synthetic_amazon.py --num-samples 10000
```

**Output:**
- 5,000 negative reviews
- 5,000 positive reviews
- Saved to: `data/synthetic-amazon/`

### Phase 2: Merge with Real IMDB Data
```bash
# Merge IMDB + synthetic Amazon
python scripts/merge_datasets.py \
    --datasets data/imdb-classifier data/synthetic-amazon \
    --output data/imdb-augmented
```

**Expected Dataset Sizes:**
- IMDB negative: 12,500
- IMDB positive: 12,500
- Synthetic negative: 5,000
- Synthetic positive: 5,000
- **Total negative: 17,500**
- **Total positive: 17,500**
- **Total training examples: 35,000** (vs current 25,000)

### Phase 3: Train on Augmented Dataset
```bash
# Train GPT-2 on augmented dataset
python src/train_gpt2_generative.py \
    --data-dir data/imdb-augmented \
    --output-dir results-gpt2-augmented \
    --model-size gpt2 \
    --epochs 3 \
    --batch-size 8 \
    --fp16
```

**Expected Improvement:**
- More training data → better generalization
- Mixed domains (movie + product reviews) → more robust features
- Current: 86.6% accuracy
- Target: 89-90%+ accuracy

## Parallel Processing Strategy

**Currently Running on nigel.birs.ca:**

1. **GPU (RTX 4070 Ti):**
   - GPT-2 full dataset training (12,500 per class)
   - VRAM: 9.6 GB / 16 GB
   - ETA: ~1.5-2 hours

2. **CPU (Ollama):**
   - Synthetic data generation (10,000 samples)
   - Using gpt-oss:20b model
   - ETA: ~38 hours

**Efficient Resource Usage:**
- GPU busy with training
- CPU busy with generation
- No resource contention

## Alternative: Faster Generation Options

If 38 hours is too long, alternatives:

### Option 1: Reduce Sample Count
```bash
# Generate 2,000 samples instead (7.8 hours)
python scripts/generate_synthetic_amazon.py --num-samples 2000
```

### Option 2: Increase Temperature (Faster but Lower Quality)
```bash
# Higher temperature = faster but more random
python scripts/generate_synthetic_amazon.py \
    --num-samples 10000 \
    --temperature 1.2
```

### Option 3: Use Real Amazon Dataset Instead
```bash
# Download pre-made Amazon Polarity dataset (3.6M samples)
python scripts/prepare_amazon.py --max-samples 10000
# Much faster: ~2 minutes vs 38 hours
```

### Option 4: Parallel Generation (Multiple Processes)
If Ollama can handle it:
```bash
# Run 4 parallel generators (divide by 4 = 9.7 hours)
for i in {0..3}; do
    python scripts/generate_synthetic_amazon.py \
        --num-samples 2500 \
        --output-dir data/synthetic-amazon-part-$i &
done
```

## Experiment Timeline

**Current Status: 2025-11-02 16:30 PST**

| Task | Status | ETA | Notes |
|------|--------|-----|-------|
| GPT-2 Full Training | 🔄 In Progress | ~18:00 PST | 12,500 samples/class |
| Synthetic Generation | 🔄 In Progress | ~Nov 4, 06:00 PST | 10,000 samples |
| Merge Datasets | ⏳ Pending | After generation | Quick (~1 minute) |
| Train on Augmented | ⏳ Pending | After merge | ~2 hours |
| Evaluation | ⏳ Pending | After training | ~10 minutes |

## Expected Results

**Current Baseline (12,500 per class):**
- GPT-2 Generative: TBD (training now)
- Previous (5,000 per class): 86.6%

**Augmented Dataset (17,500 per class):**
- Expected: 88-90%+
- Hypothesis: More data → better likelihood estimates → better classification

**Success Criteria:**
- Beat current 86.6%
- Approach or beat zero-shot GPT-2 (89.4%)
- Competitive with discriminative GPT-2 (90.2%)

## Files

**Generation Scripts:**
- `scripts/generate_synthetic_amazon.py` - Synthetic review generator
- `scripts/merge_datasets.py` - Dataset merger

**Log Files:**
- `~/gpt2-full-training.log` - GPT-2 training progress
- `~/synthetic-generation.log` - Synthetic data generation progress

**Output Directories:**
- `data/synthetic-amazon/` - Synthetic reviews
- `data/imdb-augmented/` - Merged dataset (IMDB + synthetic)
- `results-gpt2-full/` - Full dataset trained models
- `results-gpt2-augmented/` - Augmented dataset trained models

## Monitoring Commands

```bash
# Check GPT-2 training progress
ssh vincent@nigel.birs.ca "tail -50 ~/gpt2-full-training.log"

# Check synthetic generation progress
ssh vincent@nigel.birs.ca "tail -50 ~/synthetic-generation.log"

# Check GPU usage
ssh vincent@nigel.birs.ca "nvidia-smi"

# Check running processes
ssh vincent@nigel.birs.ca "ps aux | grep python"
```
