# GPT-2 Generative Classifier Implementation

## Overview

This implementation addresses the fundamental architectural issue discovered in our initial experiments: **RoBERTa (Masked Language Model) cannot compute proper sequence probabilities, but GPT-2 (autoregressive LM) can.**

## The Problem with RoBERTa

Our initial diffusion-based approach used RoBERTa, which is a **Masked Language Model (MLM)**:

- **Bidirectional**: Sees both past and future context
- **Masked prediction**: Trained to predict masked tokens, not sequences
- **Cannot compute P(text)**: No way to get true sequence probability
- **Result**: 86.9% accuracy (worse than GPT-2 zero-shot's 89.4%)

## The GPT-2 Solution

GPT-2 is a **true generative autoregressive language model**:

- **Autoregressive**: Predicts next token given all previous tokens
- **Can compute P(text) exactly**: P(text) = P(w₁) × P(w₂|w₁) × ... × P(wₙ|w₁...wₙ₋₁)
- **Proper likelihood estimates**: Can use Bayes rule for classification
- **Zero-shot already strong**: 89.4% on IMDB without any training

## Architecture

### Training (`train_gpt2_generative.py`)

**Per-Class Training:**
```
For each class:
  1. Load all training examples for that class
  2. Fine-tune GPT-2 on those examples using language modeling objective
  3. Save per-class model
```

**Key Implementation Details:**
```python
# For GPT-2 LM training, labels = input_ids (shifted internally by model)
{
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': input_ids.clone()  # Standard LM training
}

# Training objective: minimize negative log-likelihood
outputs = model(**batch)
loss = outputs.loss  # Computed by GPT-2LMHeadModel
```

**Features:**
- FP16 mixed precision (reduces VRAM by ~50%)
- Gradient accumulation (effective larger batch sizes)
- Sequential training (one class at a time)
- Works with GPT-2 Small (~2-3 GB VRAM with FP16)

### Evaluation (`eval_gpt2_generative.py`)

**Classification Process:**
```
For each test example:
  1. Compute log P(text | class_0) using class_0 model
  2. Compute log P(text | class_1) using class_1 model
  3. ...
  4. Predict class with highest log P(text | class)
```

**Two Probability Computation Methods:**

**Method 1: Exact Token-by-Token (Default, Recommended)**
```python
# Compute exact: log P(text) = sum of log P(token_i | tokens_1...i-1)
logits = model(input_ids).logits
log_probs = F.log_softmax(logits, dim=-1)

# Get log prob of each actual token
token_log_probs = log_probs.gather(dim=-1, index=target_ids)
total_log_prob = token_log_probs.sum()
```

**Method 2: Loss-Based Approximation (Faster)**
```python
# Use GPT-2's built-in loss computation
outputs = model(input_ids, labels=input_ids)
loss = outputs.loss  # Average negative log-likelihood
log_prob = -loss * num_tokens
```

## Hardware Requirements

### Current Hardware: RTX 4070 Ti SUPER (16 GB)

**GPT-2 Small (124M params) - Default "gpt2"**
- Per model: ~4-5 GB (full precision)
- Per model: ~2-3 GB (FP16)
- **FITS** ✅ for 2-class (IMDB)
- **FITS** ✅ for 4-class (AG News) if trained sequentially

**Expected VRAM Usage:**
```
Training:
- Model weights: ~500 MB
- Optimizer (AdamW): ~1 GB
- Gradients: ~500 MB
- Activations (batch=8): ~2-3 GB
- FP16 Total: ~2-3 GB per model

Inference:
- Model weights: ~500 MB (or ~250 MB in FP16)
- For 2-class: ~1 GB total (both models loaded)
- For 4-class: ~2 GB total (all models loaded)
```

## Usage

### Quick Test (100 samples, 1 epoch)

```bash
# On nigel.birs.ca
cd ~/text-diffusion
source venv/bin/activate
./experiments/test_gpt2_quick.sh
```

This will:
1. Train 2 GPT-2 models (one per class) on 100 IMDB samples each
2. Evaluate on 100 test samples
3. Save results to `results-gpt2-test/`
4. Takes ~5-10 minutes

### Full Training (IMDB)

```bash
# Train on full IMDB dataset
python src/train_gpt2_generative.py \
    --data-dir data/imdb-classifier \
    --output-dir results-gpt2-generative \
    --model-size gpt2 \
    --epochs 3 \
    --batch-size 8 \
    --fp16

# Evaluate
python src/eval_gpt2_generative.py \
    --model-dir results-gpt2-generative \
    --data-dir data/imdb-classifier \
    --output results-gpt2-generative/eval_results.json
```

Expected time: ~30-45 minutes per class (IMDB with 5000 samples)

### Full Training (AG News - 4 classes)

```bash
# Train on AG News
python src/train_gpt2_generative.py \
    --data-dir data/agnews-classifier \
    --output-dir results-gpt2-generative-agnews \
    --model-size gpt2 \
    --epochs 3 \
    --batch-size 8 \
    --fp16

# Evaluate
python src/eval_gpt2_generative.py \
    --model-dir results-gpt2-generative-agnews \
    --data-dir data/agnews-classifier \
    --output results-gpt2-generative-agnews/eval_results.json
```

Expected time: ~2-3 hours (4 classes × 30-45 min each)

### Training Options

```bash
# Use larger model (if VRAM allows)
python src/train_gpt2_generative.py \
    --model-size gpt2-medium \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --fp16

# Train single class (for debugging)
python src/train_gpt2_generative.py \
    --class-id 0 \
    --quick-test

# Custom settings
python src/train_gpt2_generative.py \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --max-samples 1000
```

## Expected Results

### Hypothesis

If architecture is the issue (RoBERTa MLM vs GPT-2 LM), we expect:

**Current Results (RoBERTa):**
- Diffusion Trained: 86.9%
- GPT-2 Zero-Shot: 89.4%
- Gap: -2.5% (diffusion worse)

**Expected Results (GPT-2):**
- GPT-2 Generative Trained: **90-92%?**
- GPT-2 Zero-Shot: 89.4%
- Gap: +0.6-2.6% (trained better)

### Key Questions

1. **Does trained GPT-2 beat zero-shot GPT-2?**
   - Need: Trained > 89.4%
   - If yes: Per-class training helps

2. **Does GPT-2 generative beat discriminative?**
   - GPT-2 Native: 90.2%
   - GPT-2 Generative: ???
   - If competitive: Generative approach is viable

3. **Does GPT-2 generative beat RoBERTa diffusion?**
   - RoBERTa Diffusion: 86.9%
   - GPT-2 Generative: ???
   - If yes: Architecture was the issue

4. **Does it generalize to AG News?**
   - Same pattern on 4-class task?
   - Consistent improvements?

## Technical Advantages of GPT-2

1. **Exact Probabilities**: Can compute P(text) exactly using chain rule
2. **Proper Generative Model**: Trained to model text distribution
3. **Interpretable Likelihoods**: Each token's contribution to P(text) is clear
4. **Composable**: Can combine with priors: P(class|text) ∝ P(text|class) × P(class)
5. **No Masking Artifacts**: No need to sample masks (source of variance in RoBERTa)

## Next Steps

1. ✅ **Implemented**: Training and evaluation scripts
2. ⏳ **Current**: Quick test to verify implementation
3. 🔜 **Next**: Full IMDB training
4. 🔜 **Then**: Compare against all baselines
5. 🔜 **Finally**: AG News validation

## Files

- `src/train_gpt2_generative.py` - Training script (per-class GPT-2 LM)
- `src/eval_gpt2_generative.py` - Evaluation script (likelihood-based classification)
- `experiments/test_gpt2_quick.sh` - Quick test script (100 samples, 1 epoch)
- `experiments/GPT2_REQUIREMENTS.md` - Detailed VRAM requirements
- `experiments/GPT2_IMPLEMENTATION.md` - This file

## References

**Key Insight:**
> "RoBERTa is a Masked Language Model (MLM), not a true generative model. It cannot compute P(text) properly because it's bidirectional and trained for masked token prediction, not sequence modeling. GPT-2 is autoregressive and can compute exact sequence probabilities."

**Related Files:**
- `experiments/SUMMARY.md` - Experiment results summary
- `experiments/STATUS.md` - Current experiment status
- `docs/CLASSIFIER_RESULTS.md` - Initial classifier results
- `docs/FIXING_CLASSIFIER.md` - Analysis of why RoBERTa failed
