# MDLM Full Implementation - Deployment Summary

## Date: November 4, 2025

## Overview
Deployed full MDLM (Masked Diffusion Language Model) training pipeline for generative classification on nigel.birs.ca.

## What Was Accomplished

### 1. Training Infrastructure Created ✅
- **train_mdlm_classifier.py**: Per-class MDLM training script
- **eval_mdlm_classifier.py**: Generative classifier evaluation script
- **run_mdlm_full_training.sh**: Complete pipeline orchestration

### 2. MDLM Configuration
- **Model**: Small DiT (768 hidden, 12 blocks, 12 heads)
- **Sequence Length**: 512 tokens
- **Batch Size**: 8 (effective 32 with gradient accumulation)
- **Epochs**: 20 per class
- **Diffusion**: Substitution-based (subs parameterization)
- **Tokenizer**: GPT-2

### 3. Training Strategy
**Per-Class Training**:
- Class 0 (Negative): 9,460 IMDB samples → `results-mdlm/class_0/`
- Class 1 (Positive): 9,673 IMDB samples → `results-mdlm/class_1/`

**Classification Method**:
```python
# Generative classification via Bayes rule
log_p_0 = -model_0.compute_nll(text)  # log P(text | negative)
log_p_1 = -model_1.compute_nll(text)  # log P(text | positive)
predicted_class = argmax(log_p_0, log_p_1)
```

### 4. Dependency Resolution
**Challenge**: MDLM requires `flash-attn==2.5.6` (GPU-specific, compiled)

**Solution**: Installing flash-attn in venv on nigel.birs.ca
- CUDA 12.8 available on RTX 4070 Ti SUPER
- Installation in progress (5-10 minutes compile time)

### 5. Screen Session Management
**Session**: `mdlm-training` (screen ID: 54767)

**Commands**:
```bash
# Check screen sessions
screen -ls

# Attach to training session
screen -r mdlm-training

# Detach from session (keeps running)
Ctrl+A, D

# View training logs
tail -f ~/text-diffusion/mdlm_training.log

# Kill screen session
screen -S mdlm-training -X quit
```

## File Locations on nigel.birs.ca

### Scripts
- `~/text-diffusion/train_mdlm_classifier.py`
- `~/text-diffusion/eval_mdlm_classifier.py`
- `~/text-diffusion/run_mdlm_full_training.sh`

### Data
- `~/text-diffusion/data/imdb-combined/train_class_0.json` (9,460 samples)
- `~/text-diffusion/data/imdb-combined/train_class_1.json` (9,673 samples)
- `~/text-diffusion/data/imdb-combined/test.json` (1,000 samples)

### Results (will be created)
- `~/text-diffusion/results-mdlm/class_0/` - Negative class MDLM checkpoints
- `~/text-diffusion/results-mdlm/class_1/` - Positive class MDLM checkpoints
- `~/text-diffusion/results-mdlm/evaluation_results.json` - Final metrics

### Logs
- `~/text-diffusion/mdlm_training.log` - Complete training output

## Expected Timeline

### Phase 1: Class 0 Training (Negative)
- **Data**: 9,460 samples
- **Epochs**: 20
- **Estimated Time**: 6-12 hours
- **Checkpoints**: Every 1000 steps

### Phase 2: Class 1 Training (Positive)
- **Data**: 9,673 samples
- **Epochs**: 20
- **Estimated Time**: 6-12 hours
- **Checkpoints**: Every 1000 steps

### Phase 3: Evaluation
- **Test Set**: 1,000 samples
- **Per-Sample NLL Computation**: ~2-5 seconds
- **Estimated Time**: 30-60 minutes

**Total Pipeline Time**: 12-24 hours

## Success Criteria

### Excellent (>90%)
- ✅ Matches/exceeds GPT-2's 90.1% accuracy
- ✅ Validates MDLM for generative classification
- ✅ State-of-the-art discrete diffusion result

### Good (85-90%)
- ✅ Competitive with GPT-2
- ✅ Confirms discrete diffusion viability
- ✅ Significant improvement over RoBERTa MLM (61.7%)

### Acceptable (80-85%)
- ⚠️ Shows promise
- ⚠️ May need hyperparameter tuning
- ✅ Still beats RoBERTa MLM approach

### Needs Improvement (<80%)
- ❌ Comparable to failed RoBERTa approach
- ❌ May indicate fundamental issues

## Comparison Context

### Current Baselines (IMDB Sentiment)
1. **GPT-2 Native (Real IMDB)**: 90.1% ⭐⭐⭐⭐⭐⭐⭐⭐⭐
2. **Simple Discrete Diffusion**: 88.5% ⭐⭐⭐⭐⭐⭐⭐⭐ ← POC SUCCESS
3. **GPT-2 Native (Real + Synthetic)**: 87.5% ⭐⭐⭐⭐⭐⭐⭐⭐
4. **RoBERTa "Diffusion" (MLM)**: 61.7% ⭐⭐⭐⭐⭐⭐

**Target**: MDLM should match or exceed Simple Diffusion (88.5%)

## Technical Architecture

### MDLM Model (DiT-based)
```python
{
    'type': 'ddit',              # Diffusion Transformer
    'hidden_size': 768,          # Small model size
    'n_blocks': 12,              # 12 transformer layers
    'n_heads': 12,               # 12 attention heads
    'length': 512,               # Max sequence length
    'parameterization': 'subs',  # Substitution diffusion
    'dropout': 0.1,
}
```

### Training Configuration
```python
{
    'max_epochs': 20,
    'batch_size': 8,
    'accumulate_grad_batches': 4,  # Effective batch = 32
    'gradient_clip_val': 1.0,
    'lr': 2e-4,
    'warmup_steps': 500,
}
```

### Diffusion Process
- **Forward**: Add substitution noise to tokens
- **Reverse**: Denoise using learned model
- **Noise Schedule**: Log-linear
- **Sampling**: DDPM predictor (100 steps)

## Next Steps (After Installation)

1. ✅ Flash-attn installation completes
2. 🔄 Restart training pipeline in screen
3. 📊 Monitor training logs
4. ⏰ Wait 12-24 hours for both classes
5. 📈 Evaluate on test set
6. 📝 Compare with baselines
7. 🎉 Celebrate if >88.5% accuracy!

## Monitoring Commands

```bash
# Check if training is running
ssh vincent@nigel.birs.ca "screen -ls"

# View live training logs
ssh vincent@nigel.birs.ca "tail -f ~/text-diffusion/mdlm_training.log"

# Check GPU usage
ssh vincent@nigel.birs.ca "nvidia-smi"

# Check recent results
ssh vincent@nigel.birs.ca "ls -lth ~/text-diffusion/results-mdlm/"

# Count checkpoints
ssh vincent@nigel.birs.ca "ls ~/text-diffusion/results-mdlm/class_*/checkpoint-* | wc -l"
```

## What Makes This Different from Simple Diffusion POC

### Simple Diffusion (88.5% - POC)
- ✅ Uses GPT-2 language model directly
- ✅ Simple NLL computation
- ✅ Fast proof-of-concept
- ⚠️ Not "true" discrete diffusion

### MDLM (This Implementation)
- ✅ TRUE discrete text diffusion (substitution-based)
- ✅ State-of-the-art DiT architecture
- ✅ Proper diffusion noise schedules
- ✅ Semi-autoregressive generation
- ⚠️ More complex (flash attention, DiT)
- ⚠️ Longer training time

**Goal**: Demonstrate that true discrete diffusion (MDLM) can match or exceed the simple approach.

## Key Innovation

**Generative Classification via Discrete Diffusion**:
- Train separate MDLM models per class
- Compute P(text | class) using diffusion likelihood
- Classify using Bayes rule
- No discriminative classifier needed

**Advantages**:
- 🎯 Directly models text generation process
- 🎯 Can generate synthetic examples
- 🎯 Interpretable likelihoods
- 🎯 Works for any number of classes

## Risk Mitigation

### If MDLM Underperforms (<85%)
1. **Increase Training**: 30-40 epochs instead of 20
2. **Larger Model**: Use medium config (1024 hidden)
3. **Hyperparameter Tuning**: Learning rate, batch size
4. **Data Augmentation**: Use more synthetic samples

### If Training Too Slow
1. **Reduce Sequence Length**: 256 instead of 512
2. **Smaller Model**: Keep current small config
3. **Fewer Epochs**: Start with 10 epochs

### If Flash Attention Issues Persist
1. **Alternative Model**: Try non-DiT MDLM backbone
2. **Fallback**: Stick with Simple Diffusion (88.5%)

## Documentation Status

### Created Files ✅
- `train_mdlm_classifier.py` - Per-class training
- `eval_mdlm_classifier.py` - Generative classification
- `run_mdlm_full_training.sh` - Full pipeline
- `MDLM_DEPLOYMENT_SUMMARY.md` - This document

### Updated Files ✅
- Deployed all scripts to nigel.birs.ca
- Fixed path resolution issues
- Set up screen session management

## Current Status

**Date**: November 4, 2025, 3:47 AM PST
**Status**: ✅ TRAINING ACTIVE - Class 0 (Negative Sentiment)
**Screen Session**: Active (64734.mdlm-training) - Detached
**GPU**: RTX 4070 Ti SUPER - 98% utilization, 10.8GB/16.4GB memory
**Training Speed**: ~8.3 iterations/second
**Progress**: Epoch 0 started, validation at step 1000
**Next Milestone**: Complete 20K steps (~6-12 hours)

### Recent Fix Applied (3:47 AM)
- **Issue**: ModelCheckpoint monitoring 'train_loss' but MDLM logs 'trainer/loss'
- **Error**: Training crashed at step 1000 during validation checkpoint
- **Fix**: Changed monitor from 'train_loss' to 'trainer/loss' in train_mdlm_classifier.py:274
- **Result**: Training restarted successfully, checkpoints now saving correctly

---

**Last Updated**: 2025-11-04 03:47 AM PST
**Location**: nigel.birs.ca:~/text-diffusion/
**Deployed By**: Claude Code (automated deployment + runtime debugging)
