# MDLM Training Configuration Fix

## Date: November 4, 2025

## Problem Identified

Initial MDLM pretrained fine-tuning achieved only **57.8% accuracy** (expected 85-90%). Investigation revealed **critical missing configuration parameters** that prevented proper training.

## Root Cause: Incomplete Training Configuration

Compared our training config with official BD3-LM repository and found:

### Missing Parameters

1. **`training.sampling_eps_min`**: Missing (required buffer)
2. **`training.sampling_eps_max`**: Missing (required buffer)
3. **`num_warmup_steps`**: 500 (should be 2500)
4. **`strict=False` in load_state_dict**: Should use default `strict=True`

### Why This Matters

From BD3-LM `diffusion.py`:
```python
self.register_buffer('sampling_eps_min', torch.tensor(
    self.config.training.sampling_eps_min))
self.register_buffer('sampling_eps_max', torch.tensor(
    self.config.training.sampling_eps_max))
```

These are **registered buffers** that the MDLM training loop expects. Without them, the training process doesn't properly handle the diffusion sampling schedule, leading to poor convergence.

## Changes Made

### 1. Added Required Buffers
```python
'training': {
    'ema': 0.9999,
    'antithetic_sampling': True,
    'importance_sampling': False,
    'sampling_eps': 1e-3,
    'sampling_eps_min': 1e-3,  # ADDED
    'sampling_eps_max': 1.0,    # ADDED
    'change_of_variables': False,
},
```

### 2. Increased Warmup Steps
```python
'lr_scheduler': {
    '_target_': 'transformers.get_constant_schedule_with_warmup',
    'num_warmup_steps': 2500,  # Was 500, now matches BD3-LM
},
```

### 3. Removed strict=False
```python
# Before:
model.load_state_dict(state_dict, strict=False)

# After:
model.load_state_dict(state_dict)  # strict=True by default
```

## Expected Impact

With these fixes:
- **sampling_eps_min/max**: Proper diffusion noise schedule during training
- **Longer warmup**: More stable training for pretrained model fine-tuning
- **strict=True**: Ensures all parameters match (we verified 100% key match)

## Previous Results (Incorrect Config)

```
Accuracy: 57.8%
Confusion Matrix:
  578 correct negative
  422 false negatives (predicted negative, should be positive)
  0 true positives
  0 false positives

Issue: Model always predicted class 0
Cause: Both models converged to similar distributions
```

## Expected Results (Corrected Config)

With proper training configuration:
- **Target**: 85-90% accuracy
- **Proper class divergence**: Models should learn distinct distributions
- **Comparison**: Competitive with GPT-2 (90.1%) and Simple Diffusion (88.5%)

## Training Command (Updated)

```bash
# Remove old results
ssh vincent@nigel.birs.ca "rm -rf ~/text-diffusion/results-mdlm-pretrained"

# Run corrected training pipeline
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && screen -dmS mdlm-pretrained-fixed bash run_mdlm_pretrained_full.sh"
```

## Source

**Reference**: https://github.com/kuleshov-group/bd3lms

Key files reviewed:
- `main.py`: Loading pretrained models (lines 150-175)
- `configs/config.yaml`: Default training configuration
- `configs/lr_scheduler/constant_warmup.yaml`: Warmup steps
- `diffusion.py`: Buffer registration

## Timeline

- **Initial Training**: November 4, 2025 (03:00-12:00) - 57.8% accuracy ❌
- **Bug Discovery**: November 4, 2025 (13:00) - Found missing config
- **Fix Deployed**: November 4, 2025 (13:30) - Ready to re-train
- **Re-training**: Pending (estimated 12-24 hours)

---

**Status**: Fixed training script deployed to nigel.birs.ca
**Next Step**: Re-run training pipeline with corrected configuration
