# Synthetic Data Generation - Monitoring Guide

## Current Status

**Started**: November 2, 2025 at 21:29 (9:29 PM)
**Method**: GPU-batched transformers with Qwen/Qwen3-8B
**Progress**: 1/625 batches complete (0.16% of negative reviews)
**Speed**: 45.91 seconds per batch (8 samples) = 5.7 seconds per sample
**ETA**: ~16 hours total for 10,000 samples

## Quick Status Commands

### Check Overall Progress
```bash
ssh vincent@nigel.birs.ca "tail -3 ~/batched-qwen-10k.log"
```

### Check GPU Usage
```bash
ssh vincent@nigel.birs.ca "nvidia-smi"
```

### Check if Process is Running
```bash
ssh vincent@nigel.birs.ca "ps aux | grep generate_synthetic_batched | grep -v grep"
```

### Watch Live Progress
```bash
ssh vincent@nigel.birs.ca "tail -f ~/batched-qwen-10k.log"
```
Press `Ctrl+C` to stop watching

## Timeline Breakdown

### Phase 1: Negative Reviews (In Progress)
- **Target**: 5,000 negative reviews
- **Batches**: 625 (8 samples per batch)
- **Current**: 1/625 (0.16%)
- **ETA**: ~8 hours

### Phase 2: Positive Reviews (After Phase 1)
- **Target**: 5,000 positive reviews
- **Batches**: 625 (8 samples per batch)
- **ETA**: ~8 hours

### Phase 3: Merge & Training (After Generation)
- **Merge datasets**: ~1 minute
- **Train 6 epochs**: ~30-40 minutes
- **Evaluate**: ~1 minute

**Total Pipeline ETA**: ~17 hours from now (complete by ~2:30 PM Nov 3)

## Expected Output Structure

After completion, you'll have:

```
data/synthetic-imdb/
├── train_class_0.json    # 5,000 negative reviews
├── train_class_1.json    # 5,000 positive reviews
└── metadata.json         # Dataset info

data/imdb-augmented/      # After merge
├── train_class_0.json    # 17,500 negative (12,500 real + 5,000 synthetic)
├── train_class_1.json    # 17,500 positive (12,500 real + 5,000 synthetic)
├── test.json             # Original IMDB test set
└── metadata.json
```

## When Generation Completes

Run this single command to merge data and train:

```bash
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && ./scripts/train_augmented_6epochs.sh"
```

This will:
1. ✅ Merge IMDB (25K) + Synthetic (10K) → 35K total
2. ✅ Train GPT-2 for 6 epochs on augmented data
3. ✅ Save models to `results-gpt2-augmented/`

## Expected Results

### Current Best (3 epochs, 25K samples):
- **Accuracy**: 90.1%
- **Baseline**: GPT-2 Discriminative 90.2%

### Target (6 epochs, 35K samples):
- **Expected**: 91-92%+
- **Goal**: Beat discriminative baseline (>90.2%)

## Technical Details

### Model Configuration
- **Model**: Qwen/Qwen3-8B (8 billion parameters)
- **Batch Size**: 8 samples per GPU batch
- **VRAM Usage**: 12.66 GB
- **Temperature**: 0.9
- **Top-p**: 0.9
- **Max Tokens**: 200 per review

### Generation Script
- **Location**: `scripts/generate_synthetic_batched.py`
- **Log File**: `~/batched-qwen-10k.log`
- **Output Dir**: `data/synthetic-imdb/`

### Why This Approach?
- **vLLM**: Failed due to PyTorch CUDA memory fragmentation issues
- **Async Ollama**: Would take ~17 hours (similar to current approach)
- **Batched Transformers**: Currently running, 16 hours ETA, more reliable

## Troubleshooting

### If Generation Stops
```bash
# Check if process died
ssh vincent@nigel.birs.ca "ps aux | grep generate_synthetic_batched | grep -v grep"

# Check for errors in log
ssh vincent@nigel.birs.ca "tail -100 ~/batched-qwen-10k.log | grep -i error"

# Restart if needed
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && source venv/bin/activate && nohup python scripts/generate_synthetic_batched.py --num-samples 10000 --batch-size 8 --model Qwen/Qwen3-8B --output-dir data/synthetic-imdb > ~/batched-qwen-10k.log 2>&1 &"
```

### If GPU Issues Occur
```bash
# Check GPU memory
ssh vincent@nigel.birs.ca "nvidia-smi"

# Kill generation and restart
ssh vincent@nigel.birs.ca "pkill -f generate_synthetic_batched"
# Wait 10 seconds for GPU memory to clear
# Then restart with command above
```

## Progress Calculation

**Speed**: 45.91 seconds per batch × 625 batches = 28,693 seconds ≈ 8 hours per class
**Total**: 8 hours × 2 classes = 16 hours

**Samples per hour**: 3600 / 5.7 = ~630 samples/hour
**Total time**: 10,000 / 630 ≈ 15.9 hours

## Summary

✅ Generation is running with Qwen3-8B (8B parameters)
✅ GPU batching working (batch size 8)
✅ Estimated completion: ~16 hours (Nov 3 at 2:30 PM)
✅ Next step: Merge datasets and train for 6 epochs
✅ Target: Beat 90.2% discriminative baseline with augmented data

**Just let it run overnight and check progress in the morning!**
