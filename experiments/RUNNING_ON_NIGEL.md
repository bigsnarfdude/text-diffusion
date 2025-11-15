# Experiments Running on nigel.birs.ca

## Status: ✅ RUNNING

**Started**: November 2, 2025, 2:15 PM
**Location**: nigel.birs.ca (RTX 4070 Ti SUPER GPU)
**Screen Session**: `text-diffusion-exp`
**Log File**: `~/text-diffusion-experiments.log`

## Progress

### Completed:
- ✅ GPT-2 Zero-Shot: **89.4% accuracy** (11 seconds)

### In Progress:
- ⏳ GPT-2 Native Classifier training (~30 minutes)
- ⏳ Diffusion Baseline (~45 minutes)
- ⏳ Diffusion Trained (~45 minutes)

### Estimated Completion: ~2-3 hours from start

## Monitoring Commands

### Check Progress
```bash
ssh vincent@nigel.birs.ca "tail -50 ~/text-diffusion-experiments.log"
```

### Watch Live Output
```bash
ssh vincent@nigel.birs.ca "tail -f ~/text-diffusion-experiments.log"
```

### Attach to Screen Session
```bash
ssh vincent@nigel.birs.ca
screen -r text-diffusion-exp
# Press Ctrl+A then D to detach
```

### Check GPU Usage
```bash
ssh vincent@nigel.birs.ca "nvidia-smi"
```

### Get Results When Complete
```bash
# List results
ssh vincent@nigel.birs.ca "ls -lh ~/text-diffusion/results/comparison/"

# Download results
scp vincent@nigel.birs.ca:~/text-diffusion/results/comparison/comparison_imdb_*.json ./results/

# View summary
ssh vincent@nigel.birs.ca "cat ~/text-diffusion-experiments.log | grep -A 20 'COMPARISON TABLE'"
```

## What's Happening

The script is running all 4 classification approaches:

1. **GPT-2 Zero-Shot** ✅ Complete
   - Method: Perplexity-based classification
   - Training: None
   - Result: **89.4% accuracy**
   - Time: 11 seconds

2. **GPT-2 Native** ⏳ In Progress
   - Method: Discriminative fine-tuning
   - Training: 3 epochs on 10,000 samples
   - Expected: ~85-90% accuracy
   - Time: ~30 minutes

3. **Diffusion Baseline** ⏳ Pending
   - Method: Likelihood-based (untrained models)
   - Training: None
   - Expected: ~50-60% accuracy
   - Time: ~45 minutes (30 samples × 1000 texts)

4. **Diffusion Trained** ⏳ Pending
   - Method: Likelihood-based (per-class trained)
   - Training: Already done (using pre-trained models)
   - Expected: ~85-90% accuracy?
   - Time: ~45 minutes

## Key Questions Being Answered

1. **Does zero-shot GPT-2 work?**
   - ✅ YES! 89.4% accuracy without any training

2. **Does diffusion training help?**
   - ⏳ Testing: Compare diffusion-baseline vs diffusion-trained
   - Need to see significant gap (p < 0.05)

3. **Is diffusion competitive?**
   - ⏳ Testing: Compare diffusion-trained vs GPT-2 native
   - Looking for accuracy within 0.05

## Results So Far

### GPT-2 Zero-Shot: 89.4%

This is surprisingly high! The pretrained GPT-2 model can classify sentiment very well just by comparing perplexities of:
- "This is a negative review: [text]"
- "This is a positive review: [text]"

**Implications**:
- IMDB sentiment might be too easy as a test task
- Zero-shot GPT-2 is a very strong baseline
- Our diffusion approach needs to beat 89.4% to be worthwhile

## Next Steps After Completion

1. **Review Results**:
   - Check comparison table
   - Examine statistical significance
   - Analyze which approach won

2. **Document Findings**:
   - Update experiments/SUMMARY.md
   - Create results visualization
   - Write up conclusions

3. **Share Results**:
   - Present comparison table
   - Discuss implications
   - Plan next experiments

## Troubleshooting

If experiments fail:

```bash
# Check screen sessions
ssh vincent@nigel.birs.ca "screen -ls"

# View full log
ssh vincent@nigel.birs.ca "cat ~/text-diffusion-experiments.log"

# Restart experiments
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && source venv/bin/activate && ./experiments/RUN_EXPERIMENTS.sh full 2>&1 | tee ~/text-diffusion-experiments.log"
```

## Hardware

- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **VRAM**: 16 GB
- **CUDA**: 12.8
- **PyTorch**: 2.9.0+cu128

This is MUCH faster than your laptop CPU!
