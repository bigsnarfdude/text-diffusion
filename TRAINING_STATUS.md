# Text Diffusion Training - Active Session

## Status: ✅ TRAINING IN PROGRESS

**Started**: Nov 1, 2025 at 3:16 PM
**Location**: nigel.birs.ca:~/text-diffusion
**Screen Session**: text-diffusion-training

## Training Configuration

- **Model**: roberta-base (125M parameters)
- **Dataset**: WikiText-2 (23,767 training examples)
- **Batch Size**: 16
- **Epochs**: 3
- **Total Steps**: 4,458 steps (1,486 per epoch)
- **Learning Rate**: 5e-5 with warmup

## Diffusion Settings

- **Masking Probabilities**: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
- **Prefix Length**: 5 tokens (preserved during training)
- **Strategy**: Variable masking for full denoising curve

## Current Progress (as of last check)

- **Steps Completed**: ~1000 / 4458 (22%)
- **Epoch Progress**: 0.67 / 3.0
- **Training Speed**: ~11.3 it/s
- **Estimated Total Time**: 6.5 hours
- **Estimated Completion**: ~9:45 PM

## Loss Trends

```
Step 100:  loss = 4.21
Step 200:  loss = 4.43
Step 300:  loss = 4.25
Step 400:  loss = 4.00
Step 500:  loss = 4.09
Step 600:  loss = 3.88
Step 700:  loss = 4.19
Step 800:  loss = 4.02
Step 900:  loss = 3.86
Step 1000: eval_loss = 3.91 ✅
```

**Trend**: Loss is decreasing nicely from ~4.2 → ~3.9

## Checkpoints

Checkpoints are being saved every 1000 steps to:
```
~/text-diffusion/results-full/checkpoint-1000/
~/text-diffusion/results-full/checkpoint-2000/  (expected)
~/text-diffusion/results-full/checkpoint-3000/  (expected)
~/text-diffusion/results-full/checkpoint-4000/  (expected)
~/text-diffusion/results-full/final-model/      (at completion)
```

## Monitoring Commands

### Quick Status Check
```bash
./monitor_training.sh
```

### Watch Live Training
```bash
ssh vincent@nigel.birs.ca 'tail -f ~/text-diffusion/training.log'
```

### Attach to Screen (Interactive)
```bash
ssh vincent@nigel.birs.ca
screen -r text-diffusion-training
# Press Ctrl+A then D to detach
```

### Check Checkpoints
```bash
ssh vincent@nigel.birs.ca 'ls -lh ~/text-diffusion/results-full/'
```

### View Recent Losses
```bash
ssh vincent@nigel.birs.ca "cd ~/text-diffusion && grep 'loss' training.log | tail -20"
```

## What to Expect

### Expected Loss Progression

**100% Masked (Hardest)**:
- Start: ~10.0
- Mid: ~4.0-5.0
- End: ~2.5

**50% Masked (Medium)**:
- Start: ~6.0
- Mid: ~3.0-4.0
- End: ~1.5

**10% Masked (Easiest)**:
- Start: ~3.0
- Mid: ~2.0
- End: ~1.0

### Good Training Signs ✅
- Loss decreases steadily
- No sudden spikes
- Eval loss < training loss (not overfitting)
- Speed remains consistent (~11 it/s)

### Warning Signs ⚠️
- Loss increasing or stuck
- Sudden large spikes
- Speed dropping significantly
- Out of memory errors

## After Training Completes

### 1. Generate Text
```bash
ssh vincent@nigel.birs.ca
cd ~/text-diffusion
source venv/bin/activate

python3 generate.py \
  --checkpoint results-full/final-model \
  --prefix "Machine learning is" \
  --num-samples 5 \
  --sampling topk \
  --temperature 0.7
```

### 2. Try Different Schedules
```bash
# Linear (default)
python3 generate.py --checkpoint results-full/final-model --schedule linear

# Cosine (better quality)
python3 generate.py --checkpoint results-full/final-model --schedule cosine

# Exponential (faster/creative)
python3 generate.py --checkpoint results-full/final-model --schedule exponential
```

### 3. Compare Sampling Methods
```bash
# Greedy (deterministic)
python3 generate.py --checkpoint results-full/final-model --sampling greedy

# Top-k (balanced)
python3 generate.py --checkpoint results-full/final-model --sampling topk --top-k 50

# Nucleus (high quality)
python3 generate.py --checkpoint results-full/final-model --sampling nucleus --top-p 0.9
```

### 4. Download Model Locally
```bash
# Download trained model
scp -r vincent@nigel.birs.ca:~/text-diffusion/results-full/final-model ./

# Generate locally
python3 generate.py --checkpoint ./final-model --prefix "Your text here"
```

## Troubleshooting

### Training Stopped?
```bash
# Check if screen session exists
ssh vincent@nigel.birs.ca "screen -ls"

# Check last log lines
ssh vincent@nigel.birs.ca "tail -50 ~/text-diffusion/training.log"

# Restart if needed
ssh vincent@nigel.birs.ca
cd ~/text-diffusion
screen -dmS text-diffusion-training bash -c \
  'source venv/bin/activate && python3 train.py --model-name roberta-base --epochs 3 --batch-size 16 --output-dir results-full 2>&1 | tee -a training.log'
```

### Out of Memory?
```bash
# Kill training
ssh vincent@nigel.birs.ca "screen -S text-diffusion-training -X quit"

# Restart with smaller batch size
ssh vincent@nigel.birs.ca
cd ~/text-diffusion
screen -dmS text-diffusion-training bash -c \
  'source venv/bin/activate && python3 train.py --model-name roberta-base --epochs 3 --batch-size 8 --output-dir results-full 2>&1 | tee training.log'
```

### Check GPU Usage
```bash
ssh vincent@nigel.birs.ca "nvidia-smi"
```

## Files and Directories

```
~/text-diffusion/
├── training.log              # Full training output
├── results-full/             # Checkpoints directory
│   ├── checkpoint-1000/      # Saved checkpoint
│   ├── checkpoint-2000/      # Saved checkpoint (pending)
│   ├── ...
│   └── final-model/          # Final trained model (pending)
├── venv/                     # Python virtual environment
├── train.py                  # Training script
├── generate.py               # Generation script
├── config.py                 # Configuration
└── data_collator.py          # Variable masking implementation
```

## Next Steps After Training

1. ✅ **Generate samples** to verify quality
2. ✅ **Experiment with schedules** (linear, cosine, exponential)
3. ✅ **Try different sampling** (greedy, top-k, nucleus)
4. ✅ **Test different temperatures** (0.3, 0.7, 1.2)
5. ✅ **Compare to expectations** (coherence, diversity)
6. ✅ **Download model** for local use
7. 🔄 **Fine-tune on your data** (optional)

## Learning Resources

- **LEARNING_GUIDE.md**: Comprehensive technical deep-dive (800 lines)
- **PROJECT_SUMMARY.md**: Architecture and design decisions
- **QUICKSTART.md**: Quick start and common commands
- **README.md**: Project overview

---

**Training is progressing well!** Check back in ~6 hours for completion.

Monitor with: `./monitor_training.sh`
