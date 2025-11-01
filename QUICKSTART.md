# Quick Start Guide - Text Diffusion on nigel.birs.ca

## Setup (5 minutes)

```bash
# 1. Copy files to nigel
cd /Users/vincent/development/text-diffusion
scp -r * vincent@nigel.birs.ca:~/text-diffusion/

# 2. SSH to nigel
ssh vincent@nigel.birs.ca

# 3. Navigate and install dependencies
cd ~/text-diffusion
python3 -m pip install --user torch transformers datasets accelerate matplotlib

# 4. Test installation
python3 -c "import torch; print('PyTorch:', torch.__version__)"
```

## Understanding Before Training (10 minutes)

Run the visualization experiments to see how it works:

```bash
# See what masking does at different levels
python3 experiments/masking_viz.py
```

This will show you:
- What the model sees at 10%, 50%, 100% masking
- Why we need ALL masking levels (not just one)
- How prefix preservation enables conditional generation
- Trade-offs between different denoising schedules

## Quick Test Training (30 minutes)

Train a tiny model to verify everything works:

```bash
# Quick test mode: 1 epoch, reduced steps
python3 train.py --quick-test

# Expected output:
# - Model loads (distilroberta-base)
# - Dataset tokenizes (wikitext-2)
# - Training runs ~100 steps
# - Checkpoints save to ./results/
```

**What to watch for:**
- Loss should decrease (starting ~8-10, dropping to ~3-5)
- Each step shows current loss
- Eval runs every 50 steps
- Total time: ~20-30 minutes on CPU

## Generate Text (5 minutes)

Once training finishes, generate some text:

```bash
# Basic generation with linear schedule
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "Machine learning is" \
  --num-samples 3 \
  --sampling topk \
  --temperature 0.7

# Try different schedules
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "The quick brown fox" \
  --schedule cosine \
  --sampling nucleus
```

**You'll see:**
- Step-by-step denoising process
- Text gradually becomes coherent
- Multiple diverse samples from same prefix

## Full Training (2-3 hours)

For better quality, train longer:

```bash
# Full training: 3 epochs, all data
python3 train.py \
  --epochs 3 \
  --batch-size 16 \
  --output-dir results-full

# Or larger model (better quality, slower)
python3 train.py \
  --model-name roberta-base \
  --epochs 3 \
  --batch-size 8 \
  --output-dir results-roberta-base
```

## Experiments to Try

### 1. Compare Denoising Schedules

```bash
PREFIX="The future of artificial intelligence"

# Linear (default)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --schedule linear \
  --num-samples 1

# Cosine (more refinement)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --schedule cosine \
  --num-samples 1

# Exponential (fast then slow)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --schedule exponential \
  --num-samples 1
```

### 2. Compare Sampling Methods

```bash
PREFIX="Scientists have discovered"

# Greedy (deterministic)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --sampling greedy

# Top-k (balanced)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --sampling topk --top-k 50

# Nucleus (dynamic)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --sampling nucleus --top-p 0.9
```

### 3. Temperature Control

```bash
PREFIX="The weather today is"

# Conservative (temp=0.3)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --temperature 0.3

# Balanced (temp=0.7)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --temperature 0.7

# Creative (temp=1.2)
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "$PREFIX" \
  --temperature 1.2
```

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python3 train.py --quick-test --batch-size 4

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
python3 train.py --quick-test
```

### "Training loss not decreasing"
- Check if data collator is actually masking (should see [MASK] tokens)
- Verify different masking levels are being used
- Try longer training or higher learning rate

### "Generated text is repetitive"
- Increase temperature (try 0.8-1.0)
- Use top-k or nucleus sampling instead of greedy
- Train longer for better model quality
- Increase number of denoising steps

### "Generated text ignores prefix"
- Check prefix is being preserved during generation
- Try longer prefix (more context)
- Verify model was trained with prefix preservation

## Understanding Output

### Training Logs
```
Step 100: {'loss': 3.456, 'learning_rate': 4.5e-05}
  → Loss decreasing = good
  → Learning rate following warmup schedule

Eval: {'eval_loss': 3.234}
  → Lower than train loss = not overfitting
```

### Generation Output
```
STEP 1 (100% masked):
  Machine learning [MASK] [MASK] [MASK]...
  → Starting point, fully masked

STEP 5 (50% masked):
  Machine learning is a subset of [MASK] intelligence...
  → Structure emerging, content still rough

STEP 10 (0% masked):
  Machine learning is a subset of artificial intelligence that...
  → Complete, coherent text
```

## Next Steps

1. **Understand the code**:
   - Read `data_collator.py` - the key innovation
   - Read `generate.py` - the inference algorithm
   - Modify and experiment!

2. **Try your domain**:
   - Replace WikiText with your dataset
   - Adjust masking strategies for your text structure
   - Add constraints for formatted output

3. **Analyze behavior**:
   - What masking levels learn what?
   - How do layers differ in their representations?
   - What happens with different model sizes?

4. **Optimize**:
   - Parallel decoding (predict multiple positions at once)
   - Adaptive schedules (based on confidence)
   - Classifier guidance (controlled generation)

## Resources

- Code: `~/text-diffusion/`
- Results: `~/text-diffusion/results/`
- Experiments: `~/text-diffusion/experiments/`
- Logs: `~/text-diffusion/results/logs/`

Happy experimenting! 🚀
