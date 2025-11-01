# Text Diffusion Learning Project

A toy implementation of discrete text diffusion using RoBERTa for learning and experimentation.

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Or with virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train small model (fast, for learning)
python train.py --quick-test

# Generate text
python generate.py --checkpoint results/final-model

# Experiment with masking strategies
python experiments/masking_viz.py
```

## Project Structure

```
text-diffusion/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # All hyperparameters
├── data_collator.py            # The magic: variable masking for training
├── train.py                     # Training script with visualization
├── generate.py                  # Iterative denoising generation
└── experiments/
    ├── masking_viz.py          # Visualize masking strategies
    ├── schedule_comparison.py   # Compare denoising schedules
    └── layer_analysis.py        # Analyze model layer behavior
```

## Core Concepts

### 1. Training: Variable Masking
- Each batch randomly gets masking rate: 10%, 20%, ..., 90%, 100%
- Model learns to denoise at ALL corruption levels
- Key insight: Same model, different masking = different denoising tasks

### 2. Generation: Iterative Refinement
- Start: 100% masked (except prefix)
- Step 1: Predict all positions → 90% masked
- Step 2: Predict remaining → 80% masked
- ...
- Step 10: Final polish → Complete text

### 3. Why It Works
- High masking (90-100%): Model learns structure, frequent words
- Medium masking (40-60%): Context-dependent content
- Low masking (10-20%): Fine details, rare tokens

## Experiments to Try

### Masking Strategies
```bash
# Linear schedule (default)
python generate.py --schedule linear

# Cosine schedule (more time at ends)
python generate.py --schedule cosine

# Exponential decay
python generate.py --schedule exponential
```

### Sampling Methods
```bash
# Greedy (deterministic)
python generate.py --sampling greedy

# Top-k sampling
python generate.py --sampling topk --k 50

# Nucleus sampling
python generate.py --sampling nucleus --p 0.9

# Temperature control
python generate.py --temperature 0.7
```

### Layer Analysis
```bash
# Which layers learn what?
python experiments/layer_analysis.py --checkpoint results/checkpoint-1000
```

## Learning Path

### Phase 1: Understand Training (Today)
1. Run quick training with visualization
2. Watch how different masking levels affect loss
3. Inspect what model predicts at each masking level

### Phase 2: Understand Generation (Tomorrow)
1. Generate with step-by-step output
2. See how text evolves through denoising
3. Try different schedules and compare

### Phase 3: Experiment (This Week)
1. Try different model sizes (distilroberta, roberta-base, roberta-large)
2. Compare masking strategies
3. Analyze layer representations

### Phase 4: Your Domain (Next Week)
1. Fine-tune on your specific text domain
2. Add constraints for structured generation
3. Evaluate on your metrics

## Key Files Explained

### `data_collator.py` - The Training Secret
This is where the magic happens. Standard masked LM training uses 15% masking.
We use VARIABLE masking (10% to 100%) so the model learns the full denoising curve.

### `train.py` - Training Loop
Standard HuggingFace Trainer with:
- Logging for each masking level
- Checkpointing
- Visualization of predictions

### `generate.py` - Iterative Generation
The inference algorithm:
1. Start fully masked
2. For each denoising step:
   - Forward pass → get logits
   - Sample tokens for masked positions
   - Fill in predictions
   - Re-mask fewer positions
3. Return final text

## Monitoring Training

Training logs show loss per masking level:
```
Step 100:
  mask_1.0: 8.234  (100% masked - hardest)
  mask_0.5: 3.456  (50% masked - medium)
  mask_0.1: 1.234  (10% masked - easiest)
```

Good training: All levels decrease, with 100% masked having highest loss.

## Common Issues

### "Model ignoring prefix"
- Check PREFIX_LEN in generate.py
- Increase prefix length during generation
- Train with longer preserved prefixes

### "Repetitive generation"
- Increase temperature (try 0.7-1.0)
- Use top-k or nucleus sampling
- Add more denoising steps

### "Incoherent text"
- Train longer (model underfitted)
- Increase model size
- Use more denoising steps at inference

## Documentation

### Quick References
- **README.md** - This file: project overview and basic usage
- **DEPLOYMENT.md** - Setup and deployment guide
- **GENERATION_VALIDATION.md** - Sample outputs and quality assessment

### Deep Dives
- **LEARNING_GUIDE.md** - Comprehensive technical explanation (800 lines)
  - Mathematical intuition
  - Implementation details
  - Debugging strategies
  - Advanced topics
- **PROJECT_SUMMARY.md** - Architecture, design decisions, success criteria
- **PROJECT_INDEX.md** - Complete navigation guide

### Scripts
- **deploy.sh** - Deployment script for remote servers
- **monitor_training.sh** - Training progress monitoring
- **requirements.txt** - All dependencies

## Resources

### Original Work
- Original blog: https://nathan.rs/posts/roberta-diffusion/

### Papers
- RoBERTa paper: https://arxiv.org/abs/1907.11692
- D3PM (discrete diffusion): https://arxiv.org/abs/2107.03006
- BERT: https://arxiv.org/abs/1810.04805

### Tools
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org/docs/stable/index.html

## Next Steps

### Immediate (Today)
1. Setup: `pip install -r requirements.txt`
2. Visualize: `python experiments/masking_viz.py`
3. Train: `python train.py --quick-test`
4. Generate: `python generate.py --checkpoint results/final-model`

### Short-term (This Week)
1. Read LEARNING_GUIDE.md for deep understanding
2. Try different sampling strategies and schedules
3. Experiment with hyperparameters

### Medium-term (Research)
1. Train with more epochs for better quality
2. Experiment with non-uniform masking distributions
3. Add classifier guidance for controlled generation
4. Implement parallel decoding (faster generation)

## Contributing

This is a learning project. Feel free to:
- Experiment with different approaches
- Try different model architectures
- Test on different datasets
- Share your findings!

Happy learning! 🚀
