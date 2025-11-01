# Deployment Guide

## Local Setup

### Prerequisites
- Python 3.8+
- GPU with CUDA (optional, but recommended for training)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/bigsnarfdude/text-diffusion.git
cd text-diffusion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

```bash
# Run visualization experiments (no GPU needed)
python experiments/masking_viz.py

# Quick training test (5-10 minutes)
python train.py --quick-test

# Generate samples
python generate.py --checkpoint results/final-model --prefix "Your text here"
```

## Remote Server Deployment

### Using the deployment script

```bash
# Set your server details
export REMOTE_HOST=user@your-server.com
export REMOTE_DIR=~/text-diffusion

# Deploy
./deploy.sh
```

### Manual deployment

```bash
# Copy files to server
scp -r * user@server:~/text-diffusion/

# SSH to server
ssh user@server
cd ~/text-diffusion

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training
python train.py --model-name roberta-base --epochs 3 --batch-size 16 --output-dir results-full
```

### Training in screen session (recommended for long training)

```bash
# Start screen session
screen -S text-diffusion-training

# Activate environment and train
cd ~/text-diffusion
source venv/bin/activate
python train.py --model-name roberta-base --epochs 3 --batch-size 16 --output-dir results-full 2>&1 | tee training.log

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r text-diffusion-training
```

### Monitoring training

```bash
# Set your server details
export REMOTE_HOST=user@your-server.com
export REMOTE_DIR=~/text-diffusion

# Run monitor
./monitor_training.sh

# Or manually
ssh user@server 'tail -f ~/text-diffusion/training.log'
```

## Training Options

### Quick test (5-10 minutes)
```bash
python train.py --quick-test
```
- Model: distilroberta-base
- Epochs: 1
- Purpose: Test that everything works

### Standard training (30-60 minutes)
```bash
python train.py --epochs 3 --batch-size 16
```
- Model: distilroberta-base
- Epochs: 3
- Purpose: Toy model for learning

### Full training (2-4 hours)
```bash
python train.py --model-name roberta-base --epochs 10 --batch-size 16 --output-dir results-full
```
- Model: roberta-base (125M parameters)
- Epochs: 10+
- Purpose: Better quality for experimentation

### Production quality (days)
```bash
python train.py --model-name roberta-large --epochs 20 --batch-size 8 --output-dir results-prod
```
- Model: roberta-large (355M parameters)
- Epochs: 20+
- Dataset: Consider using cleaner corpus than WikiText
- Purpose: High-quality generation

## Generation

### Basic generation
```bash
python generate.py \
  --checkpoint results/final-model \
  --prefix "Your prompt here" \
  --num-samples 5
```

### Try different schedules
```bash
# Linear (default)
python generate.py --checkpoint results/final-model --schedule linear

# Cosine (better quality)
python generate.py --checkpoint results/final-model --schedule cosine

# Exponential (faster)
python generate.py --checkpoint results/final-model --schedule exponential
```

### Try different sampling methods
```bash
# Greedy (deterministic)
python generate.py --checkpoint results/final-model --sampling greedy

# Top-k (balanced)
python generate.py --checkpoint results/final-model --sampling topk --top-k 50

# Nucleus (high quality)
python generate.py --checkpoint results/final-model --sampling nucleus --top-p 0.9
```

### Adjust temperature
```bash
# Conservative (less random)
python generate.py --checkpoint results/final-model --temperature 0.5

# Balanced
python generate.py --checkpoint results/final-model --temperature 0.7

# Creative (more random)
python generate.py --checkpoint results/final-model --temperature 1.0
```

## Troubleshooting

### CUDA out of memory
```bash
# Reduce batch size
python train.py --batch-size 8  # or 4

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
python train.py --quick-test
```

### Training loss not decreasing
- Check data collator is masking correctly
- Verify different masking levels are being used
- Try longer training or higher learning rate

### Generated text is repetitive
- Increase temperature (try 0.8-1.0)
- Use top-k or nucleus sampling instead of greedy
- Train longer for better model quality

### Generated text ignores prefix
- Check prefix length in generate.py
- Try longer prefix (more context)
- Verify model was trained with prefix preservation

## Performance Tips

### GPU Training
- Use largest batch size that fits in memory
- Enable mixed precision training (add to train.py)
- Monitor GPU utilization with `nvidia-smi`

### CPU Training
- Reduce batch size to 4-8
- Reduce model size (use distilroberta-base)
- Expect 5-10x slower than GPU

### Faster Training
- Use distilroberta-base instead of roberta-base
- Reduce epochs for quick experiments
- Use --quick-test flag for testing code changes

### Better Quality
- Train for more epochs (10-20+)
- Use roberta-base or roberta-large
- Use cleaner dataset than WikiText
- Increase denoising steps during generation

## Next Steps

After successful deployment:
1. Read LEARNING_GUIDE.md for deep understanding
2. Experiment with different hyperparameters
3. Try training on your own dataset
4. Compare different generation strategies
5. Analyze model behavior at different masking levels
