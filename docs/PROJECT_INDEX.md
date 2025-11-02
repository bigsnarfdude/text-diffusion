# Text Diffusion Project - Complete Index

## 📋 Quick Navigation

### 🚀 Getting Started (5 minutes)
1. **QUICKSTART.md** - Deployment and first steps
2. Run `./deploy_to_nigel.sh` to deploy
3. Run `python3 experiments/masking_viz.py` to understand the concepts

### 📚 Understanding the Project (30 minutes)
1. **README.md** - Project overview and basic usage
2. **PROJECT_SUMMARY.md** - Architecture and design decisions
3. **LEARNING_GUIDE.md** - Deep technical dive (800 lines!)

### 💻 Core Code (1350 lines total)
- **config.py** (200 lines) - All hyperparameters
- **data_collator.py** (350 lines) - Variable masking (KEY INNOVATION)
- **train.py** (200 lines) - Training with HuggingFace Trainer
- **generate.py** (250 lines) - Iterative denoising generation
- **experiments/masking_viz.py** (350 lines) - Visualization tools

## 🎯 Learning Path

### Day 1: Understand Concepts
```bash
# Read overview
cat README.md

# Visualize masking strategies
python3 experiments/masking_viz.py

# See what you're building
cat PROJECT_SUMMARY.md
```

### Day 2: Run Quick Test
```bash
# Local training
python3 train.py --quick-test  # 20-30 minutes

# Or deploy to remote server (optional)
./deploy.sh
ssh user@your-server
cd ~/text-diffusion
source venv/bin/activate
python3 train.py --quick-test
```

### Day 3: Generate & Experiment
```bash
# Generate text
python3 generate.py \
  --checkpoint results/final-model \
  --prefix "Machine learning is" \
  --num-samples 5

# Try different schedules
python3 generate.py --checkpoint results/final-model --schedule cosine
python3 generate.py --checkpoint results/final-model --schedule exponential

# Try different sampling
python3 generate.py --checkpoint results/final-model --sampling nucleus --top-p 0.9
```

### Week 2: Deep Understanding
```bash
# Read comprehensive guide
cat LEARNING_GUIDE.md

# Study the key innovation
cat data_collator.py

# Understand generation algorithm
cat generate.py
```

## 📖 Documentation Guide

### For Quick Setup
**File**: DEPLOYMENT.md
**Read time**: 10 minutes
**Contains**:
- Local and remote setup instructions
- Training options (quick test to full training)
- Generation examples
- Troubleshooting guide

### For Understanding Concepts
**File**: PROJECT_SUMMARY.md (400 lines)
**Read time**: 15 minutes
**Contains**:
- Architecture diagrams
- Key innovations explained
- Design decisions with rationale
- Expected results

### For Deep Technical Knowledge
**File**: LEARNING_GUIDE.md (800 lines)
**Read time**: 1 hour
**Contains**:
- Mathematical intuition
- Component deep-dives
- Training dynamics
- Generation algorithms
- Debugging strategies
- Advanced topics

### For Daily Use
**File**: README.md (200 lines)
**Read time**: 5 minutes
**Contains**:
- Quick commands
- Common usage patterns
- File explanations
- Resource links

## 🔑 Key Concepts

### The Innovation: Variable Masking

**Standard Masked LM (BERT):**
```
Always 15% masking → learns to fill 15% blanks
```

**Diffusion Masked LM (This Project):**
```
Variable masking (10-100%) → learns full denoising curve
```

**Why it matters:**
- Standard: Can't generate from scratch (needs real text to mask)
- Diffusion: Can generate from 100% masked (pure generation)

### The Algorithm: Iterative Denoising

**Training:**
```python
for batch in data:
    mask_prob = random.choice([1.0, 0.9, 0.8, ..., 0.1])
    masked_text = apply_masks(batch, mask_prob)
    predictions = model(masked_text)
    loss = cross_entropy(predictions, original_text)
```

**Generation:**
```python
text = [PREFIX] + [MASK] * length
for mask_prob in [1.0, 0.9, 0.8, ..., 0.1]:
    predictions = model(text)
    fill_masks(text, predictions)
    re_mask(text, mask_prob)
return text
```

## 🧪 Experiments Included

### 1. Masking Visualization
**Script**: `experiments/masking_viz.py`
**What it shows**:
- Text appearance at 10%, 50%, 100% masking
- Prefix preservation in action
- Schedule comparison graphs

**Run**: `python3 experiments/masking_viz.py`
**Time**: 2 minutes
**Output**: Console + PNG graph

### 2. Schedule Comparison
**Command**: Generate with different `--schedule` flags
**Schedules**:
- Linear: Uniform unmasking (default)
- Cosine: More time at ends
- Exponential: Fast then slow

**Compare**: Quality, coherence, creativity

### 3. Sampling Methods
**Command**: Generate with different `--sampling` flags
**Methods**:
- Greedy: Deterministic (same every time)
- Top-k: Balanced diversity/quality
- Nucleus: Dynamic cutoff

**Compare**: Diversity, quality, consistency

### 4. Temperature Control
**Command**: Generate with different `--temperature` values
**Settings**:
- 0.3: Conservative, common words
- 0.7: Balanced (default)
- 1.2: Creative, diverse

**Compare**: Creativity vs. coherence

## 📊 Expected Results

### Quick Test Training (30 min)
- Model: distilroberta-base
- Time: 20-30 minutes (CPU)
- Loss: ~3.0 (100% masked) → ~1.5 (10% masked)
- Quality: Coherent but simple

### Full Training (2-3 hours)
- Model: roberta-base
- Time: 2-3 hours (CPU)
- Loss: ~2.5 (100% masked) → ~1.0 (10% masked)
- Quality: Good coherence, varied vocabulary

### Generation
- Speed: ~2 sec/sample (10 steps, CPU)
- Coherence: Good with proper training
- Diversity: High with top-k/nucleus sampling

## 🛠️ Technical Stack

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace library
- **Datasets**: Data loading utilities
- **NumPy**: Numerical operations

### Model Architecture
- **Base**: RoBERTa (encoder-only transformer)
- **Task**: Masked Language Modeling
- **Innovation**: Variable masking for diffusion

### Training Framework
- **Trainer**: HuggingFace Trainer API
- **Optimization**: AdamW with warmup
- **Monitoring**: TensorBoard-compatible logging

## 🎓 What You'll Learn

### Conceptual Understanding
1. How diffusion models work for text
2. Difference between diffusion and autoregressive generation
3. Why variable masking enables generation
4. Trade-offs in sampling strategies

### Implementation Skills
1. HuggingFace ecosystem (Transformers, Datasets, Trainer)
2. Custom data collators for special training needs
3. Iterative generation algorithms
4. PyTorch model inference and sampling

### Experimental Design
1. Hyperparameter tuning strategies
2. Ablation study design
3. Quality evaluation methods
4. Debugging ML training issues

## 🔍 File-by-File Guide

### config.py (200 lines)
**Purpose**: Central configuration management
**Key classes**:
- `TrainingConfig`: All training hyperparameters
- `GenerationConfig`: All generation settings
**Functions**:
- `parse_training_args()`: CLI argument parsing
- `parse_generation_args()`: CLI argument parsing
- `get_mask_schedule()`: Schedule generation

### data_collator.py (350 lines)
**Purpose**: THE KEY INNOVATION - variable masking
**Key class**: `DiffusionDataCollator`
**What it does**:
1. Randomly select masking probability per batch
2. Mask that percentage of tokens
3. Create labels for training
**Also includes**: `VisualizableDiffusionCollator` for debugging

### train.py (200 lines)
**Purpose**: Training script with HuggingFace Trainer
**Key functions**:
- `load_and_prepare_data()`: Dataset loading
- `setup_model()`: Model initialization
- `train()`: Main training loop
**Output**: Trained model checkpoints in `results/`

### generate.py (250 lines)
**Purpose**: Iterative denoising generation
**Key class**: `DiffusionGenerator`
**Algorithm**:
1. Start with fully masked text
2. For each step: predict → fill → re-mask
3. Return final denoised text
**Supports**: Multiple sampling strategies, schedules

### experiments/masking_viz.py (350 lines)
**Purpose**: Visualization and understanding
**Visualizations**:
1. Masking levels (10-100%)
2. Prefix preservation
3. Schedule comparison
**Output**: Console logs + PNG graphs

## 🚀 Deployment Instructions

### Local Development
```bash
cd text-diffusion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 experiments/masking_viz.py
```

### Deploy to Remote Server
```bash
# Set your server details
export REMOTE_HOST=user@your-server.com
export REMOTE_DIR=~/text-diffusion

# Run deployment script
./deploy.sh
```

This script:
1. Tests SSH connection
2. Copies all files
3. Sets up virtual environment
4. Installs dependencies
5. Verifies installation

### Manual Deployment
```bash
# Copy files
scp -r * user@your-server:~/text-diffusion/

# SSH and setup
ssh user@your-server
cd ~/text-diffusion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test
python3 -c "import torch, transformers, datasets"
```

## 📈 Success Metrics

### Training is working when:
- ✅ Loss decreases steadily
- ✅ All masking levels show improvement
- ✅ 100% masked loss > 50% masked > 10% masked
- ✅ No sudden spikes or divergence

### Generation is working when:
- ✅ Text is grammatically correct
- ✅ Content is coherent and on-topic
- ✅ Prefix conditioning is respected
- ✅ Multiple samples show diversity
- ✅ Intermediate steps show gradual refinement

## 🐛 Common Issues & Solutions

### "Import error: No module named torch"
```bash
python3 -m pip install --user torch transformers datasets
```

### "CUDA out of memory"
```bash
# Reduce batch size
python3 train.py --quick-test --batch-size 4

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
```

### "Generated text ignoring prefix"
- Check `prefix_length` in `src/generate.py`
- Verify model was trained with prefix preservation
- Try longer prefix (more context)

### "Repetitive generation"
- Increase temperature: `--temperature 0.8`
- Use top-k/nucleus: `--sampling topk --top-k 50`
- Train longer

### "Incoherent generation"
- Lower temperature: `--temperature 0.5`
- More denoising steps: `--steps 20`
- Train longer or use larger model

## 🎯 Next Steps

### Immediate (Today)
1. ✅ Project created and verified
2. ⏭️ Setup local or remote environment
3. ⏭️ Run visualization experiments
4. ⏭️ Quick test training

### Short-term (This Week)
1. Read LEARNING_GUIDE.md thoroughly
2. Run full training (3 epochs)
3. Generate and evaluate samples
4. Try all experiment variations

### Medium-term (This Month)
1. Fine-tune on your domain data
2. Implement advanced features (parallel decoding, adaptive schedules)
3. Compare to GPT-2 baseline
4. Write up findings

### Research Directions
1. Optimal masking distribution (not uniform)
2. Confidence-based adaptive schedules
3. Classifier guidance for controlled generation
4. Structured output constraints

## 📚 References & Resources

### Papers
- **RoBERTa**: Liu et al., 2019 - https://arxiv.org/abs/1907.11692
- **D3PM**: Austin et al., 2021 - https://arxiv.org/abs/2107.03006
- **BERT**: Devlin et al., 2018 - https://arxiv.org/abs/1810.04805

### Implementations
- **Original**: https://nathan.rs/posts/roberta-diffusion/
- **This project**: Toy implementation for learning

### Documentation
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Datasets**: https://huggingface.co/docs/datasets

## 💡 Tips for Learning

1. **Start with visualization**: Run `masking_viz.py` first to understand concepts
2. **Read code top-to-bottom**: Start with `src/config.py`, then `src/data_collator.py`
3. **Modify and experiment**: Change hyperparameters, see what happens
4. **Watch training closely**: Monitor logs, verify loss decreases
5. **Generate frequently**: Test generation at different training checkpoints
6. **Ask questions**: Why does this work? What if I change X?

## 🏆 Project Statistics

- **Total lines of code**: ~1350 (core) + 800 (docs)
- **Documentation**: 2000+ lines across 4 guides
- **Time to implement**: Complete from scratch
- **Time to understand**: 1-2 hours reading docs
- **Time to train (quick)**: 30 minutes
- **Time to train (full)**: 2-3 hours

## ✨ Project Highlights

1. **Well-structured toy implementation**: Fully documented, modular, tested
2. **Comprehensive documentation**: 4 guides covering all levels
3. **Learning-focused**: Visualization tools, step-by-step walkthroughs
4. **Ready to deploy**: One-command deployment script
5. **Extensible**: Clean architecture for experimentation

---

**Happy learning and experimenting!** 🚀

For questions or issues, refer to:
- LEARNING_GUIDE.md for technical deep-dives
- QUICKSTART.md for deployment help
- Code comments for implementation details
