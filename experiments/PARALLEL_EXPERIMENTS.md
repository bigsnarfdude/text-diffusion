# Parallel Experiments on RTX 4070 Ti

## Current GPU Usage
- **In Use**: 7,076 MB (43%)
- **Free**: ~9,300 MB (57%)
- **GPU Util**: 97%

## Available Capacity for Parallel Experiments

We can run another experiment in parallel! Here are good options:

### Option 1: Different Dataset (Recommended)
Run the same 4-approach comparison on a different dataset while IMDB runs.

**Datasets to try:**
```bash
# SST-2 (Stanford Sentiment Treebank) - binary sentiment
# Smaller than IMDB, might be faster
python scripts/prepare_sst2.py
./experiments/RUN_EXPERIMENTS.sh full --dataset sst2

# AG News - 4-class topic classification
python scripts/prepare_agnews.py
./experiments/RUN_EXPERIMENTS.sh full --dataset agnews

# TREC - Question classification (6 classes)
python scripts/prepare_trec.py
./experiments/RUN_EXPERIMENTS.sh full --dataset trec
```

**Why this is good:**
- Tests generalization of our approach
- Multiple tasks = more convincing evidence
- Runs in parallel on same GPU

### Option 2: Hyperparameter Sweep for Diffusion
Test different settings for our diffusion classifier:

**Variables to test:**
- Number of likelihood samples (5, 10, 20, 30)
- Mask probability (0.10, 0.15, 0.20, 0.25)
- Different masking strategies

```bash
# Test different likelihood samples
for samples in 5 10 20 30; do
    python experiments/test_diffusion_settings.py \
        --num-likelihood-samples $samples \
        --output results/hyperparam-sweep/
done
```

**Why this is good:**
- Optimizes our approach
- Might find settings that beat GPT-2's 89.4%
- Relatively quick (inference only, no training)

### Option 3: Different Model Sizes
Compare model size impact:

```bash
# Test with different base models
for model in distilroberta-base roberta-base roberta-large; do
    python src/train_generative_classifier.py \
        --model $model \
        --output results-$model
done
```

**Why this is good:**
- Larger models might improve likelihood discrimination
- Tests if our approach scales
- Training needed, so longer running

### Option 4: Ablation Studies
Test what components matter:

**Experiments:**
1. Single model for all classes vs per-class models
2. Pretrained vs random init
3. Different training epochs
4. Class balancing strategies

### Option 5: Quick Validation Tasks
Fast experiments to validate findings:

```bash
# Quick test on multiple small datasets
for dataset in cola mrpc rte; do
    python experiments/compare_all_approaches.py \
        --dataset $dataset \
        --quick \
        --output results/validation/$dataset
done
```

## Recommended: Option 1 (Different Dataset)

Start with AG News (4-class topic classification) because:
- ✅ Different task type (topics vs sentiment)
- ✅ More classes (4 vs 2) - harder problem
- ✅ Tests if approach generalizes
- ✅ Can run in parallel with IMDB
- ✅ ~30 minutes to prepare + run

### How to Run in Parallel

```bash
# Terminal 1 (already running): IMDB experiment
# Monitor: ssh vincent@nigel.birs.ca "tail -f ~/text-diffusion-experiments.log"

# Terminal 2: Start AG News experiment
ssh vincent@nigel.birs.ca
cd ~/text-diffusion
source venv/bin/activate

# Prepare AG News data
python scripts/prepare_agnews.py --max-samples 5000 --max-test 1000

# Run in separate screen
screen -dmS agnews-experiments bash -c 'cd ~/text-diffusion && source venv/bin/activate && ./experiments/RUN_EXPERIMENTS.sh full --dataset agnews --data-dir data/agnews-classifier --output results/comparison-agnews 2>&1 | tee ~/agnews-experiments.log'
```

### Monitor Both Experiments

```bash
# Check IMDB progress
ssh vincent@nigel.birs.ca "tail -20 ~/text-diffusion-experiments.log"

# Check AG News progress
ssh vincent@nigel.birs.ca "tail -20 ~/agnews-experiments.log"

# Check GPU usage
ssh vincent@nigel.birs.ca "nvidia-smi"
```

## Expected GPU Usage with Both

- **IMDB**: ~7 GB
- **AG News**: ~5-6 GB (4 classes instead of 2)
- **Total**: ~12-13 GB / 16 GB (75-80%)
- **Still safe**: 3-4 GB buffer

## What We'll Learn

Running on multiple datasets shows:
1. **Generalization**: Does our approach work beyond sentiment?
2. **Scalability**: How does it handle more classes?
3. **Robustness**: Consistent performance across tasks?

This makes the research MUCH more convincing than just IMDB results!
