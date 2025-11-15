# Experiments Status - Running on nigel.birs.ca

## ✅ Both Experiments Running in Parallel!

**GPU**: NVIDIA RTX 4070 Ti SUPER (16 GB)
**Usage**: 10.3 GB / 16.4 GB (63%)
**Utilization**: 100%

### Experiment 1: IMDB (Binary Sentiment Classification)

**Screen**: `text-diffusion-exp`
**Log**: `~/text-diffusion-experiments.log`
**Dataset**: 2 classes, 10,000 train, 1,000 test

**Progress:**
- ✅ GPT-2 Zero-Shot: **89.4% accuracy** (11 seconds)
- ⏳ GPT-2 Native: Training epoch 1/3 (28% complete)
- ⏳ Diffusion Baseline: Pending
- ⏳ Diffusion Trained: Pending

### Experiment 2: AG News (4-Class Topic Classification)

**Screen**: `agnews-exp`
**Log**: `~/agnews-experiments.log`
**Dataset**: 4 classes (World, Sports, Business, Sci/Tech), 12,000 train, 1,000 test

**Progress:**
- ⏳ Training diffusion models (4 classes)
- ⏳ Then: Run 4 comparison approaches

## Why Two Datasets?

### Research Validation
1. **IMDB** (sentiment): Binary classification, easier task
2. **AG News** (topics): 4-way classification, harder task

**Testing:**
- Does approach generalize beyond sentiment?
- How does it scale to more classes?
- Consistent performance across task types?

This makes the research **much more convincing** than just IMDB results!

## Monitor Commands

```bash
# Check IMDB progress
ssh vincent@nigel.birs.ca "tail -30 ~/text-diffusion-experiments.log"

# Check AG News progress
ssh vincent@nigel.birs.ca "tail -30 ~/agnews-experiments.log"

# Check GPU usage
ssh vincent@nigel.birs.ca "nvidia-smi"

# Watch IMDB live
ssh vincent@nigel.birs.ca "tail -f ~/text-diffusion-experiments.log"
```

## Expected Results

### IMDB (Binary)
- GPT-2 Zero-Shot: ✅ 89.4%
- GPT-2 Native: ~88-90%
- Diffusion Baseline: ~50-60%
- Diffusion Trained: ~85-90%?

### AG News (4-Class)
- GPT-2 Zero-Shot: ~75-85%
- GPT-2 Native: ~85-90%
- Diffusion Baseline: ~40-50%
- Diffusion Trained: ???

## Core Research Questions

### Q1: Does likelihood-based classification work?
**Test**: Diffusion baseline vs random
- IMDB: Need > 50%
- AG News: Need > 25%

### Q2: Does per-class training enable discrimination?
**Test**: Diffusion-trained vs diffusion-baseline
- **This is our core hypothesis**
- Need statistically significant gap (p < 0.05)
- Need substantial accuracy improvement

### Q3: Is it competitive?
**Test**: Diffusion-trained vs GPT-2 approaches
- Not about "beating" them
- About showing it's a **viable alternative**

### Q4: Does it generalize?
**Test**: Performance across both datasets
- Similar patterns on IMDB and AG News?
- Consistent improvements from training?

## Estimated Completion

**IMDB**: ~2 hours from now
**AG News**: ~3 hours from now (4 classes = more training)

Results will be in:
- `~/text-diffusion/results/comparison/comparison_imdb_*.json`
- `~/text-diffusion/results/comparison-agnews/comparison_agnews_*.json`

## Key Finding So Far

**GPT-2 zero-shot achieved 89.4% on IMDB** without any training! This is surprisingly strong, which means:
- IMDB sentiment is a relatively easy task
- Our approach needs to match this to be worthwhile
- Makes AG News (harder task) even more important

## Next Steps

Once experiments complete:
1. Download results
2. Create comparison visualizations
3. Analyze statistical significance
4. Draw conclusions about research viability
