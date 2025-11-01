# Text Diffusion Learning Guide

## What You're Building

A toy GPT-2 scale text diffusion model that learns to generate text through iterative denoising.

**Key Insight**: Instead of training a model to predict the next token (GPT-2), we train it to denoise masked text at varying corruption levels. Then we generate by starting with fully masked text and iteratively removing masks.

## Core Concepts

### 1. Standard Masked Language Modeling (BERT/RoBERTa)

**Standard approach:**
```
Input:  "The quick [MASK] fox jumps"
Target: "The quick brown fox jumps"
```
- Always masks 15% of tokens
- Model learns to fill in blanks
- **Not generative** - needs real text to mask

### 2. Diffusion-Based Approach (This Project)

**Our approach:**
```
Training:
  10% masked: "The quick [MASK] fox jumps over the lazy dog"
  50% masked: "The [MASK] brown [MASK] [MASK] over [MASK] [MASK] dog"
  100% masked: "[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]"
```
- **Variable masking** (10% to 100%)
- Model learns to denoise at ALL levels
- **Generative** - can start from 100% masked

**Generation:**
```
Step 1 (100% masked): [MASK] [MASK] [MASK] [MASK]
Step 2 (80% masked):  The [MASK] [MASK] fox
Step 3 (60% masked):  The quick [MASK] fox
Step 4 (40% masked):  The quick brown fox
Step 5 (20% masked):  The quick brown fox
...
Step 10 (0% masked):  The quick brown fox jumps
```

### 3. Why This Works

**Mathematical intuition:**
- Model learns p(token | context, corruption_level)
- At 100% corruption: Predicts from minimal context (frequent words, structure)
- At 50% corruption: Predicts from partial context (semantic meaning)
- At 10% corruption: Predicts from full context (rare words, details)

**The magic:** Same model, different masking = different denoising tasks

## Implementation Details

### Component 1: Data Collator (The Key Innovation)

**File:** `data_collator.py`

**What it does:**
```python
# For each batch during training:
1. Randomly select mask_prob from [1.0, 0.9, 0.8, ..., 0.1]
2. Mask that percentage of tokens (except prefix)
3. Labels = original unmasked tokens
4. Model learns to predict originals from masked input
```

**Why it matters:**
- Standard MLM: Always 15% masking → model only learns one denoising level
- Diffusion MLM: Variable masking → model learns ENTIRE denoising curve

### Component 2: Training Loop (Standard HuggingFace)

**File:** `train.py`

**What it does:**
```python
# Standard supervised learning:
for batch in data:
    masked_input, targets = data_collator(batch)  # Variable masking here
    predictions = model(masked_input)
    loss = cross_entropy(predictions, targets)
    loss.backward()
    optimizer.step()
```

**What model learns:**
- Input: "[MASK] [MASK] brown [MASK]" + mask_prob=0.5
- Output: Probability distribution over vocabulary for each mask
- Model implicitly learns corruption level from how many masks it sees

### Component 3: Generation (Iterative Denoising)

**File:** `generate.py`

**Algorithm:**
```python
# Start with fully masked continuation
text = [PREFIX] + [MASK] * N

# Iteratively denoise
for mask_prob in [1.0, 0.9, 0.8, ..., 0.1]:
    # Forward pass
    logits = model(text)

    # Sample tokens for masked positions
    for pos in masked_positions:
        text[pos] = sample(logits[pos], temperature, top_k)

    # Re-mask for next iteration
    text = re_mask(text, mask_prob)

return text
```

**Key decisions:**
1. **Schedule**: How fast to unmask (linear, cosine, exponential)
2. **Sampling**: How to pick tokens (greedy, top-k, nucleus)
3. **Temperature**: How random to be (0.0=deterministic, 1.0+=creative)

## Training Deep Dive

### What Gets Learned at Each Masking Level?

**High Masking (90-100%)**
- Very limited context
- Model learns: Sentence structure, common words, basic grammar
- Example: "[MASK] [MASK] [MASK] the [MASK]" → probably "... the ..."

**Medium Masking (40-60%)**
- Partial context available
- Model learns: Semantic relationships, context-dependent words
- Example: "Machine [MASK] is a [MASK] of AI" → "learning" and "subset"

**Low Masking (10-20%)**
- Almost full context
- Model learns: Rare words, specific phrasings, fine details
- Example: "The capital of France is [MASK]" → "Paris"

### Training Dynamics

**Expected loss curves:**
```
Masking Level    Initial Loss    Final Loss
100%             ~10.0           ~6.0        (hardest)
50%              ~6.0            ~3.0        (medium)
10%              ~3.0            ~1.5        (easiest)
```

**Good training:**
- All levels decrease over time
- 100% masked always highest loss (it's harder!)
- Smooth decrease (no sudden spikes)

**Bad training:**
- Some levels not decreasing → collator bug or insufficient data
- Sudden spikes → learning rate too high
- All levels converge to same value → model not learning corruption-dependent denoising

## Generation Deep Dive

### Denoising Schedules

**Linear (default):**
```python
[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
```
- Uniform unmasking rate
- Equal time at all corruption levels
- **Use for**: General-purpose generation, understanding the process

**Cosine:**
```python
[1.0, 0.99, 0.95, 0.88, 0.76, 0.59, 0.38, 0.19, 0.05, 0.01]
```
- More time at high and low corruption (ends)
- Faster through medium corruption (middle)
- **Use for**: Higher quality, more careful generation

**Exponential:**
```python
[1.0, 0.74, 0.55, 0.41, 0.30, 0.22, 0.17, 0.12, 0.09, 0.07]
```
- Fast unmasking early (structural decisions)
- Slow refinement late (details)
- **Use for**: Creative generation, exploring different structures

### Sampling Strategies

**Greedy (deterministic):**
```python
token = argmax(logits)
```
- Always pick highest probability
- Same input → same output
- **Use for**: Testing, debugging, need reproducibility

**Top-k:**
```python
top_k_tokens = topk(logits, k=50)
token = sample(softmax(top_k_tokens))
```
- Sample from top k most likely
- Good balance diversity/quality
- **Use for**: General generation (k=40-60)

**Nucleus (top-p):**
```python
sorted_probs = sort(softmax(logits))
cumsum_probs = cumsum(sorted_probs)
nucleus = sorted_probs[cumsum_probs <= p]
token = sample(nucleus)
```
- Dynamic cutoff based on probability mass
- Adapts to prediction confidence
- **Use for**: High-quality diverse generation (p=0.9-0.95)

**Temperature:**
```python
logits = logits / temperature
probs = softmax(logits)
```
- temperature < 1.0: More conservative (sharper distribution)
- temperature = 1.0: Use model's natural probabilities
- temperature > 1.0: More random (flatter distribution)
- **Use for**: Controlling creativity (0.5=safe, 1.0=balanced, 1.5=creative)

## Experimentation Guide

### Experiment 1: Understanding Masking

**Goal:** See what the model sees during training

**Run:**
```bash
python3 experiments/masking_viz.py
```

**What to observe:**
- How much text is visible at each masking level
- How prefix preservation works
- Why we need all masking levels

**Key insight:** Training with all levels → model learns full denoising curve

### Experiment 2: Compare Schedules

**Goal:** See how schedule affects generation quality

**Run:**
```bash
# Same prefix, different schedules
PREFIX="Machine learning is"

python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --schedule linear
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --schedule cosine
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --schedule exponential
```

**What to observe:**
- Coherence of generated text
- Creativity vs. quality trade-off
- Which schedule works best for your use case

**Expected:** Cosine often highest quality, exponential most creative

### Experiment 3: Sampling Method Impact

**Goal:** Understand sampling strategy trade-offs

**Run:**
```bash
PREFIX="The future of artificial intelligence"

python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --sampling greedy
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --sampling topk --top-k 50
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --sampling nucleus --top-p 0.9
```

**What to observe:**
- Greedy: Same output every time (deterministic)
- Top-k: Diverse but sometimes off-topic
- Nucleus: Best balance for most tasks

### Experiment 4: Temperature Effects

**Goal:** Control generation creativity

**Run:**
```bash
PREFIX="Scientists have discovered"

python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --temperature 0.3
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --temperature 0.7
python3 generate.py --checkpoint results/final-model --prefix "$PREFIX" --temperature 1.2
```

**What to observe:**
- Low temp: Conservative, common words
- Medium temp: Balanced quality/diversity
- High temp: Creative but sometimes incoherent

**Expected:** temp=0.5-0.8 best for most tasks

## Common Patterns & Debugging

### Issue: "Model ignoring prefix"

**Symptoms:**
- Generated text doesn't relate to prefix
- Output is generic/random

**Causes:**
1. PREFIX_LEN not set correctly in generate.py
2. Model not trained with prefix preservation
3. Prefix too short (< 3 tokens)

**Solutions:**
```python
# In generate.py, verify:
prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
prefix_length = len(prefix_ids)
# Use this exact value to avoid masking prefix

# In train.py, verify:
config.prefix_length = 5  # Or higher
# Must match generation expectations
```

### Issue: "Repetitive generation"

**Symptoms:**
- Same words/phrases repeated
- Boring/predictable output

**Causes:**
1. Temperature too low
2. Greedy sampling
3. Model undertrained

**Solutions:**
```bash
# Increase temperature
python3 generate.py --temperature 0.8  # Up from 0.3

# Use top-k or nucleus
python3 generate.py --sampling topk --top-k 50

# Train longer
python3 train.py --epochs 5  # Up from 3
```

### Issue: "Incoherent generation"

**Symptoms:**
- Text doesn't make sense
- Grammar errors
- Random topics

**Causes:**
1. Temperature too high
2. Model undertrained
3. Too few denoising steps

**Solutions:**
```bash
# Lower temperature
python3 generate.py --temperature 0.5  # Down from 1.0

# More denoising steps
python3 generate.py --steps 20  # Up from 10

# Train longer or use larger model
python3 train.py --model-name roberta-base --epochs 5
```

## Advanced Topics

### Parallel Decoding

**Current:** Fill in masks sequentially (slower but safer)
**Alternative:** Fill multiple masks at once (faster but may degrade quality)

```python
# Instead of:
for pos in masked_positions:
    fill(pos)
    re_mask_others()

# Could do:
fill(all_masked_positions)  # Parallel
re_mask_some()
```

**Trade-off:** Speed vs. quality (tokens don't get to condition on each other)

### Adaptive Schedules

**Current:** Fixed schedule regardless of content
**Alternative:** Adjust based on model confidence

```python
# If model is confident, unmask more aggressively
if avg_confidence > threshold:
    unmask_more_tokens()
else:
    unmask_fewer_tokens()  # Take more time to refine
```

**Benefit:** Faster for easy text, slower for hard text

### Classifier Guidance

**Current:** No control over content beyond prefix
**Alternative:** Guide generation toward desired attributes

```python
# Steer generation toward specific properties
score = classifier(generated_text)  # E.g., sentiment, topic
adjust_logits(logits, score)  # Boost tokens that increase desired score
```

**Use cases:** Sentiment control, topic steering, style transfer

## Next Steps for Your Learning

### Week 1: Understand the Basics
- [ ] Run masking visualization
- [ ] Train quick-test model (30 min)
- [ ] Generate text and observe denoising steps
- [ ] Read through data_collator.py line-by-line

### Week 2: Experiment with Settings
- [ ] Try all three denoising schedules
- [ ] Compare sampling methods
- [ ] Test different temperatures
- [ ] Train full 3-epoch model

### Week 3: Modify the Code
- [ ] Add custom masking distribution (non-uniform)
- [ ] Implement parallel decoding
- [ ] Add confidence-based adaptive schedule
- [ ] Visualize layer activations

### Week 4: Domain Adaptation
- [ ] Load your own dataset
- [ ] Adjust prefix lengths for your use case
- [ ] Add constraints for structured output
- [ ] Implement domain-specific metrics

## References

**Original Blog Post:**
https://nathan.rs/posts/roberta-diffusion/

**Key Papers:**
- RoBERTa: https://arxiv.org/abs/1907.11692
- Discrete Diffusion (D3PM): https://arxiv.org/abs/2107.03006
- BERT Masked LM: https://arxiv.org/abs/1810.04805

**HuggingFace Docs:**
- Transformers: https://huggingface.co/docs/transformers
- Datasets: https://huggingface.co/docs/datasets
- Training: https://huggingface.co/docs/transformers/training

## Questions to Ponder

1. **Why does variable masking matter?**
   - Think: What if we only trained at 50% masking?
   - Answer: Model wouldn't learn how to denoise 100% or 10% masked text

2. **Why iterative denoising vs. one-shot?**
   - Think: Could we just predict all masks at once?
   - Answer: Yes, but quality is worse (tokens can't condition on each other)

3. **Why start from pretrained RoBERTa?**
   - Think: Could we train from scratch?
   - Answer: Yes, but much slower (need to learn language model + denoising)

4. **How is this different from GPT-2?**
   - GPT-2: p(next_token | previous_tokens) - autoregressive
   - This: p(token | context, corruption) - diffusion
   - Trade-off: GPT faster generation, diffusion more flexible editing

5. **When would you use this vs. GPT?**
   - Use diffusion for: Editing, filling gaps, controlled generation
   - Use GPT for: Fast generation, long-form text, chatbots

Happy learning! 🚀
