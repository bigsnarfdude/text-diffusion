# Comparison: text-diffusion vs tiny-diffusion

## Executive Summary

This document compares our `~/development/text-diffusion` experiments with the `tiny-diffusion` repository to identify learning opportunities and potential improvements.

**Key Finding**: The two projects take fundamentally different approaches to discrete text diffusion:
- **Our approach**: Leverages pretrained MLM (RoBERTa/MDLM) with variable masking
- **tiny-diffusion**: Trains from scratch with true discrete diffusion on character-level data

Both are valid but serve different purposes. We can learn from tiny-diffusion's:
1. Clean discrete diffusion implementation
2. Parallel decoding strategies
3. Visualization approaches
4. Efficient training patterns

---

## Architecture Comparison

### Model Architecture

| Aspect | text-diffusion (Ours) | tiny-diffusion |
|--------|----------------------|----------------|
| **Base Model** | Pretrained RoBERTa/MDLM | Trained from scratch |
| **Tokenization** | WordPiece (32k vocab) | Character-level (128 ASCII) |
| **Size** | 82M params (RoBERTa) | 10.8M params |
| **Layers** | 12 transformer blocks | 6 transformer blocks |
| **Attention** | Bidirectional (RoBERTa) | Bidirectional (no causal mask) |
| **Embeddings** | Learned position emb | Rotary position embeddings (RoPE) |
| **Normalization** | LayerNorm | RMS Normalization (functional) |
| **Activation** | GELU | ReLU² |
| **Attention Type** | Standard multi-head | QK-normalized attention |

**Key Differences**:
- We rely on transfer learning; they train from scratch
- We use subword tokens; they use characters
- They use more modern architectural choices (RoPE, RMS norm, ReLU²)

---

## Diffusion Approach Comparison

### Masking Strategy

#### Our Approach (Variable Masking)
```python
# DiffusionDataCollator
mask_probs = [1.0, 0.9, 0.8, ..., 0.1]  # 10 levels
per_batch_masking()  # Different rate each batch
```

**Characteristics**:
- 10 discrete masking levels
- Random selection per batch
- Prefix preservation (first N tokens)
- Special token protection

#### tiny-diffusion Approach (Linear Schedule)
```python
# MaskedDiffusionSchedule
timesteps = 128
mask_prob(t) = t / 128  # Continuous schedule
```

**Characteristics**:
- 128 timestep levels (much finer granularity)
- Linear interpolation
- Context preservation (first 16 tokens)
- Smooth transition from 0% → 100% masked

**What we can learn**:
- **Finer-grained timesteps** (128 vs 10) might improve generation quality
- **Continuous schedule** easier to reason about than discrete levels
- **Context length tuning** (they use 16, we use variable)

---

### Training Objective

#### Our Approach
```python
# Standard MLM loss on masked tokens
loss = cross_entropy(predictions[masked], labels[masked])
# Loss computed at variable masking levels
```

#### tiny-diffusion Approach
```python
# Identical objective, different schedule
loss = cross_entropy(predictions[masked], labels[masked])
# Loss computed at sampled timestep t
```

**Similarity**: Both use the same loss function (masked token prediction)

**What we can learn**:
- Their time embedding approach (adds timestep info to each position)
- We could add explicit timestep conditioning to our models

---

## Generation/Decoding Comparison

### Our Approach: Schedule-Based Denoising

```python
# generate.py
for step in range(num_steps):
    mask_prob = schedule(step)  # Linear/cosine/exponential
    predictions = model(input_ids)

    # Sample based on strategy
    if strategy == "greedy":
        tokens = argmax(predictions)
    elif strategy == "top_k":
        tokens = sample_top_k(predictions, k=50)
    elif strategy == "nucleus":
        tokens = sample_nucleus(predictions, p=0.9)

    # Replace all masked tokens at once
    input_ids[masked] = tokens[masked]
```

**Characteristics**:
- Fixed number of steps (typically 10)
- Replace all masked tokens each step
- Schedule determines masking rate
- Three sampling strategies

---

### tiny-diffusion Approach: Confidence-Based Denoising

```python
# sample.py - Method 1: Confidence threshold
for step in range(max_steps):
    logits = model(tokens, timestep)
    confidence = softmax(logits).max(dim=-1)

    # Only unmask tokens above confidence threshold
    unmask_mask = (confidence >= threshold) & is_masked

    if not unmask_mask.any():
        # Force unmask highest confidence token
        unmask_mask[confidence.argmax()] = True

    tokens[unmask_mask] = sample(logits[unmask_mask])
```

```python
# Method 2: Top-K parallel decoding
for step in range(max_steps):
    logits = model(tokens, timestep)
    confidence = softmax(logits).max(dim=-1)

    # Unmask exactly K highest-confidence tokens
    top_k_indices = confidence.topk(k).indices
    tokens[top_k_indices] = sample(logits[top_k_indices])
```

**Characteristics**:
- **Variable steps** (stops when all tokens unmasked)
- **Selective unmasking** (only high-confidence tokens)
- **Quality control** via confidence threshold
- **Progressive refinement** (high-confidence first)

---

### 🔑 KEY LEARNING OPPORTUNITY: Confidence-Based Decoding

**Why this is better than our approach**:

1. **Adaptive speed**: Fast for easy sequences, slow for hard ones
2. **Quality control**: Only commit to high-confidence predictions
3. **Progressive refinement**: Easier tokens first, harder ones later
4. **Natural stopping**: Converges when all tokens confident

**How we could integrate this**:

```python
# New file: src/confidence_generate.py
def generate_confidence_based(
    model,
    prefix_ids,
    max_steps=50,
    confidence_threshold=0.7,
    temperature=1.0
):
    """
    Generate using confidence-based parallel decoding.
    Only unmask tokens with prediction confidence >= threshold.
    """
    # Start with all masks after prefix
    tokens = create_masked_sequence(prefix_ids, seq_len=256)

    for step in range(max_steps):
        # Get predictions
        outputs = model(tokens)
        logits = outputs.logits

        # Compute confidence (max softmax probability)
        probs = F.softmax(logits / temperature, dim=-1)
        confidence, predicted = probs.max(dim=-1)

        # Find masked positions above threshold
        is_masked = (tokens == MASK_TOKEN_ID)
        high_confidence = confidence >= confidence_threshold
        unmask_positions = is_masked & high_confidence

        # If nothing meets threshold, unmask highest confidence
        if not unmask_positions.any():
            masked_confidence = confidence.clone()
            masked_confidence[~is_masked] = -1  # Ignore non-masked
            best_idx = masked_confidence.argmax()
            unmask_positions[best_idx] = True

        # Update tokens at high-confidence positions
        tokens[unmask_positions] = predicted[unmask_positions]

        # Stop if all unmasked
        if not (tokens == MASK_TOKEN_ID).any():
            break

    return tokens
```

**Benefits for our project**:
- Better control over generation quality
- More interpretable (can see which tokens model is confident about)
- Natural early stopping (no need to guess num_steps)
- Could improve classification (only use high-confidence predictions)

---

## Visualization Comparison

### Our Visualization (tools/visualize_generation.py)

```python
# Create GIF showing each denoising step
for step in steps:
    # Show text with masked tokens as [MASK]
    display_text(detokenize(tokens))
    save_frame()

create_gif(frames)
```

**Output**: Animated GIF showing progressive denoising

**Strengths**:
- Simple and clear
- Works with our tokenizer
- Good for debugging

---

### tiny-diffusion Visualization (animations/diffusion-process.py)

```python
# More sophisticated visualization
for step in steps:
    # Display masked positions as █ blocks
    text = [char if not masked else '█' for char, masked in zip(text, mask)]

    # Color code by confidence
    colors = get_confidence_colors(confidence_scores)

    # Annotate with timestep info
    ax.set_title(f"Step {step}/{total}, {percent_complete}% complete")
```

**Output**: Annotated animation with:
- Block characters for masks (█)
- Timestep information
- Completion percentage
- Cleaner visual presentation

**What we can learn**:
- Use block characters (█) instead of [MASK] text (cleaner)
- Add timestep/progress information
- Color-code by confidence
- Add completion percentage

---

### Proposed Enhanced Visualization for Our Project

```python
# tools/visualize_generation_v2.py
def create_enhanced_animation(
    model,
    prefix,
    num_steps=10,
    confidence_threshold=0.7
):
    """Enhanced visualization with confidence coloring."""
    frames = []

    for step in range(num_steps):
        # Generate predictions
        outputs = model(tokens)
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, _ = probs.max(dim=-1)

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Render text with color coding
        for i, (token_id, conf) in enumerate(zip(tokens, confidence)):
            if token_id == MASK_TOKEN_ID:
                char = '█'
                color = 'gray'
            else:
                char = tokenizer.decode([token_id])
                color = confidence_to_color(conf)  # Red=low, Green=high

            ax.text(x, y, char, color=color, fontsize=12)

        # Add metadata
        masked_count = (tokens == MASK_TOKEN_ID).sum()
        ax.set_title(
            f"Step {step}/{num_steps} | "
            f"{masked_count} tokens remaining | "
            f"Avg confidence: {confidence.mean():.2%}"
        )

        frames.append(fig)

    create_gif(frames, 'enhanced_animation.gif')
```

---

## Code Quality & Patterns Comparison

### Our Strengths

1. **Comprehensive Documentation**
   - 1000+ lines of docs across multiple files
   - PROJECT_SUMMARY.md, LEARNING_GUIDE.md, etc.
   - Better than tiny-diffusion's single README

2. **Modular Architecture**
   - Clean separation: config, data, train, generate, classifier
   - Easy to extend and experiment
   - Well-organized src/ directory

3. **Multiple Approaches**
   - RoBERTa, GPT-2, MDLM implementations
   - Comparative evaluation
   - More comprehensive experimental framework

4. **Production Patterns**
   - Deployment scripts
   - Monitoring tools
   - Screen session management

---

### tiny-diffusion's Strengths

1. **Simpler Codebase**
   - 3 main files (model, training, sample)
   - Easier to understand for newcomers
   - No abstraction overhead

2. **Modern Architecture Choices**
   - Rotary position embeddings (better than learned)
   - RMS normalization (more efficient)
   - QK-normalized attention (more stable)
   - Flash attention integration

3. **Clean Training Loop**
   - Simple, readable training code
   - No framework dependencies (plain PyTorch)
   - Easy to debug

4. **Efficient Implementation**
   - Pre-computed rotary embeddings
   - Functional normalization (no params)
   - `@torch.inference_mode()` for sampling
   - Minimal memory overhead

**What we can learn**:
- Simplify our training loop (less abstraction)
- Consider modern architectural improvements
- Use functional patterns where possible
- Add proper inference mode decorators

---

## Training Efficiency Comparison

### Our Approach

**RoBERTa Classifier**:
- **Time**: 2-3 hours (full training on IMDB)
- **Hardware**: CPU/single GPU
- **Dataset**: 17,500 samples per class
- **Batch size**: 16
- **Epochs**: 3

**MDLM Classifier**:
- **Time**: 7-12 hours per class
- **Hardware**: GPU (on nigel.birs.ca)
- **Dataset**: Same as RoBERTa
- **Batch size**: 8 (effective 32 with grad accum)
- **Epochs**: 20-164

---

### tiny-diffusion Approach

**Character-level Generation**:
- **Time**: 30 minutes (4× A100 GPUs)
- **Dataset**: 1.1M characters (Tiny Shakespeare)
- **Batch size**: Not specified (likely 64-128)
- **Steps**: 20,000

**Why they're faster**:
1. Smaller model (10.8M vs 82M params)
2. Smaller dataset (1.1MB vs 50MB+ IMDB)
3. Character-level (simpler than WordPiece)
4. From-scratch training (no pretrained overhead)
5. Better hardware (4× A100)

---

## Key Learnings & Recommendations

### 1. ✅ Implement Confidence-Based Decoding

**Priority**: HIGH
**Effort**: Medium
**Impact**: HIGH

**Implementation**:
```bash
# Create new generation module
cp src/generate.py src/confidence_generate.py

# Add confidence-based method
# Add to generate.py or create separate file
```

**Benefits**:
- Better generation quality
- Natural stopping criterion
- Interpretable (see model confidence)
- Could improve classifier performance

---

### 2. ✅ Increase Timestep Granularity

**Priority**: MEDIUM
**Effort**: LOW
**Impact**: MEDIUM

**Current**: 10 masking levels [1.0, 0.9, ..., 0.1]
**Proposed**: 100 or 128 levels (match tiny-diffusion)

**Implementation**:
```python
# src/data_collator.py
# Change from:
mask_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# To:
mask_probs = [i/100 for i in range(1, 101)]  # 100 levels
# Or
mask_probs = [i/128 for i in range(1, 129)]  # 128 levels (match tiny-diffusion)
```

**Benefits**:
- Smoother denoising curve
- Better training signal
- More granular control during generation

**Considerations**:
- Slightly slower training (more variety in batches)
- May need to increase batch size to sample all levels equally

---

### 3. ✅ Add Explicit Time Conditioning

**Priority**: MEDIUM
**Effort**: HIGH
**Impact**: MEDIUM-HIGH

**Current**: Model doesn't know which timestep it's at
**Proposed**: Add timestep embeddings like tiny-diffusion

**Implementation**:
```python
# New file: src/time_conditioned_model.py
class TimeConditionedRoBERTa(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        # Add time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, input_ids, timestep=None, **kwargs):
        # Get base embeddings
        hidden_states = self.roberta.embeddings(input_ids)

        # Add time conditioning
        if timestep is not None:
            time_emb = self.time_embed(timestep.unsqueeze(-1))
            hidden_states = hidden_states + time_emb.unsqueeze(1)

        # Continue with transformer
        outputs = self.roberta.encoder(hidden_states, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)

        return MaskedLMOutput(logits=logits, ...)
```

**Benefits**:
- Model knows corruption level
- Can learn different strategies for different timesteps
- Better theoretical grounding (true diffusion model)

**Considerations**:
- Requires model architecture change
- Need to retrain from scratch
- More complex than current approach

---

### 4. ✅ Improve Visualization

**Priority**: LOW
**Effort**: LOW
**Impact**: MEDIUM

**Proposed enhancements**:
1. Use block characters (█) instead of [MASK] text
2. Add confidence color coding
3. Show timestep and progress information
4. Create side-by-side comparison view

**Implementation**:
```bash
# Enhance existing tool
vim tools/visualize_generation.py

# Add new features:
# - Block character rendering
# - Confidence color mapping
# - Progress annotations
# - Metadata display
```

---

### 5. ✅ Modern Architecture Improvements

**Priority**: LOW (for new models)
**Effort**: HIGH
**Impact**: MEDIUM

**Considerations**:
- Rotary position embeddings (better than learned)
- RMS normalization (more efficient)
- ReLU² activation (slight improvement)
- Flash attention (2-3x faster)

**Note**: These require training from scratch, so not practical for current RoBERTa-based work. Consider for future experiments.

---

### 6. ⚠️ Character-Level Experiments

**Priority**: LOW
**Effort**: MEDIUM
**Impact**: LOW (for our use case)

**Proposal**: Try character-level diffusion like tiny-diffusion

**Pros**:
- Simpler tokenization
- No OOV issues
- Easier to visualize
- Smaller vocab (128 vs 32k)

**Cons**:
- Longer sequences (256 chars << 256 WordPiece tokens)
- Less semantic meaning per token
- Harder to match pretrained model performance
- Not suited for classification tasks

**Recommendation**: Only if doing pure research/learning, not for practical applications.

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)

1. **Add confidence-based generation**
   ```bash
   cd ~/development/text-diffusion
   cp src/generate.py src/confidence_generate.py
   # Implement confidence-based decoding
   # Test with existing trained models
   ```

2. **Increase timestep granularity**
   ```bash
   # Edit src/data_collator.py
   # Change to 100 or 128 levels
   # Retrain with finer schedule
   ```

3. **Enhance visualization**
   ```bash
   # Edit tools/visualize_generation.py
   # Add block characters and confidence colors
   # Generate new sample animations
   ```

---

### Phase 2: Experimental (1 week)

4. **Time-conditioned model**
   ```bash
   # Create new model architecture
   # Train from scratch with time embeddings
   # Compare with baseline
   ```

5. **Comparison study**
   ```bash
   # Train models with different configurations
   # Compare: 10 vs 100 timesteps
   # Compare: schedule-based vs confidence-based
   # Document findings
   ```

---

### Phase 3: Long-term (if interested)

6. **Character-level experiments**
   ```bash
   # Implement character tokenizer
   # Train tiny model from scratch
   # Compare with WordPiece approach
   ```

7. **Modern architecture**
   ```bash
   # Build model with RoPE, RMS norm, etc.
   # Train from scratch
   # Benchmark against RoBERTa
   ```

---

## Conclusion

**What tiny-diffusion does better**:
1. ✅ Confidence-based parallel decoding (adaptive quality)
2. ✅ Finer timestep granularity (128 vs 10)
3. ✅ Explicit time conditioning (model knows timestep)
4. ✅ Modern architectural choices (RoPE, RMS norm)
5. ✅ Cleaner visualization (block chars, colors, metadata)
6. ✅ Simpler codebase (easier to understand)

**What we do better**:
1. ✅ Transfer learning from pretrained models
2. ✅ Multiple experimental approaches (RoBERTa, GPT-2, MDLM)
3. ✅ Comprehensive documentation (1000+ lines)
4. ✅ Production-ready tooling (deployment, monitoring)
5. ✅ Real classification task (94% accuracy)
6. ✅ Subword tokenization (better for real text)

**Recommended immediate actions**:
1. **Implement confidence-based generation** ← Biggest impact
2. **Increase timestep granularity** ← Easy improvement
3. **Enhance visualization** ← Better debugging

**Future considerations**:
- Time-conditioned models (if starting new experiments)
- Modern architecture (for next generation of models)
- Character-level (for pure research/learning)

---

## References

- **tiny-diffusion**: `/Users/vincent/Downloads/tiny-diffusion-main/`
- **text-diffusion**: `~/development/text-diffusion/`
- **This document**: `~/development/text-diffusion/COMPARISON_WITH_TINY_DIFFUSION.md`
