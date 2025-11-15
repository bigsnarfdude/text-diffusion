# TRUE Text Diffusion Generative Classifier - Implementation Plan

## Goal
Build a generative classification system using TRUE discrete text diffusion that rivals or exceeds GPT-2's 90.1% accuracy on IMDB sentiment classification.

## Current Status: Research Phase

### Discovered Approaches (2024 State-of-the-Art)

#### 1. MDLM (Masked Diffusion Language Model) - NeurIPS 2024 ⭐ BEST CHOICE
**Source**: https://github.com/kuleshov-group/mdlm
**Paper**: https://arxiv.org/abs/2406.07524

**Advantages**:
- ✅ State-of-the-art: 17% improvement over previous diffusion methods
- ✅ 25-30x faster decoding than SSD-LM
- ✅ Industry adoption: Powers ByteDance Seed Diffusion, Nvidia Genmol
- ✅ Active development and maintenance
- ✅ Pre-trained models available
- ✅ Simple masked diffusion approach

**Key Features**:
- Simplified Rao-Blackwellized objective
- Mixture of masked language modeling losses
- Semi-autoregressive generation
- Multiple sampling strategies (ancestral, analytic, efficient)
- Supports DiT, transformer, and Mamba architectures

**Unknown**:
- ❓ Can it compute P(text) for classification?
- ❓ Does it support likelihood-based classification?

**Next Steps**:
1. Clone repo and examine code for likelihood computation
2. Check if model exposes log probability of sequences
3. Test if we can adapt for per-class training

---

#### 2. SSD-LM (Semi-autoregressive Simplex-based Diffusion)
**Source**: https://arxiv.org/abs/2210.17432

**Advantages**:
- ✅ Performs diffusion in natural vocabulary space (not learned latent)
- ✅ Matches/outperforms GPT-2 on quality and diversity
- ✅ Allows classifier guidance with off-the-shelf classifiers

**Disadvantages**:
- ❌ Slower than MDLM (25-30x slower decoding)
- ❌ Less recent (2022 vs 2024)

---

#### 3. D3PM (Discrete Denoising Diffusion Probabilistic Models)
**Paper**: https://arxiv.org/abs/2107.03006

**Advantages**:
- ✅ Foundational approach for discrete diffusion
- ✅ Explicit transition matrices for token changes
- ✅ Strong theoretical foundation

**Disadvantages**:
- ❌ More complex to implement
- ❌ Older approach (2021)
- ❌ Superseded by MDLM and others

---

#### 4. SEDD (Score Entropy Discrete Diffusion)
**Note**: Won ICML 2024 Best Paper Award

**Advantages**:
- ✅ 6-8x better generative perplexity than GPT-2
- ✅ 32x fewer network evaluations
- ✅ Strong theoretical contribution

**Unknown**:
- ❓ Implementation availability
- ❓ Ease of adaptation for classification

---

## Recommended Approach: MDLM-based Generative Classifier

### Phase 1: Feasibility Check (Today/Tomorrow)

**Objective**: Determine if MDLM can compute P(text) for classification

**Tasks**:
1. Clone MDLM repository
   ```bash
   git clone https://github.com/kuleshov-group/mdlm.git
   cd mdlm
   conda env create -f requirements.yaml
   conda activate mdlm
   ```

2. Examine codebase for likelihood computation:
   - Check model API for `log_prob()` or similar
   - Look for perplexity evaluation code (computes P(text))
   - Review training objective (should reveal if P(text) is computable)

3. Test likelihood computation:
   ```python
   # Pseudocode - verify MDLM can do this
   from mdlm import MDLM

   model = MDLM.load_pretrained('kuleshov-group/mdlm-owt')
   text = "This movie was amazing!"
   log_prob = model.compute_log_probability(text)  # Can MDLM do this?
   ```

**Decision Point**:
- ✅ If MDLM supports P(text): Proceed to Phase 2
- ❌ If MDLM cannot compute P(text): Fall back to SSD-LM or custom implementation

---

### Phase 2: Per-Class Model Training (2-3 days)

**Objective**: Train separate MDLM models for each class

**Data**: Combined IMDB + synthetic (19,133 samples)
- Class 0 (negative): 9,460 samples
- Class 1 (positive): 9,673 samples

**Training Configuration**:
```bash
# Train negative class model
python main.py \
  model=small \
  data=imdb_negative \
  parameterization=subs \
  model.length=512 \
  training.epochs=20 \
  training.batch_size=32 \
  output_dir=results-mdlm/class_0

# Train positive class model
python main.py \
  model=small \
  data=imdb_positive \
  parameterization=subs \
  model.length=512 \
  training.epochs=20 \
  training.batch_size=32 \
  output_dir=results-mdlm/class_1
```

**Expected Training Time**:
- Small model (~100M params)
- 20 epochs × 9,500 samples
- Estimate: 6-12 hours per class on GPU
- Total: 12-24 hours for both classes

---

### Phase 3: Classification Implementation (1 day)

**Objective**: Implement generative classifier using trained MDLM models

```python
class MDLMGenerativeClassifier:
    def __init__(self, class_models):
        self.models = class_models  # [model_0, model_1]
        self.num_classes = len(class_models)

    def classify(self, text):
        """
        Classify using Bayes rule:
        P(class | text) ∝ P(text | class) × P(class)
        """
        log_probs = []

        for class_id in range(self.num_classes):
            # Compute log P(text | class_i) using MDLM
            log_prob = self.models[class_id].compute_log_probability(text)
            log_probs.append(log_prob)

        # Assume uniform prior: P(class) = 1/K
        # Classification: argmax P(text | class)
        predicted_class = np.argmax(log_probs)

        return predicted_class, log_probs
```

**Key Implementation Details**:
1. Each MDLM model computes P(text | class_i)
2. Use Bayes rule for classification
3. No need for discriminative classifier - pure generative approach

---

### Phase 4: Evaluation & Comparison (1 day)

**Objective**: Evaluate MDLM classifier vs GPT-2 baseline

**Metrics**:
- Accuracy (primary metric)
- Precision, Recall, F1 per class
- Confusion matrix
- Inference speed (samples/second)
- Calibration (likelihood quality)

**Baseline Comparison**:
| Model | Architecture | Accuracy | Speed |
|-------|-------------|----------|-------|
| GPT-2 Native | Autoregressive | 90.1% | Fast |
| RoBERTa "Diffusion" | MLM (broken) | 61.7% | Slow |
| **MDLM Diffusion** | True discrete diffusion | **TBD** | ? |

**Success Criteria**:
- ✅ Excellent: >90% (matches or exceeds GPT-2)
- ✅ Good: 85-90% (competitive with GPT-2)
- ⚠️ Acceptable: 80-85% (shows promise, needs tuning)
- ❌ Failed: <80% (fundamental issues)

---

## Alternative: Custom Discrete Diffusion Implementation

**If MDLM doesn't support P(text) computation, implement from scratch:**

### Minimal Discrete Diffusion for Classification

**Core Components**:

1. **Forward Process** (text → noise):
   ```python
   def forward_diffusion(text, t, vocab_size):
       """Add noise to text at timestep t"""
       # Replace tokens with random vocab items
       # Noise increases with t: t=0 (no noise) → t=T (pure noise)
       noise_rate = t / T
       noisy_text = randomly_mask(text, prob=noise_rate)
       return noisy_text
   ```

2. **Reverse Process** (noise → text):
   ```python
   def reverse_diffusion(noisy_text, t, model):
       """Denoise text at timestep t"""
       # Predict original token at each position
       predictions = model(noisy_text, t)
       # Sample from predictions
       denoised_text = sample_from_predictions(predictions)
       return denoised_text
   ```

3. **Training Objective**:
   ```python
   def train_step(text, model):
       """Train denoising model"""
       # Random timestep
       t = random.randint(0, T)

       # Forward process: add noise
       noisy_text = forward_diffusion(text, t)

       # Predict original tokens
       predictions = model(noisy_text, t)

       # Loss: cross-entropy with true tokens
       loss = cross_entropy(predictions, text)
       return loss
   ```

4. **Likelihood Computation**:
   ```python
   def compute_log_probability(text, model, num_samples=100):
       """Compute P(text) via importance sampling"""
       log_prob = 0.0

       for _ in range(num_samples):
           # Sample random timestep and noise
           t = random.randint(0, T)
           noisy_text = forward_diffusion(text, t)

           # Compute denoising probability
           pred_probs = model(noisy_text, t)
           token_probs = pred_probs.gather(text)  # P(token | noisy)

           log_prob += torch.log(token_probs).sum()

       return log_prob / num_samples
   ```

**Implementation Timeline**:
- 2-3 days for core diffusion framework
- 1 day for per-class training integration
- 1 day for likelihood-based classification
- **Total: 4-5 days**

---

## Timeline Summary

### Option A: MDLM-based (if P(text) supported)
- **Day 1**: Feasibility check, MDLM setup
- **Day 2-3**: Per-class training (12-24 hours)
- **Day 4**: Classification implementation
- **Day 5**: Evaluation and comparison
- **Total: 5 days**

### Option B: Custom Implementation (if MDLM insufficient)
- **Day 1-3**: Implement discrete diffusion framework
- **Day 4**: Per-class training
- **Day 5-6**: Classification and evaluation
- **Total: 6 days**

---

## Risk Assessment

### Technical Risks

1. **MDLM may not compute P(text)** (High probability)
   - **Impact**: Need custom implementation
   - **Mitigation**: Have fallback plan ready

2. **Discrete diffusion may underperform** (Medium probability)
   - **Impact**: May not reach 90% accuracy
   - **Mitigation**: Test on subset first, iterate quickly

3. **Training time longer than expected** (Medium probability)
   - **Impact**: Delays evaluation
   - **Mitigation**: Use smaller model for initial tests

4. **Likelihood computation too slow for inference** (Low probability)
   - **Impact**: Impractical for production
   - **Mitigation**: Optimize sampling, reduce timesteps

### Resource Risks

1. **GPU availability**
   - **Current**: nigel.birs.ca available
   - **Backup**: Use smaller model if needed

2. **Data quality**
   - **Current**: High-quality IMDB + synthetic
   - **Backup**: Can use real IMDB only if synthetic hurts

---

## Success Metrics

### Research Success:
- ✅ Prove discrete diffusion CAN compete with GPT-2 for generative classification
- ✅ Achieve >85% accuracy (within 5% of GPT-2's 90.1%)
- ✅ Demonstrate P(text) computation advantage over RoBERTa MLM

### Practical Success:
- ✅ Reasonable inference speed (<5 seconds per sample)
- ✅ Stable training and convergence
- ✅ Reproducible results

### Documentation Success:
- ✅ Clear comparison: GPT-2 vs RoBERTa MLM vs True Diffusion
- ✅ Architectural analysis explaining performance differences
- ✅ Actionable recommendations for future work

---

## Next Immediate Actions (Priority Order)

1. ⏸️ **Wait for RoBERTa + combined training** (16 hours remaining)
   - Will complete tomorrow evening
   - Provides final data point on MLM approach

2. 🔍 **Clone and examine MDLM** (1-2 hours)
   ```bash
   cd /Users/vincent/development/
   git clone https://github.com/kuleshov-group/mdlm.git
   cd mdlm
   # Examine for P(text) computation capability
   ```

3. 📊 **Test MDLM likelihood** (2-3 hours)
   - Load pre-trained model
   - Test if it can compute P("sample text")
   - Verify we can adapt for classification

4. ⚖️ **Make decision** (30 minutes)
   - If MDLM works: Proceed with MDLM-based classifier
   - If MDLM insufficient: Plan custom discrete diffusion

5. 🚀 **Begin implementation** (Day 2+)
   - Set up per-class training pipeline
   - Prepare data loaders for IMDB + synthetic
   - Start training first class model

---

## Open Questions

1. **Can MDLM compute P(text)?**
   - Need to examine codebase
   - Check for log_prob or perplexity evaluation methods

2. **What's the best discrete diffusion architecture for classification?**
   - MDLM seems fastest and most practical
   - But need likelihood computation for generative classification

3. **Should we use pre-trained MDLM or train from scratch?**
   - Pre-trained: Faster, but may not match IMDB domain
   - From scratch: Slower, but domain-specific

4. **What sampling strategy works best for classification?**
   - MDLM offers: ancestral (D3PM), analytic (SEDD), efficient
   - Need to test which gives best likelihood estimates

5. **How many diffusion timesteps do we need?**
   - More steps: Better quality, slower inference
   - Fewer steps: Faster, but may lose accuracy
   - Need to find sweet spot (100-1000 steps?)

---

## Resources

### Papers
- **MDLM**: https://arxiv.org/abs/2406.07524 (NeurIPS 2024)
- **SSD-LM**: https://arxiv.org/abs/2210.17432
- **D3PM**: https://arxiv.org/abs/2107.03006
- **Diffusion-LM**: https://arxiv.org/abs/2205.14217

### Code Repositories
- **MDLM**: https://github.com/kuleshov-group/mdlm
- **Awesome DLMs**: https://github.com/VILA-Lab/Awesome-DLMs

### Related Work
- Survey: "Diffusion models in text generation" (PMC)
- Blog: "Diffusion Language Models: The New Paradigm" (HuggingFace)

---

**Status**: Research phase complete, awaiting RoBERTa results and MDLM feasibility check

**Last Updated**: 2025-11-03 (Evening)

**Next Milestone**: MDLM feasibility decision (tomorrow)
