# How to Fix the Generative Classifier

## Problem Diagnosis

Current results: 52.6% accuracy (random guessing)

**Root Cause:** Both class models learned nearly identical P(text) distributions because:
1. Positive and negative reviews share most vocabulary
2. MLM objective doesn't emphasize discriminative features
3. Models trained independently (no signal to diverge)
4. Insufficient training (3 epochs, 2000 samples)

## Proposed Fixes (Ranked by Likelihood of Success)

### Fix 1: MORE TRAINING + FULL DATA ⭐ (Try First)

**Hypothesis:** Models need more data/epochs to specialize

```bash
# Use full IMDB dataset
python scripts/prepare_imdb.py  # No --max-samples (use all 25k)

# Train for 10 epochs instead of 3
python src/train_generative_classifier.py \
  --epochs 10 \
  --batch-size 16

# Expected:
# - More data → better class-specific patterns
# - More epochs → models diverge more
```

**Expected Improvement:** 60-70% accuracy if this works

---

### Fix 2: BETTER LIKELIHOOD ESTIMATION ⭐⭐

**Hypothesis:** Current likelihood estimates are too noisy

**Changes needed in `src/classifier/inference.py`:**

```python
# Current: 5 samples, 15% masking
def compute_log_likelihood(text, class_id, num_samples=5, mask_prob=0.15):
    ...

# Improved: Multiple strategies averaged
def compute_log_likelihood_robust(text, class_id):
    likelihoods = []

    # Strategy 1: Low masking (10%)
    for _ in range(10):
        ll = compute_ll(text, class_id, mask_prob=0.10)
        likelihoods.append(ll)

    # Strategy 2: Medium masking (15%)
    for _ in range(10):
        ll = compute_ll(text, class_id, mask_prob=0.15)
        likelihoods.append(ll)

    # Strategy 3: High masking (20%)
    for _ in range(10):
        ll = compute_ll(text, class_id, mask_prob=0.20)
        likelihoods.append(ll)

    # Average 30 samples instead of 5
    return np.mean(likelihoods)
```

**Expected Improvement:** 55-60% accuracy

---

### Fix 3: CONTRASTIVE TRAINING ⭐⭐⭐ (Most Promising)

**Hypothesis:** Models need explicit signal to diverge

**New training approach in `src/classifier/trainer.py`:**

```python
# Instead of training models independently,
# train them jointly with contrastive loss

class ContrastivePerClassTrainer:
    def train_all_classes_contrastive(self):
        # Initialize both models
        model_0 = RobertaForMaskedLM(...)
        model_1 = RobertaForMaskedLM(...)

        for epoch in epochs:
            for batch in dataloader:
                # Get positive and negative samples
                pos_texts = batch['positive_texts']
                neg_texts = batch['negative_texts']

                # Standard MLM loss
                loss_0 = mlm_loss(model_0, neg_texts)
                loss_1 = mlm_loss(model_1, pos_texts)

                # CONTRASTIVE LOSS: Push models apart
                # Model 0 should give lower likelihood to positive texts
                # Model 1 should give lower likelihood to negative texts
                contrast_loss = (
                    log_likelihood(model_0, pos_texts) +  # Should be low
                    log_likelihood(model_1, neg_texts)    # Should be low
                )

                # Combined objective
                total_loss = loss_0 + loss_1 + 0.5 * contrast_loss
                total_loss.backward()
```

**Expected Improvement:** 70-80% accuracy

---

### Fix 4: FOCUS ON SENTIMENT TOKENS

**Hypothesis:** Most tokens are irrelevant, focus on sentiment words

```python
# Identify sentiment-bearing tokens
SENTIMENT_WORDS = [
    'good', 'bad', 'great', 'terrible', 'amazing', 'awful',
    'love', 'hate', 'best', 'worst', 'excellent', 'horrible',
    'wonderful', 'disappointing', 'fantastic', 'boring'
]

# Modified likelihood: only compute on sentiment tokens
def compute_sentiment_likelihood(text, class_id):
    tokens = tokenize(text)

    # Find sentiment token positions
    sentiment_positions = [
        i for i, tok in enumerate(tokens)
        if tok.lower() in SENTIMENT_WORDS
    ]

    # Mask only sentiment tokens
    masked_ids = mask_positions(tokens, sentiment_positions)

    # Compute likelihood only on sentiment tokens
    outputs = model(masked_ids)
    loss = outputs.loss  # Only computed on sentiment positions

    return -loss
```

**Expected Improvement:** 65-75% accuracy

---

### Fix 5: USE GPT-STYLE AUTOREGRESSIVE MODEL

**Hypothesis:** MLM likelihood estimation is fundamentally wrong

**Replace RoBERTa with GPT:**

```python
from transformers import GPT2LMHeadModel

# Train separate GPT models per class
model_negative = GPT2LMHeadModel.from_pretrained('gpt2')
model_positive = GPT2LMHeadModel.from_pretrained('gpt2')

# Compute TRUE likelihood (not estimated)
def compute_true_likelihood(text, model):
    tokens = tokenize(text)

    # Autoregressive likelihood
    log_prob = 0
    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]

        probs = model(context)
        log_prob += log(probs[target])

    return log_prob
```

**Expected Improvement:** 75-85% accuracy (but much slower)

---

### Fix 6: HYBRID APPROACH

**Hypothesis:** Combine generative + discriminative

```python
# Train generative models as before
generative_score = P(text|class_0) vs P(text|class_1)

# ALSO train a small discriminative classifier
discriminative_model = RobertaForSequenceClassification(...)
discriminative_score = discriminative_model(text)

# Ensemble
final_prediction = (
    0.5 * generative_score +
    0.5 * discriminative_score
)
```

**Expected Improvement:** 80-90% accuracy

---

## Recommended Sequence

### Phase 1: Quick Wins (1 hour)
1. ✅ Fix 1: Train with full data (25k samples) and 10 epochs
2. ✅ Fix 2: Better likelihood estimation (30 samples vs 5)

### Phase 2: If Phase 1 Fails (2-3 hours)
3. ✅ Fix 3: Implement contrastive training
4. ✅ Fix 4: Focus on sentiment tokens

### Phase 3: If Phase 2 Fails (4-6 hours)
5. ✅ Fix 5: Switch to GPT-style autoregressive
6. ✅ Fix 6: Hybrid approach

---

## Quick Test: Fix 1 (More Training)

```bash
# On nigel
cd ~/text-diffusion

# Prepare full dataset (no limits)
python scripts/prepare_imdb.py

# Train with full data, 10 epochs
python src/train_generative_classifier.py \
  --epochs 10 \
  --batch-size 16

# Evaluate with better likelihood estimation
python src/evaluate_classifier.py \
  --model-dir results-generative-classifier \
  --num-likelihood-samples 20
```

If this gets to 65-70% accuracy, we're on the right track. If still ~52%, need architectural changes (Fix 3+).

---

## Expected Outcomes

| Fix | Expected Accuracy | Effort | Likelihood |
|-----|------------------|---------|------------|
| More training (Fix 1) | 60-70% | Low | Medium |
| Better likelihood (Fix 2) | 55-60% | Low | Low |
| Contrastive training (Fix 3) | 70-80% | Medium | High |
| Sentiment tokens (Fix 4) | 65-75% | Medium | Medium |
| Autoregressive (Fix 5) | 75-85% | High | High |
| Hybrid (Fix 6) | 80-90% | High | Very High |

**Recommendation:** Try Fix 1 + Fix 2 first (low effort). If fails, implement Fix 3 (contrastive training).
