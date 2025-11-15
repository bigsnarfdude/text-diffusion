# Comprehensive Classification Experiments

This directory contains rigorous experiments comparing different text classification approaches.

## Motivation

We previously claimed success with a diffusion-based generative classifier, but this was **naive** without proper baselines. We need to compare against:

1. **Baseline GPT-2 (Zero-shot)**: Can pretrained GPT-2 classify without any training, just using perplexity?
2. **Fine-tuned GPT-2 (Native)**: How well does standard discriminative fine-tuning work?
3. **Diffusion Baseline (Untrained)**: Does our approach work with no training?
4. **Diffusion Trained**: Does per-class training actually help?

## Experimental Design

### Approaches

#### 1. GPT-2 Zero-Shot (No Training)

**Method**: Prompt-based classification using perplexity
- For each class c and text x:
  - Compute perplexity of: "This is a [class] review: [text]"
  - Lower perplexity = better match
- Classify: argmin_c perplexity(x | prompt_c)

**Strengths**:
- No training required
- Uses strong pretrained language model
- Simple and interpretable

**Weaknesses**:
- Sensitive to prompt engineering
- May not adapt well to domain-specific tasks

#### 2. GPT-2 Native Classifier (Fine-tuned)

**Method**: Standard discriminative fine-tuning
- Add classification head to GPT-2
- Fine-tune on (text, label) pairs
- Standard cross-entropy loss

**Strengths**:
- Well-established baseline
- Direct optimization of classification objective
- Fast at inference time

**Weaknesses**:
- Requires labeled training data
- May overfit on small datasets
- Ignores generative capabilities

#### 3. Diffusion Baseline (Untrained)

**Method**: Likelihood-based classification with pretrained models
- For each class c, use pretrained RoBERTa MLM
- Compute P(x | class_c) via masked token prediction
- Classify: argmax_c P(x | class_c)

**Strengths**:
- No training required
- Tests core hypothesis: can pretrained MLMs discriminate by likelihood?

**Weaknesses**:
- May have poor discrimination without adaptation
- Same model used for all classes

#### 4. Diffusion Trained (Our Approach)

**Method**: Per-class fine-tuning + likelihood-based classification
- Train separate RoBERTa MLM for each class
- Each model learns P(x | class_c) from class-specific data
- Classify: argmax_c P(x | class_c) using trained models

**Strengths**:
- Each model specializes on class-specific distribution
- Generative models can capture complex patterns
- Theoretically sound (Bayes rule)

**Weaknesses**:
- Requires training N models (where N = number of classes)
- Computationally expensive at inference
- More hyperparameters to tune

### Evaluation Metrics

For each approach, we measure:

1. **Accuracy**: Overall classification accuracy
2. **Precision/Recall/F1**: Per-class and macro-averaged
3. **Runtime**: Total time for test set classification
4. **Statistical Significance**: McNemar's test for pairwise comparisons

### Statistical Testing

We use **McNemar's test** for comparing paired predictions:
- Null hypothesis: Two approaches have same error rate
- Test statistic: (|n_01 - n_10| - 1)^2 / (n_01 + n_10)
  - n_01 = approach 1 wrong, approach 2 correct
  - n_10 = approach 1 correct, approach 2 wrong
- Report p-values and significance at α = 0.05

## Usage

### Setup

1. Prepare data:
```bash
python scripts/prepare_imdb.py
```

2. Train diffusion models (for approach 4):
```bash
python src/train_generative_classifier.py \
    --data-dir data/imdb-classifier \
    --output-dir results-generative-classifier \
    --epochs 3 \
    --mask-prob 0.15
```

### Run Experiments

#### Quick test (100 samples):
```bash
python experiments/compare_all_approaches.py \
    --dataset imdb \
    --data-dir data/imdb-classifier \
    --output results/comparison \
    --quick
```

#### Full comparison (all approaches):
```bash
python experiments/compare_all_approaches.py \
    --dataset imdb \
    --data-dir data/imdb-classifier \
    --diffusion-model-dir results-generative-classifier \
    --output results/comparison
```

#### Specific approaches only:
```bash
# Compare only GPT-2 approaches
python experiments/compare_all_approaches.py \
    --approaches gpt2-zeroshot gpt2-native \
    --dataset imdb

# Compare only diffusion approaches
python experiments/compare_all_approaches.py \
    --approaches diffusion-baseline diffusion-trained \
    --dataset imdb
```

### Expected Output

The script will output:

1. **Per-approach results**: Accuracy, precision, recall, F1, runtime
2. **Comparison table**: Side-by-side metrics
3. **Statistical significance**: Pairwise McNemar's test results
4. **JSON results**: Detailed results saved to `results/comparison/`

Example output:
```
================================================================================
COMPARISON TABLE
================================================================================

Approach                          Accuracy  Precision     Recall         F1   Time (s)
--------------------------------------------------------------------------------
gpt2-zeroshot                       0.7234     0.7189     0.7234     0.7211     145.23
gpt2-native                         0.8912     0.8923     0.8912     0.8917      23.45
diffusion-baseline                  0.5123     0.5234     0.5123     0.5178     312.67
diffusion-trained                   0.9145     0.9167     0.9145     0.9156     298.34

================================================================================
PAIRWISE STATISTICAL SIGNIFICANCE (McNemar's Test)
================================================================================

gpt2-zeroshot vs gpt2-native:
  Both correct: 6834
  gpt2-zeroshot only: 512
  gpt2-native only: 1678
  Both wrong: 976
  p-value: 0.0000 ***

gpt2-native vs diffusion-trained:
  Both correct: 8234
  gpt2-native only: 678
  diffusion-trained only: 911
  Both wrong: 177
  p-value: 0.0234 ***
```

## Key Questions to Answer

1. **Does zero-shot GPT-2 work?**
   - Can we classify without any training using perplexity?
   - How far behind is this from fine-tuned approaches?

2. **Is standard fine-tuning sufficient?**
   - Does GPT-2 native classifier outperform everything?
   - When is the added complexity of generative classification worth it?

3. **Does training help diffusion?**
   - Is there a significant gap between diffusion-baseline and diffusion-trained?
   - This tests our core hypothesis: per-class training improves likelihood discrimination

4. **Best overall approach?**
   - Which approach gives best accuracy?
   - Which approach gives best accuracy/compute trade-off?

## Hypotheses

### H1: GPT-2 zero-shot beats random but loses to fine-tuned
- **Prediction**: Accuracy > 0.6 for binary classification (vs 0.5 random)
- **Rationale**: Pretrained models understand sentiment, but not optimized for classification

### H2: GPT-2 native is strong baseline
- **Prediction**: Accuracy > 0.85 on IMDB
- **Rationale**: Well-established approach with direct optimization

### H3: Diffusion baseline performs poorly
- **Prediction**: Accuracy ≈ random (0.5-0.6)
- **Rationale**: Same model for all classes can't discriminate well

### H4: Diffusion trained beats diffusion baseline significantly
- **Prediction**: Accuracy gap > 0.2, p < 0.05
- **Rationale**: This is our core claim - per-class training enables discrimination

### H5: Diffusion trained competitive with GPT-2 native
- **Prediction**: Accuracy within 0.05 of GPT-2 native
- **Rationale**: If true, validates generative approach as viable alternative

## Analysis Plan

After running experiments:

1. **Verify hypotheses**: Check if predictions match results
2. **Error analysis**: What types of examples does each approach fail on?
3. **Efficiency analysis**: Accuracy vs. compute trade-offs
4. **Confidence analysis**: Are probability estimates well-calibrated?

## Notes

- All experiments use **identical test sets** for fair comparison
- Statistical tests account for **paired predictions** (same examples)
- Runtime measurements include **inference only** (not training time)
- Results saved with **full configuration** for reproducibility
