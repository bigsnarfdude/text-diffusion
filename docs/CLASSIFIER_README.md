# Generative Text Classifier using Diffusion Models

A novel approach to text classification using **generative modeling** instead of discriminative classifiers. We train separate diffusion models for each class and classify by comparing likelihoods.

## 🎯 Key Idea

**Traditional Classifier:**
- Train one model: `P(class|text)` directly
- Uses cross-entropy loss on class labels
- Discriminative approach

**Generative Classifier (This Implementation):**
- Train N models: `P(text|class_i)` for each class
- Classify using Bayes rule: `argmax_c P(class|text) ∝ P(text|class) * P(class)`
- Generative approach using diffusion models

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Download and prepare IMDB dataset
python scripts/prepare_imdb.py

# Output: data/imdb-classifier/
#   - train_class_0.json (negative examples)
#   - train_class_1.json (positive examples)
#   - test.json (test set with labels)
#   - metadata.json (dataset info)
```

### 2. Train Classifier

```bash
# Quick test (100 samples per class, 1 epoch) - 5 minutes
python src/train_generative_classifier.py --quick-test

# Full training (all data, 3 epochs) - 2-3 hours
python src/train_generative_classifier.py --epochs 3 --batch-size 8

# Train single class for debugging
python src/train_generative_classifier.py --class-id 0 --epochs 1
```

### 3. Evaluate

```bash
# Evaluate on full test set
python src/evaluate_classifier.py --model-dir results-generative-classifier

# Quick test on subset
python src/evaluate_classifier.py \
  --model-dir results-generative-classifier \
  --max-samples 100

# Detailed error analysis
python src/evaluate_classifier.py \
  --model-dir results-generative-classifier \
  --analyze-errors
```

### 4. Visualize

```bash
# Create visualizations
python tools/visualize_classification.py \
  --model-dir results-generative-classifier \
  --output-dir results-classifier-viz

# Outputs:
#   - confusion_matrix.png
#   - confidence_distribution.png
#   - likelihood_comparison.png
```

## 📊 How It Works

### Training Phase

For each class (e.g., positive/negative sentiment):

1. **Collect class-specific data**: All examples from that class
2. **Train diffusion model**: Using variable masking (10%-100%)
3. **Save model**: Separate model checkpoint for each class

```python
# Train on positive examples only
texts_positive = ["Great movie!", "Loved it!", ...]
model_positive = train_diffusion(texts_positive)

# Train on negative examples only
texts_negative = ["Terrible film.", "Waste of time.", ...]
model_negative = train_diffusion(texts_negative)
```

### Classification Phase

For each new text:

1. **Compute likelihoods**: `P(text|class)` using each class's model
2. **Apply Bayes rule**: `P(class|text) ∝ P(text|class) * P(class)`
3. **Predict**: `argmax_c P(class_c|text)`

```python
# New text
text = "This movie was amazing!"

# Compute log likelihoods
log_p_positive = model_positive.likelihood(text)  # e.g., -2.3
log_p_negative = model_negative.likelihood(text)  # e.g., -5.7

# Higher likelihood → predicted class
predicted = "positive"  # -2.3 > -5.7
```

### Likelihood Estimation

We estimate `P(text|class)` by:

1. Randomly mask 15% of tokens
2. Compute model's prediction loss on masked tokens
3. Return negative loss as log likelihood estimate
4. Average over multiple masking samples for robustness

## 🏗️ Architecture

```
src/classifier/
├── __init__.py          # Module exports
├── data.py              # Dataset loading and preparation
├── trainer.py           # Per-class model training
├── inference.py         # Likelihood-based classification
└── metrics.py           # Evaluation metrics

src/
├── train_generative_classifier.py   # Main training script
└── evaluate_classifier.py           # Evaluation script

tools/
└── visualize_classification.py      # Visualization tool

scripts/
└── prepare_imdb.py                  # IMDB data preparation
```

## 📈 Expected Results

### Quick Test (--quick-test)
- **Training time**: 5-10 minutes
- **Accuracy**: ~60-70% (limited data)
- **Purpose**: Verify everything works

### Full Training (3 epochs)
- **Training time**: 2-3 hours (CPU)
- **Accuracy**: ~75-85% (depends on model size)
- **Purpose**: Real classification performance

### Comparison to Discriminative Classifier
- **RoBERTa Fine-tuning**: ~90-95% accuracy
- **Generative Classifier**: ~75-85% accuracy
- **Trade-off**: Lower accuracy, but **explainable** (can see likelihoods)

## 🎓 Why This Approach?

### Advantages

1. **Interpretability**
   - Can see `P(text|class)` for each class
   - Understand why model chose a particular class
   - Useful for debugging and trust

2. **Uncertainty Quantification**
   - Get probability distribution over classes
   - Know when model is uncertain
   - Variance from multiple likelihood samples

3. **No Class Imbalance Issues**
   - Each class has its own model
   - No need to balance training data
   - Models don't compete during training

4. **Extensible**
   - Easy to add new classes: just train new model
   - No need to retrain everything
   - Can use different model sizes per class

### Disadvantages

1. **Lower Accuracy**
   - Typically 5-15% lower than discriminative classifiers
   - Not state-of-the-art for pure performance

2. **Slower Inference**
   - Must run N forward passes (one per class)
   - Multiple masking samples per class
   - ~10x slower than single classifier

3. **More Training Time**
   - Train N models instead of 1
   - Total training time = N × single model time

4. **More Storage**
   - Store N models instead of 1
   - Each model ~500MB for RoBERTa-base

## 🔬 Experiments to Try

### Different Model Sizes
```bash
# Faster training, lower quality
python src/train_generative_classifier.py --model distilroberta-base

# Better quality, slower
python src/train_generative_classifier.py --model roberta-large
```

### Different Likelihood Estimation
```bash
# More likelihood samples = more robust, slower
python src/evaluate_classifier.py \
  --model-dir results-generative-classifier \
  --num-likelihood-samples 10

# Different masking probability
python src/evaluate_classifier.py \
  --model-dir results-generative-classifier \
  --mask-prob 0.20  # Mask 20% instead of 15%
```

### Class Priors
Edit `src/classifier/inference.py` to set non-uniform priors:
```python
classifier = GenerativeClassifier(
    ...,
    class_priors=[0.3, 0.7]  # e.g., if 70% of real-world data is positive
)
```

## 📚 Implementation Details

### Variable Masking for Diffusion

Key insight: Train with **variable masking rates** (10%-100%) instead of fixed 15%:

```python
# Standard BERT/RoBERTa
mask_prob = 0.15  # Always 15%

# Diffusion (our approach)
mask_prob = random.choice([0.1, 0.2, 0.3, ..., 1.0])
```

This teaches the model to denoise at **all corruption levels**, which gives better likelihood estimates.

### Likelihood Computation

```python
def compute_log_likelihood(text, class_model):
    # 1. Tokenize
    input_ids = tokenizer(text)

    # 2. Randomly mask 15% of tokens
    masked_ids, labels = apply_masking(input_ids, mask_prob=0.15)

    # 3. Forward pass
    outputs = class_model(masked_ids)
    loss = cross_entropy(outputs, labels)

    # 4. Negative loss = log likelihood estimate
    return -loss
```

### Multiple Samples

We average over multiple masking samples for robustness:

```python
log_likelihoods = []
for _ in range(num_samples):  # e.g., 5 samples
    ll = compute_log_likelihood(text, class_model)
    log_likelihoods.append(ll)

mean_ll = np.mean(log_likelihoods)
```

## 🐛 Troubleshooting

### "Data file not found"
```bash
# Make sure you prepared the data first
python scripts/prepare_imdb.py
```

### "Model not found"
```bash
# Train models before evaluating
python src/train_generative_classifier.py --quick-test
```

### "CUDA out of memory"
```bash
# Reduce batch size
python src/train_generative_classifier.py --batch-size 4

# Or force CPU
export CUDA_VISIBLE_DEVICES=""
```

### Low Accuracy
- Train longer: `--epochs 5` or `--epochs 10`
- Use larger model: `--model roberta-base` or `--model roberta-large`
- Use more likelihood samples during evaluation: `--num-likelihood-samples 10`

## 📖 References

### Generative Classification
- **"Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression"** (Ng & Jordan, 2002)
- Explains fundamental difference between generative and discriminative approaches

### Text Diffusion
- **"Structured Denoising Diffusion Models in Discrete State-Spaces"** (Austin et al., 2021)
- D3PM paper - discrete diffusion for text

### This Implementation
- Based on RoBERTa diffusion from: https://nathan.rs/posts/roberta-diffusion/
- Extended with classification capabilities

## 💡 Future Improvements

1. **Parallel Inference**: Batch process all classes simultaneously
2. **Confidence Calibration**: Calibrate probability outputs
3. **Active Learning**: Use uncertainty to select informative examples
4. **Multi-class Extensions**: Test on datasets with >2 classes
5. **Hybrid Approach**: Combine with discriminative classifier

## ✅ Next Steps

After training and evaluating:

1. **Analyze errors**: Use `--analyze-errors` flag
2. **Try different settings**: Experiment with hyperparameters
3. **Compare to baseline**: Train standard RoBERTa classifier
4. **Visualize results**: Use visualization tools
5. **Test on your data**: Adapt data preparation script

---

**Happy Classifying!** 🎉
