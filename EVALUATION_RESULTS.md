# GPT-2 Generative Classifier - Complete Evaluation Results

**Date:** November 3, 2025
**Status:** ✅ All experiments completed successfully

---

## Executive Summary

Three GPT-2 generative classifier models were trained and evaluated on IMDB sentiment classification:

1. **Generative (Baseline)**: 86.6% accuracy
2. **Full IMDB**: 90.1% accuracy ⭐ **BEST PERFORMER**
3. **Augmented (IMDB + Synthetic)**: 87.5% accuracy

**Key Finding:** The full IMDB training significantly outperformed both the baseline and augmented approaches. Surprisingly, adding synthetic data **decreased** performance by 2.6% compared to the full model.

---

## Detailed Results

### Overall Performance Comparison

| Model       | Accuracy | Precision | Recall  | F1 Score |
|-------------|----------|-----------|---------|----------|
| Generative  | 86.60%   | 86.73%    | 86.60%  | 86.59%   |
| Full        | **90.10%** | **90.15%** | **90.10%** | **90.10%** |
| Augmented   | 87.50%   | 87.81%    | 87.50%  | 87.47%   |

### Per-Class Performance

#### Negative Class

| Model       | Precision | Recall  | F1 Score |
|-------------|-----------|---------|----------|
| Generative  | 84.53%    | 89.60%  | 86.99%   |
| Full        | **88.78%** | **91.80%** | **90.27%** |
| Augmented   | 84.40%    | **92.00%** | 88.04%   |

#### Positive Class

| Model       | Precision | Recall  | F1 Score |
|-------------|-----------|---------|----------|
| Generative  | 88.94%    | 83.60%  | 86.19%   |
| Full        | **91.51%** | **88.40%** | **89.93%** |
| Augmented   | 91.21%    | 83.00%  | 86.91%   |

### Confusion Matrices

**Generative:**
```
                 Predicted Neg  Predicted Pos
  Actual Neg:         448             52
  Actual Pos:          82            418
```

**Full (Best):**
```
                 Predicted Neg  Predicted Pos
  Actual Neg:         459             41
  Actual Pos:          58            442
```

**Augmented:**
```
                 Predicted Neg  Predicted Pos
  Actual Neg:         460             40
  Actual Pos:          85            415
```

---

## Training Details

### Generative (Baseline)
- **Training Data**: ~500 examples per class (small subset)
- **Examples**: 500 negative + 500 positive
- **Training Time**: ~2 minutes
- **Purpose**: Quick baseline test

### Full IMDB
- **Training Data**: Complete IMDB dataset
- **Examples**: 17,345 negative + 17,446 positive
- **Epochs**: 6
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Final Loss**: ~1.27
- **FP16**: Enabled
- **Training Time**: ~30-40 minutes

### Augmented (IMDB + Synthetic)
- **Training Data**: IMDB + Synthetic generated data
- **Source Datasets**:
  - `data/imdb-classifier`
  - `data/synthetic-imdb`
- **Examples**: 17,345 negative + 17,446 positive (same as Full)
- **Epochs**: 6
- **Batch Size**: 8
- **Learning Rate**: 5e-5
- **Final Loss**: ~1.28
- **FP16**: Enabled
- **Synthetic Data**: Generated using Qwen3 model (see logs/)

---

## Key Insights

### 1. Full IMDB Training Wins
✅ **Best Overall Model: Full (90.10% accuracy)**

The full IMDB training achieved the highest performance across all metrics:
- +3.5% improvement over Generative baseline
- +2.6% better than Augmented approach
- Most balanced performance across both classes

### 2. Synthetic Data Did NOT Help
❌ **Augmented model underperformed vs. Full model**

Performance deltas:
- Generative → Full: **+3.50%** (86.60% → 90.10%)
- Full → Augmented: **-2.60%** (90.10% → 87.50%)
- Generative → Augmented: +0.90% (86.60% → 87.50%)

**Possible reasons for synthetic data degradation:**
- Quality of synthetic data may be lower than real IMDB reviews
- Distribution mismatch between synthetic and real data
- Model may have learned artifacts from synthetic generation
- Need to investigate synthetic data quality (see `data/synthetic-imdb-cleaned/`)

### 3. Class Balance Analysis
All models show good class balance:

| Model       | Negative F1 | Positive F1 | Difference |
|-------------|-------------|-------------|------------|
| Generative  | 87.0%       | 86.2%       | 0.8%       |
| Full        | 90.3%       | 89.9%       | 0.3% ✅     |
| Augmented   | 88.0%       | 86.9%       | 1.1%       |

**Full model has the best class balance** with only 0.3% difference between classes.

### 4. Negative Class Benefits from Augmentation
Interestingly, the augmented model achieved:
- **Best negative recall: 92.0%** (vs 91.8% for Full)
- But sacrificed positive class recall (83.0% vs 88.4%)

This suggests synthetic data may have introduced bias toward negative predictions.

---

## Error Analysis

### Full Model (Best) Errors:
- **False Negatives**: 58 (11.6%)
- **False Positives**: 41 (8.2%)
- Total errors: 99/1000 (9.9%)

### Augmented Model Errors:
- **False Negatives**: 85 (17.0%) ⚠️ 27 more than Full
- **False Positives**: 40 (8.0%)
- Total errors: 125/1000 (12.5%)

**The augmented model struggles more with false negatives**, suggesting it's overly conservative in predicting positive sentiment.

---

## Recommendations

### 1. Deploy Full Model (90.1% accuracy)
The full IMDB model should be the production choice:
- Highest overall accuracy
- Best class balance
- No complexity from synthetic data

### 2. Investigate Synthetic Data Quality
Before attempting more augmentation:
- ✅ Review `data/synthetic-imdb-cleaned/` contents
- ✅ Check generation logs in `logs/synthetic-*.log`
- ✅ Compare synthetic vs. real review distributions
- ✅ Consider using higher-quality generation models

### 3. Alternative Augmentation Strategies
If augmentation is still desired:
- Try paraphrasing real reviews instead of full generation
- Use back-translation for data augmentation
- Filter synthetic data by quality metrics
- Mix smaller proportions of synthetic data (e.g., 10-20%)

### 4. Further Improvements
- Experiment with larger base models (GPT2-medium, GPT2-large)
- Try different training schedules (more epochs, learning rate tuning)
- Implement ensemble methods across multiple training runs
- Fine-tune on domain-specific data if available

---

## Files and Locations

### Result Directories
- `results-gpt2-generative/` - Baseline model (86.6%)
- `results-gpt2-full/` - Best model (90.1%)
- `results-gpt2-augmented/` - Augmented model (87.5%)

### Evaluation Results
- `results-gpt2-generative/eval_results.json`
- `results-gpt2-full/eval_results.json`
- `results-gpt2-augmented/eval_results.json`

### Training Logs
- `training.log` - Quick test runs
- `training_full.log` - Full training runs
- `logs/synthetic-*.log` - Synthetic data generation logs

### Data Directories
- `data/imdb-classifier/` - Original IMDB data
- `data/synthetic-imdb/` - Generated synthetic reviews
- `data/synthetic-imdb-cleaned/` - Cleaned synthetic data
- `data/imdb-augmented/` - Merged IMDB + synthetic

---

## Conclusion

The **Full IMDB model (90.1% accuracy)** is the clear winner. Adding synthetic data unexpectedly degraded performance by 2.6%, suggesting that:

1. Real data quality is superior to generated data
2. The full IMDB dataset is already sufficient
3. Synthetic data augmentation needs more careful curation

**Next Steps:** Deploy the full model and investigate why synthetic augmentation failed to improve results.
