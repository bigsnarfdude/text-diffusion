"""
Generative Classifier using Text Diffusion

This module implements a generative classifier that trains separate diffusion models
per class and classifies by comparing likelihoods.

Key idea:
- Train one RoBERTa diffusion model per class (e.g., positive/negative sentiment)
- At inference: compute P(x|class) for each class using the diffusion model
- Classify using Bayes rule: argmax_c P(class|x) ∝ P(x|class) * P(class)
"""

from .data import ClassificationDataset, load_classification_data
from .trainer import PerClassTrainer
from .inference import GenerativeClassifier
from .metrics import compute_metrics, ClassificationMetrics

__all__ = [
    'ClassificationDataset',
    'load_classification_data',
    'PerClassTrainer',
    'GenerativeClassifier',
    'compute_metrics',
    'ClassificationMetrics',
]
