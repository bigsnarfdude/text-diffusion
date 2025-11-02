"""
Evaluation metrics for generative classifier.
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    num_samples: int


def compute_metrics(
    predictions: List[int],
    labels: List[int],
    class_names: List[str]
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: Predicted class IDs
        labels: True class IDs
        class_names: Names of classes

    Returns:
        ClassificationMetrics object
    """
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    # Weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Per-class breakdown
    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    metrics = ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision_avg),
        recall=float(recall_avg),
        f1=float(f1_avg),
        confusion_matrix=cm,
        per_class_metrics=per_class,
        num_samples=len(labels)
    )

    return metrics


def print_metrics(metrics: ClassificationMetrics, class_names: List[str]):
    """
    Pretty print classification metrics.

    Args:
        metrics: ClassificationMetrics object
        class_names: Names of classes
    """
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}\n")

    print(f"Samples: {metrics.num_samples}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision (weighted): {metrics.precision:.4f}")
    print(f"Recall (weighted): {metrics.recall:.4f}")
    print(f"F1 Score (weighted): {metrics.f1:.4f}\n")

    print("Per-Class Metrics:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for class_name in class_names:
        m = metrics.per_class_metrics[class_name]
        print(f"{class_name:<15} "
              f"{m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} "
              f"{m['support']:>10}")

    print(f"\nConfusion Matrix:")
    print(f"{'':>15}", end='')
    for name in class_names:
        print(f"{name:>12}", end='')
    print()

    for i, name in enumerate(class_names):
        print(f"{name:>15}", end='')
        for j in range(len(class_names)):
            print(f"{metrics.confusion_matrix[i, j]:>12}", end='')
        print()

    print()


def print_classification_report(
    predictions: List[int],
    labels: List[int],
    class_names: List[str]
):
    """
    Print sklearn classification report.

    Args:
        predictions: Predicted class IDs
        labels: True class IDs
        class_names: Names of classes
    """
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*60}\n")

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )
    print(report)


def analyze_errors(
    texts: List[str],
    predictions: List[int],
    labels: List[int],
    probabilities: List[Dict[str, float]],
    class_names: List[str],
    num_examples: int = 5
):
    """
    Analyze classification errors.

    Args:
        texts: Original texts
        predictions: Predicted class IDs
        labels: True class IDs
        probabilities: Prediction probabilities
        class_names: Names of classes
        num_examples: Number of error examples to show
    """
    # Find errors
    errors = [
        (i, texts[i], labels[i], predictions[i], probabilities[i])
        for i in range(len(texts))
        if labels[i] != predictions[i]
    ]

    if not errors:
        print("\n✅ No classification errors!\n")
        return

    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS ({len(errors)} errors)")
    print(f"{'='*60}\n")

    # Show examples
    for i, (idx, text, true_label, pred_label, probs) in enumerate(errors[:num_examples]):
        print(f"Example {i+1}/{min(num_examples, len(errors))}:")
        print(f"  Text: {text[:150]}...")
        print(f"  True: {class_names[true_label]}")
        print(f"  Predicted: {class_names[pred_label]}")
        print(f"  Probabilities:")
        for class_name, prob in probs.items():
            marker = " ✓" if class_name == class_names[true_label] else ""
            print(f"    {class_name}: {prob:.4f}{marker}")
        print()

    # Error breakdown by true class
    print("Errors by True Class:")
    for class_id, class_name in enumerate(class_names):
        class_errors = [e for e in errors if e[2] == class_id]
        total_class = sum(1 for l in labels if l == class_id)
        error_rate = len(class_errors) / total_class if total_class > 0 else 0
        print(f"  {class_name}: {len(class_errors)}/{total_class} "
              f"({error_rate:.2%} error rate)")

    print()
