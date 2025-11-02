#!/usr/bin/env python3
"""
Visualize Generative Classifier Results

Creates visualizations showing:
- Likelihood distributions per class
- Confidence vs correctness
- Confusion matrix heatmap

Usage:
    python tools/visualize_classification.py --model-dir results-generative-classifier
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.trainer import PerClassTrainer
from src.classifier.inference import GenerativeClassifier
from src.classifier.data import load_test_examples, load_metadata


def visualize_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    class_names: List[str],
    output_path: str
):
    """Create confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  ✅ Saved confusion matrix to {output_path}")


def visualize_confidence_distribution(
    probabilities: List[Dict[str, float]],
    labels: List[int],
    predictions: List[int],
    class_names: List[str],
    output_path: str
):
    """
    Visualize distribution of prediction confidence.
    Shows separate distributions for correct vs incorrect predictions.
    """
    # Extract max probabilities (confidence)
    confidences = []
    correctness = []

    for i, probs in enumerate(probabilities):
        max_prob = max(probs.values())
        confidences.append(max_prob)
        correctness.append(predictions[i] == labels[i])

    correct_conf = [c for c, corr in zip(confidences, correctness) if corr]
    incorrect_conf = [c for c, corr in zip(confidences, correctness) if not corr]

    plt.figure(figsize=(10, 6))

    plt.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
    plt.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')

    plt.xlabel('Prediction Confidence (Max Probability)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  ✅ Saved confidence distribution to {output_path}")


def visualize_likelihood_comparison(
    classifier: GenerativeClassifier,
    texts: List[str],
    labels: List[int],
    class_names: List[str],
    output_path: str,
    num_samples: int = 10
):
    """
    Visualize likelihood distributions for sample texts.
    Shows log P(text|class) for each class.
    """
    # Sample a few examples from each class
    examples_per_class = []
    for class_id in range(len(class_names)):
        class_indices = [i for i, l in enumerate(labels) if l == class_id]
        sampled = np.random.choice(class_indices, min(num_samples // 2, len(class_indices)), replace=False)
        examples_per_class.extend([(texts[i], labels[i]) for i in sampled])

    # Compute likelihoods
    fig, axes = plt.subplots(len(examples_per_class), 1, figsize=(10, 3 * len(examples_per_class)))

    if len(examples_per_class) == 1:
        axes = [axes]

    for idx, (text, true_label) in enumerate(examples_per_class):
        # Get explanation
        explanation = classifier.explain_prediction(text, num_samples=5)

        # Extract likelihoods
        class_lls = [ll_info['mean_log_likelihood'] for ll_info in explanation['log_likelihoods']]
        class_stds = [ll_info['std_log_likelihood'] for ll_info in explanation['log_likelihoods']]

        # Plot
        ax = axes[idx]
        x_pos = np.arange(len(class_names))

        colors = ['green' if i == true_label else 'blue' for i in range(len(class_names))]

        ax.bar(x_pos, class_lls, yerr=class_stds, color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Log Likelihood')
        ax.set_title(f'Example {idx+1} (True: {class_names[true_label]})\n"{text[:50]}..."', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  ✅ Saved likelihood comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Generative Classifier Results'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory with trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imdb-classifier',
        help='Directory with prepared data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results-classifier-viz',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=500,
        help='Max test samples to visualize'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATIVE CLASSIFIER VISUALIZATION")
    print(f"{'='*60}\n")

    # Load metadata
    metadata = load_metadata(args.data_dir)
    class_names = metadata['class_names']

    # Load trained models
    print(f"Loading models...")
    trainer = PerClassTrainer(
        model_name=metadata.get('model_name', 'distilroberta-base'),
        data_dir=args.data_dir,
        output_dir=args.model_dir
    )
    trainer.load_trained_models()

    # Create classifier
    classifier = GenerativeClassifier(
        class_models=trainer.class_models,
        tokenizer=trainer.tokenizer,
        class_names=class_names
    )

    # Load test data
    print(f"Loading test data...")
    texts, labels = load_test_examples(args.data_dir, max_samples=args.max_samples)
    print(f"Loaded {len(texts)} test examples\n")

    # Classify
    print("Classifying examples...")
    predictions, probabilities = classifier.classify_batch(texts, show_progress=True)

    print(f"\nCreating visualizations...\n")

    # 1. Confusion matrix
    visualize_confusion_matrix(
        predictions, labels, class_names,
        output_path / 'confusion_matrix.png'
    )

    # 2. Confidence distribution
    visualize_confidence_distribution(
        probabilities, labels, predictions, class_names,
        output_path / 'confidence_distribution.png'
    )

    # 3. Likelihood comparison (on subset)
    print("  Computing likelihood comparisons (this may take a minute)...")
    visualize_likelihood_comparison(
        classifier, texts, labels, class_names,
        output_path / 'likelihood_comparison.png',
        num_samples=min(10, len(texts))
    )

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Visualizations saved to: {args.output_dir}/")
    print(f"  confusion_matrix.png")
    print(f"  confidence_distribution.png")
    print(f"  likelihood_comparison.png")
    print()


if __name__ == '__main__':
    main()
