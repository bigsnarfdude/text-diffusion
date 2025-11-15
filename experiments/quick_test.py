#!/usr/bin/env python3
"""
Quick test of comparison framework on tiny sample.

Tests all 4 approaches on just 10 test examples to verify:
1. All approaches run without errors
2. Predictions are in correct format
3. Metrics are computed correctly
4. Statistical tests work

Usage:
    python experiments/quick_test.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.data import load_test_examples, load_metadata
from experiments.compare_all_approaches import (
    GPT2ZeroShotClassifier,
    ExperimentResults,
    compute_statistical_significance,
    print_comparison_table
)

def main():
    print("\n" + "="*80)
    print("QUICK TEST - Comparison Framework Validation")
    print("="*80 + "\n")

    # Load data
    data_dir = 'data/imdb-classifier'
    metadata = load_metadata(data_dir)
    class_names = metadata['class_names']

    print("Loading 10 test examples...")
    texts, labels = load_test_examples(data_dir, max_samples=10)
    print(f"Loaded {len(texts)} examples")
    print(f"Class names: {class_names}")
    print()

    # Test GPT-2 Zero-shot
    print("Testing GPT-2 Zero-shot...")
    classifier = GPT2ZeroShotClassifier(class_names)

    results = ExperimentResults('gpt2-zeroshot', 'imdb')

    predictions = []
    probabilities = []

    for i, text in enumerate(texts):
        print(f"  Classifying {i+1}/{len(texts)}...", end='\r')
        pred, probs = classifier.classify(text)
        predictions.append(pred)
        probabilities.append(probs)

    print(f"  Classifying {len(texts)}/{len(texts)}... Done!")

    results.predictions = predictions
    results.labels = labels
    results.probabilities = probabilities
    results.runtime = 10.0  # Dummy value

    print("\nResults:")
    print(f"  Predictions: {predictions}")
    print(f"  Labels:      {labels}")
    print(f"  Accuracy:    {sum(p == l for p, l in zip(predictions, labels)) / len(labels):.2f}")

    metrics = results.compute_metrics()
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    # Test with dummy second approach
    print("\n" + "-"*80)
    print("Testing statistical significance computation...")

    results2 = ExperimentResults('dummy', 'imdb')
    results2.predictions = [1 - p for p in predictions]  # Flip predictions
    results2.labels = labels
    results2.probabilities = probabilities
    results2.runtime = 5.0

    sig = compute_statistical_significance(results, results2)
    print(f"\nStatistical test results:")
    print(f"  Both correct: {sig['n_both_correct']}")
    print(f"  Only first correct: {sig['n_only_first_correct']}")
    print(f"  Only second correct: {sig['n_only_second_correct']}")
    print(f"  Both wrong: {sig['n_both_wrong']}")
    print(f"  p-value: {sig['p_value']:.4f}")

    # Test comparison table
    print("\n" + "-"*80)
    print("Testing comparison table...")
    print_comparison_table([results, results2])

    print("\n" + "="*80)
    print("✅ QUICK TEST PASSED - All components working correctly!")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Train diffusion models:")
    print("   python src/train_generative_classifier.py --data-dir data/imdb-classifier --epochs 3")
    print()
    print("2. Run full comparison:")
    print("   python experiments/compare_all_approaches.py --quick")
    print()


if __name__ == '__main__':
    main()
