#!/usr/bin/env python3
"""
Evaluate Generative Classifier

Evaluates trained generative classifier on test set.

Usage:
    # Evaluate on full test set
    python src/evaluate_classifier.py --model-dir results-generative-classifier

    # Quick test on small subset
    python src/evaluate_classifier.py --model-dir results-generative-classifier --max-samples 50

    # Show detailed error analysis
    python src/evaluate_classifier.py --model-dir results-generative-classifier --analyze-errors
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.trainer import PerClassTrainer
from src.classifier.inference import GenerativeClassifier
from src.classifier.data import load_test_examples, load_metadata
from src.classifier.metrics import (
    compute_metrics,
    print_metrics,
    print_classification_report,
    analyze_errors
)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Generative Classifier'
    )

    # Model args
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

    # Evaluation args
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of test samples (for quick testing)'
    )
    parser.add_argument(
        '--num-likelihood-samples',
        type=int,
        default=5,
        help='Number of masking samples for likelihood estimation'
    )
    parser.add_argument(
        '--mask-prob',
        type=float,
        default=0.15,
        help='Masking probability for likelihood estimation'
    )

    # Analysis args
    parser.add_argument(
        '--analyze-errors',
        action='store_true',
        help='Show detailed error analysis'
    )
    parser.add_argument(
        '--num-error-examples',
        type=int,
        default=5,
        help='Number of error examples to show'
    )

    # Output args
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Save results to JSON file'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, auto-detected if not specified)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GENERATIVE CLASSIFIER EVALUATION")
    print(f"{'='*60}\n")

    # Load metadata
    metadata = load_metadata(args.data_dir)
    class_names = metadata['class_names']

    # Load trained models
    print(f"Loading models from {args.model_dir}...")
    trainer = PerClassTrainer(
        model_name=metadata.get('model_name', 'distilroberta-base'),
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        device=args.device
    )
    trainer.load_trained_models()

    # Create classifier
    print("Creating classifier...")
    classifier = GenerativeClassifier(
        class_models=trainer.class_models,
        tokenizer=trainer.tokenizer,
        class_names=class_names,
        device=trainer.device
    )

    # Load test data
    print(f"Loading test data...")
    texts, labels = load_test_examples(args.data_dir, max_samples=args.max_samples)
    print(f"Loaded {len(texts)} test examples\n")

    # Classify
    print("Classifying test examples...")
    print(f"  Likelihood samples per text: {args.num_likelihood_samples}")
    print(f"  Masking probability: {args.mask_prob}\n")

    predictions, probabilities = classifier.classify_batch(
        texts=texts,
        num_samples=args.num_likelihood_samples,
        mask_prob=args.mask_prob,
        show_progress=True
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels, class_names)

    # Print results
    print_metrics(metrics, class_names)
    print_classification_report(predictions, labels, class_names)

    # Error analysis
    if args.analyze_errors:
        analyze_errors(
            texts=texts,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            class_names=class_names,
            num_examples=args.num_error_examples
        )

    # Save results
    if args.save_results:
        results = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'per_class_metrics': metrics.per_class_metrics,
            'num_samples': metrics.num_samples,
            'config': {
                'model_dir': args.model_dir,
                'data_dir': args.data_dir,
                'num_likelihood_samples': args.num_likelihood_samples,
                'mask_prob': args.mask_prob,
            }
        }

        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {save_path}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
