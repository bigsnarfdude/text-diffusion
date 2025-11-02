#!/usr/bin/env python3
"""
Train Generative Classifier

Trains separate diffusion models for each class, which can then be used
for classification by comparing likelihoods.

Usage:
    # Quick test
    python src/train_generative_classifier.py --quick-test

    # Full training
    python src/train_generative_classifier.py --epochs 3 --batch-size 8

    # Train single class for debugging
    python src/train_generative_classifier.py --class-id 0 --epochs 1
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.trainer import PerClassTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train Generative Classifier using Text Diffusion'
    )

    # Model args
    parser.add_argument(
        '--model',
        type=str,
        default='distilroberta-base',
        help='Base model (distilroberta-base, roberta-base, roberta-large)'
    )

    # Data args
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imdb-classifier',
        help='Directory with prepared data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results-generative-classifier',
        help='Output directory for trained models'
    )

    # Training args
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Training epochs per class'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )

    # Testing/debugging args
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with small subset (100 samples per class, 1 epoch)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max samples per class (for testing)'
    )
    parser.add_argument(
        '--class-id',
        type=int,
        default=None,
        help='Train only a single class (for debugging)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu, auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = PerClassTrainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )

    # Train
    if args.class_id is not None:
        # Train single class (for debugging)
        print(f"\n🔍 DEBUG MODE: Training single class {args.class_id}\n")
        trainer.train_single_class(
            class_id=args.class_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples
        )
    else:
        # Train all classes
        trainer.train_all_classes(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples_per_class=args.max_samples,
            quick_test=args.quick_test
        )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModels saved to: {args.output_dir}/")
    print("\nNext steps:")
    print(f"  1. Evaluate: python src/evaluate_classifier.py --model-dir {args.output_dir}")
    print(f"  2. Visualize: python tools/visualize_classification.py")
    print()


if __name__ == '__main__':
    main()
