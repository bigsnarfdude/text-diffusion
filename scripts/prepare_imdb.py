#!/usr/bin/env python3
"""
Prepare IMDB dataset for generative classification.

Downloads IMDB sentiment dataset and organizes it for per-class training.

Output structure:
    data/imdb-classifier/
        metadata.json              # Dataset info
        train_class_0.json         # Negative training examples
        train_class_1.json         # Positive training examples
        test.json                  # Combined test set with labels

Usage:
    python scripts/prepare_imdb.py
    python scripts/prepare_imdb.py --output-dir data/my-classifier --max-samples 1000
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from datasets import load_dataset
from tqdm import tqdm


def prepare_imdb_data(
    output_dir: str = 'data/imdb-classifier',
    max_samples_per_class: int = None,
    max_test_samples: int = None
):
    """
    Download and prepare IMDB dataset.

    Args:
        output_dir: Where to save prepared data
        max_samples_per_class: Limit training samples per class
        max_test_samples: Limit test samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PREPARING IMDB DATASET")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir}")
    if max_samples_per_class:
        print(f"Max samples per class: {max_samples_per_class}")
    if max_test_samples:
        print(f"Max test samples: {max_test_samples}")
    print()

    # Load IMDB dataset
    print("Downloading IMDB dataset...")
    dataset = load_dataset('imdb')

    train_data = dataset['train']
    test_data = dataset['test']

    print(f"✅ Downloaded:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples\n")

    # Class names
    class_names = ['negative', 'positive']
    num_classes = 2

    # Prepare training data (split by class)
    print("Preparing training data...")

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        print(f"  Processing class {class_id} ({class_name})...")

        # Filter examples for this class
        class_examples = [
            ex['text'] for ex in tqdm(train_data, desc=f"    Filtering")
            if ex['label'] == class_id
        ]

        # Limit if requested
        if max_samples_per_class:
            class_examples = class_examples[:max_samples_per_class]

        # Save
        class_file = output_path / f'train_class_{class_id}.json'
        with open(class_file, 'w') as f:
            json.dump({
                'class_id': class_id,
                'class_name': class_name,
                'num_examples': len(class_examples),
                'texts': class_examples
            }, f, indent=2)

        print(f"    ✅ Saved {len(class_examples)} examples to {class_file}")

    # Prepare test data (keep labels for evaluation)
    print("\nPreparing test data...")

    test_texts = []
    test_labels = []

    for ex in tqdm(test_data, desc="  Processing"):
        test_texts.append(ex['text'])
        test_labels.append(ex['label'])

    # Limit if requested
    if max_test_samples:
        test_texts = test_texts[:max_test_samples]
        test_labels = test_labels[:max_test_samples]

    # Save
    test_file = output_path / 'test.json'
    with open(test_file, 'w') as f:
        json.dump({
            'num_examples': len(test_texts),
            'texts': test_texts,
            'labels': test_labels
        }, f, indent=2)

    print(f"  ✅ Saved {len(test_texts)} examples to {test_file}")

    # Save metadata
    print("\nSaving metadata...")

    metadata = {
        'dataset': 'imdb',
        'task': 'sentiment_classification',
        'num_classes': num_classes,
        'class_names': class_names,
        'splits': {
            'train': {
                class_name: len([ex for ex in train_data if ex['label'] == i])
                for i, class_name in enumerate(class_names)
            },
            'test': len(test_texts)
        }
    }

    if max_samples_per_class:
        metadata['max_samples_per_class'] = max_samples_per_class
    if max_test_samples:
        metadata['max_test_samples'] = max_test_samples

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Saved metadata to {metadata_file}")

    print(f"\n{'='*60}")
    print("✅ DATA PREPARATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Data saved to: {output_dir}/")
    print("\nFiles created:")
    print(f"  metadata.json          - Dataset information")
    print(f"  train_class_0.json     - Negative training examples")
    print(f"  train_class_1.json     - Positive training examples")
    print(f"  test.json              - Test set with labels")
    print("\nNext step:")
    print(f"  python src/train_generative_classifier.py --quick-test")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Prepare IMDB dataset for generative classification'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/imdb-classifier',
        help='Output directory for prepared data'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Max training samples per class (for testing)'
    )
    parser.add_argument(
        '--max-test',
        type=int,
        default=None,
        help='Max test samples (for testing)'
    )

    args = parser.parse_args()

    prepare_imdb_data(
        output_dir=args.output_dir,
        max_samples_per_class=args.max_samples,
        max_test_samples=args.max_test
    )


if __name__ == '__main__':
    main()
