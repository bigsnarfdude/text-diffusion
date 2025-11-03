#!/usr/bin/env python3
"""
Merge Multiple Classification Datasets

Combines multiple prepared classification datasets (real + synthetic) into one.

Usage:
    # Merge IMDB + synthetic
    python scripts/merge_datasets.py \
        --datasets data/imdb-classifier data/synthetic-amazon \
        --output data/imdb-augmented

    # Merge multiple datasets
    python scripts/merge_datasets.py \
        --datasets data/imdb-classifier data/synthetic-amazon data/amazon-classifier \
        --output data/mega-dataset
"""

import argparse
import json
from pathlib import Path
from typing import List


def merge_datasets(
    dataset_dirs: List[str],
    output_dir: str
):
    """
    Merge multiple classification datasets.

    Args:
        dataset_dirs: List of dataset directories to merge
        output_dir: Output directory for merged dataset
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MERGING CLASSIFICATION DATASETS")
    print(f"{'='*60}\n")
    print(f"Input datasets: {len(dataset_dirs)}")
    for i, d in enumerate(dataset_dirs, 1):
        print(f"  {i}. {d}")
    print(f"Output directory: {output_dir}\n")

    # Load first dataset to get structure
    first_dataset = Path(dataset_dirs[0])
    with open(first_dataset / 'metadata.json') as f:
        base_metadata = json.load(f)

    num_classes = base_metadata['num_classes']
    class_names = base_metadata['class_names']

    print(f"Dataset structure:")
    print(f"  Classes: {num_classes} ({class_names})\n")

    # Merge each class
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        print(f"Merging class {class_id} ({class_name})...")

        all_texts = []
        dataset_counts = []

        # Load from each dataset
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir)
            class_file = dataset_path / f'train_class_{class_id}.json'

            if class_file.exists():
                with open(class_file) as f:
                    data = json.load(f)
                    texts = data['texts']
                    all_texts.extend(texts)
                    dataset_counts.append({
                        'source': dataset_dir,
                        'count': len(texts)
                    })
                    print(f"  + {len(texts):,} from {dataset_dir}")
            else:
                print(f"  ! Missing {class_file}")

        # Save merged class
        output_file = output_path / f'train_class_{class_id}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'class_id': class_id,
                'class_name': class_name,
                'num_examples': len(all_texts),
                'texts': all_texts,
                'sources': dataset_counts
            }, f, indent=2)

        print(f"  ✅ Total: {len(all_texts):,} examples saved to {output_file}\n")

    # Create merged metadata
    metadata = {
        'dataset': 'merged',
        'task': base_metadata.get('task', 'classification'),
        'num_classes': num_classes,
        'class_names': class_names,
        'source_datasets': dataset_dirs,
        'splits': {
            'train': {}
        }
    }

    # Count total per class
    for class_id in range(num_classes):
        class_file = output_path / f'train_class_{class_id}.json'
        with open(class_file) as f:
            data = json.load(f)
            metadata['splits']['train'][class_names[class_id]] = data['num_examples']

    # Save metadata
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {metadata_file}\n")

    # Copy test set from first dataset if available
    test_file = Path(dataset_dirs[0]) / 'test.json'
    if test_file.exists():
        import shutil
        output_test = output_path / 'test.json'
        shutil.copy(test_file, output_test)
        print(f"✅ Copied test set from {dataset_dirs[0]}\n")

    print(f"{'='*60}")
    print("✅ DATASET MERGE COMPLETE")
    print(f"{'='*60}\n")
    print(f"Merged dataset saved to: {output_dir}/")
    print("\nDataset statistics:")
    for class_name, count in metadata['splits']['train'].items():
        print(f"  {class_name}: {count:,} examples")
    print(f"  Total: {sum(metadata['splits']['train'].values()):,} examples\n")
    print("Next step:")
    print(f"  python src/train_gpt2_generative.py --data-dir {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple classification datasets'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='List of dataset directories to merge'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for merged dataset'
    )

    args = parser.parse_args()

    merge_datasets(
        dataset_dirs=args.datasets,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
