"""
Classification datasets for generative classifier.

Handles loading and preparing class-specific datasets for training
separate diffusion models per class.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ClassificationDataset(Dataset):
    """
    Dataset for a single class in generative classification.

    Each class gets its own dataset containing only examples from that class.
    We train a separate diffusion model on each class's dataset.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        """
        Args:
            texts: List of text examples for this class
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns tokenized text.
        The data collator will handle masking during training.
        """
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Remove batch dimension
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def load_classification_data(
    data_dir: str,
    class_id: int,
    split: str = 'train'
) -> List[str]:
    """
    Load text examples for a specific class.

    Args:
        data_dir: Directory containing prepared data
        class_id: Which class to load (0, 1, etc.)
        split: 'train', 'val', or 'test'

    Returns:
        List of text strings for this class
    """
    data_path = Path(data_dir) / f'{split}_class_{class_id}.json'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Run data preparation script first:\n"
            f"  python scripts/prepare_imdb.py"
        )

    with open(data_path, 'r') as f:
        data = json.load(f)

    texts = data['texts']

    print(f"Loaded {len(texts)} examples for class {class_id} ({split})")
    return texts


def load_metadata(data_dir: str) -> Dict:
    """
    Load dataset metadata (class names, counts, etc.)

    Args:
        data_dir: Directory containing prepared data

    Returns:
        Dictionary with metadata
    """
    metadata_path = Path(data_dir) / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Run data preparation script first:\n"
            f"  python scripts/prepare_imdb.py"
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def create_class_datasets(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    split: str = 'train',
    max_length: int = 256,
    max_samples_per_class: int = None
) -> Dict[int, ClassificationDataset]:
    """
    Create datasets for all classes.

    Args:
        data_dir: Directory containing prepared data
        tokenizer: HuggingFace tokenizer
        split: 'train', 'val', or 'test'
        max_length: Maximum sequence length
        max_samples_per_class: Limit samples per class (for testing)

    Returns:
        Dictionary mapping class_id -> ClassificationDataset
    """
    metadata = load_metadata(data_dir)
    num_classes = metadata['num_classes']

    datasets = {}

    for class_id in range(num_classes):
        # Load texts for this class
        texts = load_classification_data(data_dir, class_id, split)

        # Optionally limit samples
        if max_samples_per_class is not None:
            texts = texts[:max_samples_per_class]

        # Create dataset
        datasets[class_id] = ClassificationDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length
        )

    return datasets


def load_test_examples(
    data_dir: str,
    max_samples: int = None
) -> Tuple[List[str], List[int]]:
    """
    Load test examples with labels for evaluation.

    Args:
        data_dir: Directory containing prepared data
        max_samples: Limit number of test examples

    Returns:
        (texts, labels) where labels are class IDs
    """
    test_path = Path(data_dir) / 'test.json'

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            f"Run data preparation script first:\n"
            f"  python scripts/prepare_imdb.py"
        )

    with open(test_path, 'r') as f:
        data = json.load(f)

    texts = data['texts']
    labels = data['labels']

    if max_samples is not None:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    print(f"Loaded {len(texts)} test examples")
    return texts, labels
