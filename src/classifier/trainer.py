"""
Trainer for generative classification.
Trains separate diffusion model for each class.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    RobertaForMaskedLM,
    AutoTokenizer,
    TrainingArguments
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_collator import DiffusionDataCollator
from src.train import DiffusionTrainer  # Use custom trainer that removes mask_prob
from src.classifier.data import (
    ClassificationDataset,
    load_classification_data,
    load_metadata
)


class PerClassTrainer:
    """
    Trains separate diffusion models for each class using generative modeling.

    At inference, we compute p(text|class) for each class and choose the class
    with highest likelihood.
    """

    def __init__(
        self,
        model_name: str = 'distilroberta-base',
        data_dir: str = 'data/imdb-classifier',
        output_dir: str = 'results-generative-classifier',
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load metadata
        self.metadata = load_metadata(str(data_dir))
        self.num_classes = self.metadata['num_classes']
        self.class_names = self.metadata['class_names']

        # Will hold trained models
        self.class_models = {}

        print(f"\n{'='*60}")
        print(f"GENERATIVE CLASSIFIER TRAINER")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Classes: {self.num_classes} ({self.class_names})")
        print(f"Data: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

    def train_all_classes(
        self,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_samples_per_class: Optional[int] = None,
        quick_test: bool = False
    ):
        """
        Train separate diffusion model for each class.

        Args:
            epochs: Number of training epochs per class
            batch_size: Training batch size
            learning_rate: Learning rate
            max_samples_per_class: Limit samples per class (for testing)
            quick_test: Use tiny subset for quick testing
        """
        if quick_test:
            epochs = 1
            max_samples_per_class = 100
            print("⚠️  QUICK TEST MODE: Using small subset\n")

        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]

            print(f"\n{'='*60}")
            print(f"TRAINING CLASS {class_id}: {class_name.upper()}")
            print(f"{'='*60}\n")

            # Load class-specific data
            print(f"Loading training data for class {class_id}...")
            texts = load_classification_data(str(self.data_dir), class_id, split='train')

            if max_samples_per_class:
                texts = texts[:max_samples_per_class]

            print(f"  Loaded {len(texts)} examples")
            if texts:
                print(f"  Sample: {texts[0][:100]}...\n")

            # Create dataset
            dataset = ClassificationDataset(texts, self.tokenizer)

            # Initialize fresh model for this class
            print(f"Initializing model from {self.model_name}...")
            model = RobertaForMaskedLM.from_pretrained(self.model_name)
            model.to(self.device)

            # Setup training
            class_output_dir = self.output_dir / f'class-{class_id}'
            class_output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(class_output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_steps=50,
                save_steps=500,
                save_total_limit=2,
                logging_dir=str(class_output_dir / 'logs'),
                report_to='none',  # Disable wandb
                remove_unused_columns=False,
            )

            # Create data collator with variable masking (the key innovation!)
            data_collator = DiffusionDataCollator(
                tokenizer=self.tokenizer
                # mask_probs defaults to [1.0, 0.9, ..., 0.1] for variable masking
            )

            # Create trainer (using DiffusionTrainer to handle mask_prob)
            trainer = DiffusionTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            # Train!
            print(f"Starting training for class {class_id}...")
            trainer.train()

            # Save final model
            final_model_path = class_output_dir / 'final-model'
            print(f"\nSaving final model to {final_model_path}...")
            trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))

            # Store in memory
            self.class_models[class_id] = model

            print(f"✅ Completed training for class {class_id}")

        # Save overall metadata
        self._save_training_metadata(epochs, batch_size, learning_rate)

        print(f"\n{'='*60}")
        print(f"✅ ALL CLASSES TRAINED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\nModels saved to: {self.output_dir}/")
        for i, name in enumerate(self.class_names):
            print(f"  class-{i}/final-model/ ({name})")
        print(f"\nNext step: python src/evaluate_classifier.py")
        print()

    def train_single_class(
        self,
        class_id: int,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_samples: Optional[int] = None
    ):
        """Train model for a single class (useful for debugging)."""
        class_name = self.class_names[class_id]

        print(f"\n{'='*60}")
        print(f"TRAINING SINGLE CLASS {class_id}: {class_name.upper()}")
        print(f"{'='*60}\n")

        # Load class-specific data
        texts = load_classification_data(str(self.data_dir), class_id, split='train')

        if max_samples:
            texts = texts[:max_samples]

        # Create dataset
        dataset = ClassificationDataset(texts, self.tokenizer)

        # Initialize model
        model = RobertaForMaskedLM.from_pretrained(self.model_name)
        model.to(self.device)

        # Setup training
        class_output_dir = self.output_dir / f'class-{class_id}'
        class_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(class_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            logging_dir=str(class_output_dir / 'logs'),
            report_to='none',
            remove_unused_columns=False,
        )

        data_collator = DiffusionDataCollator(
            tokenizer=self.tokenizer
        )

        trainer = DiffusionTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save
        final_model_path = class_output_dir / 'final-model'
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))

        self.class_models[class_id] = model

        print(f"✅ Completed training for class {class_id}")

    def _save_training_metadata(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """Save training configuration."""
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'training': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
            },
            'data_dir': str(self.data_dir),
        }

        metadata_path = self.output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Training metadata saved to {metadata_path}")

    def load_trained_models(self):
        """Load previously trained models from disk."""
        print(f"\nLoading trained models from {self.output_dir}...")

        for class_id in range(self.num_classes):
            model_path = self.output_dir / f'class-{class_id}' / 'final-model'

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Train first with: python src/train_generative_classifier.py"
                )

            print(f"  Loading class {class_id} model...")
            model = RobertaForMaskedLM.from_pretrained(str(model_path))
            model.to(self.device)
            model.eval()

            self.class_models[class_id] = model

        print(f"✅ Loaded {len(self.class_models)} class models\n")
