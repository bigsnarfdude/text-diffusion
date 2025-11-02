#!/usr/bin/env python3
"""
Training script for RoBERTa Diffusion Model

This trains RoBERTa as a generative model using variable masking.
The magic happens in the data collator, which applies different masking
rates to teach the model the full denoising curve.
"""

import os
import sys
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset

from src.config import TrainingConfig, parse_training_args
from src.data_collator import DiffusionDataCollator


class DiffusionTrainer(Trainer):
    """
    Custom Trainer that removes mask_prob before passing to model.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to remove mask_prob from inputs."""
        # Remove mask_prob before passing to model
        inputs.pop("mask_prob", None)
        return super().training_step(model, inputs, num_items_in_batch)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to remove mask_prob from inputs."""
        # Remove mask_prob before passing to model
        inputs.pop("mask_prob", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class MaskingLevelCallback(TrainerCallback):
    """
    Callback to log loss per masking level.

    This helps you see if the model is learning to denoise
    at all corruption levels (not just one).
    """

    def __init__(self):
        self.mask_level_losses = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track loss by masking level if available."""
        if logs is None:
            return

        # You can extend this to track per-level metrics
        # For now, just ensure basic logging works
        pass


def load_and_prepare_data(config: TrainingConfig):
    """
    Load dataset and tokenize.

    Args:
        config: Training configuration

    Returns:
        Tuple of (tokenizer, train_dataset, eval_dataset)
    """
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    tokenizer = RobertaTokenizerFast.from_pretrained(config.model_name)

    # Load dataset
    print(f"Loading dataset: {config.dataset_name}/{config.dataset_config}")
    dataset = load_dataset(config.dataset_name, config.dataset_config)

    # Tokenize
    def tokenize_function(examples):
        # Remove empty texts
        texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": []}

        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors=None,  # Return lists, not tensors
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # Filter out empty sequences
    def filter_empty(example):
        return len(example["input_ids"]) > 0

    tokenized_datasets = tokenized_datasets.filter(filter_empty)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(f"Train size: {len(train_dataset):,}")
    print(f"Eval size: {len(eval_dataset):,}")
    print(f"Sequence length: {config.max_length}")
    print(f"{'='*80}\n")

    return tokenizer, train_dataset, eval_dataset


def setup_model(config: TrainingConfig):
    """
    Load pre-trained RoBERTa for masked LM.

    Args:
        config: Training configuration

    Returns:
        RoBERTa model
    """
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")

    print(f"Loading model: {config.model_name}")
    model = RobertaForMaskedLM.from_pretrained(config.model_name)

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    print(f"{'='*80}\n")

    return model


def train(config: TrainingConfig):
    """
    Main training function.

    Args:
        config: Training configuration
    """
    print("\n" + "="*80)
    print("ROBERTA DIFFUSION TRAINING")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Masking levels: {config.mask_probs}")
    if config.quick_test:
        print("\n⚡ QUICK TEST MODE - Reduced steps for fast iteration")
    print("="*80 + "\n")

    # Load data
    tokenizer, train_dataset, eval_dataset = load_and_prepare_data(config)

    # Setup model
    model = setup_model(config)

    # Setup diffusion data collator (THE KEY COMPONENT)
    print(f"\n{'='*80}")
    print("DIFFUSION DATA COLLATOR")
    print(f"{'='*80}")
    print("This is where the magic happens!")
    print(f"Masking probabilities: {config.mask_probs}")
    print(f"Prefix length (preserved): {config.prefix_length}")
    print("\nEach batch will randomly use one of these masking levels.")
    print("This teaches the model to denoise at ALL corruption levels.")
    print(f"{'='*80}\n")

    data_collator = DiffusionDataCollator(
        tokenizer=tokenizer,
        mask_probs=config.mask_probs,
        prefix_length=config.prefix_length,
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Disable wandb for simplicity
        remove_unused_columns=False,  # Keep mask_prob for logging
    )

    # Setup trainer (use custom DiffusionTrainer)
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[MaskingLevelCallback()],
    )

    # Train!
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint before exit...")
        trainer.save_model(os.path.join(config.output_dir, "checkpoint-interrupted"))

    # Save final model
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")

    final_model_path = os.path.join(config.output_dir, "final-model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Model saved to: {final_model_path}")
    print(f"Checkpoints in: {config.output_dir}")

    # Create 'latest' symlink for easy access
    latest_link = os.path.join(config.output_dir, "checkpoint-latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(final_model_path, latest_link)
    print(f"Created symlink: checkpoint-latest -> final-model")

    print("\n✅ Training complete!")
    print(f"\nNext steps:")
    print(f"  1. Generate text: python generate.py --checkpoint {final_model_path}")
    print(f"  2. Visualize: python experiments/masking_viz.py")
    print(f"  3. Analyze layers: python experiments/layer_analysis.py\n")


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) > 1:
        config = parse_training_args()
    else:
        # Default config for interactive use
        print("No arguments provided, using default configuration.")
        print("Run with --help to see available options.\n")
        config = TrainingConfig(quick_test=True)

    # Train
    train(config)


if __name__ == "__main__":
    main()
