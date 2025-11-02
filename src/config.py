"""
Configuration for text diffusion training and generation.
All hyperparameters in one place for easy experimentation.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Model
    model_name: str = "distilroberta-base"  # Smaller/faster for learning
    # Alternative options:
    # - "roberta-base" (larger, better quality)
    # - "roberta-large" (higher quality, slower)

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 128

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500

    # Logging & Checkpointing
    output_dir: str = "./results"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3

    # Diffusion-specific
    mask_probs: List[float] = None  # Will be [1.0, 0.9, ..., 0.1]
    prefix_length: int = 5  # Preserve first N tokens during training

    # Quick test mode (for learning/debugging)
    quick_test: bool = False

    def __post_init__(self):
        if self.mask_probs is None:
            self.mask_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        if self.quick_test:
            print("🚀 Quick test mode enabled!")
            self.num_train_epochs = 1
            self.per_device_train_batch_size = 8
            self.logging_steps = 10
            self.eval_steps = 50
            self.save_steps = 100


@dataclass
class GenerationConfig:
    """Generation/inference hyperparameters."""

    # Model
    checkpoint_path: str = "./results/checkpoint-latest"
    device: str = "cuda"  # or "cpu"

    # Generation settings
    prefix: str = "The quick brown fox"
    max_length: int = 64
    num_samples: int = 5

    # Denoising schedule
    schedule_type: str = "linear"  # linear, cosine, exponential
    num_steps: int = 10

    # Sampling strategy
    sampling_method: str = "topk"  # greedy, topk, nucleus
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # Visualization
    show_steps: bool = True  # Print intermediate denoising steps

    def get_mask_schedule(self) -> List[float]:
        """Generate masking probability schedule for denoising."""
        import numpy as np

        if self.schedule_type == "linear":
            # Linear: 1.0 → 0.0
            return np.linspace(1.0, 0.0, self.num_steps + 1)[:-1].tolist()

        elif self.schedule_type == "cosine":
            # Cosine: More time at high/low corruption
            t = np.linspace(0, np.pi / 2, self.num_steps)
            return (1.0 - np.sin(t)).tolist()

        elif self.schedule_type == "exponential":
            # Exponential decay: Fast at first, slow at end
            return (np.exp(-3 * np.linspace(0, 1, self.num_steps))).tolist()

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def parse_training_args() -> TrainingConfig:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train RoBERTa diffusion model")

    parser.add_argument("--model-name", type=str, default="distilroberta-base",
                       help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test mode (fewer steps)")

    args = parser.parse_args()

    config = TrainingConfig()
    config.model_name = args.model_name
    config.output_dir = args.output_dir
    config.num_train_epochs = args.epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_length = args.max_length
    config.quick_test = args.quick_test

    return config


def parse_generation_args() -> GenerationConfig:
    """Parse command-line arguments for generation."""
    parser = argparse.ArgumentParser(description="Generate text with RoBERTa diffusion")

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--prefix", type=str, default="The quick brown fox",
                       help="Text prefix for conditional generation")
    parser.add_argument("--max-length", type=int, default=64,
                       help="Maximum generation length")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to generate")
    parser.add_argument("--schedule", type=str, default="linear",
                       choices=["linear", "cosine", "exponential"],
                       help="Denoising schedule type")
    parser.add_argument("--steps", type=int, default=10,
                       help="Number of denoising steps")
    parser.add_argument("--sampling", type=str, default="topk",
                       choices=["greedy", "topk", "nucleus"],
                       help="Sampling method")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--hide-steps", action="store_true",
                       help="Don't show intermediate denoising steps")

    args = parser.parse_args()

    config = GenerationConfig()
    config.checkpoint_path = args.checkpoint
    config.prefix = args.prefix
    config.max_length = args.max_length
    config.num_samples = args.num_samples
    config.schedule_type = args.schedule
    config.num_steps = args.steps
    config.sampling_method = args.sampling
    config.temperature = args.temperature
    config.top_k = args.top_k
    config.top_p = args.top_p
    config.device = args.device
    config.show_steps = not args.hide_steps

    return config


if __name__ == "__main__":
    # Demo: Print default configs
    print("=== Default Training Config ===")
    train_config = TrainingConfig()
    for key, value in train_config.__dict__.items():
        print(f"{key:30s}: {value}")

    print("\n=== Default Generation Config ===")
    gen_config = GenerationConfig()
    for key, value in gen_config.__dict__.items():
        print(f"{key:30s}: {value}")

    print("\n=== Masking Schedules ===")
    for schedule_type in ["linear", "cosine", "exponential"]:
        gen_config.schedule_type = schedule_type
        schedule = gen_config.get_mask_schedule()
        print(f"{schedule_type:12s}: {[f'{x:.2f}' for x in schedule[:5]]} ...")
