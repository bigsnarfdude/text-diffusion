#!/usr/bin/env python3
"""
TRUE GPU-Batched Synthetic Review Generation using gpt-oss:20b

Uses transformers directly with proper batching for maximum GPU utilization.

Key improvements:
- Loads gpt-oss:20b once on GPU
- Generates 8-16 samples simultaneously per forward pass
- True GPU batching (not API calls)
- Estimated time: ~3-4 hours for 10K samples

Usage:
    python scripts/generate_synthetic_batched.py --num-samples 10000 --batch-size 8
"""

import argparse
import json
from pathlib import Path
from typing import List
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class BatchedGenerator:
    """Generate synthetic reviews with true GPU batching."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device: str = "cuda",
        torch_dtype = torch.float16
    ):
        print(f"\nLoading {model_name}...")
        print("This may take a few minutes for a 20B parameter model...\n")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"✅ Model loaded on {device}")
        print(f"   VRAM usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    def create_prompt(self, sentiment: str) -> str:
        """Create generation prompt."""
        if sentiment == "positive":
            return """Write a positive movie review in the style of IMDB user reviews. The review should be 3-5 sentences about a specific movie, mentioning plot, acting, or production quality. Make it enthusiastic and personal. Write only the review text, no labels.

Review:"""
        else:
            return """Write a negative movie review in the style of IMDB user reviews. The review should be 3-5 sentences about a specific movie, mentioning plot, acting, or production quality. Make it critical and disappointed. Write only the review text, no labels.

Review:"""

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate a batch of reviews simultaneously."""

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100  # Prompt length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )

        # Decode
        reviews = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)

            # Extract just the review (after "Review:")
            if "Review:" in text:
                review = text.split("Review:")[-1].strip()
            else:
                review = text.strip()

            # Clean up
            review = review.strip('"\'')

            # Validate
            if len(review.split()) >= 10:
                reviews.append(review)
            else:
                reviews.append(None)  # Mark as failed

        return reviews

    def generate_dataset(
        self,
        sentiment: str,
        num_samples: int,
        batch_size: int = 8,
        temperature: float = 0.9
    ) -> List[str]:
        """Generate full dataset with batching."""

        reviews = []
        prompt = self.create_prompt(sentiment)

        # Calculate number of batches
        num_batches = (num_samples + batch_size - 1) // batch_size

        print(f"Generating {num_samples} {sentiment} reviews...")
        print(f"  Batch size: {batch_size}")
        print(f"  Total batches: {num_batches}\n")

        for batch_idx in tqdm(range(num_batches), desc=f"{sentiment.capitalize()}"):
            # Determine batch size (last batch might be smaller)
            current_batch_size = min(batch_size, num_samples - len(reviews))

            # Create batch of prompts
            batch_prompts = [prompt] * current_batch_size

            # Generate batch
            batch_reviews = self.generate_batch(
                batch_prompts,
                temperature=temperature
            )

            # Add valid reviews
            for review in batch_reviews:
                if review is not None:
                    reviews.append(review)

            # Show progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Generated {len(reviews)}/{num_samples} reviews")

        return reviews


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic reviews with GPU batching'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Total number of samples'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic-imdb',
        help='Output directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-oss-20b',
        help='Model to use'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.9,
        help='Generation temperature'
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GPU-BATCHED SYNTHETIC GENERATION")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Total samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}\n")

    # Initialize generator
    generator = BatchedGenerator(model_name=args.model)

    start_time = time.time()

    # Generate negative reviews
    print(f"{'='*60}")
    negative_reviews = generator.generate_dataset(
        sentiment="negative",
        num_samples=args.num_samples // 2,
        batch_size=args.batch_size,
        temperature=args.temperature
    )

    # Save negative
    with open(output_path / 'train_class_0.json', 'w') as f:
        json.dump({
            'class_id': 0,
            'class_name': 'negative',
            'num_examples': len(negative_reviews),
            'texts': negative_reviews,
            'synthetic': True,
            'source': args.model
        }, f, indent=2)

    print(f"\n✅ Saved {len(negative_reviews)} negative reviews\n")

    # Generate positive reviews
    print(f"{'='*60}")
    positive_reviews = generator.generate_dataset(
        sentiment="positive",
        num_samples=args.num_samples // 2,
        batch_size=args.batch_size,
        temperature=args.temperature
    )

    # Save positive
    with open(output_path / 'train_class_1.json', 'w') as f:
        json.dump({
            'class_id': 1,
            'class_name': 'positive',
            'num_examples': len(positive_reviews),
            'texts': positive_reviews,
            'synthetic': True,
            'source': args.model
        }, f, indent=2)

    print(f"\n✅ Saved {len(positive_reviews)} positive reviews\n")

    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'dataset': 'synthetic_imdb',
            'task': 'sentiment_classification',
            'domain': 'movie_reviews',
            'num_classes': 2,
            'class_names': ['negative', 'positive'],
            'synthetic': True,
            'generator': args.model,
            'batch_size': args.batch_size,
            'temperature': args.temperature,
            'splits': {
                'train': {
                    'negative': len(negative_reviews),
                    'positive': len(positive_reviews)
                }
            }
        }, f, indent=2)

    elapsed = time.time() - start_time
    total_samples = len(negative_reviews) + len(positive_reviews)

    print(f"{'='*60}")
    print("✅ GENERATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Total samples: {total_samples}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Speed: {elapsed/total_samples:.2f} seconds per sample")
    print(f"Throughput: {total_samples*3600/elapsed:.0f} samples/hour\n")


if __name__ == '__main__':
    main()
