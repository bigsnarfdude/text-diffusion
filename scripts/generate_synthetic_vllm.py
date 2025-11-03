#!/usr/bin/env python3
"""
VLLM-Powered Synthetic Review Generation using gpt-oss:20b

Uses vLLM for maximum GPU batching and throughput.

Key improvements over transformers:
- vLLM's PagedAttention for efficient memory usage
- Continuous batching for maximum throughput
- Automatic batch scheduling
- Estimated time: ~1-2 hours for 10K samples

Usage:
    python scripts/generate_synthetic_vllm.py --num-samples 10000 --batch-size 32
"""

import argparse
import json
from pathlib import Path
from typing import List
import time

from vllm import LLM, SamplingParams
from tqdm import tqdm


class VLLMGenerator:
    """Generate synthetic reviews with vLLM batching."""

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        print(f"\nLoading {model_name} with vLLM...")
        print("This may take a few minutes for a 20B parameter model...\n")

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16"  # Required for gpt-oss mxfp4 quantization
        )

        print(f"✅ Model loaded with vLLM\n")

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
        max_tokens: int = 200,
        temperature: float = 0.9,
        top_p: float = 0.9
    ) -> List[str]:
        """Generate a batch of reviews using vLLM."""

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=None
        )

        # Generate with vLLM (handles batching automatically)
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract and validate reviews
        reviews = []
        for output in outputs:
            text = output.outputs[0].text.strip()

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
        batch_size: int = 32,
        temperature: float = 0.9
    ) -> List[str]:
        """Generate full dataset with vLLM batching."""

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
        description='Generate synthetic reviews with vLLM'
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
        default=32,
        help='Batch size for generation (vLLM handles larger batches)'
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
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.7,  # Reduced from 0.9 to avoid OOM with gpt-oss:20b
        help='GPU memory utilization (0.0-1.0)'
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("VLLM-POWERED SYNTHETIC GENERATION")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Total samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}\n")

    # Initialize generator
    generator = VLLMGenerator(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

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
            'source': f"{args.model}-vllm"
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
            'source': f"{args.model}-vllm"
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
            'generator': f"{args.model}-vllm",
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
