#!/usr/bin/env python3
"""
Generate Synthetic Movie Reviews using Ollama gpt-oss:20b

Uses the local Ollama model on nigel.birs.ca to generate synthetic IMDB-style
movie reviews for data augmentation.

Usage:
    # Generate 10,000 samples (5,000 per class)
    python scripts/generate_synthetic_reviews.py --num-samples 10000

    # Quick test with 100 samples
    python scripts/generate_synthetic_reviews.py --num-samples 100 --quick-test
"""

import argparse
import json
import requests
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import random


class OllamaGenerator:
    """Generate synthetic reviews using Ollama gpt-oss:20b."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b"
    ):
        self.ollama_url = ollama_url
        self.model = model

    def generate_review(
        self,
        sentiment: str,  # "positive" or "negative"
        product_category: str = None,
        temperature: float = 0.9,
        max_retries: int = 3
    ) -> str:
        """Generate a single synthetic review."""

        # Create prompt based on sentiment - MOVIE REVIEWS for IMDB domain
        if sentiment == "positive":
            prompt = """Write a positive movie review in the style of IMDB user reviews.
The review should be 3-5 sentences about a specific movie, mentioning plot, acting, or production quality.
Make it enthusiastic and personal, like a real movie fan wrote it.
Write only the review text, no labels or prefixes.

Review:"""
        else:
            prompt = """Write a negative movie review in the style of IMDB user reviews.
The review should be 3-5 sentences about a specific movie, mentioning plot, acting, or production quality.
Make it disappointed and critical, like a real movie fan wrote it.
Write only the review text, no labels or prefixes.

Review:"""

        # Call Ollama API
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": 0.9,
                            "max_tokens": 256
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    review_text = result.get('response', '').strip()

                    # Clean up the response
                    # Remove any quotes or prefixes
                    review_text = review_text.strip('"\'')

                    # Remove common prefixes
                    prefixes_to_remove = [
                        "Review:", "Amazon Review:", "Product Review:",
                        "Here's the review:", "Here is the review:"
                    ]
                    for prefix in prefixes_to_remove:
                        if review_text.startswith(prefix):
                            review_text = review_text[len(prefix):].strip()

                    # Validate length (Amazon reviews should be substantial)
                    if len(review_text.split()) >= 10:  # At least 10 words
                        return review_text
                    else:
                        # Too short, retry
                        continue

                else:
                    print(f"API error {response.status_code}: {response.text}")
                    time.sleep(1)

            except Exception as e:
                print(f"Generation error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1)

        # Fallback if all retries fail
        return None

    def generate_batch(
        self,
        sentiment: str,
        num_samples: int,
        temperature: float = 0.9,
        show_progress: bool = True
    ) -> List[str]:
        """Generate multiple synthetic reviews."""

        reviews = []

        iterator = tqdm(
            range(num_samples),
            desc=f"Generating {sentiment} reviews",
            disable=not show_progress
        )

        for i in iterator:
            review = self.generate_review(sentiment, temperature=temperature)

            if review is not None:
                reviews.append(review)
            else:
                # Generation failed, use a placeholder warning
                print(f"Warning: Failed to generate review {i} for {sentiment}")

            # Rate limiting - be nice to Ollama
            time.sleep(0.1)  # 100ms delay between generations

        return reviews


def prepare_synthetic_dataset(
    output_dir: str,
    num_samples: int = 10000,
    temperature: float = 0.9
):
    """
    Generate synthetic Amazon reviews using Ollama.

    Args:
        output_dir: Where to save the synthetic dataset
        num_samples: Total number of samples (split 50/50 between classes)
        temperature: Generation temperature (higher = more creative)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING SYNTHETIC IMDB MOVIE REVIEWS")
    print(f"{'='*60}\n")
    print(f"Model: Ollama gpt-oss:20b")
    print(f"Total samples: {num_samples}")
    print(f"Per class: {num_samples // 2}")
    print(f"Output directory: {output_dir}")
    print(f"Temperature: {temperature}\n")

    # Initialize generator
    generator = OllamaGenerator()

    # Test connection
    print("Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running\n")
        else:
            print(f"⚠️  Ollama returned status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    # Generate negative reviews (class 0)
    print("Generating negative reviews...")
    negative_reviews = generator.generate_batch(
        sentiment="negative",
        num_samples=num_samples // 2,
        temperature=temperature
    )

    # Save negative reviews
    negative_file = output_path / 'train_class_0.json'
    with open(negative_file, 'w') as f:
        json.dump({
            'class_id': 0,
            'class_name': 'negative',
            'num_examples': len(negative_reviews),
            'texts': negative_reviews,
            'synthetic': True,
            'source': 'ollama-gpt-oss-20b'
        }, f, indent=2)

    print(f"✅ Saved {len(negative_reviews)} negative reviews to {negative_file}\n")

    # Generate positive reviews (class 1)
    print("Generating positive reviews...")
    positive_reviews = generator.generate_batch(
        sentiment="positive",
        num_samples=num_samples // 2,
        temperature=temperature
    )

    # Save positive reviews
    positive_file = output_path / 'train_class_1.json'
    with open(positive_file, 'w') as f:
        json.dump({
            'class_id': 1,
            'class_name': 'positive',
            'num_examples': len(positive_reviews),
            'texts': positive_reviews,
            'synthetic': True,
            'source': 'ollama-gpt-oss-20b'
        }, f, indent=2)

    print(f"✅ Saved {len(positive_reviews)} positive reviews to {positive_file}\n")

    # Save metadata
    metadata = {
        'dataset': 'synthetic_imdb',
        'task': 'sentiment_classification',
        'domain': 'movie_reviews',
        'num_classes': 2,
        'class_names': ['negative', 'positive'],
        'synthetic': True,
        'generator': 'ollama-gpt-oss-20b',
        'temperature': temperature,
        'splits': {
            'train': {
                'negative': len(negative_reviews),
                'positive': len(positive_reviews)
            }
        }
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {metadata_file}\n")

    print(f"{'='*60}")
    print("✅ SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Generated {len(negative_reviews) + len(positive_reviews)} synthetic reviews")
    print(f"Negative: {len(negative_reviews)}")
    print(f"Positive: {len(positive_reviews)}\n")
    print("Files created:")
    print(f"  {output_dir}/metadata.json")
    print(f"  {output_dir}/train_class_0.json")
    print(f"  {output_dir}/train_class_1.json\n")
    print("Next steps:")
    print(f"  # Merge with real data")
    print(f"  python scripts/merge_datasets.py \\")
    print(f"      --datasets data/imdb-classifier {output_dir} \\")
    print(f"      --output data/imdb-augmented")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Amazon reviews using Ollama'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Total number of synthetic samples to generate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/synthetic-imdb',
        help='Output directory for synthetic data'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.9,
        help='Generation temperature (higher = more creative)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with 100 samples'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.num_samples = 100
        print("⚡ QUICK TEST MODE ⚡")
        print("Generating 100 samples (50 per class)\n")

    # Generate synthetic data
    prepare_synthetic_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        temperature=args.temperature
    )


if __name__ == '__main__':
    main()
