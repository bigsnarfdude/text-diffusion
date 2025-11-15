#!/usr/bin/env python3
"""
Compare confidence-based vs schedule-based generation.

Tests the new confidence-based approach against the original schedule-based
approach to see differences in quality, speed, and convergence.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

from src.generate import DiffusionGenerator
from src.confidence_generate import ConfidenceBasedGenerator
from src.config import GenerationConfig


def test_both_methods(checkpoint_path: str, device: str = "cpu"):
    """
    Compare confidence-based vs schedule-based generation.

    Args:
        checkpoint_path: Path to trained model
        device: Device to use (cpu/cuda)
    """
    print("="*80)
    print("COMPARING GENERATION METHODS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("="*80 + "\n")

    # Load model once
    print("Loading model...")
    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_path)
    model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded\n")

    # Test prompts
    test_prompts = [
        "The movie was",
        "This film is",
        "I think the director",
    ]

    # Generation settings
    config = GenerationConfig(
        checkpoint_path=checkpoint_path,
        prefix="",  # Will be set per prompt
        max_length=50,
        num_samples=1,
        schedule_type="linear",
        num_steps=10,
        sampling_method="topk",
        top_k=50,
        temperature=0.8,
        device=device,
        show_steps=False,
    )

    # Confidence settings
    confidence_threshold = 0.7
    max_steps = 30

    # Test each prompt with both methods
    for i, prefix in enumerate(test_prompts, 1):
        print("\n" + "="*80)
        print(f"TEST {i}/3: '{prefix}'")
        print("="*80)

        # Method 1: Schedule-based (original)
        print("\n📋 METHOD 1: Schedule-Based Generation")
        print("-" * 80)
        config.prefix = prefix
        generator_schedule = DiffusionGenerator(model, tokenizer, config)

        start_time = time.time()
        result_schedule = generator_schedule.generate(
            prefix=prefix,
            max_length=config.max_length,
            show_steps=False
        )
        time_schedule = time.time() - start_time

        print(f"Time: {time_schedule:.2f}s")
        print(f"Steps: {config.num_steps} (fixed)")
        print(f"Result: {result_schedule}")

        # Method 2: Confidence-based (new)
        print("\n🎯 METHOD 2: Confidence-Based Generation")
        print("-" * 80)
        generator_confidence = ConfidenceBasedGenerator(model, tokenizer, config)

        start_time = time.time()
        result_confidence = generator_confidence.generate(
            prefix=prefix,
            max_length=config.max_length,
            confidence_threshold=confidence_threshold,
            max_steps=max_steps,
            show_steps=False
        )
        time_confidence = time.time() - start_time

        print(f"Time: {time_confidence:.2f}s")
        print(f"Threshold: {confidence_threshold:.1%}")
        print(f"Result: {result_confidence}")

        # Comparison
        print("\n📊 COMPARISON:")
        print("-" * 80)
        speedup = time_schedule / time_confidence if time_confidence > 0 else 0
        print(f"Schedule-based: {time_schedule:.2f}s")
        print(f"Confidence-based: {time_confidence:.2f}s")
        print(f"Speedup: {speedup:.2f}x {'(confidence faster)' if speedup > 1 else '(schedule faster)'}")
        print(f"\nOutputs {'identical' if result_schedule == result_confidence else 'different'}")

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\n✅ Tested {len(test_prompts)} prompts with both methods")
    print(f"\n📋 Schedule-based:")
    print(f"   - Fixed {config.num_steps} steps per generation")
    print(f"   - Predictable runtime")
    print(f"   - Re-masks randomly each step")
    print(f"\n🎯 Confidence-based:")
    print(f"   - Variable steps (stops when converged)")
    print(f"   - Quality control via threshold")
    print(f"   - Progressive refinement (easy tokens first)")
    print(f"\n💡 Next Steps:")
    print(f"   1. Try different confidence thresholds (0.5, 0.6, 0.7, 0.8)")
    print(f"   2. Test on longer sequences")
    print(f"   3. Evaluate classification performance with confidence-based")
    print(f"   4. Create visualization comparing both methods")
    print("="*80 + "\n")


def main():
    """Main entry point."""
    # Check for checkpoint argument
    if len(sys.argv) < 2:
        print("Usage: python test_confidence_vs_schedule.py <checkpoint_path> [device]")
        print("\nExample:")
        print("  python test_confidence_vs_schedule.py results-generative-classifier/class-0/final-model")
        print("  python test_confidence_vs_schedule.py results-generative-classifier/class-1/final-model cpu")
        print("\nAvailable checkpoints:")
        import glob
        checkpoints = glob.glob("results*/*/final-model") + glob.glob("results*/checkpoint-*")
        for cp in sorted(checkpoints)[:5]:
            print(f"  - {cp}")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, using CPU")
        device = "cpu"

    # Run comparison
    test_both_methods(checkpoint_path, device)


if __name__ == "__main__":
    main()
