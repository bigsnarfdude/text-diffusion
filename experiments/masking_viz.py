#!/usr/bin/env python3
"""
Visualize how different masking strategies affect the text.

This helps you understand what the model sees during training
and how masking probability affects the denoising task.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import RobertaTokenizerFast
from src.data_collator import VisualizableDiffusionCollator


def visualize_masking_levels():
    """Show how different masking probabilities affect text."""
    print("\n" + "="*80)
    print("MASKING LEVEL VISUALIZATION")
    print("="*80)
    print("\nThis shows what the model sees at different corruption levels.")
    print("Understanding this helps you grasp how diffusion training works.")
    print("="*80 + "\n")

    # Setup
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Test text
    text = (
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data without being explicitly programmed. "
        "It uses statistical techniques to give computer systems the ability "
        "to progressively improve their performance on a specific task."
    )

    print(f"ORIGINAL TEXT:")
    print(f"  {text}\n")

    # Tokenize
    example = tokenizer(text, truncation=True, max_length=64, padding="max_length")

    # Test different masking levels
    masking_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

    for mask_prob in masking_levels:
        print(f"\n{'='*80}")
        print(f"MASKING PROBABILITY: {mask_prob:.0%} (Model must recover {mask_prob:.0%} of tokens)")
        print(f"{'='*80}")

        # Create collator with fixed masking
        collator = VisualizableDiffusionCollator(
            tokenizer=tokenizer,
            mask_probs=[mask_prob],
            prefix_length=0,  # No prefix preservation for this demo
        )

        # Process
        batch = collator([example])

        # Show result
        masked_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        print(f"MASKED: {masked_text}")

        # Stats
        n_masked = (batch["input_ids"][0] == tokenizer.mask_token_id).sum().item()
        n_total = (batch["input_ids"][0] != tokenizer.pad_token_id).sum().item()
        actual_rate = n_masked / n_total if n_total > 0 else 0
        print(f"STATS: {n_masked}/{n_total} tokens masked ({actual_rate:.1%})")

    print("\n" + "="*80)
    print("OBSERVATIONS")
    print("="*80)
    print("""
Key insights from this visualization:

1. HIGH MASKING (80-100%): Model learns structure and common words
   - Must infer meaning from very limited context
   - Learns statistical patterns and frequent tokens
   - Like guessing a sentence with only 1-2 words visible

2. MEDIUM MASKING (40-60%): Model learns context-dependent content
   - Has enough context to make informed predictions
   - Learns semantic relationships between words
   - Most "interesting" learning happens here

3. LOW MASKING (10-20%): Model learns fine details
   - Almost all context is available
   - Learns rare words, specific phrasings
   - Like proofreading with a few words missing

Training with ALL levels → model learns the full denoising curve!
    """)
    print("="*80 + "\n")


def visualize_prefix_preservation():
    """Show how prefix preservation works."""
    print("\n" + "="*80)
    print("PREFIX PRESERVATION VISUALIZATION")
    print("="*80)
    print("\nPrefix preservation allows conditional generation during training.")
    print("The model learns to continue text based on a given prefix.")
    print("="*80 + "\n")

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    text = "Machine learning enables computers to learn from data and improve their performance."

    print(f"ORIGINAL TEXT:")
    print(f"  {text}\n")

    example = tokenizer(text, truncation=True, max_length=64, padding="max_length")

    # Test different prefix lengths
    prefix_lengths = [0, 3, 5, 10]

    for prefix_len in prefix_lengths:
        print(f"\n{'='*80}")
        print(f"PREFIX LENGTH: {prefix_len} tokens (preserved, never masked)")
        print(f"{'='*80}")

        collator = VisualizableDiffusionCollator(
            tokenizer=tokenizer,
            mask_probs=[0.8],  # 80% masking
            prefix_length=prefix_len,
        )

        batch = collator([example])

        # Show prefix
        if prefix_len > 0:
            prefix_ids = batch["input_ids"][0, :prefix_len]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            print(f"PREFIX (never masked): '{prefix_text}'")

        # Show masked result
        masked_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
        print(f"RESULT: {masked_text}")

    print("\n" + "="*80)
    print("WHY PREFIX PRESERVATION MATTERS")
    print("="*80)
    print("""
Prefix preservation enables:

1. CONDITIONAL GENERATION: Model learns to continue given text
   - Training: "Machine learning [MASK] [MASK]..." → predict rest
   - Generation: Use same pattern to extend new prefixes

2. CONTROLLED OUTPUTS: Start generation with desired context
   - Example: "The airplane safety protocol" → aviation-specific text
   - Model conditions on prefix to maintain relevance

3. BETTER LEARNING: Model uses prefix as anchor point
   - Doesn't have to guess everything from scratch
   - Learns relationships between prefix and continuation

For this project: prefix_length=5 is a good default
    """)
    print("="*80 + "\n")


def visualize_schedule_comparison():
    """Compare different denoising schedules."""
    print("\n" + "="*80)
    print("DENOISING SCHEDULE COMPARISON")
    print("="*80)
    print("\nThe schedule determines how quickly we unmask during generation.")
    print("Different schedules emphasize different parts of the denoising process.")
    print("="*80 + "\n")

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    num_steps = 10

    # Generate schedules
    schedules = {}

    # Linear
    schedules['linear'] = np.linspace(1.0, 0.0, num_steps + 1)[:-1]

    # Cosine
    t = np.linspace(0, np.pi / 2, num_steps)
    schedules['cosine'] = 1.0 - np.sin(t)

    # Exponential
    schedules['exponential'] = np.exp(-3 * np.linspace(0, 1, num_steps))

    # Print schedules
    print("Schedule comparison (masking probability at each step):\n")
    print(f"{'Step':<8}", end="")
    for name in schedules:
        print(f"{name:<15}", end="")
    print()
    print("-" * 50)

    for step in range(num_steps):
        print(f"{step+1:<8}", end="")
        for name in schedules:
            print(f"{schedules[name][step]:<15.3f}", end="")
        print()

    # Plot
    plt.figure(figsize=(10, 6))
    for name, schedule in schedules.items():
        plt.plot(range(1, num_steps + 1), schedule, marker='o', label=name, linewidth=2)

    plt.xlabel('Denoising Step', fontsize=12)
    plt.ylabel('Masking Probability', fontsize=12)
    plt.title('Denoising Schedule Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = 'experiments/schedule_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Plot saved to: {output_path}")

    print("\n" + "="*80)
    print("SCHEDULE CHARACTERISTICS")
    print("="*80)
    print("""
LINEAR (default):
  - Uniform unmasking rate
  - Equal time at all corruption levels
  - Good general-purpose choice
  - Predictable behavior

COSINE:
  - More time at high and low corruption
  - Faster through medium corruption
  - Better quality at ends
  - Good for careful generation

EXPONENTIAL:
  - Fast unmasking at first
  - Slow refinement at end
  - Emphasizes structural decisions early
  - Good for creative generation

TIP: Start with linear, then experiment!
    """)
    print("="*80 + "\n")


def main():
    """Run all visualizations."""
    print("\n" + "="*80)
    print("TEXT DIFFUSION - MASKING STRATEGY EXPERIMENTS")
    print("="*80)
    print("\nThese experiments help you understand:")
    print("  1. How masking affects text during training")
    print("  2. Why prefix preservation matters")
    print("  3. How different schedules affect generation")
    print("="*80 + "\n")

    try:
        visualize_masking_levels()
        visualize_prefix_preservation()
        visualize_schedule_comparison()

        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print("\n✅ You should now understand:")
        print("  • What the model sees at different masking levels")
        print("  • How prefix preservation enables conditional generation")
        print("  • Trade-offs between different denoising schedules")
        print("\n💡 Next steps:")
        print("  • Run training: python train.py --quick-test")
        print("  • Generate text: python generate.py --checkpoint results/checkpoint-latest")
        print("  • Try different schedules during generation")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
