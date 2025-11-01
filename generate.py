#!/usr/bin/env python3
"""
Text Generation with RoBERTa Diffusion

This implements the iterative denoising process:
1. Start with fully masked text (except prefix)
2. For each step: predict masked tokens, fill in best guesses, re-mask fewer
3. Repeat until fully denoised
"""

import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

from config import GenerationConfig, parse_generation_args


class DiffusionGenerator:
    """
    Iterative text generator using diffusion-trained RoBERTa.

    The generation process:
    - Start: [PREFIX] [MASK] [MASK] [MASK] ... (100% masked)
    - Step 1: Predict → fill in 10% → 90% still masked
    - Step 2: Predict → fill in 10% → 80% still masked
    - ...
    - Step 10: Final predictions → 0% masked → complete text
    """

    def __init__(
        self,
        model: RobertaForMaskedLM,
        tokenizer: RobertaTokenizerFast,
        config: GenerationConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        self.model.to(self.device)
        self.model.eval()

        # Get masking schedule
        self.mask_schedule = config.get_mask_schedule()

    def generate(
        self,
        prefix: str,
        max_length: int,
        show_steps: bool = True,
    ) -> str:
        """
        Generate text via iterative denoising.

        Args:
            prefix: Starting text (not masked)
            max_length: Total length including prefix
            show_steps: Whether to print intermediate steps

        Returns:
            Generated text
        """
        # Tokenize prefix
        prefix_ids = self.tokenizer.encode(
            prefix,
            add_special_tokens=True,
            return_tensors="pt"
        )[0].to(self.device)

        prefix_length = len(prefix_ids)

        if prefix_length >= max_length:
            print(f"⚠️  Warning: Prefix length ({prefix_length}) >= max_length ({max_length})")
            print(f"   Returning prefix only.")
            return prefix

        # Initialize: prefix + fully masked continuation
        num_to_generate = max_length - prefix_length
        mask_token_id = self.tokenizer.mask_token_id

        input_ids = torch.cat([
            prefix_ids,
            torch.full((num_to_generate,), mask_token_id, device=self.device)
        ]).unsqueeze(0)  # Add batch dimension

        if show_steps:
            print("\n" + "="*80)
            print("ITERATIVE DENOISING GENERATION")
            print("="*80)
            print(f"Prefix: {prefix}")
            print(f"Target length: {max_length} tokens")
            print(f"Denoising steps: {len(self.mask_schedule)}")
            print(f"Schedule: {self.config.schedule_type}")
            print(f"Sampling: {self.config.sampling_method} (temp={self.config.temperature})")
            print("="*80 + "\n")

            initial_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"STEP 0 (100% masked):")
            print(f"  {initial_text[:150]}...")
            print()

        # Iterative denoising
        for step, mask_prob in enumerate(self.mask_schedule, 1):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Find currently masked positions (after prefix)
            mask_positions = (input_ids[0, prefix_length:] == mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions = mask_positions + prefix_length  # Adjust for prefix

            if len(mask_positions) == 0:
                break  # No more masks to fill

            # Sample tokens for masked positions
            for pos in mask_positions:
                pos = pos.item()
                token_logits = logits[0, pos]  # [vocab_size]

                # Apply sampling strategy
                sampled_token = self._sample_token(token_logits)
                input_ids[0, pos] = sampled_token

            # Show progress
            if show_steps:
                current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                n_masked = (input_ids[0, prefix_length:] == mask_token_id).sum().item()
                mask_pct = (n_masked / num_to_generate) * 100
                print(f"STEP {step} ({mask_pct:.0f}% masked):")
                print(f"  {current_text[:150]}...")
                print()

            # Re-mask for next iteration (except last step)
            if step < len(self.mask_schedule):
                # Determine how many to re-mask
                next_mask_prob = self.mask_schedule[step] if step < len(self.mask_schedule) else 0.0
                n_to_mask = int(num_to_generate * next_mask_prob)

                if n_to_mask > 0:
                    # Randomly select positions to re-mask (after prefix)
                    maskable_positions = torch.arange(prefix_length, max_length, device=self.device)
                    indices = torch.randperm(len(maskable_positions))[:n_to_mask]
                    positions_to_mask = maskable_positions[indices]
                    input_ids[0, positions_to_mask] = mask_token_id

        # Final result
        final_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if show_steps:
            print("="*80)
            print("FINAL RESULT:")
            print("="*80)
            print(final_text)
            print("="*80 + "\n")

        return final_text

    def _sample_token(self, logits: torch.Tensor) -> int:
        """
        Sample a token from logits using configured strategy.

        Args:
            logits: Unnormalized logits [vocab_size]

        Returns:
            Sampled token ID
        """
        # Apply temperature
        logits = logits / self.config.temperature

        if self.config.sampling_method == "greedy":
            # Deterministic: always pick highest probability
            return logits.argmax().item()

        elif self.config.sampling_method == "topk":
            # Top-k sampling
            top_k = min(self.config.top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_k_indices[sampled_idx].item()

        elif self.config.sampling_method == "nucleus":
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            # Keep at least one token
            sorted_indices_to_remove[0] = False

            # Set logits of removed tokens to -inf
            sorted_logits[sorted_indices_to_remove] = float('-inf')

            # Sample from remaining distribution
            probs = F.softmax(sorted_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return sorted_indices[sampled_idx].item()

        else:
            raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")

    def generate_multiple(
        self,
        prefix: str,
        max_length: int,
        num_samples: int,
        show_steps: bool = False,
    ) -> List[str]:
        """
        Generate multiple samples with the same prefix.

        Args:
            prefix: Starting text
            max_length: Total length
            num_samples: Number of samples to generate
            show_steps: Whether to show steps (only for first sample)

        Returns:
            List of generated texts
        """
        samples = []

        for i in range(num_samples):
            print(f"\n{'='*80}")
            print(f"GENERATING SAMPLE {i+1}/{num_samples}")
            print(f"{'='*80}")

            text = self.generate(
                prefix=prefix,
                max_length=max_length,
                show_steps=(show_steps and i == 0),  # Only show first
            )
            samples.append(text)

            if not show_steps:
                print(f"Sample {i+1}: {text[:100]}...")

        return samples


def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load trained model and tokenizer.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {checkpoint_path}")

    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_path)
    model = RobertaForMaskedLM.from_pretrained(checkpoint_path)

    print(f"Model loaded on: {device}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    return model, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) > 1:
        config = parse_generation_args()
    else:
        print("ERROR: Must provide --checkpoint argument")
        print("\nUsage: python generate.py --checkpoint path/to/checkpoint")
        print("\nExample:")
        print("  python generate.py --checkpoint results/checkpoint-latest \\")
        print("    --prefix 'The quick brown fox' \\")
        print("    --num-samples 5 \\")
        print("    --sampling topk \\")
        print("    --temperature 0.7")
        sys.exit(1)

    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        config.device = "cpu"

    # Load model
    model, tokenizer = load_model(config.checkpoint_path, config.device)

    # Create generator
    generator = DiffusionGenerator(model, tokenizer, config)

    # Generate samples
    print("\n" + "="*80)
    print("GENERATION SETTINGS")
    print("="*80)
    print(f"Prefix: '{config.prefix}'")
    print(f"Max length: {config.max_length}")
    print(f"Num samples: {config.num_samples}")
    print(f"Schedule: {config.schedule_type}")
    print(f"Sampling: {config.sampling_method}")
    print(f"Temperature: {config.temperature}")
    if config.sampling_method == "topk":
        print(f"Top-k: {config.top_k}")
    if config.sampling_method == "nucleus":
        print(f"Top-p: {config.top_p}")
    print("="*80)

    samples = generator.generate_multiple(
        prefix=config.prefix,
        max_length=config.max_length,
        num_samples=config.num_samples,
        show_steps=config.show_steps,
    )

    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  {sample}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
