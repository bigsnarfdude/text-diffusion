#!/usr/bin/env python3
"""
Confidence-Based Text Generation with RoBERTa Diffusion

Inspired by tiny-diffusion's parallel decoding approach:
- Only unmask tokens with high prediction confidence
- Progressive refinement (easy tokens first, hard tokens later)
- Natural stopping when all tokens are confident
- Quality control via confidence threshold
"""

import sys
import os
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

from src.config import GenerationConfig, parse_generation_args


class ConfidenceBasedGenerator:
    """
    Confidence-based iterative text generator using diffusion-trained RoBERTa.

    Key differences from schedule-based generation:
    1. Selective unmasking: Only unmask high-confidence predictions
    2. Adaptive speed: Fast for easy sequences, slow for hard ones
    3. Quality control: Confidence threshold controls output quality
    4. Natural stopping: Converges when all tokens confident

    Algorithm (from tiny-diffusion):
    -------
    while any tokens masked and steps < max_steps:
        1. Predict all masked positions
        2. Compute confidence = max(softmax(logits)) for each position
        3. Find masked positions where confidence >= threshold
        4. If none qualify, unmask single highest-confidence token
        5. Sample and fill in high-confidence positions
        6. Continue to next iteration
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

    def generate(
        self,
        prefix: str,
        max_length: int,
        confidence_threshold: float = 0.7,
        max_steps: int = 50,
        show_steps: bool = True,
    ) -> str:
        """
        Generate text via confidence-based iterative denoising.

        Args:
            prefix: Starting text (not masked)
            max_length: Total length including prefix
            confidence_threshold: Only unmask tokens with confidence >= this (0.0-1.0)
                                 Higher = slower but higher quality
                                 Lower = faster but potentially lower quality
                                 Recommended: 0.5-0.8
            max_steps: Maximum denoising iterations (safety limit)
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
            print("CONFIDENCE-BASED ITERATIVE DENOISING")
            print("="*80)
            print(f"Prefix: {prefix}")
            print(f"Target length: {max_length} tokens")
            print(f"Confidence threshold: {confidence_threshold:.1%}")
            print(f"Max steps: {max_steps}")
            print(f"Temperature: {self.config.temperature}")
            print(f"Sampling: {self.config.sampling_method}")
            print("="*80 + "\n")

            initial_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"STEP 0 (100% masked):")
            print(f"  {initial_text[:150]}...")
            print()

        # Iterative denoising
        step = 0
        total_unmasked = 0

        while step < max_steps:
            step += 1

            # Find currently masked positions (after prefix)
            mask_positions = (input_ids[0, prefix_length:] == mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions = mask_positions + prefix_length  # Adjust for prefix offset

            if len(mask_positions) == 0:
                if show_steps:
                    print(f"✅ All tokens unmasked! Converged in {step-1} steps.\n")
                break  # All tokens unmasked - done!

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Compute confidence for all masked positions
            # Confidence = max probability after softmax
            probs = F.softmax(logits[0, mask_positions] / self.config.temperature, dim=-1)
            confidence, predicted = probs.max(dim=-1)  # [num_masked]

            # Find high-confidence positions
            high_confidence_mask = confidence >= confidence_threshold
            high_confidence_positions = mask_positions[high_confidence_mask]

            # If no positions meet threshold, force unmask the highest confidence one
            if len(high_confidence_positions) == 0:
                # Find the single highest-confidence masked position
                best_idx = confidence.argmax()
                high_confidence_positions = mask_positions[best_idx].unsqueeze(0)
                high_confidence_mask = torch.zeros_like(confidence, dtype=torch.bool)
                high_confidence_mask[best_idx] = True

                if show_steps:
                    print(f"   ⚠️  No tokens met threshold, forcing unmask of highest "
                          f"confidence: {confidence[best_idx]:.1%}")

            # Sample tokens for high-confidence positions
            num_to_unmask = len(high_confidence_positions)

            for i, pos in enumerate(high_confidence_positions):
                pos = pos.item()
                token_logits = logits[0, pos]

                # Sample token (with temperature and sampling strategy)
                sampled_token = self._sample_token(token_logits)
                input_ids[0, pos] = sampled_token

            total_unmasked += num_to_unmask

            # Show progress
            if show_steps:
                current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                n_masked = (input_ids[0, prefix_length:] == mask_token_id).sum().item()
                mask_pct = (n_masked / num_to_generate) * 100
                avg_conf = confidence[high_confidence_mask].mean().item()

                print(f"STEP {step} ({mask_pct:.0f}% masked):")
                print(f"  Unmasked: {num_to_unmask} tokens (avg confidence: {avg_conf:.1%})")
                print(f"  Remaining: {n_masked} masked tokens")
                print(f"  Text: {current_text[:150]}...")
                print()

        # Check if we hit max steps
        if step >= max_steps:
            n_masked = (input_ids[0, prefix_length:] == mask_token_id).sum().item()
            if n_masked > 0 and show_steps:
                print(f"⚠️  Hit max steps ({max_steps}), {n_masked} tokens still masked")
                print(f"   Consider increasing max_steps or lowering confidence_threshold")
                print()

        # Final result
        final_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if show_steps:
            print("="*80)
            print("FINAL RESULT:")
            print("="*80)
            print(f"Converged in {step} steps")
            print(f"Total tokens unmasked: {total_unmasked}")
            print(f"Average steps per token: {step / total_unmasked:.2f}")
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
        confidence_threshold: float = 0.7,
        max_steps: int = 50,
        show_steps: bool = False,
    ) -> List[str]:
        """
        Generate multiple samples with the same prefix.

        Args:
            prefix: Starting text
            max_length: Total length
            num_samples: Number of samples to generate
            confidence_threshold: Confidence threshold for unmasking
            max_steps: Max denoising iterations per sample
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
                confidence_threshold=confidence_threshold,
                max_steps=max_steps,
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
    # Parse arguments (reuse existing config)
    if len(sys.argv) > 1:
        config = parse_generation_args()
    else:
        print("ERROR: Must provide --checkpoint argument")
        print("\nUsage: python confidence_generate.py --checkpoint path/to/checkpoint")
        print("\nExample:")
        print("  python confidence_generate.py --checkpoint results/checkpoint-latest \\")
        print("    --prefix 'The quick brown fox' \\")
        print("    --num-samples 3 \\")
        print("    --sampling topk \\")
        print("    --temperature 0.7")
        print("\nNew confidence-based parameters:")
        print("  Set via environment variables:")
        print("    CONFIDENCE_THRESHOLD=0.7  # Default: 0.7 (range: 0.0-1.0)")
        print("    MAX_STEPS=50              # Default: 50")
        sys.exit(1)

    # Get confidence parameters from environment (or use defaults)
    confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.7'))
    max_steps = int(os.environ.get('MAX_STEPS', '50'))

    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        config.device = "cpu"

    # Load model
    model, tokenizer = load_model(config.checkpoint_path, config.device)

    # Create generator
    generator = ConfidenceBasedGenerator(model, tokenizer, config)

    # Generate samples
    print("\n" + "="*80)
    print("CONFIDENCE-BASED GENERATION SETTINGS")
    print("="*80)
    print(f"Prefix: '{config.prefix}'")
    print(f"Max length: {config.max_length}")
    print(f"Num samples: {config.num_samples}")
    print(f"Confidence threshold: {confidence_threshold:.1%}")
    print(f"Max steps: {max_steps}")
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
        confidence_threshold=confidence_threshold,
        max_steps=max_steps,
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
