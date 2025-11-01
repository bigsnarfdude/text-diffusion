"""
Diffusion Data Collator - The Magic of Variable Masking

This is THE KEY COMPONENT that makes diffusion training work.
Unlike standard masked LM (15% masking), we use VARIABLE masking rates
to teach the model the full denoising curve.

Key insight: Same model, different masking levels = different denoising tasks
"""

import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DiffusionDataCollator:
    """
    Data collator for diffusion-based masked language modeling.

    For each batch:
    1. Randomly select a masking probability (10%, 20%, ..., 100%)
    2. Mask that percentage of tokens (except prefix)
    3. Model learns to predict original tokens from masked input

    This trains the model to denoise at ALL corruption levels.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_probs: List[float] = None  # [1.0, 0.9, 0.8, ..., 0.1]
    prefix_length: int = 5  # Never mask first N tokens
    mlm_probability: float = None  # Ignored, computed per-batch
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mask_probs is None:
            self.mask_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        # Verify mask token exists
        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token for diffusion training")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of examples with random masking.

        Args:
            examples: List of dicts with 'input_ids' and optionally 'attention_mask'

        Returns:
            Dict with:
                - input_ids: Token IDs with masks applied
                - attention_mask: Attention mask
                - labels: Original token IDs (targets for prediction)
                - mask_prob: Masking probability used (for logging)
        """
        # Step 1: Standard batching and padding
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors=self.return_tensors,
        )

        # Step 2: Randomly select masking probability for this batch
        # This is the KEY: different batches see different corruption levels
        mask_prob = random.choice(self.mask_probs)

        # Step 3: Prepare labels (original tokens are the targets)
        # Clone input_ids before masking - this is what we're trying to predict
        batch["labels"] = batch["input_ids"].clone()

        # Step 4: Create masking
        input_ids = batch["input_ids"]
        batch_size, seq_length = input_ids.shape

        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, mask_prob)

        # Step 5: Never mask special tokens or padding
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(
                    ids, already_has_special_tokens=True
                )
                for ids in input_ids.tolist()
            ],
            dtype=torch.bool
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Step 6: Never mask prefix tokens (for conditional generation)
        # This allows model to use prefix as context during training
        if self.prefix_length > 0:
            prefix_mask = torch.zeros_like(probability_matrix, dtype=torch.bool)
            prefix_mask[:, :self.prefix_length] = True
            probability_matrix.masked_fill_(prefix_mask, value=0.0)

        # Step 7: Apply masking
        # Random mask based on probability matrix
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Replace masked tokens with [MASK] token
        # (Unlike BERT which does 80% MASK, 10% random, 10% unchanged)
        # We do 100% MASK for simplicity and following the reference implementation
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # Step 8: Set labels to -100 for tokens we're not predicting
        # Only predict the masked tokens (not the preserved prefix or special tokens)
        labels = batch["labels"]
        labels[~masked_indices] = -100

        # Store masking probability for logging
        batch["mask_prob"] = torch.tensor([mask_prob] * batch_size)

        return batch


class VisualizableDiffusionCollator(DiffusionDataCollator):
    """
    Extended collator that can visualize what it's doing.
    Useful for understanding and debugging.
    """

    def visualize_batch(self, examples: List[Dict[str, Any]], num_examples: int = 3):
        """
        Show what the collator does to a batch.

        Args:
            examples: Batch of examples
            num_examples: How many examples to visualize
        """
        print("\n" + "=" * 80)
        print("DIFFUSION COLLATOR VISUALIZATION")
        print("=" * 80)

        # Process batch
        batch = self.__call__(examples)

        # Show settings
        print(f"\nSettings:")
        print(f"  Mask probabilities: {self.mask_probs}")
        print(f"  Prefix length: {self.prefix_length}")
        print(f"  Batch size: {batch['input_ids'].shape[0]}")
        print(f"  Sequence length: {batch['input_ids'].shape[1]}")
        print(f"  Masking probability this batch: {batch['mask_prob'][0].item():.1%}")

        # Show examples
        print(f"\nShowing first {num_examples} examples:")
        print("-" * 80)

        for i in range(min(num_examples, len(examples))):
            print(f"\nExample {i + 1}:")

            # Original text
            original_ids = batch["labels"][i].clone()
            original_ids[original_ids == -100] = self.tokenizer.pad_token_id
            original_text = self.tokenizer.decode(original_ids, skip_special_tokens=True)
            print(f"  ORIGINAL: {original_text[:100]}...")

            # Masked input
            masked_text = self.tokenizer.decode(
                batch["input_ids"][i],
                skip_special_tokens=False
            )
            print(f"  MASKED:   {masked_text[:100]}...")

            # Statistics
            n_masked = (batch["input_ids"][i] == self.tokenizer.mask_token_id).sum().item()
            n_total = (batch["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()
            actual_mask_rate = n_masked / n_total if n_total > 0 else 0
            print(f"  STATS:    {n_masked}/{n_total} masked ({actual_mask_rate:.1%})")

        print("\n" + "=" * 80 + "\n")

        return batch


def test_collator():
    """Test the diffusion collator with dummy data."""
    from transformers import RobertaTokenizerFast

    print("Testing Diffusion Data Collator")
    print("=" * 80)

    # Setup
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Create test examples
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
    ]

    examples = [
        tokenizer(text, truncation=True, max_length=32, padding="max_length")
        for text in texts
    ]

    # Test standard collator
    print("\n1. Standard Collator (random masking):")
    collator = DiffusionDataCollator(tokenizer=tokenizer, prefix_length=3)
    batch = collator(examples)
    print(f"   Batch keys: {batch.keys()}")
    print(f"   Input shape: {batch['input_ids'].shape}")
    print(f"   Mask probability: {batch['mask_prob'][0].item():.1%}")

    # Test visualizable collator
    print("\n2. Visualizable Collator:")
    viz_collator = VisualizableDiffusionCollator(
        tokenizer=tokenizer,
        mask_probs=[0.5],  # Fixed 50% for visualization
        prefix_length=3
    )
    viz_collator.visualize_batch(examples)

    # Test different masking levels
    print("\n3. Testing Different Masking Levels:")
    for mask_prob in [0.1, 0.5, 1.0]:
        test_collator = DiffusionDataCollator(
            tokenizer=tokenizer,
            mask_probs=[mask_prob],
            prefix_length=3
        )
        batch = test_collator(examples)
        n_masked = (batch["input_ids"][0] == tokenizer.mask_token_id).sum().item()
        n_total = (batch["input_ids"][0] != tokenizer.pad_token_id).sum().item()
        actual_rate = n_masked / n_total if n_total > 0 else 0
        print(f"   {mask_prob:.1%} masking → {actual_rate:.1%} actual ({n_masked}/{n_total} tokens)")


if __name__ == "__main__":
    test_collator()
