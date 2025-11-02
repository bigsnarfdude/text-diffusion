#!/usr/bin/env python3
"""
Compare RoBERTa Diffusion vs GPT-2 Autoregressive Generation

This script generates text with both models and creates a side-by-side
visualization showing the different generation paradigms:
- RoBERTa Diffusion: Iterative refinement (all positions simultaneously)
- GPT-2: Autoregressive (left-to-right, one token at a time)
"""

import argparse
import sys
import os
import time
from typing import List, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import (
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
)
from PIL import Image, ImageDraw, ImageFont

from src.config import GenerationConfig


class ModelComparison:
    """
    Compare RoBERTa Diffusion and GPT-2 generation side-by-side.
    """

    def __init__(
        self,
        roberta_model: RobertaForMaskedLM,
        roberta_tokenizer: RobertaTokenizerFast,
        gpt2_model: GPT2LMHeadModel,
        gpt2_tokenizer: GPT2TokenizerFast,
        config: GenerationConfig,
        width: int = 1400,
        height: int = 1000,
        font_size: int = 20,
    ):
        self.roberta_model = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.gpt2_model = gpt2_model
        self.gpt2_tokenizer = gpt2_tokenizer
        self.config = config
        self.device = config.device

        self.roberta_model.to(self.device).eval()
        self.gpt2_model.to(self.device).eval()

        self.width = width
        self.height = height
        self.font_size = font_size

        # Load font
        try:
            self.font = ImageFont.truetype("/Library/Fonts/Monaco.dfont", font_size)
            self.title_font = ImageFont.truetype("/Library/Fonts/Monaco.dfont", font_size + 4)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
                self.title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size + 4)
            except:
                print("⚠️  Could not load monospace font, using default")
                self.font = ImageFont.load_default()
                self.title_font = ImageFont.load_default()

        # Color scheme
        self.bg_color = (15, 15, 35)
        self.roberta_color = (100, 200, 255)  # Light blue
        self.gpt2_color = (255, 150, 100)     # Light orange
        self.text_color = (220, 220, 220)
        self.mask_color = (100, 100, 120)
        self.new_color = (100, 255, 150)
        self.divider_color = (60, 60, 80)

    def generate_roberta_diffusion(
        self,
        prefix: str,
        max_length: int,
    ) -> Tuple[List[str], float]:
        """
        Generate text with RoBERTa diffusion and capture snapshots.

        Returns:
            (snapshots, generation_time)
        """
        print(f"\n{'='*80}")
        print("ROBERTA DIFFUSION GENERATION")
        print(f"{'='*80}\n")

        # Tokenize prefix
        prefix_ids = self.roberta_tokenizer.encode(
            prefix,
            add_special_tokens=True,
            return_tensors="pt"
        )[0].to(self.device)

        prefix_length = len(prefix_ids)
        num_to_generate = max_length - prefix_length
        mask_token_id = self.roberta_tokenizer.mask_token_id

        # Initialize
        input_ids = torch.cat([
            prefix_ids,
            torch.full((num_to_generate,), mask_token_id, device=self.device)
        ]).unsqueeze(0)

        # Get schedule
        mask_schedule = self.config.get_mask_schedule()
        snapshots = []

        # Initial snapshot
        snapshots.append(self.roberta_tokenizer.decode(input_ids[0], skip_special_tokens=False))

        t0 = time.time()

        # Iterative denoising
        for step, mask_prob in enumerate(mask_schedule, 1):
            with torch.no_grad():
                outputs = self.roberta_model(input_ids)
                logits = outputs.logits

            # Find masked positions
            mask_positions = (input_ids[0, prefix_length:] == mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions = mask_positions + prefix_length

            if len(mask_positions) == 0:
                break

            # Sample tokens
            for pos in mask_positions:
                pos = pos.item()
                token_logits = logits[0, pos]

                # Apply temperature
                if self.config.temperature != 1.0:
                    token_logits = token_logits / self.config.temperature

                # Top-k sampling
                top_k = min(self.config.top_k, token_logits.size(-1))
                indices_to_remove = token_logits < torch.topk(token_logits, top_k)[0][..., -1, None]
                token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(token_logits, dim=-1)
                sampled_token = torch.multinomial(probs, 1).item()
                input_ids[0, pos] = sampled_token

            # Save snapshot
            snapshots.append(self.roberta_tokenizer.decode(input_ids[0], skip_special_tokens=False))

            # Re-mask for next iteration
            if step < len(mask_schedule):
                next_mask_prob = mask_schedule[step] if step < len(mask_schedule) else 0.0
                n_to_mask = int(num_to_generate * next_mask_prob)

                if n_to_mask > 0:
                    maskable_positions = torch.arange(prefix_length, max_length, device=self.device)
                    indices = torch.randperm(len(maskable_positions))[:n_to_mask]
                    positions_to_mask = maskable_positions[indices]
                    input_ids[0, positions_to_mask] = mask_token_id

        elapsed = time.time() - t0

        final_text = self.roberta_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"✓ Generated in {elapsed:.2f}s over {len(snapshots)} steps")
        print(f"  Result: {final_text[:100]}...")

        return snapshots, elapsed

    def generate_gpt2_autoregressive(
        self,
        prefix: str,
        max_length: int,
    ) -> Tuple[List[str], float]:
        """
        Generate text with GPT-2 and capture token-by-token snapshots.

        Returns:
            (snapshots, generation_time)
        """
        print(f"\n{'='*80}")
        print("GPT-2 AUTOREGRESSIVE GENERATION")
        print(f"{'='*80}\n")

        # Tokenize prefix
        encoding = self.gpt2_tokenizer(
            prefix,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        t0 = time.time()

        # Generate
        output_ids = self.gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            pad_token_id=self.gpt2_tokenizer.eos_token_id,
        )

        elapsed = time.time() - t0

        # Create token-by-token snapshots
        full_ids = output_ids[0]
        L = full_ids.size(0)

        snapshots = []
        # Start with prefix
        prefix_len = input_ids.size(1)
        snapshots.append(self.gpt2_tokenizer.decode(full_ids[:prefix_len], skip_special_tokens=True))

        # Add one token at a time
        for i in range(prefix_len + 1, L + 1):
            snapshots.append(self.gpt2_tokenizer.decode(full_ids[:i], skip_special_tokens=True))

        final_text = self.gpt2_tokenizer.decode(full_ids, skip_special_tokens=True)
        print(f"✓ Generated in {elapsed:.2f}s over {len(snapshots)} tokens")
        print(f"  Result: {final_text[:100]}...")

        return snapshots, elapsed

    def create_comparison_frame(
        self,
        frame_num: int,
        total_frames: int,
        roberta_text: str,
        gpt2_text: str,
        roberta_total_time: float,
        gpt2_total_time: float,
        roberta_steps: int,
        gpt2_steps: int,
        roberta_current_step: int,
        gpt2_current_step: int,
    ) -> Image.Image:
        """Create a single comparison frame."""
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Main title
        title = "Text Generation Comparison"
        draw.text((20, 20), title, fill=self.text_color, font=self.title_font)

        # Frame number
        draw.text((self.width - 200, 20), f"Frame {frame_num}/{total_frames}", fill=self.text_color, font=self.font)

        # Divider line
        mid_y = self.height // 2
        draw.line([(0, mid_y), (self.width, mid_y)], fill=self.divider_color, width=3)

        # RoBERTa section (top half)
        y_offset = 80
        draw.text((20, y_offset), "RoBERTa Diffusion (Iterative Refinement)", fill=self.roberta_color, font=self.title_font)

        # Calculate elapsed time based on current step
        roberta_elapsed = (roberta_current_step / roberta_steps) * roberta_total_time if roberta_steps > 0 else roberta_total_time
        draw.text((20, y_offset + 30), f"Step {roberta_current_step}/{roberta_steps} | Time: {roberta_elapsed:.2f}s / {roberta_total_time:.2f}s", fill=self.text_color, font=self.font)

        # RoBERTa text (wrap and display)
        text_y = y_offset + 70
        self._draw_wrapped_text(draw, roberta_text, 20, text_y, mid_y - 20, is_roberta=True)

        # GPT-2 section (bottom half)
        y_offset = mid_y + 20
        draw.text((20, y_offset), "GPT-2 (Autoregressive, Left-to-Right)", fill=self.gpt2_color, font=self.title_font)

        # Calculate elapsed time based on current token
        gpt2_elapsed = (gpt2_current_step / gpt2_steps) * gpt2_total_time if gpt2_steps > 0 else gpt2_total_time
        draw.text((20, y_offset + 30), f"Token {gpt2_current_step}/{gpt2_steps} | Time: {gpt2_elapsed:.2f}s / {gpt2_total_time:.2f}s", fill=self.text_color, font=self.font)

        # GPT-2 text
        text_y = y_offset + 70
        self._draw_wrapped_text(draw, gpt2_text, 20, text_y, self.height - 20, is_roberta=False)

        return img

    def _draw_wrapped_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int,
        y: int,
        max_y: int,
        is_roberta: bool,
    ):
        """Draw text with wrapping, handling special tokens."""
        # Clean up text
        if is_roberta:
            # Replace mask tokens with placeholder
            text = text.replace('<s>', '').replace('</s>', '')
            text = text.replace('<mask>', '[MASK]')
            # RoBERTa uses Ġ for spaces
            text = text.replace('Ġ', ' ')

        # Word wrap
        words = text.split()
        lines = []
        current_line = []
        max_chars = 70

        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Draw lines
        line_height = self.font_size + 8
        for i, line in enumerate(lines):
            if y + i * line_height > max_y:
                break

            # Color masked tokens differently
            if '[MASK]' in line and is_roberta:
                # Draw word by word with different colors
                x_offset = x
                for word in line.split():
                    if word == '[MASK]':
                        color = self.mask_color
                    else:
                        color = self.text_color
                    draw.text((x_offset, y + i * line_height), word + ' ', fill=color, font=self.font)
                    # Rough approximation of word width
                    x_offset += len(word + ' ') * (self.font_size // 2)
            else:
                color = self.text_color
                draw.text((x, y + i * line_height), line, fill=color, font=self.font)

    def create_comparison_animation(
        self,
        prompt: str,
        max_length: int,
        output_path: str = "comparison_animation.gif",
    ):
        """
        Generate with both models and create comparison animation.
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"Max length: {max_length}")
        print(f"Output: {output_path}")
        print(f"{'='*80}\n")

        # Generate with both models
        roberta_snaps, roberta_time = self.generate_roberta_diffusion(prompt, max_length)
        gpt2_snaps, gpt2_time = self.generate_gpt2_autoregressive(prompt, max_length)

        # Create frames
        print(f"\n{'='*80}")
        print("CREATING COMPARISON FRAMES")
        print(f"{'='*80}\n")

        frames = []
        max_frames = max(len(roberta_snaps), len(gpt2_snaps))

        for i in range(max_frames):
            # Get text for this frame (use last if exceeded)
            roberta_idx = min(i, len(roberta_snaps) - 1)
            gpt2_idx = min(i, len(gpt2_snaps) - 1)

            roberta_text = roberta_snaps[roberta_idx]
            gpt2_text = gpt2_snaps[gpt2_idx]

            frame = self.create_comparison_frame(
                frame_num=i + 1,
                total_frames=max_frames,
                roberta_text=roberta_text,
                gpt2_text=gpt2_text,
                roberta_total_time=roberta_time,
                gpt2_total_time=gpt2_time,
                roberta_steps=len(roberta_snaps),
                gpt2_steps=len(gpt2_snaps),
                roberta_current_step=roberta_idx,
                gpt2_current_step=gpt2_idx,
            )
            frames.append(frame)

            if (i + 1) % 5 == 0 or i == max_frames - 1:
                print(f"✓ Created frame {i + 1}/{max_frames}")

        # Save as GIF
        print(f"\n{'='*80}")
        print("SAVING ANIMATION")
        print(f"{'='*80}\n")

        # Add pauses at beginning and end
        frames_with_pause = (
            [frames[0]] * 3 +  # Pause at start
            frames +            # All frames
            [frames[-1]] * 5    # Pause at end
        )

        frames_with_pause[0].save(
            output_path,
            save_all=True,
            append_images=frames_with_pause[1:],
            duration=800,  # 800ms per frame
            loop=0,
        )

        print(f"✓ Saved: {output_path}")
        print(f"  Total frames: {len(frames)}")
        print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

        # Also save individual frames
        frames_dir = Path("comparison_frames")
        frames_dir.mkdir(exist_ok=True)

        for i, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{i:03d}.png")

        print(f"  Individual frames: {frames_dir}/")

        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"RoBERTa Diffusion:")
        print(f"  Time: {roberta_time:.2f}s")
        print(f"  Steps: {len(roberta_snaps)}")
        print(f"  Method: Iterative refinement (all positions)")
        print()
        print(f"GPT-2:")
        print(f"  Time: {gpt2_time:.2f}s")
        print(f"  Tokens: {len(gpt2_snaps)}")
        print(f"  Method: Autoregressive (left-to-right)")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RoBERTa Diffusion vs GPT-2 generation"
    )

    # Model checkpoints
    parser.add_argument(
        "--roberta-checkpoint",
        required=True,
        help="Path to RoBERTa diffusion checkpoint"
    )
    parser.add_argument(
        "--gpt2-checkpoint",
        default="gpt2",
        help="GPT-2 checkpoint (default: gpt2)"
    )

    # Generation settings
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for both models"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length"
    )

    # Diffusion settings
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of denoising steps for RoBERTa"
    )
    parser.add_argument(
        "--schedule",
        choices=["linear", "cosine", "exponential"],
        default="linear",
        help="Denoising schedule"
    )

    # Sampling settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus (top-p) sampling"
    )

    # Output
    parser.add_argument(
        "--output",
        default="comparison_animation.gif",
        help="Output GIF path"
    )

    # Device
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROBERTA DIFFUSION VS GPT-2 COMPARISON")
    print("="*80)

    # Load RoBERTa
    print(f"\nLoading RoBERTa from: {args.roberta_checkpoint}")
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_checkpoint)
    roberta_model = RobertaForMaskedLM.from_pretrained(args.roberta_checkpoint)

    # Load GPT-2
    print(f"Loading GPT-2 from: {args.gpt2_checkpoint}")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_checkpoint)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(args.gpt2_checkpoint)

    print(f"Device: {args.device}")

    # Create config
    config = GenerationConfig(
        checkpoint_path=args.roberta_checkpoint,
        device=args.device,
        prefix=args.prompt,
        max_length=args.max_length,
        num_steps=args.num_steps,
        schedule_type=args.schedule,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sampling_method="topk",
    )

    # Create comparison
    comparison = ModelComparison(
        roberta_model=roberta_model,
        roberta_tokenizer=roberta_tokenizer,
        gpt2_model=gpt2_model,
        gpt2_tokenizer=gpt2_tokenizer,
        config=config,
    )

    # Generate comparison animation
    comparison.create_comparison_animation(
        prompt=args.prompt,
        max_length=args.max_length,
        output_path=args.output,
    )

    print(f"\n✅ Comparison complete! View: {args.output}\n")


if __name__ == "__main__":
    main()
