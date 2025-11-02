#!/usr/bin/env python3
"""
Create an animated visualization of text diffusion generation.

Shows the iterative denoising process as text gradually emerges from noise.
"""

import sys
from typing import List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from config import GenerationConfig, parse_generation_args


class GenerationVisualizer:
    """
    Visualize text diffusion generation step-by-step.
    Creates frames showing masked tokens gradually being revealed.
    """

    def __init__(
        self,
        model: RobertaForMaskedLM,
        tokenizer: RobertaTokenizerFast,
        config: GenerationConfig,
        width: int = 1200,
        height: int = 800,
        font_size: int = 24,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

        self.model.to(self.device)
        self.model.eval()

        # Image settings
        self.width = width
        self.height = height
        self.font_size = font_size

        # Try to load a nice monospace font
        try:
            self.font = ImageFont.truetype("/Library/Fonts/Monaco.dfont", font_size)
        except:
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            except:
                print("⚠️  Could not load monospace font, using default")
                self.font = ImageFont.load_default()

        # Get masking schedule
        self.mask_schedule = config.get_mask_schedule()

        # Color scheme
        self.bg_color = (15, 15, 35)  # Dark blue background
        self.prefix_color = (100, 200, 255)  # Light blue for prefix
        self.mask_color = (100, 100, 120)  # Gray for masked tokens
        self.new_color = (100, 255, 150)  # Green for newly revealed tokens
        self.old_color = (200, 200, 200)  # White for previously revealed tokens
        self.text_color = (220, 220, 220)  # Light gray for general text

    def wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max_width characters."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length <= max_width:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def create_frame(
        self,
        step: int,
        total_steps: int,
        tokens: List[str],
        mask_positions: set,
        newly_revealed: set,
        prefix_length: int,
    ) -> Image.Image:
        """Create a single frame showing the current generation state."""
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Title
        title = f"Text Diffusion - Step {step}/{total_steps}"
        mask_pct = len(mask_positions) / (len(tokens) - prefix_length) * 100 if len(tokens) > prefix_length else 0
        subtitle = f"Masked: {mask_pct:.0f}% | Revealed: {100-mask_pct:.0f}%"

        draw.text((20, 20), title, fill=self.text_color, font=self.font)
        draw.text((20, 60), subtitle, fill=self.prefix_color, font=self.font)

        # Progress bar
        bar_x, bar_y = 20, 100
        bar_width, bar_height = self.width - 40, 30

        # Background bar
        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
            fill=(40, 40, 60),
            outline=(80, 80, 100),
            width=2,
        )

        # Progress fill
        progress = step / total_steps
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            draw.rectangle(
                [bar_x, bar_y, bar_x + fill_width, bar_y + bar_height],
                fill=(100, 200, 255),
            )

        # Text content
        text_y = 160
        line_height = self.font_size + 10

        # Prepare text with color coding
        text_parts = []
        for i, token in enumerate(tokens):
            # Clean up token (remove Ġ prefix that RoBERTa uses for spaces)
            display_token = token.replace('Ġ', ' ').replace('<s>', '').replace('</s>', '')

            if not display_token:
                continue

            # Determine color based on token state
            if i < prefix_length:
                color = self.prefix_color
                marker = ""
            elif i in mask_positions:
                color = self.mask_color
                display_token = "[MASK]"
                marker = ""
            elif i in newly_revealed:
                color = self.new_color
                marker = "✨"
            else:
                color = self.old_color
                marker = ""

            text_parts.append((display_token, color, marker))

        # Render text with wrapping
        current_text = ""
        current_line_parts = []
        max_chars_per_line = 50

        y = text_y
        x = 20

        for display_token, color, marker in text_parts:
            # Simple wrapping
            if len(current_text) + len(display_token) > max_chars_per_line:
                # Draw current line
                for token, token_color, token_marker in current_line_parts:
                    text_to_draw = token_marker + token
                    draw.text((x, y), text_to_draw, fill=token_color, font=self.font)
                    # Estimate width (rough approximation)
                    x += len(text_to_draw) * (self.font_size // 2)

                # New line
                y += line_height
                x = 20
                current_text = ""
                current_line_parts = []

                # Stop if we're running out of space
                if y > self.height - 100:
                    break

            current_text += display_token
            current_line_parts.append((display_token, color, marker))

        # Draw remaining line
        if current_line_parts and y < self.height - 100:
            for token, token_color, token_marker in current_line_parts:
                text_to_draw = token_marker + token
                draw.text((x, y), text_to_draw, fill=token_color, font=self.font)
                x += len(text_to_draw) * (self.font_size // 2)

        # Legend
        legend_y = self.height - 80
        draw.text((20, legend_y), "Legend:", fill=self.text_color, font=self.font)
        draw.text((120, legend_y), "Prefix", fill=self.prefix_color, font=self.font)
        draw.text((250, legend_y), "[MASK]", fill=self.mask_color, font=self.font)
        draw.text((380, legend_y), "✨ New", fill=self.new_color, font=self.font)
        draw.text((510, legend_y), "Revealed", fill=self.old_color, font=self.font)

        return img

    def generate_with_visualization(
        self,
        prefix: str,
        max_length: int,
        output_dir: str = "visualization_frames",
    ) -> Tuple[str, List[Image.Image]]:
        """
        Generate text and create visualization frames.

        Returns:
            (generated_text, frames)
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print("GENERATING WITH VISUALIZATION")
        print(f"{'='*80}")
        print(f"Prefix: {prefix}")
        print(f"Output: {output_dir}/")
        print(f"{'='*80}\n")

        # Tokenize prefix
        prefix_ids = self.tokenizer.encode(
            prefix,
            add_special_tokens=True,
            return_tensors="pt"
        )[0].to(self.device)

        prefix_length = len(prefix_ids)

        # Initialize with prefix + masks
        total_length = max_length
        num_to_generate = total_length - prefix_length

        # Start fully masked
        mask_id = self.tokenizer.mask_token_id
        current_ids = torch.cat([
            prefix_ids,
            torch.full((num_to_generate,), mask_id, device=self.device)
        ])

        frames = []
        all_mask_positions = set(range(prefix_length, total_length))
        previous_revealed = set()

        # Create initial frame (all masked)
        tokens = [self.tokenizer.decode([tid]) for tid in current_ids]
        frame = self.create_frame(
            step=0,
            total_steps=len(self.mask_schedule),
            tokens=tokens,
            mask_positions=all_mask_positions,
            newly_revealed=set(),
            prefix_length=prefix_length,
        )
        frames.append(frame)
        frame.save(output_path / f"frame_000.png")
        print(f"✓ Frame 0: 100% masked")

        # Iterative denoising
        for step_idx, mask_ratio in enumerate(self.mask_schedule, 1):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(current_ids.unsqueeze(0))
                logits = outputs.logits[0]

            # Sample predictions for masked positions
            for pos in list(all_mask_positions):
                pos_logits = logits[pos]

                # Apply temperature and sampling
                if self.config.temperature != 1.0:
                    pos_logits = pos_logits / self.config.temperature

                if self.config.sampling_method == "greedy":
                    pred_id = pos_logits.argmax()
                elif self.config.sampling_method == "topk":
                    top_k = min(self.config.top_k, pos_logits.size(-1))
                    indices_to_remove = pos_logits < torch.topk(pos_logits, top_k)[0][..., -1, None]
                    pos_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(pos_logits, dim=-1)
                    pred_id = torch.multinomial(probs, 1)[0]
                elif self.config.sampling_method == "nucleus":
                    sorted_logits, sorted_indices = torch.sort(pos_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    pos_logits[indices_to_remove] = float('-inf')
                    probs = F.softmax(pos_logits, dim=-1)
                    pred_id = torch.multinomial(probs, 1)[0]
                else:
                    pred_id = pos_logits.argmax()

                current_ids[pos] = pred_id

            # Determine what's still masked
            num_to_mask = int(num_to_generate * mask_ratio)

            if num_to_mask > 0 and step_idx < len(self.mask_schedule):
                # Get positions that can be masked (everything except prefix)
                maskable_positions = list(range(prefix_length, total_length))

                # Randomly select positions to re-mask
                import random
                random.shuffle(maskable_positions)
                positions_to_mask = set(maskable_positions[:num_to_mask])

                # Mask them
                for pos in positions_to_mask:
                    current_ids[pos] = mask_id

                newly_revealed = all_mask_positions - positions_to_mask - previous_revealed
                all_mask_positions = positions_to_mask
            else:
                # Final step - reveal everything
                newly_revealed = all_mask_positions - previous_revealed
                all_mask_positions = set()

            # Create frame
            tokens = [self.tokenizer.decode([tid]) for tid in current_ids]
            frame = self.create_frame(
                step=step_idx,
                total_steps=len(self.mask_schedule),
                tokens=tokens,
                mask_positions=all_mask_positions,
                newly_revealed=newly_revealed,
                prefix_length=prefix_length,
            )
            frames.append(frame)
            frame.save(output_path / f"frame_{step_idx:03d}.png")

            mask_pct = len(all_mask_positions) / num_to_generate * 100 if num_to_generate > 0 else 0
            print(f"✓ Frame {step_idx}: {mask_pct:.0f}% masked, {len(newly_revealed)} newly revealed")

            previous_revealed.update(newly_revealed)

        # Generate final text
        final_text = self.tokenizer.decode(current_ids, skip_special_tokens=True)

        print(f"\n{'='*80}")
        print("FINAL TEXT:")
        print(f"{'='*80}")
        print(final_text)
        print(f"{'='*80}\n")

        return final_text, frames


def create_gif(frames: List[Image.Image], output_path: str, duration: int = 800):
    """Create an animated GIF from frames."""
    if not frames:
        print("⚠️  No frames to create GIF")
        return

    print(f"\nCreating animated GIF: {output_path}")

    # Save as GIF with the first frame repeated at start and end for better looping
    frames_with_pause = (
        [frames[0]] * 2 +  # Pause on first frame
        frames[1:] +        # All frames
        [frames[-1]] * 3    # Pause on last frame
    )

    frames_with_pause[0].save(
        output_path,
        save_all=True,
        append_images=frames_with_pause[1:],
        duration=duration,
        loop=0,
    )

    print(f"✓ GIF saved: {output_path}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {duration}ms per frame")


def main():
    """Main visualization script."""
    config = parse_generation_args()

    print("\n" + "="*80)
    print("TEXT DIFFUSION VISUALIZATION")
    print("="*80)

    # Load model
    print(f"\nLoading model from: {config.checkpoint_path}")

    try:
        model = RobertaForMaskedLM.from_pretrained(config.checkpoint_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(config.checkpoint_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTrying to load from base model...")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        print("⚠️  Using base model (not trained on your data)")

    print(f"Model loaded on: {config.device}")

    # Create visualizer
    visualizer = GenerationVisualizer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        width=1200,
        height=800,
        font_size=22,
    )

    # Generate with visualization
    output_dir = "visualization_frames"
    final_text, frames = visualizer.generate_with_visualization(
        prefix=config.prefix,
        max_length=config.max_length,
        output_dir=output_dir,
    )

    # Create animated GIF
    gif_path = "text_diffusion_animation.gif"
    create_gif(frames, gif_path, duration=800)

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Frames saved to: {output_dir}/")
    print(f"Animated GIF: {gif_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
