# Text Diffusion Visualization Guide

## What You've Created

An animated visualization that shows text diffusion in action - watch as text gradually emerges from noise through iterative denoising!

## Generated Files

- **`text_diffusion_animation.gif`** - Animated GIF showing the complete process
- **`visualization_frames/`** - Individual PNG frames (frame_000.png through frame_010.png)

## What the Visualization Shows

### Color Legend

- **Light Blue** - Prefix (your input prompt that stays fixed)
- **Gray [MASK]** - Tokens still masked (unknown)
- **✨ Green** - Newly revealed tokens in this step
- **White** - Previously revealed tokens

### Progress Elements

1. **Title Bar**: Shows current step (e.g., "Step 5/10")
2. **Progress Bar**: Visual indicator of completion (blue bar fills up)
3. **Masked Percentage**: Shows how much text is still masked vs. revealed
4. **Text Area**: The actual text with color-coded tokens

## How to Use

### Basic Usage

```bash
# Activate environment
cd ~/text-diffusion
source venv/bin/activate

# Generate visualization
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Your prompt here" \
  --max-length 50 \
  --sampling topk
```

### Examples

```bash
# Scientific topic
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Quantum computing is" \
  --max-length 60

# Creative writing
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Once upon a time" \
  --temperature 1.0 \
  --max-length 70

# More deterministic (greedy)
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "The future of technology" \
  --sampling greedy \
  --max-length 50
```

### Command-Line Options

- `--checkpoint PATH` - Path to trained model (required)
- `--prefix "TEXT"` - Starting text/prompt (required)
- `--max-length N` - Total length including prefix (default: 64)
- `--sampling METHOD` - Sampling method: greedy, topk, nucleus (default: topk)
- `--temperature T` - Sampling temperature 0.1-2.0 (default: 0.7)
- `--schedule SCHED` - Denoising schedule: linear, cosine, exponential (default: linear)
- `--num-steps N` - Number of denoising steps (default: 10)

## Understanding the Process

### Step-by-Step Breakdown

**Frame 0 (100% masked):**
```
Machine learning is [MASK] [MASK] [MASK] [MASK] [MASK] ...
```
Everything after prefix is unknown.

**Frame 5 (60% masked):**
```
Machine learning is ✨a ✨approach that allows AI [MASK] [MASK] [MASK] ...
```
Some tokens revealed (green), others still masked.

**Frame 10 (0% masked):**
```
Machine learning is a approach that allows AI to apply its knowledge...
```
Complete text revealed!

### What Makes Good Visualizations

**Best Results:**
- Moderate length (40-70 tokens)
- Clear, specific prompts
- Temperature 0.7-1.0 for diversity
- 10-15 denoising steps

**Things to Avoid:**
- Very long sequences (>100 tokens) - gets cluttered
- Temperature too low (<0.3) - repetitive
- Temperature too high (>1.5) - incoherent

## Output Files

### Individual Frames

Located in `visualization_frames/`:
- `frame_000.png` - Initial state (100% masked)
- `frame_001.png` - First denoising step
- ...
- `frame_010.png` - Final result (0% masked)

Each frame is 1200x800 pixels, PNG format.

### Animated GIF

`text_diffusion_animation.gif`:
- All frames combined
- 800ms per frame
- Pauses on first/last frame
- Loops infinitely
- ~200-400KB file size

## Customization

### Change Image Size

Edit the script at line ~30:
```python
visualizer = GenerationVisualizer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    width=1600,    # Change width
    height=1000,   # Change height
    font_size=28,  # Change font size
)
```

### Change Animation Speed

Edit the `create_gif()` call at line ~435:
```python
create_gif(frames, gif_path, duration=500)  # 500ms per frame (faster)
# or
create_gif(frames, gif_path, duration=1200) # 1200ms per frame (slower)
```

### Change Color Scheme

Edit the color definitions at lines ~60-66:
```python
self.bg_color = (15, 15, 35)      # Dark blue background
self.prefix_color = (100, 200, 255)  # Light blue
self.mask_color = (100, 100, 120)    # Gray
self.new_color = (100, 255, 150)     # Green (newly revealed)
self.old_color = (200, 200, 200)     # White (old tokens)
```

## Tips for Sharing

### For Presentations

1. Generate with interesting prompts
2. Use moderate length (50-60 tokens)
3. Consider slower animation (1000ms per frame)
4. Export individual frames if you need slides

### For Social Media

1. Keep it short (40-50 tokens)
2. Use relatable prompts ("The meaning of life", "Why AI")
3. Standard GIF settings work well
4. File size is small enough for Twitter/Reddit

### For Documentation

1. Use technical prompts relevant to your topic
2. Include the command you used
3. Show both GIF and key individual frames
4. Explain what colors mean

## Troubleshooting

### "Font not found" warning
- Script falls back to default font
- Install Monaco (macOS) or DejaVu Sans Mono (Linux)
- Or edit script to specify your font path

### Frames look cluttered
- Reduce `--max-length` (try 40-50)
- Reduce `font_size` in script
- Increase image `width`

### Text wrapping weird
- Adjust `max_chars_per_line` in `create_frame()` method
- Or increase image width

### GIF too large
- Reduce number of steps (--num-steps 5)
- Reduce frame duration
- Reduce image dimensions

### Text not revealing smoothly
- Increase `--num-steps` (try 15-20)
- Try different `--schedule` (cosine often looks smoother)

## Example Outputs

### What You'll See

The visualization clearly shows:

1. **Initial chaos** - All [MASK] tokens
2. **Gradual structure** - Common words appear first
3. **Progressive refinement** - Context fills in
4. **Final polish** - Details and rare words

### Color Evolution

- Frame 0: Mostly gray (all masked)
- Frame 5: Mix of green (new), white (old), gray (still masked)
- Frame 10: Mostly white (all revealed)

## Integration with Training

### After Training New Model

```bash
# Train your model
python train.py --model-name roberta-base --epochs 5

# Visualize its output
python visualize_generation.py \
  --checkpoint results/final-model \
  --prefix "Your domain-specific prompt"
```

### Compare Different Models

Generate visualizations from different checkpoints:

```bash
# Early checkpoint
python visualize_generation.py \
  --checkpoint results-full/checkpoint-1000 \
  --prefix "Machine learning is"

# Late checkpoint
python visualize_generation.py \
  --checkpoint results-full/checkpoint-4000 \
  --prefix "Machine learning is"

# Compare the GIFs side-by-side
```

## Advanced Use Cases

### Batch Generation

Create multiple visualizations:

```bash
for prompt in "AI is" "Science is" "The future"; do
  python visualize_generation.py \
    --checkpoint results-full/final-model \
    --prefix "$prompt" \
    --max-length 50

  mv text_diffusion_animation.gif "viz_${prompt// /_}.gif"
done
```

### Extract Best Frame

If you just want one nice image (not animated):

```bash
# Generate visualization
python visualize_generation.py --checkpoint results-full/final-model --prefix "AI is"

# Use frame 7-8 (middle of process, most interesting)
cp visualization_frames/frame_007.png best_visualization.png
```

## What It Demonstrates

This visualization is perfect for:

- **Understanding diffusion models** - See the iterative refinement process
- **Teaching ML concepts** - Visual explanation of text generation
- **Model debugging** - Identify where generation fails
- **Presentations** - Eye-catching demonstration
- **Research** - Compare different sampling/scheduling strategies

## Next Steps

### Experiment With

1. **Different prompts** - Try creative, scientific, philosophical topics
2. **Sampling strategies** - Compare greedy vs. topk vs. nucleus
3. **Temperatures** - See how it affects diversity
4. **Schedules** - Linear vs. cosine vs. exponential
5. **Model checkpoints** - Early vs. late training

### Create Comparisons

Generate side-by-side visualizations showing:
- Same prompt, different temperatures
- Same prompt, different sampling methods
- Same prompt, different schedules
- Same prompt, different model checkpoints

### Share Your Findings

The visualization makes it easy to share:
- "Look how text emerges from noise!"
- "Compare greedy (boring) vs. stochastic (creative) sampling"
- "See how cosine schedule focuses more on endpoints"

---

**Have fun visualizing text diffusion!** 🎨✨

The GIF is located at: `text_diffusion_animation.gif`
