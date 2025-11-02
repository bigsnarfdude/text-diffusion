# Text Diffusion Visualization - What's New

## 🎉 New Features Added

### 1. Animated GIF Generation (`visualize_generation.py`)

A complete visualization tool that creates animated GIFs showing the iterative denoising process:

**What it does:**
- Creates 11 frames (Step 0-10) showing text gradually emerging from [MASK] tokens
- Color-codes tokens: Blue (prefix), Gray (masked), Green (newly revealed), White (previously revealed)
- Shows progress bar and percentage of masked vs. revealed tokens
- Outputs both individual PNG frames and animated GIF

**Usage:**
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Your prompt here" \
  --max-length 50 \
  --sampling topk \
  --temperature 0.8
```

**Output:**
- `text_diffusion_animation.gif` - Animated GIF (800ms per frame)
- `visualization_frames/frame_000.png` through `frame_010.png`

### 2. Interactive HTML Viewer (`view_animation.html`)

A beautiful, modern web interface for viewing the animation:

**Features:**
- 🎨 Glassmorphism design with gradient background
- 🔄 Replay animation button
- 🖼️ View individual frames in a grid
- 📊 Interactive legend explaining colors
- ⌨️ Keyboard shortcuts (R = replay, F = toggle frames)
- 📱 Responsive design (works on mobile)
- 📥 Download GIF button
- 🔽 Collapsible sections (Technical Details, Generate Your Own)

**How to use:**
```bash
# Local machine
open view_animation.html

# Remote server (like nigel)
python3 -m http.server 8000
# Then visit: http://localhost:8000/view_animation.html
```

### 3. Complete Documentation (`VISUALIZATION_GUIDE.md`)

Comprehensive guide covering:
- How to create visualizations
- Customization options (colors, size, fonts, animation speed)
- Command-line examples
- Troubleshooting
- Advanced use cases (batch generation, comparisons)

### 4. Updated README

The main README now prominently features:
- Embedded GIF at the top showing the process
- New "🎬 See It In Action!" section
- Links to interactive viewer and visualization guide
- Updated Quick Start with visualization commands
- Updated Project Structure showing new files

## 📁 Files Added

```
text-diffusion/
├── visualize_generation.py         # Main visualization script (~450 lines)
├── view_animation.html             # Interactive viewer (~400 lines)
├── text_diffusion_animation.gif    # Example animation (255KB)
├── VISUALIZATION_GUIDE.md          # Complete usage guide (~400 lines)
├── VISUALIZATION_README.md         # This file
└── visualization_frames/           # Individual PNG frames
    ├── frame_000.png               # Step 0 (100% masked)
    ├── frame_001.png               # Step 1 (90% masked)
    ├── ...
    └── frame_010.png               # Step 10 (0% masked, complete)
```

## 🎨 How It Works

### The Visualization Process

1. **Load Model**: Uses your trained RoBERTa diffusion model
2. **Initialize**: Create fully masked sequence (except prefix)
3. **Iterative Denoising**: 10 steps from 100% masked → 0% masked
4. **Frame Creation**: At each step, render a 1200x800 PNG showing:
   - Title with step number
   - Progress bar
   - Masked percentage
   - Color-coded text
   - Legend
5. **GIF Assembly**: Combine frames into animated GIF

### Color Coding

- **Light Blue (RGB: 100, 200, 255)** - Your input prefix (fixed)
- **Gray (RGB: 100, 100, 120)** - [MASK] tokens (unknown)
- **Green ✨ (RGB: 100, 255, 150)** - Newly revealed tokens (this step)
- **White (RGB: 200, 200, 200)** - Previously revealed tokens
- **Dark Blue Background (RGB: 15, 15, 35)** - Professional look

## 🚀 Quick Examples

### Basic Visualization
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Machine learning is"
```

### Creative (High Temperature)
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Once upon a time" \
  --temperature 1.2 \
  --sampling nucleus
```

### Deterministic (Greedy)
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "The future of AI" \
  --sampling greedy
```

### Different Schedule
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Scientists discovered" \
  --schedule cosine
```

## 📊 Example Output

**Generated Text:**
```
Machine learning is a approach that allows AI to apply its
knowledge and data at either a specific point or before.
According applying general computer theories on AI, modern
AI can become and be sufficiently intelligent that to think
beyond its own work.
```

**What the Animation Shows:**

- **Frame 0**: `Machine learning is [MASK] [MASK] [MASK]...` (100% masked)
- **Frame 5**: `Machine learning is ✨a ✨approach that allows AI [MASK]...` (60% masked)
- **Frame 10**: `Machine learning is a approach that allows AI to apply...` (0% masked)

## 🎯 Use Cases

### For Learning
- **Understand diffusion models**: See iterative refinement in action
- **Compare strategies**: Generate multiple animations with different settings
- **Debug generation**: Identify where model gets stuck or fails

### For Presentations
- **Eye-catching demos**: Animated GIF grabs attention
- **Explain concepts**: Visual explanation of text generation
- **Compare approaches**: Side-by-side animations showing different methods

### For Research
- **Analyze behavior**: Step-by-step inspection of generation process
- **Evaluate quality**: See when and how text becomes coherent
- **Compare checkpoints**: Visualize improvement during training

## 🛠️ Customization

### Change Image Size
Edit `visualize_generation.py` line ~430:
```python
visualizer = GenerationVisualizer(
    width=1600,    # Default: 1200
    height=1000,   # Default: 800
    font_size=28,  # Default: 22
)
```

### Change Animation Speed
Edit `visualize_generation.py` line ~435:
```python
create_gif(frames, gif_path, duration=500)  # Faster (default: 800)
```

### Change Colors
Edit `visualize_generation.py` lines ~60-66:
```python
self.prefix_color = (255, 100, 100)  # Red prefix
self.new_color = (255, 255, 100)     # Yellow newly revealed
```

## 📈 Performance

**Generation Time:**
- Model inference: ~5-10 seconds (10 steps on GPU)
- Frame rendering: ~2-3 seconds per frame
- GIF assembly: ~1 second
- **Total: ~30-40 seconds** for complete visualization

**File Sizes:**
- Individual frames: ~50-100KB each (PNG)
- Complete animation: ~200-400KB (GIF)
- All frames (11 total): ~1-2MB

**Requirements:**
- Python packages: torch, transformers, Pillow
- Memory: ~2GB for model + generation
- Disk space: ~5MB per visualization (frames + GIF)

## 🎓 Learning Value

This visualization is incredibly valuable for understanding diffusion models because:

1. **Makes abstract concrete**: See the "denoising" actually happen
2. **Reveals model behavior**: Watch which tokens appear first (common words) vs. last (rare tokens)
3. **Shows progressive refinement**: Not left-to-right like GPT, but simultaneous improvement
4. **Highlights stochasticity**: Different runs produce different intermediate states
5. **Demonstrates schedules**: See how linear vs. cosine schedules differ in practice

## 🌟 What Makes This Special

Unlike typical text generation where you only see:
```
Input: "Machine learning is"
Output: "Machine learning is a field of AI..."
```

You now see the **entire journey**:
```
Step 0:  Machine learning is [MASK] [MASK] [MASK] [MASK] ...
Step 1:  Machine learning is [MASK] of [MASK] AI ...
Step 3:  Machine learning is a of AI AI ...
Step 5:  Machine learning is a approach of AI ...
Step 10: Machine learning is a approach that allows AI ...
```

This is **the key insight of diffusion models** - iterative refinement rather than autoregressive generation!

## 🔗 Resources

- **Main README**: Overview and quick start
- **VISUALIZATION_GUIDE.md**: Detailed usage instructions
- **LEARNING_GUIDE.md**: Deep dive into diffusion theory
- **Interactive Viewer**: `view_animation.html` in your browser

## 🎊 Summary

You now have:
- ✅ Animated GIF generation script
- ✅ Beautiful interactive HTML viewer
- ✅ Complete documentation
- ✅ Example animations
- ✅ Updated README showcasing the visualization

**Next steps:**
1. Open `view_animation.html` in Chrome to see the example
2. Generate your own visualization with custom prompts
3. Share the animation to explain diffusion models to others!

---

**Created:** November 2025
**Purpose:** Make text diffusion models tangible and understandable through visualization
**Impact:** Transform abstract algorithms into concrete, shareable demonstrations

Enjoy watching text emerge from noise! 🎨✨
