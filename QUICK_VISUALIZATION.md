# Quick Visualization Reference

## 🎬 One-Line Commands

### View Existing Animation
```bash
open view_animation.html
```

### Create New Animation
```bash
python visualize_generation.py --checkpoint results-full/final-model --prefix "Your text here"
```

### On Remote Server (like nigel)
```bash
# Start server
python3 -m http.server 8000

# Then visit in browser:
# http://nigel.birs.ca:8000/view_animation.html
```

## 🎨 Common Presets

### Creative Writing
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Once upon a time" \
  --temperature 1.2 \
  --sampling nucleus
```

### Technical/Scientific
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Quantum computing is" \
  --temperature 0.7 \
  --sampling topk
```

### Deterministic/Repeatable
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "Machine learning is" \
  --sampling greedy \
  --temperature 0.5
```

### Smooth Animation (Cosine)
```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "The future of AI" \
  --schedule cosine
```

## 🎯 What Each Color Means

- 🔵 **Light Blue** = Your input prefix (stays fixed)
- ⚪ **White** = Previously revealed tokens
- ✨ **Green** = Newly revealed this step
- ⚫ **Gray** = [MASK] tokens (not yet revealed)

## 📊 Progress Indicators

- **Progress Bar** (top) = How far through denoising (0-100%)
- **Masked %** = Percentage of tokens still masked
- **Revealed %** = Percentage of tokens now visible

## ⌨️ Keyboard Shortcuts (in HTML viewer)

- **R** = Replay animation
- **F** = Toggle frames view

## 📁 Output Files

Every visualization creates:
- `text_diffusion_animation.gif` - The animation
- `visualization_frames/frame_000.png` through `frame_010.png` - Individual steps

## 🔧 Quick Customization

### Longer/Shorter Text
```bash
--max-length 40   # Shorter (faster)
--max-length 80   # Longer (more text)
```

### More/Fewer Steps
```bash
--num-steps 5     # Faster, less smooth
--num-steps 20    # Slower, very smooth
```

### Temperature Control
```bash
--temperature 0.3   # Conservative (less creative)
--temperature 1.0   # Balanced
--temperature 1.5   # Creative (more chaotic)
```

## 🚀 Full Command Template

```bash
python visualize_generation.py \
  --checkpoint results-full/final-model \
  --prefix "YOUR PROMPT" \
  --max-length 50 \
  --num-steps 10 \
  --sampling [greedy|topk|nucleus] \
  --temperature 0.8 \
  --schedule [linear|cosine|exponential]
```

## 📚 More Info

- **Complete Guide**: `VISUALIZATION_GUIDE.md`
- **Main README**: `README.md`
- **Deep Dive**: `LEARNING_GUIDE.md`

---

**Pro Tip:** Start with defaults, then experiment with temperature and sampling to see how they affect generation!
