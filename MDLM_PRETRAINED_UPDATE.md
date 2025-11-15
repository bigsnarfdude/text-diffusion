# MDLM Pretrained Model Update

## Date: November 4, 2025

## Problem Identified

Initial MDLM training failed with **51% accuracy** (random guessing):
- Trained from **scratch** (random initialization)
- Only 19K IMDB reviews (~10M tokens)
- Cannot learn language + sentiment from scratch
- 169M parameters need billions of tokens to converge

## Solution: Use Pretrained MDLM

### Pretrained Model Selected
**Model**: `kuleshov-group/mdlm-owt-noeos`

**Specifications**:
- Parameters: 200M (0.2B)
- Pretrained on: OpenWebText (33 billion tokens)
- Training steps: 1 million
- Architecture: Diffusion Transformer (DiT)
- Special feature: No BOS/EOS tokens (better for variable-length text)

**Why This Model**:
1. ✅ Larger than mdlm-owt (200M vs 130M)
2. ✅ No EOS tokens = handles variable-length IMDB reviews naturally
3. ✅ Fits in 16.4GB VRAM (~5-6GB with training overhead)
4. ✅ Pretrained on 33B tokens = already knows language
5. ✅ Just needs fine-tuning on sentiment

### Code Changes

Updated `train_mdlm_classifier.py` (lines 264-282):

```python
# Initialize MDLM model from pretrained checkpoint
print("\n[6/6] Initializing MDLM model from pretrained checkpoint...")
pretrained_model = 'kuleshov-group/mdlm-owt-noeos'
print(f"Loading pretrained weights from {pretrained_model}...")

# First create a fresh model with our config
model = Diffusion(config, tokenizer=tokenizer)

# Load pretrained weights from HuggingFace
import transformers
state_dict = transformers.AutoModelForMaskedLM.from_pretrained(
    pretrained_model,
    trust_remote_code=True
).state_dict()

# Load weights into our model
model.load_state_dict(state_dict, strict=False)

print(f"✅ Pretrained model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
```

**Pattern from**: https://github.com/kuleshov-group/bd3lms/blob/main/main.py

### Expected Results

**Current Baselines**:
- GPT-2 Native (pretrained + fine-tuned): 90.1% ✅
- Simple Diffusion (GPT-2 NLL): 88.5% ✅
- MDLM from scratch: 51.0% ❌

**Expected with Pretrained MDLM**:
- Target: **85-90%** accuracy
- Competitive with GPT-2 approaches
- Validates true discrete diffusion for classification

### Why This Should Work

**Analogy**:
```
Random MDLM + 19K reviews = Failed (51%)
Pretrained GPT-2 + 19K reviews = Success (90.1%)
Pretrained MDLM + 19K reviews = Expected success (85-90%)
```

**Key Insight**:
- Both GPT-2 and MDLM need pretraining on billions of tokens
- Fine-tuning works when base model already knows language
- Original failure was using random initialization

### Next Steps

1. **Test pretrained loading** (quick sanity check)
2. **Train Class 0** (negative sentiment) with pretrained MDLM
3. **Train Class 1** (positive sentiment) with pretrained MDLM
4. **Evaluate** on test set (1,000 samples)
5. **Compare** with baselines

### Resources

- **HuggingFace Model**: https://huggingface.co/kuleshov-group/mdlm-owt-noeos
- **BD3-LMs Repo**: https://github.com/kuleshov-group/bd3lms
- **MDLM Collection**: https://huggingface.co/collections/kuleshov-group/mdlm-6671bee1cc71f0dce4f2d00a
- **Paper**: https://arxiv.org/abs/2406.07524 (NeurIPS 2024)

### Technical Details

**VRAM Usage**:
- Model weights (BF16): ~400 MB
- Activations (batch=8, seq=512): ~2-3 GB
- Optimizer states: ~1.6 GB
- **Total**: ~5-6 GB (plenty of room on 16.4GB GPU)

**Training Configuration**:
- Batch size: 8 (effective 32 with grad accumulation)
- Learning rate: 3e-4
- Max steps: 20,000 per class
- Precision: BF16 mixed precision
- Sequence length: 512 tokens

---

**Status**: Ready to deploy and test
**Updated by**: Claude Code
**Location**: nigel.birs.ca:~/text-diffusion/
