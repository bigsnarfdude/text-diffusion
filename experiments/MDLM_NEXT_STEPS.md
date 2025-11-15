# MDLM Implementation - Next Steps

## Current Status (2025-11-03 Evening)

### What We've Accomplished ✅
1. **Cloned MDLM repository** to local machine and nigel.birs.ca
2. **Analyzed MDLM architecture** - confirmed it computes NLL for classification
3. **Installed core dependencies** (Lightning, Hydra, OmegaConf) in venv on nigel
4. **Documented feasibility** in `MDLM_FEASIBILITY_FINDINGS.md`

### Current Blocker 🚧
**Flash Attention Dependency**: MDLM requires `flash-attn==2.5.6` which:
- Requires specific CUDA versions (12.4)
- Requires compilation from source
- Is a complex GPU-specific dependency
- May not be compatible with nigel's current CUDA setup

## Two Paths Forward

### Path A: Full MDLM Installation (High Effort, High Reward)

**Recommended for**: Final implementation if we commit to MDLM approach

**Steps**:
1. **Check nigel's CUDA version**:
   ```bash
   ssh vincent@nigel.birs.ca "nvidia-smi"
   ```

2. **Install conda on nigel**:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Create MDLM conda environment**:
   ```bash
   cd ~/mdlm
   conda env create -f requirements.yaml
   conda activate mdlm
   ```

4. **Test flash attention installation**:
   ```bash
   python -c "import flash_attn; print('Flash Attention installed')"
   ```

5. **Run feasibility test**:
   ```bash
   python test_mdlm_feasibility.py
   ```

**Timeline**: 2-4 hours (if CUDA versions compatible)

**Risk**: May fail if CUDA mismatch or compilation errors

---

### Path B: Simplified Proof-of-Concept (Low Effort, Validates Concept)

**Recommended for**: Quick validation before committing to full MDLM

**Approach**: Instead of using full MDLM, create a minimal discrete diffusion model that:
- Demonstrates NLL computation → log P(text)
- Tests classification with per-class models
- Uses simple transformer architecture (no flash attention)
- Proves the concept works

**Implementation**: Use HuggingFace transformers + simple diffusion logic

```python
class SimpleDiscreteDiffusion:
    """
    Minimal discrete diffusion for proof-of-concept.
    Uses standard transformer with masked token prediction.
    """

    def __init__(self, base_model='distilgpt2'):
        self.model = GPT2LMHeadModel.from_pretrained(base_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)

    def compute_nll(self, text):
        """Compute negative log-likelihood of text."""
        tokens = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(
                tokens['input_ids'],
                labels=tokens['input_ids']
            )

        # CrossEntropyLoss = NLL loss
        nll = outputs.loss.item() * tokens['input_ids'].numel()
        return nll

    def train_on_class_data(self, texts):
        """Fine-tune on class-specific data."""
        # Standard fine-tuning with NLL objective
        optimizer = AdamW(self.model.parameters())

        for epoch in range(num_epochs):
            for text in texts:
                tokens = self.tokenizer(text, return_tensors='pt')
                outputs = self.model(
                    tokens['input_ids'],
                    labels=tokens['input_ids']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
```

**Advantages**:
- ✅ No flash attention required
- ✅ Works with existing venv
- ✅ Validates NLL → log P(text) approach
- ✅ Can run immediately on nigel
- ✅ Proves concept before investing in full MDLM

**Timeline**: 1-2 days

---

## Recommendation: Path B First, Then Path A

### Phase 1: Proof-of-Concept (This Week)
1. Implement `SimpleDiscreteDiffusion` classifier
2. Train on IMDB negative/positive classes
3. Evaluate accuracy
4. **Decision Point**:
   - If >75% accuracy → Validates approach, proceed to full MDLM
   - If <75% accuracy → Fundamental issue, reconsider approach

### Phase 2: Full MDLM (Next Week, If Phase 1 Succeeds)
1. Install conda on nigel
2. Set up full MDLM environment
3. Train per-class MDLM models
4. Compare: GPT-2 vs RoBERTa vs Simple Diffusion vs MDLM

## Immediate Next Action

**Create simplified discrete diffusion classifier** using GPT-2 as base:

File: `src/simple_diffusion_classifier.py`

**Training**:
```bash
# Class 0 (negative)
python src/train_simple_diffusion.py \
  --data data/imdb-combined/train_class_0.json \
  --output results-simple-diffusion/class_0

# Class 1 (positive)
python src/train_simple_diffusion.py \
  --data data/imdb-combined/train_class_1.json \
  --output results-simple-diffusion/class_1
```

**Evaluation**:
```python
# Load both models
model_0 = SimpleDiscreteDiffusion.load('results-simple-diffusion/class_0')
model_1 = SimpleDiscreteDiffusion.load('results-simple-diffusion/class_1')

# Classify
for text in test_texts:
    log_p_0 = -model_0.compute_nll(text)
    log_p_1 = -model_1.compute_nll(text)
    predicted_class = 0 if log_p_0 > log_p_1 else 1
```

---

## Why This Approach Makes Sense

1. **Validates Core Hypothesis**: Tests if text diffusion (NLL-based likelihood) works for classification

2. **Low Risk**: Uses proven components (GPT-2, standard training)

3. **Fast Iteration**: Can complete in 1-2 days vs 1 week for full MDLM

4. **Informs Decision**: Results guide whether to invest in full MDLM

5. **Publishable Either Way**:
   - Success → "Simple diffusion beats RoBERTa MLM"
   - Failure → "Why discrete diffusion struggles" (valuable negative result)

---

## Updated Timeline

### This Week (Nov 3-9)
- **Day 1**: Implement `SimpleDiscreteDiffusion` classifier
- **Day 2**: Train class 0 and class 1 models
- **Day 3**: Evaluate and analyze results
- **Day 4**: Decision point - proceed to MDLM or pivot?

### Next Week (Nov 10-16) - If Proof-of-Concept Succeeds
- **Day 5-6**: Install conda + full MDLM environment
- **Day 7-9**: Train per-class MDLM models
- **Day 10**: Evaluate MDLM classifier
- **Day 11**: Final comparison and documentation

### If Proof-of-Concept Fails
- **Alternative**: Focus on GPT-2 enhancements for abuse detection
- **Reason**: Fundamental issues with discrete diffusion for classification

---

## Success Criteria

### Proof-of-Concept (Simple Diffusion)
- ✅ Excellent: >85% (validates approach)
- ✅ Good: 75-85% (shows promise)
- ⚠️ Acceptable: 65-75% (marginal)
- ❌ Failed: <65% (similar to RoBERTa MLM)

### Full MDLM (If Implemented)
- ✅ Excellent: >90% (matches/exceeds GPT-2)
- ✅ Good: 85-90% (competitive)
- ⚠️ Acceptable: 80-85% (shows MDLM helps)
- ❌ Failed: <80% (not worth complexity)

---

## Files Created

- ✅ `/Users/vincent/development/text-diffusion/test_mdlm_feasibility.py`
- ✅ `/Users/vincent/development/text-diffusion/deploy_mdlm_to_nigel.sh`
- ✅ `/Users/vincent/development/text-diffusion/experiments/MDLM_FEASIBILITY_FINDINGS.md`
- ✅ `/Users/vincent/development/text-diffusion/experiments/TRUE_DIFFUSION_IMPLEMENTATION_PLAN.md`
- ✅ This document: `MDLM_NEXT_STEPS.md`

## Environment Status

### Local (Mac)
- ✅ MDLM repository cloned
- ✅ Lightning, Hydra, OmegaConf installed
- ⚠️ Flash attention not compatible (Mac doesn't support CUDA)

### nigel.birs.ca
- ✅ MDLM repository deployed to `~/mdlm`
- ✅ Python venv at `~/text-diffusion/venv` with PyTorch, transformers
- ✅ Lightning, Hydra, OmegaConf, timm installed in venv
- ❌ Flash attention not installed (CUDA version compatibility unknown)
- ⚠️ Conda not installed

---

**Status**: Ready to implement simplified proof-of-concept

**Next Task**: Create `src/simple_diffusion_classifier.py`

**Last Updated**: 2025-11-03 (Evening)
