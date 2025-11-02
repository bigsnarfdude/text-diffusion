# Code-Level Comparison: Original vs Ours

## Side-by-Side Code Examples

### 1. Training - Variable Masking Collator

#### Original (`finetune.py` - inline function)
```python
def diffusion_collator(features):
    """features: list of dicts with 'input_ids' and 'attention_mask'.

    Returns a batch dict:
      - input_ids: (B, MAX_LEN) with some tokens replaced by <mask>
      - attention_mask: (B, MAX_LEN) unchanged
      - labels: (B, MAX_LEN) where unmasked = -100, masked = original token IDs
    """
    batch_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    batch_attention = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    labels = batch_input_ids.clone()

    # Sample mask probability p for this batch
    p = float(mask_probs[torch.randint(low=0, high=len(mask_probs), size=(1,))])

    # Build boolean mask for positions that cannot be masked
    special_ids = set(tokenizer.all_special_ids)
    is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= batch_input_ids == sid

    pos_idxs = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    is_prefix = pos_idxs < PREFIX_LEN

    mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)

    # Draw random and mask
    rand = torch.rand_like(batch_input_ids, dtype=torch.float)
    mask_positions = (rand < p) & mask_candidate

    batch_input_ids[mask_positions] = tokenizer.mask_token_id
    labels[~mask_positions] = -100

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": labels,
    }
```

#### Ours (`src/data_collator.py` - reusable class)
```python
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
    mask_probs: List[float] = None
    prefix_length: int = 5
    mlm_probability: float = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mask_probs is None:
            self.mask_probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        if self.tokenizer.mask_token is None:
            raise ValueError("Tokenizer must have a mask token")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Step 1: Standard batching and padding
        batch = self.tokenizer.pad(examples, padding=True, return_tensors=self.return_tensors)

        # Step 2: Randomly select masking probability
        mask_prob = random.choice(self.mask_probs)

        # Step 3: Prepare labels
        batch["labels"] = batch["input_ids"].clone()

        # Step 4: Create probability matrix
        probability_matrix = torch.full(batch["input_ids"].shape, mask_prob)

        # Step 5: Never mask special tokens
        special_tokens_mask = torch.tensor([
            self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            for ids in batch["input_ids"].tolist()
        ], dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Step 6: Never mask prefix
        if self.prefix_length > 0:
            prefix_mask = torch.zeros_like(probability_matrix, dtype=torch.bool)
            prefix_mask[:, :self.prefix_length] = True
            probability_matrix.masked_fill_(prefix_mask, value=0.0)

        # Step 7: Apply masking
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch["labels"][~masked_indices] = -100
        batch["input_ids"][masked_indices] = self.tokenizer.mask_token_id
        batch["mask_prob"] = mask_prob

        return batch
```

**Key Differences:**
- ✅ **Ours is a reusable class** (not inline function)
- ✅ **Better documentation** (detailed docstring)
- ✅ **Configurable** (dataclass with defaults)
- ✅ **Returns mask_prob** for logging
- ✅ **Error handling** (validates mask token exists)

---

### 2. Generation - Iterative Denoising

#### Original (`inference.py`)
```python
# Hardcoded settings at top
MAX_LEN = 256
PREFIX_LEN = 16
N_STEPS = 10

# Build mask schedule
mask_probs = [i / N_STEPS for i in range(N_STEPS - 1, -1, -1)]

# Initialize
current_ids = torch.full((1, MAX_LEN), fill_value=mask_id, dtype=torch.long)
current_ids[0, :PREFIX_LEN] = context_ids

# Denoising loop
for p_mask in mask_probs:
    # Forward
    with torch.no_grad():
        outputs = model(input_ids=current_ids, attention_mask=current_attention)
        logits = outputs.logits

    # Sample each position
    pred_ids = torch.zeros((1, MAX_LEN), dtype=torch.long, device=DEVICE)
    for i in range(MAX_LEN):
        logit_vec = logits[0, i, :]
        filtered = top_k_top_p_filtering(logit_vec, top_k=50, top_p=0.95)
        probs = torch.softmax(filtered, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        pred_ids[0, i] = sampled

    # Final reveal or re-mask
    if p_mask == 0.0:
        new_ids = current_ids.clone()
        new_ids[0, PREFIX_LEN:] = pred_ids[0, PREFIX_LEN:]
        current_ids = new_ids
        break

    # Re-mask fraction p_mask
    positions = torch.arange(MAX_LEN, device=DEVICE)
    can_modify = positions >= PREFIX_LEN
    rand = torch.rand(MAX_LEN, device=DEVICE)
    mask_positions = (rand < p_mask) & can_modify

    next_ids = current_ids.clone()
    for i in range(PREFIX_LEN, MAX_LEN):
        if mask_positions[i]:
            next_ids[0, i] = mask_id
        else:
            next_ids[0, i] = pred_ids[0, i]

    current_ids = next_ids
```

#### Ours (`src/generate.py` - DiffusionGenerator class)
```python
class DiffusionGenerator:
    """Iterative text generator using diffusion-trained RoBERTa."""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.mask_schedule = config.get_mask_schedule()  # Flexible!

    def generate(self, prefix: str, max_length: int, show_steps: bool = True) -> str:
        # Tokenize prefix
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=True, return_tensors="pt")[0]
        prefix_length = len(prefix_ids)

        # Initialize
        num_to_generate = max_length - prefix_length
        mask_token_id = self.tokenizer.mask_token_id

        input_ids = torch.cat([
            prefix_ids,
            torch.full((num_to_generate,), mask_token_id, device=self.device)
        ]).unsqueeze(0)

        # Iterative denoising
        for step, mask_prob in enumerate(self.mask_schedule, 1):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # Find currently masked positions
            mask_positions = (input_ids[0, prefix_length:] == mask_token_id).nonzero(as_tuple=True)[0]
            mask_positions = mask_positions + prefix_length

            if len(mask_positions) == 0:
                break

            # Sample tokens for masked positions
            for pos in mask_positions:
                pos = pos.item()
                token_logits = logits[0, pos]
                sampled_token = self._sample_token(token_logits)  # Uses config
                input_ids[0, pos] = sampled_token

            # Re-mask for next iteration
            if step < len(self.mask_schedule):
                next_mask_prob = self.mask_schedule[step]
                n_to_mask = int(num_to_generate * next_mask_prob)

                if n_to_mask > 0:
                    maskable_positions = torch.arange(prefix_length, max_length, device=self.device)
                    indices = torch.randperm(len(maskable_positions))[:n_to_mask]
                    positions_to_mask = maskable_positions[indices]
                    input_ids[0, positions_to_mask] = mask_token_id

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample token based on config (greedy/topk/nucleus)."""
        if self.config.sampling_method == "greedy":
            return logits.argmax().item()

        elif self.config.sampling_method == "topk":
            # Apply temperature
            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature

            # Top-k filtering
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()

        elif self.config.sampling_method == "nucleus":
            # Apply temperature
            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature

            # Nucleus (top-p) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
```

**Key Differences:**
- ✅ **Ours is a class** (reusable, not script)
- ✅ **Flexible scheduling** (linear, cosine, exponential via config)
- ✅ **Multiple sampling methods** (greedy, top-k, nucleus)
- ✅ **Temperature control** (not in original)
- ✅ **Cleaner code** (separate _sample_token method)
- ✅ **Better variable names** (self-documenting)

---

### 3. Configuration - CLI Arguments

#### Original
```python
# finetune.py - Hardcoded at top
N_STEPS = 10
NUM_EPOCHS = 30
BATCH_SIZE = 16
MAX_LEN = 256
PREFIX_LEN = 16

# To change: Must edit source code
```

```python
# inference.py - Hardcoded
MODEL_DIR = "weights/roberta-diffusion-16s40e"
MAX_LEN = 256
PREFIX_LEN = 16
N_STEPS = 10

# Only one CLI arg
parser.add_argument("prompt", type=str, help="Text prompt")
parser.add_argument("--animation", action="store_false")
```

#### Ours (`src/config.py`)
```python
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Model
    model_name: str = "distilroberta-base"

    # Training
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500

    # Diffusion
    max_length: int = 64
    mask_probs: List[float] = None
    prefix_length: int = 5

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # Output
    output_dir: str = "./results"
    logging_steps: int = 100
    save_steps: int = 500

def parse_training_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train RoBERTa diffusion model")

    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--quick-test", action="store_true")
    # ... many more options

    args = parser.parse_args()

    # Build config from args
    config = TrainingConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        # ...
    )

    return config
```

```python
@dataclass
class GenerationConfig:
    """Generation/inference hyperparameters."""
    # Model
    checkpoint_path: str = "./results/checkpoint-latest"
    device: str = "cuda"

    # Generation
    prefix: str = "The quick brown fox"
    max_length: int = 64
    num_samples: int = 5

    # Denoising
    num_steps: int = 10
    schedule_type: str = "linear"  # linear, cosine, exponential

    # Sampling
    sampling_method: str = "topk"  # greedy, topk, nucleus
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

    def get_mask_schedule(self) -> List[float]:
        """Get masking schedule based on schedule_type."""
        if self.schedule_type == "linear":
            return self.get_linear_schedule(self.num_steps)
        elif self.schedule_type == "cosine":
            return self.get_cosine_schedule(self.num_steps)
        elif self.schedule_type == "exponential":
            return self.get_exponential_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def get_linear_schedule(self, num_steps: int) -> List[float]:
        """Linear schedule: [0.9, 0.8, ..., 0.1, 0.0]"""
        return [i / num_steps for i in range(num_steps - 1, -1, -1)]

    def get_cosine_schedule(self, num_steps: int) -> List[float]:
        """Cosine schedule: smoother transitions."""
        import math
        return [
            0.5 * (1 + math.cos(math.pi * i / num_steps))
            for i in range(num_steps + 1)
        ]

    def get_exponential_schedule(self, num_steps: int) -> List[float]:
        """Exponential schedule: fast early, slow late."""
        return [0.9 ** i for i in range(num_steps + 1)]

def parse_generation_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate text with diffusion")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix", default="The quick brown fox")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--schedule", choices=["linear", "cosine", "exponential"], default="linear")
    parser.add_argument("--sampling", choices=["greedy", "topk", "nucleus"], default="topk")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    # ... more options

    args = parser.parse_args()

    config = GenerationConfig(
        checkpoint_path=args.checkpoint,
        prefix=args.prefix,
        max_length=args.max_length,
        num_samples=args.num_samples,
        schedule_type=args.schedule,
        sampling_method=args.sampling,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    return config
```

**Key Differences:**
- ✅ **Dataclass configs** (type-safe, documented)
- ✅ **Full CLI for everything** (no source editing needed)
- ✅ **Multiple schedules** (linear, cosine, exponential)
- ✅ **Validation** (checks for valid values)
- ✅ **Defaults** (sensible values provided)
- ✅ **Extensible** (easy to add new options)

---

## Summary: What Makes Our Implementation Better

### 1. **Modularity**
- Original: Everything in one file
- Ours: Separate modules (config, collator, train, generate)

### 2. **Reusability**
- Original: Functions tied to scripts
- Ours: Classes that can be imported and reused

### 3. **Flexibility**
- Original: Edit source to change settings
- Ours: CLI arguments for everything

### 4. **Documentation**
- Original: Minimal comments
- Ours: Extensive docstrings, type hints, examples

### 5. **Features**
- Original: Basic diffusion only
- Ours: Multiple schedules, sampling methods, visualization

### 6. **Error Handling**
- Original: Basic or none
- Ours: Validation, helpful error messages

### 7. **Testing**
- Original: None
- Ours: Validation report, quick-test mode

---

## Conclusion

**Same algorithm, much better code!**

We took the research prototype and transformed it into a well-structured learning implementation:
- ✅ Better organization
- ✅ More features
- ✅ Easier to use
- ✅ Better documented
- ✅ More maintainable
- ✅ More extensible

Perfect for learning, experimentation, and understanding diffusion models! 🚀
