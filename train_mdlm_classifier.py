#!/usr/bin/env python3
"""
MDLM-based Generative Classifier for IMDB Sentiment

This script trains per-class MDLM models for generative classification.

Usage:
    # Train class 0 (negative) model
    python train_mdlm_classifier.py --class_id=0 --data_file=data/imdb-combined/train_class_0.json --output_dir=results-mdlm/class_0

    # Train class 1 (positive) model
    python train_mdlm_classifier.py --class_id=1 --data_file=data/imdb-combined/train_class_1.json --output_dir=results-mdlm/class_1
"""

import argparse
import json
import os
import sys

# Add MDLM to path
sys.path.insert(0, os.path.expanduser('~/mdlm'))

import torch
import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import Dataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Import MDLM components
from diffusion import Diffusion
from dataloader import get_dataloaders


def load_imdb_class_data(data_file):
    """Load single-class IMDB data from JSON file."""
    print(f"Loading data from {data_file}...")

    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract texts - data format: {"texts": ["text1", "text2", ...]}
    if isinstance(data, dict) and 'texts' in data:
        texts = data['texts']
    elif isinstance(data, list):
        # Alternative format: [{"text": "..."}, ...]
        texts = [item['text'] for item in data]
    else:
        raise ValueError(f"Unexpected data format in {data_file}")

    print(f"Loaded {len(texts)} samples")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({'text': texts})

    return dataset


def create_mdlm_config(class_id, output_dir, seq_length=512):
    """Create MDLM configuration for per-class training."""

    config = OmegaConf.create({
        # Model configuration
        'model': {
            'type': 'ddit',
            'hidden_size': 768,      # Small model
            'cond_dim': 128,
            'length': seq_length,
            'n_blocks': 12,
            'n_heads': 12,
            'scale_by_sigma': True,
            'dropout': 0.1,
            'tie_word_embeddings': False,
        },

        # Core MDLM settings
        'mode': 'train',
        'diffusion': 'absorbing_state',
        'backbone': 'dit',
        'parameterization': 'subs',
        'time_conditioning': False,
        'T': 0,  # Continuous time
        'subs_masking': False,
        'seed': 1,

        # Noise schedule
        'noise': {
            'type': 'loglinear',
            't_min': 0.001,
        },

        # Sampling configuration
        'sampling': {
            'predictor': 'ddpm_cache',
            'steps': 128,
            'noise_removal': True,
            'num_sample_batches': 2,
            'num_sample_log': 2,
            'semi_ar': False,
            'stride_length': 1,
            'num_strides': 1,
        },

        # Training configuration
        'training': {
            'ema': 0.9999,
            'antithetic_sampling': True,
            'importance_sampling': False,
            'sampling_eps': 1e-3,
            'sampling_eps_min': 1e-3,  # CRITICAL: Required buffer for MDLM
            'sampling_eps_max': 1.0,    # CRITICAL: Required buffer for MDLM
            'change_of_variables': False,
        },

        # Loader configuration
        'loader': {
            'global_batch_size': 32,  # Effective batch size
            'eval_global_batch_size': 32,
            'batch_size': 8,
            'eval_batch_size': 8,
            'num_workers': 2,  # Required for persistent_workers
            'pin_memory': True,
        },

        # Optimizer configuration
        'optim': {
            'lr': 3e-4,
            'weight_decay': 0,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
        },

        # Trainer configuration
        'trainer': {
            'accelerator': 'cuda',
            'devices': 1,
            'num_nodes': 1,
            'accumulate_grad_batches': 4,
            'gradient_clip_val': 1.0,
            'precision': 'bf16',
            'num_sanity_val_steps': 2,
            'max_steps': 50000,  # Increased for better convergence (~50 epochs for 9k samples)
            'log_every_n_steps': 50,
            'limit_train_batches': 1.0,
            'limit_val_batches': 1.0,
            'val_check_interval': 1000,
        },

        # Checkpointing
        'checkpointing': {
            'save_dir': output_dir,
            'resume_from_ckpt': False,
            'resume_ckpt_path': None,
        },

        # Eval configuration
        'eval': {
            'checkpoint_path': '',
            'disable_ema': False,
            'compute_generative_perplexity': False,
            'perplexity_batch_size': 8,
            'compute_perplexity_on_sanity': False,
            'gen_ppl_eval_model_name_or_path': 'gpt2',
            'generate_samples': False,
        },

        # Data configuration
        'data': {
            'tokenizer_name_or_path': 'gpt2',
            'cache_dir': '/tmp/mdlm_cache',
            'wrap': True,
            'streaming': False,
        },

        # LR Scheduler configuration (Hydra format for transformers scheduler)
        'lr_scheduler': {
            '_target_': 'transformers.get_constant_schedule_with_warmup',
            'num_warmup_steps': 2500,  # Match BD3-LM training recipe
        },
    })

    return config


def train_mdlm_class_model(class_id, data_file, output_dir, seq_length=512):
    """Train MDLM model on single-class data."""

    print("="*60)
    print(f"Training MDLM Model for Class {class_id}")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print("\n[1/6] Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded: vocab_size = {len(tokenizer)}")

    # Load class-specific data
    print(f"\n[2/6] Loading class {class_id} data...")
    dataset = load_imdb_class_data(data_file)

    # Create MDLM config
    print("\n[3/6] Creating MDLM configuration...")
    config = create_mdlm_config(class_id, output_dir, seq_length)

    # Save config
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)
    print(f"✅ Config saved to {config_path}")

    # Tokenize dataset
    print("\n[4/6] Tokenizing dataset...")
    def tokenize_function(examples):
        # Tokenize without return_tensors (let DataLoader handle batching)
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=seq_length,
            padding='max_length'
        )
        return {
            'input_ids': outputs['input_ids'],
            'attention_mask': outputs['attention_mask']
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Set format to pytorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    print(f"✅ Tokenized {len(tokenized_dataset)} samples")

    # Create data loaders
    print("\n[5/6] Creating data loaders...")
    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory
    )

    # Use 10% of training data for validation
    val_size = len(tokenized_dataset) // 10
    val_dataset = tokenized_dataset.select(range(val_size))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory
    )

    print(f"✅ Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches")

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

    # Register buffers before loading (they're in the pretrained state_dict)
    # This matches how BD3-LM handles pretrained loading
    if 'sampling_eps_min' in state_dict:
        model.register_buffer('sampling_eps_min', state_dict['sampling_eps_min'])
        model.register_buffer('sampling_eps_max', state_dict['sampling_eps_max'])
        # Remove from state_dict since we registered them manually
        state_dict = {k: v for k, v in state_dict.items()
                     if k not in ['sampling_eps_min', 'sampling_eps_max']}

    # Load weights into our model
    model.load_state_dict(state_dict, strict=False)

    print(f"✅ Pretrained model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='mdlm-class-{step:06d}',
        save_top_k=3,
        monitor='trainer/loss',  # MDLM logs as 'trainer/loss' not 'train_loss'
        mode='min',
        every_n_train_steps=1000,  # Save every 1000 steps
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup trainer
    trainer = L.Trainer(
        max_steps=config.trainer.max_steps,
        accelerator=config.trainer.accelerator if torch.cuda.is_available() else 'cpu',
        devices=config.trainer.devices,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        log_every_n_steps=config.trainer.log_every_n_steps,
        precision=config.trainer.precision,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        val_check_interval=config.trainer.val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=output_dir,
    )

    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    print("\n" + "="*60)
    print(f"✅ Training Complete for Class {class_id}")
    print("="*60)
    print(f"\nModel saved to: {output_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='Train MDLM model for single class')
    parser.add_argument('--class_id', type=int, required=True, help='Class ID (0 or 1)')
    parser.add_argument('--data_file', type=str, required=True, help='Path to class data JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--seq_length', type=int, default=512, help='Maximum sequence length (default: 512)')

    args = parser.parse_args()

    # Validate inputs
    assert args.class_id in [0, 1], "class_id must be 0 or 1"
    assert os.path.exists(args.data_file), f"Data file not found: {args.data_file}"

    # Train model
    train_mdlm_class_model(
        class_id=args.class_id,
        data_file=args.data_file,
        output_dir=args.output_dir,
        seq_length=args.seq_length
    )


if __name__ == '__main__':
    main()
