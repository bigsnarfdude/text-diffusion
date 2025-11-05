#!/usr/bin/env python3
"""
MDLM Generative Classifier - Evaluation Script

This script evaluates a two-class MDLM generative classifier on IMDB test data.

Usage:
    python eval_mdlm_classifier.py \\
        --model_0=results-mdlm/class_0/best_checkpoint.ckpt \\
        --model_1=results-mdlm/class_1/best_checkpoint.ckpt \\
        --test_file=data/imdb-combined/test.json
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# Add MDLM to path
sys.path.insert(0, os.path.expanduser('~/mdlm'))

import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from omegaconf import OmegaConf

# Import MDLM
from diffusion import Diffusion


class MDLMGenerativeClassifier:
    """Generative classifier using per-class MDLM models."""

    def __init__(self, model_paths, tokenizer, device='cuda'):
        """
        Initialize classifier with per-class MDLM models.

        Args:
            model_paths: List of paths to class model checkpoints
            tokenizer: Tokenizer to use
            device: Device to run on
        """
        self.device = device
        self.tokenizer = tokenizer
        self.num_classes = len(model_paths)

        print(f"Loading {self.num_classes} class models...")

        self.models = []
        for i, model_path in enumerate(model_paths):
            print(f"  Loading class {i} model from {model_path}...")

            # Load config
            config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
            config = OmegaConf.load(config_path)

            # Load model
            model = Diffusion.load_from_checkpoint(
                model_path,
                config=config,
                tokenizer=tokenizer
            )
            model.to(device)
            model.eval()

            self.models.append(model)
            print(f"    ✅ Loaded class {i} model")

        print(f"✅ All {self.num_classes} models loaded\n")

    def compute_log_prob(self, text, model):
        """
        Compute log P(text) using MDLM model.

        Uses the model's _loss method which computes NLL.
        log P(text) = -NLL / num_tokens (normalized by sequence length)
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Compute NLL
        with torch.no_grad():
            loss_output = model._loss(input_ids, attention_mask)

        # Extract total NLL and normalize by actual sequence length
        total_nll = loss_output.nlls.sum().item()
        num_tokens = attention_mask.sum().item()

        # Normalize to get per-token log probability
        normalized_log_prob = -total_nll / num_tokens

        return normalized_log_prob

    def classify(self, text):
        """
        Classify text using Bayes rule with uniform priors.

        P(class | text) ∝ P(text | class) * P(class)
        log P(class | text) = log P(text | class) + log P(class)

        Returns:
            predicted_class: int (0 or 1)
            log_probs: list of normalized log probabilities per class
        """
        log_probs = []

        for class_id, model in enumerate(self.models):
            log_prob = self.compute_log_prob(text, model)
            log_probs.append(log_prob)

        # Add uniform prior (balanced classes)
        log_prior = np.log(1.0 / self.num_classes)
        log_probs_with_prior = [lp + log_prior for lp in log_probs]

        # Classification: argmax log P(class | text)
        predicted_class = int(np.argmax(log_probs_with_prior))

        return predicted_class, log_probs

    def evaluate(self, test_data):
        """
        Evaluate classifier on test data.

        Args:
            test_data: List of dicts with 'text' and 'label' keys

        Returns:
            results: Dict with accuracy, precision, recall, f1, etc.
        """
        print("="*60)
        print("Evaluating MDLM Generative Classifier")
        print("="*60)

        true_labels = []
        predicted_labels = []
        all_log_probs = []

        for item in tqdm(test_data, desc="Classifying"):
            text = item['text']
            true_label = item['label']

            # Classify
            predicted_label, log_probs = self.classify(text)

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            all_log_probs.append(log_probs)

        # Compute metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average='binary',
            zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average=None,
            zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class_metrics': {
                'negative': {
                    'precision': per_class_precision[0],
                    'recall': per_class_recall[0],
                    'f1': per_class_f1[0]
                },
                'positive': {
                    'precision': per_class_precision[1],
                    'recall': per_class_recall[1],
                    'f1': per_class_f1[1]
                }
            },
            'confusion_matrix': cm.tolist(),
            'num_samples': len(test_data)
        }

        return results


def print_results(results):
    """Print evaluation results in nice format."""

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    for class_name in ['negative', 'positive']:
        metrics = results['per_class_metrics'][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"         negative        positive")
    print(f"       negative      {cm[0][0]:<15} {cm[0][1]}")
    print(f"       positive      {cm[1][0]:<15} {cm[1][1]}")

    print(f"\nTest Samples: {results['num_samples']}")

    # Compare with baselines
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINES")
    print("="*60)

    baselines = [
        ("GPT-2 Native (Real IMDB)", 0.901),
        ("Simple Discrete Diffusion", 0.885),
        ("GPT-2 Native (Real + Synthetic)", 0.875),
        ("RoBERTa Diffusion (Real IMDB)", 0.617),
    ]

    # Add MDLM result
    baselines_with_mdlm = [
        (f"MDLM Diffusion (Current)", results['accuracy'])
    ] + baselines

    # Sort by accuracy
    baselines_with_mdlm.sort(key=lambda x: x[1], reverse=True)

    print("\nAccuracy Comparison:")
    for name, acc in baselines_with_mdlm:
        stars = "⭐" * max(1, int(acc * 10))
        marker = " 👈 YOU ARE HERE" if "MDLM" in name else ""
        print(f"  {name:<40} {acc:.3f} ({acc*100:.1f}%) {stars}{marker}")

    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)

    acc = results['accuracy']
    if acc >= 0.90:
        print("🎉 EXCELLENT: ≥90% accuracy - MDLM matches/exceeds GPT-2!")
        print("   → State-of-the-art generative classification")
    elif acc >= 0.85:
        print("✅ GOOD: 85-90% accuracy - MDLM competitive with GPT-2")
        print("   → Validates true discrete diffusion approach")
    elif acc >= 0.80:
        print("⚠️  ACCEPTABLE: 80-85% accuracy - Shows promise")
        print("   → May need hyperparameter tuning")
    else:
        print("❌ NEEDS IMPROVEMENT: <80% accuracy")
        print("   → Check training convergence and data quality")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MDLM generative classifier')
    parser.add_argument('--model_0', type=str, required=True, help='Path to class 0 checkpoint')
    parser.add_argument('--model_1', type=str, required=True, help='Path to class 1 checkpoint')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--output_file', type=str, default='results-mdlm/evaluation_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Validate inputs
    assert os.path.exists(args.model_0), f"Model 0 checkpoint not found: {args.model_0}"
    assert os.path.exists(args.model_1), f"Model 1 checkpoint not found: {args.model_1}"
    assert os.path.exists(args.test_file), f"Test file not found: {args.test_file}"

    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded\n")

    # Initialize classifier
    classifier = MDLMGenerativeClassifier(
        model_paths=[args.model_0, args.model_1],
        tokenizer=tokenizer,
        device=args.device
    )

    # Load test data
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r') as f:
        data = json.load(f)

    # Handle different data formats
    if isinstance(data, dict) and 'texts' in data and 'labels' in data:
        # Format: {"texts": [...], "labels": [...]}
        test_data = [{'text': text, 'label': label}
                     for text, label in zip(data['texts'], data['labels'])]
    elif isinstance(data, list):
        # Format: [{"text": "...", "label": 0}, ...]
        test_data = data
    else:
        raise ValueError(f"Unexpected test data format in {args.test_file}")

    print(f"✅ Loaded {len(test_data)} test samples\n")

    # Evaluate
    results = classifier.evaluate(test_data)

    # Print results
    print_results(results)

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {args.output_file}")


if __name__ == '__main__':
    main()
