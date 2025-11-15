#!/usr/bin/env python3
"""
Comprehensive Comparison of Text Classification Approaches

Compares 4 different approaches on the same classification tasks:

1. Baseline GPT-2 (zero-shot): Prompt-based classification with pretrained GPT-2
2. Fine-tuned GPT-2 (native): Standard discriminative fine-tuning with classification head
3. Diffusion Baseline (untrained): Likelihood-based classification with untrained models
4. Diffusion Trained: Likelihood-based classification with per-class trained models

All approaches evaluated on identical test sets with statistical significance testing.

Usage:
    # Run all experiments
    python experiments/compare_all_approaches.py --dataset imdb --output results/comparison

    # Run specific approaches only
    python experiments/compare_all_approaches.py --dataset imdb --approaches gpt2-zeroshot diffusion-trained

    # Quick test with fewer samples
    python experiments/compare_all_approaches.py --dataset imdb --max-samples 100 --quick
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.data import load_test_examples, load_metadata


class ExperimentResults:
    """Store results from a single experiment."""

    def __init__(self, approach: str, dataset: str):
        self.approach = approach
        self.dataset = dataset
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.runtime = 0.0
        self.config = {}

    def compute_metrics(self):
        """Compute accuracy, precision, recall, F1."""
        acc = accuracy_score(self.labels, self.predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.labels, self.predictions, average='macro'
        )

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'num_samples': len(self.labels),
            'runtime': self.runtime,
            'runtime_per_sample': self.runtime / len(self.labels) if len(self.labels) > 0 else 0
        }

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'approach': self.approach,
            'dataset': self.dataset,
            'metrics': self.compute_metrics(),
            'predictions': self.predictions,
            'labels': self.labels,
            'config': self.config
        }


class GPT2ZeroShotClassifier:
    """
    Zero-shot classification using GPT-2 with prompting.

    For each class, compute P(text|class) by prompting:
    "This is a [class] review: [text]"
    and measuring the perplexity.
    """

    def __init__(self, class_names: List[str], device: Optional[str] = None):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.class_names = class_names
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading GPT-2 model for zero-shot classification...")
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def compute_perplexity(self, text: str, class_name: str) -> float:
        """Compute perplexity of text given class prompt."""
        # Create prompt
        prompt = f"This is a {class_name} review: {text}"

        # Tokenize
        encodings = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encodings['input_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            perplexity = np.exp(loss)

        return perplexity

    def classify(self, text: str) -> Tuple[int, List[float]]:
        """
        Classify text by comparing perplexities across classes.
        Lower perplexity = higher likelihood = better match.
        """
        perplexities = []

        for class_name in self.class_names:
            ppl = self.compute_perplexity(text, class_name)
            perplexities.append(ppl)

        # Lower perplexity is better, so invert for probabilities
        # Convert to log likelihoods (negative log perplexity)
        log_likes = -np.log(perplexities)

        # Softmax to get probabilities
        exp_log_likes = np.exp(log_likes - np.max(log_likes))
        probs = exp_log_likes / exp_log_likes.sum()

        prediction = int(np.argmin(perplexities))

        return prediction, probs.tolist()

    def classify_batch(self, texts: List[str], show_progress: bool = True) -> Tuple[List[int], List[List[float]]]:
        """Classify batch of texts."""
        predictions = []
        probabilities = []

        iterator = tqdm(texts, desc="GPT-2 Zero-shot") if show_progress else texts

        for text in iterator:
            pred, probs = self.classify(text)
            predictions.append(pred)
            probabilities.append(probs)

        return predictions, probabilities


class GPT2NativeClassifier:
    """
    Standard discriminative GPT-2 classifier with classification head.
    Fine-tuned on training data.
    """

    def __init__(self, class_names: List[str], device: Optional[str] = None):
        from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading GPT-2 model for discriminative classification...")
        self.model = GPT2ForSequenceClassification.from_pretrained(
            'gpt2',
            num_labels=self.num_classes
        ).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def train(self, train_texts: List[str], train_labels: List[int], epochs: int = 3):
        """Fine-tune on training data."""
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()

        print(f"Training GPT-2 native classifier for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        self.model.eval()

    def classify(self, text: str) -> Tuple[int, List[float]]:
        """Classify single text."""
        encodings = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            prediction = int(torch.argmax(logits).cpu())

        return prediction, probs.tolist()

    def classify_batch(self, texts: List[str], show_progress: bool = True) -> Tuple[List[int], List[List[float]]]:
        """Classify batch of texts."""
        predictions = []
        probabilities = []

        iterator = tqdm(texts, desc="GPT-2 Native") if show_progress else texts

        for text in iterator:
            pred, probs = self.classify(text)
            predictions.append(pred)
            probabilities.append(probs)

        return predictions, probabilities


def run_gpt2_zeroshot(args, test_texts: List[str], test_labels: List[int],
                      class_names: List[str]) -> ExperimentResults:
    """Run GPT-2 zero-shot experiment."""
    print("\n" + "="*60)
    print("APPROACH 1: GPT-2 Zero-Shot (No Training)")
    print("="*60 + "\n")

    results = ExperimentResults('gpt2-zeroshot', args.dataset)

    classifier = GPT2ZeroShotClassifier(class_names, device=args.device)

    start_time = time.time()
    predictions, probabilities = classifier.classify_batch(test_texts)
    results.runtime = time.time() - start_time

    results.predictions = predictions
    results.labels = test_labels
    results.probabilities = probabilities
    results.config = {'model': 'gpt2', 'method': 'perplexity-based'}

    return results


def run_gpt2_native(args, train_texts: List[str], train_labels: List[int],
                   test_texts: List[str], test_labels: List[int],
                   class_names: List[str]) -> ExperimentResults:
    """Run GPT-2 native classifier experiment."""
    print("\n" + "="*60)
    print("APPROACH 2: GPT-2 Native Classifier (Fine-tuned)")
    print("="*60 + "\n")

    results = ExperimentResults('gpt2-native', args.dataset)

    classifier = GPT2NativeClassifier(class_names, device=args.device)

    # Train
    train_start = time.time()
    classifier.train(train_texts, train_labels, epochs=args.gpt2_epochs)
    train_time = time.time() - train_start

    # Test
    test_start = time.time()
    predictions, probabilities = classifier.classify_batch(test_texts)
    test_time = time.time() - test_start

    results.predictions = predictions
    results.labels = test_labels
    results.probabilities = probabilities
    results.runtime = test_time
    results.config = {
        'model': 'gpt2',
        'method': 'discriminative',
        'epochs': args.gpt2_epochs,
        'train_time': train_time,
        'test_time': test_time
    }

    return results


def run_diffusion_baseline(args, test_texts: List[str], test_labels: List[int],
                           class_names: List[str]) -> ExperimentResults:
    """Run diffusion baseline (untrained) experiment."""
    print("\n" + "="*60)
    print("APPROACH 3: Diffusion Baseline (Untrained)")
    print("="*60 + "\n")

    from transformers import RobertaForMaskedLM, AutoTokenizer
    from src.classifier.inference import GenerativeClassifier

    results = ExperimentResults('diffusion-baseline', args.dataset)

    # Load untrained models (just pretrained base)
    print("Loading pretrained models (no fine-tuning)...")
    model_name = 'distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class_models = {}
    for class_id in range(len(class_names)):
        class_models[class_id] = RobertaForMaskedLM.from_pretrained(model_name)

    classifier = GenerativeClassifier(
        class_models=class_models,
        tokenizer=tokenizer,
        class_names=class_names,
        device=args.device
    )

    start_time = time.time()
    predictions, probabilities = classifier.classify_batch(
        test_texts,
        num_samples=args.num_likelihood_samples,
        mask_prob=args.mask_prob
    )
    results.runtime = time.time() - start_time

    results.predictions = predictions
    results.labels = test_labels
    results.probabilities = [list(p.values()) for p in probabilities]
    results.config = {
        'model': model_name,
        'method': 'likelihood-based',
        'trained': False,
        'num_likelihood_samples': args.num_likelihood_samples,
        'mask_prob': args.mask_prob
    }

    return results


def run_diffusion_trained(args, test_texts: List[str], test_labels: List[int],
                          class_names: List[str]) -> ExperimentResults:
    """Run diffusion trained experiment."""
    print("\n" + "="*60)
    print("APPROACH 4: Diffusion Trained (Per-Class Fine-tuned)")
    print("="*60 + "\n")

    from src.classifier.trainer import PerClassTrainer
    from src.classifier.inference import GenerativeClassifier

    results = ExperimentResults('diffusion-trained', args.dataset)

    # Load trained models
    if not Path(args.diffusion_model_dir).exists():
        print(f"ERROR: Trained models not found at {args.diffusion_model_dir}")
        print("Please train models first using:")
        print(f"  python src/train_generative_classifier.py --data-dir {args.data_dir}")
        return None

    print(f"Loading trained models from {args.diffusion_model_dir}...")

    from src.classifier.data import load_metadata
    metadata = load_metadata(args.data_dir)

    trainer = PerClassTrainer(
        model_name=metadata.get('model_name', 'distilroberta-base'),
        data_dir=args.data_dir,
        output_dir=args.diffusion_model_dir,
        device=args.device
    )
    trainer.load_trained_models()

    classifier = GenerativeClassifier(
        class_models=trainer.class_models,
        tokenizer=trainer.tokenizer,
        class_names=class_names,
        device=trainer.device
    )

    start_time = time.time()
    predictions, probabilities = classifier.classify_batch(
        test_texts,
        num_samples=args.num_likelihood_samples,
        mask_prob=args.mask_prob
    )
    results.runtime = time.time() - start_time

    results.predictions = predictions
    results.labels = test_labels
    results.probabilities = [list(p.values()) for p in probabilities]
    results.config = {
        'model': metadata.get('model_name', 'distilroberta-base'),
        'method': 'likelihood-based',
        'trained': True,
        'num_likelihood_samples': args.num_likelihood_samples,
        'mask_prob': args.mask_prob,
        'model_dir': args.diffusion_model_dir
    }

    return results


def compute_statistical_significance(results1: ExperimentResults,
                                     results2: ExperimentResults) -> Dict:
    """
    Compute statistical significance between two approaches using McNemar's test.
    """
    # McNemar's test for paired predictions
    n_00 = 0  # Both wrong
    n_01 = 0  # 1 wrong, 2 correct
    n_10 = 0  # 1 correct, 2 wrong
    n_11 = 0  # Both correct

    for pred1, pred2, label in zip(results1.predictions, results2.predictions, results1.labels):
        correct1 = (pred1 == label)
        correct2 = (pred2 == label)

        if correct1 and correct2:
            n_11 += 1
        elif correct1 and not correct2:
            n_10 += 1
        elif not correct1 and correct2:
            n_01 += 1
        else:
            n_00 += 1

    # McNemar's test statistic
    if n_01 + n_10 == 0:
        p_value = 1.0
    else:
        statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'n_both_correct': n_11,
        'n_both_wrong': n_00,
        'n_only_first_correct': n_10,
        'n_only_second_correct': n_01,
        'mcnemar_statistic': (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10) if (n_01 + n_10) > 0 else 0,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05
    }


def print_comparison_table(all_results: List[ExperimentResults]):
    """Print comparison table of all approaches."""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80 + "\n")

    print(f"{'Approach':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time (s)':>10}")
    print("-" * 80)

    for result in all_results:
        metrics = result.compute_metrics()
        print(f"{result.approach:<30} "
              f"{metrics['accuracy']:>10.4f} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} "
              f"{metrics['runtime']:>10.2f}")

    print()


def print_pairwise_significance(all_results: List[ExperimentResults]):
    """Print pairwise statistical significance tests."""
    print("\n" + "="*80)
    print("PAIRWISE STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print("="*80 + "\n")

    n = len(all_results)
    for i in range(n):
        for j in range(i + 1, n):
            result1 = all_results[i]
            result2 = all_results[j]

            sig = compute_statistical_significance(result1, result2)

            print(f"{result1.approach} vs {result2.approach}:")
            print(f"  Both correct: {sig['n_both_correct']}")
            print(f"  {result1.approach} only: {sig['n_only_first_correct']}")
            print(f"  {result2.approach} only: {sig['n_only_second_correct']}")
            print(f"  Both wrong: {sig['n_both_wrong']}")
            print(f"  p-value: {sig['p_value']:.4f} {'***' if sig['significant_at_0.05'] else '(not significant)'}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Compare All Classification Approaches')

    # Dataset args
    parser.add_argument('--dataset', type=str, default='imdb',
                       help='Dataset name (imdb, sst2, etc.)')
    parser.add_argument('--data-dir', type=str, default='data/imdb-classifier',
                       help='Directory with prepared data')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit test samples (for quick testing)')

    # Approach selection
    parser.add_argument('--approaches', nargs='+',
                       default=['gpt2-zeroshot', 'gpt2-native', 'diffusion-baseline', 'diffusion-trained'],
                       help='Which approaches to run')

    # GPT-2 args
    parser.add_argument('--gpt2-epochs', type=int, default=3,
                       help='Training epochs for GPT-2 native classifier')

    # Diffusion args
    parser.add_argument('--diffusion-model-dir', type=str, default='results-generative-classifier',
                       help='Directory with trained diffusion models')
    parser.add_argument('--num-likelihood-samples', type=int, default=5,
                       help='Likelihood samples for diffusion approaches')
    parser.add_argument('--mask-prob', type=float, default=0.15,
                       help='Mask probability for diffusion approaches')

    # Output args
    parser.add_argument('--output', type=str, default='results/comparison',
                       help='Output directory for results')

    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer samples, epochs)')

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.max_samples = args.max_samples or 100
        args.gpt2_epochs = 1
        args.num_likelihood_samples = 3

    print("\n" + "="*80)
    print("COMPREHENSIVE CLASSIFICATION APPROACH COMPARISON")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Approaches: {', '.join(args.approaches)}")
    print(f"Output: {args.output}")
    print()

    # Load data
    print("Loading test data...")
    from src.classifier.data import load_metadata, load_train_examples

    metadata = load_metadata(args.data_dir)
    class_names = metadata['class_names']

    test_texts, test_labels = load_test_examples(args.data_dir, max_samples=args.max_samples)
    print(f"Loaded {len(test_texts)} test examples")

    # Load training data for approaches that need it
    train_texts, train_labels = None, None
    if 'gpt2-native' in args.approaches:
        print("Loading training data for GPT-2 native classifier...")
        train_texts, train_labels = load_train_examples(args.data_dir)
        print(f"Loaded {len(train_texts)} training examples")

    # Run experiments
    all_results = []

    if 'gpt2-zeroshot' in args.approaches:
        result = run_gpt2_zeroshot(args, test_texts, test_labels, class_names)
        all_results.append(result)
        print(f"\nGPT-2 Zero-shot: {result.compute_metrics()['accuracy']:.4f} accuracy")

    if 'gpt2-native' in args.approaches and train_texts is not None:
        result = run_gpt2_native(args, train_texts, train_labels, test_texts, test_labels, class_names)
        all_results.append(result)
        print(f"\nGPT-2 Native: {result.compute_metrics()['accuracy']:.4f} accuracy")

    if 'diffusion-baseline' in args.approaches:
        result = run_diffusion_baseline(args, test_texts, test_labels, class_names)
        all_results.append(result)
        print(f"\nDiffusion Baseline: {result.compute_metrics()['accuracy']:.4f} accuracy")

    if 'diffusion-trained' in args.approaches:
        result = run_diffusion_trained(args, test_texts, test_labels, class_names)
        if result is not None:
            all_results.append(result)
            print(f"\nDiffusion Trained: {result.compute_metrics()['accuracy']:.4f} accuracy")

    # Print comparison
    print_comparison_table(all_results)
    print_pairwise_significance(all_results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'comparison_{args.dataset}_{timestamp}.json'

    results_dict = {
        'dataset': args.dataset,
        'timestamp': timestamp,
        'approaches': [r.to_dict() for r in all_results],
        'config': vars(args)
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
