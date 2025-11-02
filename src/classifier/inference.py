"""
Likelihood-based classification using diffusion models.

For each text, compute P(text|class) using each class's diffusion model,
then classify using Bayes rule: argmax_c P(class|text) ∝ P(text|class) * P(class)
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from transformers import RobertaForMaskedLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GenerativeClassifier:
    """
    Classifies text by comparing likelihoods from per-class diffusion models.

    Classification strategy:
    1. For each class c, compute log P(text | class_c) using the diffusion model
    2. Apply Bayes rule: P(class|text) ∝ P(text|class) * P(class)
    3. Return argmax_c P(class_c|text)
    """

    def __init__(
        self,
        class_models: Dict[int, RobertaForMaskedLM],
        tokenizer: AutoTokenizer,
        class_names: List[str],
        class_priors: Optional[List[float]] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            class_models: Dictionary mapping class_id -> trained model
            tokenizer: HuggingFace tokenizer
            class_names: List of class names
            class_priors: Prior probabilities P(class) for each class (uniform if None)
            device: Device for inference
        """
        self.class_models = class_models
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.num_classes = len(class_models)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Set all models to eval mode
        for model in self.class_models.values():
            model.eval()
            model.to(self.device)

        # Set class priors (uniform by default)
        if class_priors is None:
            self.class_priors = [1.0 / self.num_classes] * self.num_classes
        else:
            assert len(class_priors) == self.num_classes
            assert abs(sum(class_priors) - 1.0) < 1e-6
            self.class_priors = class_priors

        self.log_priors = torch.log(torch.tensor(self.class_priors)).to(self.device)

    def compute_log_likelihood(
        self,
        text: str,
        class_id: int,
        mask_prob: float = 0.15
    ) -> float:
        """
        Compute log P(text | class) using the class's diffusion model.

        We compute this by:
        1. Masking tokens at random (e.g., 15%)
        2. Computing the model's prediction loss for masked tokens
        3. Returning negative loss as log likelihood estimate

        Args:
            text: Input text
            class_id: Which class model to use
            mask_prob: Probability of masking each token

        Returns:
            Log likelihood estimate
        """
        model = self.class_models[class_id]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            # Create masked version
            labels = input_ids.clone()

            # Randomly mask tokens
            probability_matrix = torch.full(labels.shape, mask_prob)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                labels[0].tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask).bool().unsqueeze(0), value=0.0
            )

            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens

            # Replace masked tokens with [MASK]
            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_indices] = self.tokenizer.mask_token_id

            # Forward pass
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Loss is negative log likelihood
            # Return negative loss as log likelihood
            log_likelihood = -outputs.loss.item()

        return log_likelihood

    def classify_single(
        self,
        text: str,
        num_samples: int = 5,
        mask_prob: float = 0.15
    ) -> Tuple[int, Dict[str, float]]:
        """
        Classify a single text by comparing likelihoods.

        Args:
            text: Input text to classify
            num_samples: Number of masking samples to average over
            mask_prob: Masking probability for likelihood estimation

        Returns:
            (predicted_class_id, class_probabilities_dict)
        """
        log_likelihoods = torch.zeros(self.num_classes).to(self.device)

        # Compute average log likelihood for each class over multiple samples
        for class_id in range(self.num_classes):
            ll_samples = []
            for _ in range(num_samples):
                ll = self.compute_log_likelihood(text, class_id, mask_prob)
                ll_samples.append(ll)
            log_likelihoods[class_id] = np.mean(ll_samples)

        # Apply Bayes rule: log P(class|text) = log P(text|class) + log P(class)
        log_posteriors = log_likelihoods + self.log_priors

        # Normalize to get probabilities
        posteriors = F.softmax(log_posteriors, dim=0)

        # Get prediction
        predicted_class = torch.argmax(posteriors).item()

        # Create readable probabilities dict
        probs_dict = {
            self.class_names[i]: posteriors[i].item()
            for i in range(self.num_classes)
        }

        return predicted_class, probs_dict

    def classify_batch(
        self,
        texts: List[str],
        num_samples: int = 5,
        mask_prob: float = 0.15,
        show_progress: bool = True
    ) -> Tuple[List[int], List[Dict[str, float]]]:
        """
        Classify a batch of texts.

        Args:
            texts: List of texts to classify
            num_samples: Number of masking samples per text
            mask_prob: Masking probability
            show_progress: Show progress bar

        Returns:
            (predictions, probabilities_list)
        """
        predictions = []
        probabilities = []

        iterator = tqdm(texts, desc="Classifying") if show_progress else texts

        for text in iterator:
            pred, probs = self.classify_single(text, num_samples, mask_prob)
            predictions.append(pred)
            probabilities.append(probs)

        return predictions, probabilities

    def explain_prediction(
        self,
        text: str,
        num_samples: int = 5
    ) -> Dict:
        """
        Explain why a particular prediction was made.

        Args:
            text: Input text
            num_samples: Number of likelihood samples

        Returns:
            Dictionary with detailed prediction information
        """
        # Get prediction
        predicted_class, probs = self.classify_single(text, num_samples)

        # Compute individual log likelihoods
        log_likelihoods = []
        for class_id in range(self.num_classes):
            ll_samples = [
                self.compute_log_likelihood(text, class_id, 0.15)
                for _ in range(num_samples)
            ]
            log_likelihoods.append({
                'class': self.class_names[class_id],
                'mean_log_likelihood': np.mean(ll_samples),
                'std_log_likelihood': np.std(ll_samples),
                'samples': ll_samples
            })

        explanation = {
            'text': text,
            'predicted_class': self.class_names[predicted_class],
            'predicted_class_id': predicted_class,
            'probabilities': probs,
            'log_likelihoods': log_likelihoods,
            'num_samples': num_samples
        }

        return explanation

    def print_explanation(self, explanation: Dict):
        """Pretty print an explanation."""
        print(f"\n{'='*60}")
        print("CLASSIFICATION EXPLANATION")
        print(f"{'='*60}\n")
        print(f"Text: {explanation['text'][:200]}...")
        print(f"\nPredicted: {explanation['predicted_class']}\n")
        print("Probabilities:")
        for class_name, prob in explanation['probabilities'].items():
            bar = '█' * int(prob * 40)
            print(f"  {class_name:12s}: {prob:.4f} {bar}")

        print("\nLog Likelihoods:")
        for ll_info in explanation['log_likelihoods']:
            print(f"  {ll_info['class']:12s}: "
                  f"{ll_info['mean_log_likelihood']:8.4f} ± "
                  f"{ll_info['std_log_likelihood']:.4f}")
        print()
