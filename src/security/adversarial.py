"""
Adversarial Attack and Defense Module
======================================
Based on Holla, Polepalli & Sasikumar (2025, IEEE Access)

What are adversarial attacks?
  An attacker takes a real network attack packet and makes TINY changes
  to its features (like adding 0.001 to 3 numbers) so the AI model still
  misclassifies it as "Normal" traffic — even though it IS an attack.

  Example: A DoS attack flow has features [0.5, 0.2, 0.9, ...].
  FGSM computes: which direction to slightly push each number so the
  model is maximally confused? Then it adds epsilon * that direction.
  The result looks almost identical to the original attack but the
  model now says "Normal". This is called an adversarial example.

Attacks implemented:
  1. FGSM (Fast Gradient Sign Method): Goodfellow et al. 2014
     - One-step attack: x_adv = x + epsilon * sign(gradient)
     - Very fast but weaker

  2. PGD (Projected Gradient Descent): Madry et al. 2018
     - Multi-step FGSM: repeats the step multiple times
     - Stronger but slower

Defense:
  Adversarial Training: include adversarial examples during training
  so the model learns to handle them. Holla et al. restored accuracy
  from 60% back to 88% under FGSM using this approach.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class AdversarialAttacker:
    """
    Implements FGSM and PGD adversarial attacks for testing model robustness.
    Works with PyTorch models (TransformerIDS, CNNBiLSTM).
    """

    def __init__(self, model: nn.Module, device: str = None):
        """
        Args:
            model: a PyTorch nn.Module (e.g. TransformerIDS or CNNBiLSTM)
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def fgsm_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1
    ) -> np.ndarray:
        """
        Fast Gradient Sign Method (FGSM) attack.

        Formula: x_adv = x + epsilon * sign( d(Loss)/d(x) )
        Translation: push each feature by epsilon in the direction that
        maximally increases the model's loss (makes it most wrong).

        Args:
            X:       clean input features (n_samples, n_features)
            y:       true integer labels (n_samples,)
            epsilon: perturbation magnitude (0.1 = small, 0.3 = large)
        Returns:
            X_adv:   adversarial examples with same shape as X
        """
        X_t = torch.FloatTensor(X).to(self.device).requires_grad_(True)
        y_t = torch.LongTensor(y).to(self.device)

        self.model.eval()
        logits = self.model(X_t)
        loss = self.criterion(logits, y_t)

        # Compute gradient of loss with respect to input
        loss.backward()
        grad = X_t.grad.data  # gradient shape: (n_samples, n_features)

        # FGSM: add epsilon * sign(gradient)
        X_adv = X_t + epsilon * grad.sign()

        return X_adv.detach().cpu().numpy()

    def pgd_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        alpha: float = 0.01,  # step size per iteration
        num_steps: int = 10   # number of gradient steps
    ) -> np.ndarray:
        """
        Projected Gradient Descent (PGD) attack — stronger than FGSM.

        Runs multiple FGSM steps, projecting back into epsilon-ball after each.
        Madry et al. (2018): 'adversarial examples in the worst case'.

        Args:
            X:         clean input (n_samples, n_features)
            y:         true labels (n_samples,)
            epsilon:   maximum total perturbation allowed
            alpha:     step size per iteration (usually epsilon/num_steps)
            num_steps: number of gradient steps
        Returns:
            X_adv:     adversarial examples
        """
        X_original = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        # Start from random point within epsilon-ball of original
        X_adv = X_original + torch.empty_like(X_original).uniform_(-epsilon, epsilon)
        X_adv = X_adv.detach()

        self.model.eval()
        for step in range(num_steps):
            X_adv.requires_grad_(True)
            logits = self.model(X_adv)
            loss = self.criterion(logits, y_t)
            loss.backward()
            grad = X_adv.grad.data

            # Take one step in gradient direction
            X_adv = X_adv.detach() + alpha * grad.sign()

            # Project back to epsilon-ball around original
            delta = X_adv - X_original
            delta = delta.clamp(-epsilon, epsilon)
            X_adv = (X_original + delta).detach()

        return X_adv.cpu().numpy()

    def evaluate_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        attack_type: str = 'fgsm',
        batch_size: int = 512
    ) -> dict:
        """
        Evaluates model accuracy on clean vs adversarial examples.

        Args:
            X:           clean test features (n_samples, n_features)
            y:           true labels
            epsilon:     perturbation strength
            attack_type: 'fgsm' or 'pgd'
            batch_size:  process in batches to avoid memory issues
        Returns:
            dict with clean_accuracy, adv_accuracy, accuracy_drop, etc.
        """
        self.model.eval()

        # Evaluate on clean examples
        clean_acc = self._batch_accuracy(X, y, batch_size)

        # Generate adversarial examples and evaluate
        X_adv = self._generate_in_batches(X, y, epsilon, attack_type, batch_size)
        adv_acc = self._batch_accuracy(X_adv, y, batch_size)

        return {
            "attack_type": attack_type,
            "epsilon": epsilon,
            "clean_accuracy": round(clean_acc, 4),
            "adversarial_accuracy": round(adv_acc, 4),
            "accuracy_drop": round(clean_acc - adv_acc, 4),
            "accuracy_drop_pct": round((clean_acc - adv_acc) * 100, 2),
            "n_samples": len(X)
        }

    def _batch_accuracy(self, X, y, batch_size):
        """Compute accuracy in batches to avoid OOM errors."""
        correct = 0
        total = len(X)
        for i in range(0, total, batch_size):
            X_b = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
            y_b = torch.LongTensor(y[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                preds = self.model(X_b).argmax(dim=-1)
                correct += (preds == y_b).sum().item()
        return correct / total

    def _generate_in_batches(self, X, y, epsilon, attack_type, batch_size):
        """Generate adversarial examples in batches."""
        X_adv_all = []
        for i in range(0, len(X), batch_size):
            X_b = X[i:i+batch_size]
            y_b = y[i:i+batch_size]
            if attack_type == 'fgsm':
                X_adv_b = self.fgsm_attack(X_b, y_b, epsilon)
            elif attack_type == 'pgd':
                X_adv_b = self.pgd_attack(X_b, y_b, epsilon)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}. Use 'fgsm' or 'pgd'.")
            X_adv_all.append(X_adv_b)
        return np.vstack(X_adv_all)


class AdversarialTrainer:
    """
    Adversarial Training wrapper.

    Adds adversarial examples to each training batch so the model
    learns to correctly classify both clean AND adversarial flows.

    Holla et al. (2025): This restores accuracy from ~60% back to ~88%
    under FGSM attacks with only ~25% extra training time.
    """

    def __init__(self, model: nn.Module, optimizer, criterion, device: str = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.attacker = AdversarialAttacker(model, device=self.device)

    def train_step(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        epsilon: float = 0.1,
        adv_fraction: float = 0.5  # 50% of batch will be adversarial
    ) -> float:
        """
        Single training step with adversarial data augmentation.

        Splits batch: clean_half → train normally, adv_half → FGSM → train on adversarial.

        Args:
            X_batch:      float tensor (batch_size, n_features)
            y_batch:      long tensor (batch_size,)
            epsilon:      FGSM perturbation strength
            adv_fraction: fraction of batch to convert to adversarial
        Returns:
            combined loss value
        """
        n = len(X_batch)
        n_adv = int(n * adv_fraction)
        n_clean = n - n_adv

        # --- Clean part ---
        self.model.train()
        self.optimizer.zero_grad()

        X_clean = X_batch[:n_clean]
        y_clean = y_batch[:n_clean]
        logits_clean = self.model(X_clean)
        loss_clean = self.criterion(logits_clean, y_clean)

        # --- Adversarial part ---
        X_adv_np = self.attacker.fgsm_attack(
            X_batch[n_clean:].detach().cpu().numpy(),
            y_batch[n_clean:].detach().cpu().numpy(),
            epsilon=epsilon
        )
        X_adv = torch.FloatTensor(X_adv_np).to(self.device)
        y_adv = y_batch[n_clean:]

        self.model.train()
        logits_adv = self.model(X_adv)
        loss_adv = self.criterion(logits_adv, y_adv)

        # Combined loss
        total_loss = (loss_clean * n_clean + loss_adv * n_adv) / n
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
        epsilon: float = 0.1
    ) -> float:
        """
        Run one full epoch of adversarial training.
        Returns average loss.
        """
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        n = len(X_t)
        perm = torch.randperm(n)
        X_t, y_t = X_t[perm], y_t[perm]

        total_loss = 0.0
        n_batches = (n + batch_size - 1) // batch_size

        for i in range(n_batches):
            s, e = i * batch_size, min((i+1) * batch_size, n)
            loss = self.train_step(X_t[s:e], y_t[s:e], epsilon=epsilon)
            total_loss += loss

        return total_loss / n_batches


def run_adversarial_evaluation(model, X_test, y_test, epsilons=None):
    """
    Convenience function: evaluate a model's robustness at multiple epsilon values.
    Returns a list of result dicts (one per epsilon per attack type).

    Args:
        model:   PyTorch nn.Module
        X_test:  numpy array (n_samples, n_features)
        y_test:  numpy array of integer labels
        epsilons: list of epsilon values to test (default: [0.01, 0.05, 0.1, 0.2])
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2]

    attacker = AdversarialAttacker(model)
    results = []

    print("[Adversarial Evaluation]")
    for attack in ['fgsm', 'pgd']:
        for eps in epsilons:
            r = attacker.evaluate_robustness(X_test, y_test, epsilon=eps, attack_type=attack)
            results.append(r)
            print(f"  {attack.upper()} eps={eps:.2f}: clean={r['clean_accuracy']:.3f} → adv={r['adversarial_accuracy']:.3f} (drop={r['accuracy_drop_pct']:.1f}%)")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.models.transformer_ids import TransformerIDS

    print("Testing Adversarial module...")
    model = TransformerIDS(num_features=20, num_classes=5, d_model=32, nhead=4, num_layers=1)

    X_fake = np.random.randn(100, 20).astype(np.float32)
    y_fake = np.random.randint(0, 5, 100)

    attacker = AdversarialAttacker(model)

    # FGSM
    X_adv = attacker.fgsm_attack(X_fake, y_fake, epsilon=0.1)
    print(f"  FGSM output shape: {X_adv.shape}")
    print(f"  Max perturbation: {np.abs(X_adv - X_fake).max():.4f}")

    # PGD
    X_pgd = attacker.pgd_attack(X_fake[:20], y_fake[:20], epsilon=0.1, num_steps=5)
    print(f"  PGD output shape: {X_pgd.shape}")

    # Robustness evaluation
    results = attacker.evaluate_robustness(X_fake, y_fake, epsilon=0.1, attack_type='fgsm')
    print(f"  Robustness result: {results}")

    print("Adversarial module: OK")
