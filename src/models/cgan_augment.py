"""
Conditional GAN (CGAN) for Minority Class Data Augmentation
=============================================================
Based on Tang et al. (2022, IEEE TNSM)

Problem: NSL-KDD has SEVERE class imbalance:
  - Normal: 67,343 samples (53.46%)
  - DoS:    45,927 samples (36.46%)
  - Probe:  11,656 samples (9.25%)
  - R2L:      995 samples (0.79%)  ← very few!
  - U2R:       52 samples (0.04%) ← extremely few!

If we train without fixing this, the model will almost never see U2R attacks
and will always predict Normal/DoS. It won't learn to detect rare attacks.

Solution (CGAN):
  - Train a Generator network that learns to create REALISTIC synthetic
    attack traffic for rare classes
  - The Generator gets a class label (e.g., "make a U2R attack sample")
  - A Discriminator learns to tell real traffic from fake traffic
  - After training: generate 1000 synthetic U2R samples → model now has
    enough examples to learn U2R patterns

Result: Tang et al. showed +15-35% F1-score improvement on rare classes.

Architecture:
  Generator G: (noise_dim + n_classes) → hidden → n_features
  Discriminator D: (n_features + n_classes) → hidden → real/fake score
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional


class Generator(nn.Module):
    """
    Generator network for CGAN.

    Takes:
      - z: random noise vector (latent space)
      - c: one-hot class label (condition — 'what attack type to generate')
    Produces:
      - synthetic flow features that look like the conditioned attack class
    """
    def __init__(self, noise_dim: int = 64, n_classes: int = 5, n_features: int = 41):
        super().__init__()
        input_dim = noise_dim + n_classes  # noise + class condition

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, n_features),
            nn.Tanh()  # output in [-1, 1] range (data should be normalized)
        )

    def forward(self, z, c):
        """
        Args:
            z: noise tensor (batch, noise_dim)
            c: one-hot class tensor (batch, n_classes)
        Returns:
            synthetic features (batch, n_features)
        """
        x = torch.cat([z, c], dim=1)
        return self.network(x)


class Discriminator(nn.Module):
    """
    Discriminator network for CGAN.

    Decides: is this sample REAL (from dataset) or FAKE (from Generator)?
    Also conditioned on class label — knows what attack type to expect.
    """
    def __init__(self, n_features: int = 41, n_classes: int = 5):
        super().__init__()
        input_dim = n_features + n_classes

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()  # output: probability of being real (0=fake, 1=real)
        )

    def forward(self, x, c):
        """
        Args:
            x: features tensor (batch, n_features)
            c: one-hot class tensor (batch, n_classes)
        Returns:
            real/fake probability (batch, 1)
        """
        inp = torch.cat([x, c], dim=1)
        return self.network(inp)


class CGANAugmentor:
    """
    Full CGAN training and augmentation pipeline.

    Usage:
        augmentor = CGANAugmentor(n_features=41, n_classes=5)
        augmentor.fit(X_train, y_train, epochs=100)
        X_new, y_new = augmentor.augment(target_counts={4: 1000})  # generate 1000 U2R
        X_augmented = np.vstack([X_train, X_new])
        y_augmented = np.hstack([y_train, y_new])
    """

    def __init__(
        self,
        n_features: int = 41,
        n_classes: int = 5,
        noise_dim: int = 64,
        lr: float = 2e-4,
        device: str = None
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Generator and Discriminator
        self.generator = Generator(noise_dim, n_classes, n_features).to(self.device)
        self.discriminator = Discriminator(n_features, n_classes).to(self.device)

        # Separate optimizers for G and D
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.g_losses = []
        self.d_losses = []
        self.is_trained = False
        self._data_stats = {}  # for denormalizing generated data

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        verbose: bool = True,
        print_every: int = 20
    ):
        """
        Train the CGAN on the provided dataset.

        Args:
            X:          feature matrix (n_samples, n_features) — should be normalized to [-1, 1]
            y:          integer class labels (n_samples,)
            epochs:     number of training epochs
            batch_size: GAN training batch size
            verbose:    print progress
            print_every: print every N epochs
        """
        # Store normalization stats
        self._data_min = X.min(axis=0)
        self._data_max = X.max(axis=0)
        self._data_range = self._data_max - self._data_min + 1e-8

        # Normalize X to [-1, 1] for Tanh output compatibility
        X_norm = 2 * (X - self._data_min) / self._data_range - 1

        X_t = torch.FloatTensor(X_norm).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        n_samples = len(X_t)

        # Real and fake labels with label smoothing (improves training stability)
        real_label = 0.9  # slightly less than 1 to prevent overconfidence
        fake_label = 0.0

        for epoch in range(1, epochs + 1):
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_t = X_t[perm]
            y_t = y_t[perm]

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = (n_samples + batch_size - 1) // batch_size

            for i in range(n_batches):
                s, e = i * batch_size, min((i+1)*batch_size, n_samples)
                real_x = X_t[s:e]
                real_y = y_t[s:e]
                curr_batch = len(real_x)

                # One-hot encode labels
                c = self._one_hot(real_y, self.n_classes)  # (batch, n_classes)

                # ==================
                # Train Discriminator
                # ==================
                self.d_optimizer.zero_grad()

                # D on real samples
                real_out = self.discriminator(real_x, c)
                real_labels = torch.full((curr_batch, 1), real_label, device=self.device)
                d_loss_real = self.criterion(real_out, real_labels)

                # Generate fake samples
                z = torch.randn(curr_batch, self.noise_dim, device=self.device)
                fake_x = self.generator(z, c)
                fake_out = self.discriminator(fake_x.detach(), c)
                fake_labels = torch.full((curr_batch, 1), fake_label, device=self.device)
                d_loss_fake = self.criterion(fake_out, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # ==================
                # Train Generator
                # ==================
                self.g_optimizer.zero_grad()

                # Generator wants D to classify its fakes as real
                z = torch.randn(curr_batch, self.noise_dim, device=self.device)
                fake_x = self.generator(z, c)
                g_out = self.discriminator(fake_x, c)
                target_real = torch.full((curr_batch, 1), real_label, device=self.device)
                g_loss = self.criterion(g_out, target_real)

                g_loss.backward()
                self.g_optimizer.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            self.g_losses.append(epoch_g_loss / n_batches)
            self.d_losses.append(epoch_d_loss / n_batches)

            if verbose and epoch % print_every == 0:
                print(f"  CGAN Epoch {epoch:4d}/{epochs} | G Loss: {self.g_losses[-1]:.4f} | D Loss: {self.d_losses[-1]:.4f}")

        self.is_trained = True
        return self

    def generate(self, class_label: int, n_samples: int) -> np.ndarray:
        """
        Generate synthetic samples for a specific attack class.

        Args:
            class_label: integer class to generate (0=Normal, 1=DoS, ..., 4=U2R)
            n_samples:   number of synthetic samples to generate
        Returns:
            numpy array of shape (n_samples, n_features) in original feature scale
        """
        if not self.is_trained:
            raise RuntimeError("CGAN must be trained before generating. Call fit() first.")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.noise_dim, device=self.device)
            c_labels = torch.full((n_samples,), class_label, dtype=torch.long, device=self.device)
            c = self._one_hot(c_labels, self.n_classes)
            fake_x = self.generator(z, c).cpu().numpy()

        # Denormalize from [-1, 1] back to original feature scale
        X_denorm = (fake_x + 1) / 2 * self._data_range + self._data_min
        return X_denorm

    def augment(
        self,
        X_original: np.ndarray,
        y_original: np.ndarray,
        target_counts: Dict[int, int] = None,
        min_samples_per_class: int = 1000
    ) -> tuple:
        """
        Augment dataset by generating synthetic samples for underrepresented classes.

        Args:
            X_original:          original feature matrix
            y_original:          original labels
            target_counts:       dict {class_label: target_count}, e.g. {4: 2000}
            min_samples_per_class: if target_counts not given, ensure each class has at least this many
        Returns:
            (X_new, y_new): only the NEWLY generated samples (stack with original yourself)
        """
        if target_counts is None:
            # Auto-compute: find classes below min_samples_per_class
            unique, counts = np.unique(y_original, return_counts=True)
            class_counts = dict(zip(unique, counts))
            target_counts = {
                cls: max(0, min_samples_per_class - class_counts.get(cls, 0))
                for cls in range(self.n_classes)
            }

        X_new_list = []
        y_new_list = []

        for cls, n_to_gen in target_counts.items():
            if n_to_gen > 0:
                X_gen = self.generate(cls, n_to_gen)
                X_new_list.append(X_gen)
                y_new_list.append(np.full(n_to_gen, cls, dtype=int))
                print(f"  [CGAN] Generated {n_to_gen} synthetic samples for class {cls}")

        if not X_new_list:
            return np.empty((0, self.n_features)), np.empty(0, dtype=int)

        return np.vstack(X_new_list), np.hstack(y_new_list)

    def _one_hot(self, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Convert integer labels to one-hot encoding."""
        return torch.eye(n_classes, device=self.device)[labels]

    def save(self, path_prefix: str):
        """Save generator and discriminator weights."""
        torch.save(self.generator.state_dict(), path_prefix + '_generator.pt')
        torch.save(self.discriminator.state_dict(), path_prefix + '_discriminator.pt')
        np.save(path_prefix + '_data_min.npy', self._data_min)
        np.save(path_prefix + '_data_max.npy', self._data_max)
        print(f"[CGAN] Saved to {path_prefix}_*.pt/npy")

    def load(self, path_prefix: str):
        """Load generator and discriminator weights."""
        self.generator.load_state_dict(torch.load(path_prefix + '_generator.pt', map_location=self.device))
        self.discriminator.load_state_dict(torch.load(path_prefix + '_discriminator.pt', map_location=self.device))
        self._data_min = np.load(path_prefix + '_data_min.npy')
        self._data_max = np.load(path_prefix + '_data_max.npy')
        self._data_range = self._data_max - self._data_min + 1e-8
        self.is_trained = True
        print(f"[CGAN] Loaded from {path_prefix}_*.pt/npy")
        return self


if __name__ == "__main__":
    print("Testing CGAN Augmentor...")
    n_features, n_classes = 20, 5
    # Simulate imbalanced dataset
    X = np.random.randn(500, n_features).astype(np.float32)
    y = np.array([0]*300 + [1]*150 + [2]*40 + [3]*8 + [4]*2)  # very imbalanced!

    print("  Original class distribution:", dict(zip(*np.unique(y, return_counts=True))))

    augmentor = CGANAugmentor(n_features=n_features, n_classes=n_classes, noise_dim=32)
    augmentor.fit(X, y, epochs=10, verbose=True, print_every=5)

    X_new, y_new = augmentor.augment(X, y, min_samples_per_class=100)
    print(f"  Generated {len(X_new)} synthetic samples")
    print("  New distribution:", dict(zip(*np.unique(np.hstack([y, y_new]), return_counts=True))))
    print("CGAN Augmentor: OK")
