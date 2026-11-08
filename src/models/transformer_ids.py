"""
Transformer-based Intrusion Detection System (IDS)
===================================================
Based on RTIDS (Wu et al., 2022, IEEE Access, Vol. 10, pp. 64375-64387)

The transformer architecture uses:
- Multi-head self-attention to capture long-range dependencies between traffic features
- Positional embedding to represent feature sequence order
- Feed-forward network for final classification

This model classifies network flows into 5 categories:
  Normal, DoS, Probe, R2L, U2R
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Adds positional information to feature embeddings.
    For tabular features, we treat each feature as a 'token' at position i.
    This tells the model 'feature 3 is at position 3' — giving it sequence context.
    """
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerIDS(nn.Module):
    """
    Transformer Encoder for Network Intrusion Detection.

    Architecture:
      Input (num_features) → Linear Embedding → Positional Encoding
      → N x TransformerEncoderLayer (Multi-Head Attention + FFN)
      → Global Average Pool → Classifier Head → 5 attack categories

    Parameters:
      num_features: number of input flow features (default: 41 for NSL-KDD)
      d_model: internal embedding dimension (default: 64)
      nhead: number of attention heads (default: 4)
      num_layers: number of transformer encoder layers (default: 2)
      num_classes: output classes (default: 5)
      dropout: dropout rate for regularization (default: 0.1)
    """
    def __init__(
        self,
        num_features: int = 41,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Step 1: Project each scalar feature to d_model dimensions
        # We treat each feature as a single token
        self.feature_embedding = nn.Linear(1, d_model)

        # Step 2: Positional encoding — tells the model which position each feature is at
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_features + 1)

        # Step 3: Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # (batch, seq, feature) order
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Step 4: Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, num_features)
        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        # Reshape: (batch, features) → (batch, features, 1)
        x = x.unsqueeze(-1)

        # Embed each feature scalar to d_model dimensions
        # (batch, features, 1) → (batch, features, d_model)
        x = self.feature_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer encoder
        # (batch, seq_len=features, d_model)
        x = self.transformer_encoder(x)

        # Global average pooling over the sequence (feature) dimension
        # (batch, d_model)
        x = x.mean(dim=1)

        # Classify
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        """Returns softmax probabilities for each class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def get_confidence(self, x):
        """Returns max probability (confidence score) for routing."""
        proba = self.predict_proba(x)
        return proba.max(dim=-1).values


class TransformerIDSTrainer:
    """
    Training wrapper for TransformerIDS.
    Handles training loop, validation, and model saving.
    """

    def __init__(
        self,
        num_features: int = 41,
        num_classes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerIDS(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accuracies = []

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 20, batch_size: int = 256, verbose: bool = True):
        """
        Train the transformer model.

        Args:
            X_train: numpy array of shape (n_samples, n_features)
            y_train: numpy array of integer class labels
            X_val:   optional validation features
            y_val:   optional validation labels
            epochs:  number of training epochs
            batch_size: mini-batch size
            verbose: print progress every 5 epochs
        """
        # Convert numpy arrays to torch tensors
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        n_samples = len(X_t)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            # Shuffle data each epoch
            perm = torch.randperm(n_samples)
            X_t = X_t[perm]
            y_t = y_t[perm]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_t[start:end]
                y_batch = y_t[start:end]

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                # Gradient clipping prevents exploding gradients in transformers
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)

            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                self.val_accuracies.append(val_acc)
                if verbose and epoch % 5 == 0:
                    print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                if verbose and epoch % 5 == 0:
                    print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

        return self

    def evaluate(self, X, y):
        """Returns accuracy on given data."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()
        return acc

    def predict(self, X):
        """Returns integer class predictions."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        """Returns class probabilities (n_samples, n_classes)."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=-1)
        return proba.cpu().numpy()

    def save(self, path: str):
        """Save model state dict."""
        torch.save({
            'model_state': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
        }, path)
        print(f"[TransformerIDS] Model saved to {path}")

    def load(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"[TransformerIDS] Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Quick smoke test
    print("Testing TransformerIDS...")
    trainer = TransformerIDSTrainer(num_features=41, num_classes=5)
    X_fake = np.random.randn(200, 41).astype(np.float32)
    y_fake = np.random.randint(0, 5, 200)
    trainer.train(X_fake, y_fake, epochs=5, verbose=True)
    acc = trainer.evaluate(X_fake, y_fake)
    preds = trainer.predict(X_fake[:10])
    proba = trainer.predict_proba(X_fake[:5])
    print(f"  Train acc: {acc:.4f}")
    print(f"  Sample preds: {preds}")
    print(f"  Sample proba shape: {proba.shape}")
    print("TransformerIDS: OK")
