"""
CNN-BiLSTM Hybrid Model for Network Intrusion Detection
========================================================
Based on CloudShield architecture (Djallel et al., 2023, IEEE)
Target: 99.1% accuracy on CIC-IDS2018, <3.2ms detection latency

Architecture idea:
  - CNN layers: extract LOCAL spatial patterns from feature groups
    (e.g., packet-size features next to each other → CNN sees their joint pattern)
  - BiLSTM layers: capture TEMPORAL dependencies in BOTH directions
    (forward pass = what came before; backward pass = what comes after)
  - Combined: spatial local patterns + temporal sequence context

For tabular data (NSL-KDD), we treat the 41 features as a sequence of length 41.
Each feature is a 'timestep' in the LSTM, and CNN slides over groups of features.
"""

import torch
import torch.nn as nn
import numpy as np


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM hybrid model for multi-class intrusion detection.

    Parameters:
      num_features: number of input features (41 for NSL-KDD)
      cnn_filters: number of CNN output channels (feature maps)
      cnn_kernel: size of CNN kernel (how many features each CNN window covers)
      lstm_hidden: number of LSTM hidden units
      lstm_layers: number of stacked LSTM layers
      num_classes: number of output attack categories
      dropout: dropout rate
    """
    def __init__(
        self,
        num_features: int = 41,
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden

        # ---- CNN Block ----
        # Input shape: (batch, 1, num_features)  — 1 channel, num_features length
        # Conv1d slides over features, extracting local patterns
        self.cnn_block = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=cnn_filters,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2  # 'same' padding to keep sequence length
            ),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=cnn_filters,
                out_channels=cnn_filters,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2
            ),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # After CNN: (batch, cnn_filters, num_features)
        # Permute to (batch, num_features, cnn_filters) for LSTM

        # ---- BiLSTM Block ----
        # bidirectional=True means 2 LSTMs running forward and backward
        # output size = lstm_hidden * 2 (concatenates both directions)
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,    # each timestep has cnn_filters features
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,        # forward + backward
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # ---- Attention over LSTM outputs ----
        # Instead of just taking the last hidden state, we learn to 'attend'
        # to the most relevant timesteps (features)
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # ---- Classifier Head ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, num_features)
        Returns:
            logits: tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Add channel dimension for CNN: (batch, num_features) → (batch, 1, num_features)
        x = x.unsqueeze(1)

        # CNN: (batch, 1, num_features) → (batch, cnn_filters, num_features)
        x = self.cnn_block(x)

        # Permute for LSTM: (batch, num_features, cnn_filters)
        x = x.permute(0, 2, 1)

        # BiLSTM: (batch, num_features, cnn_filters) → (batch, num_features, lstm_hidden*2)
        lstm_out, _ = self.bilstm(x)

        # Attention: compute weight for each timestep
        # (batch, num_features, 1) → softmax → weighted sum
        attn_scores = self.attention(lstm_out)      # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # normalize over sequence
        context = (lstm_out * attn_weights).sum(dim=1)    # (batch, lstm_hidden*2)

        # Classify
        logits = self.classifier(context)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def get_confidence(self, x):
        proba = self.predict_proba(x)
        return proba.max(dim=-1).values


class CNNBiLSTMTrainer:
    """Training wrapper for CNNBiLSTM model."""

    def __init__(
        self,
        num_features: int = 41,
        num_classes: int = 5,
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNBiLSTM(
            num_features=num_features,
            cnn_filters=cnn_filters,
            cnn_kernel=cnn_kernel,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_accuracies = []

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 20, batch_size: int = 256, verbose: bool = True):
        """Train the CNN-BiLSTM model."""
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)
        n_samples = len(X_t)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            perm = torch.randperm(n_samples)
            X_t = X_t[perm]
            y_t = y_t[perm]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                self.optimizer.zero_grad()
                logits = self.model(X_t[start:end])
                loss = self.criterion(logits, y_t[start:end])
                loss.backward()
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
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()
        return acc

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).argmax(dim=-1)
        return preds.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            proba = torch.softmax(self.model(X_t), dim=-1)
        return proba.cpu().numpy()

    def save(self, path: str):
        torch.save({'model_state': self.model.state_dict(),
                    'train_losses': self.train_losses,
                    'val_accuracies': self.val_accuracies}, path)
        print(f"[CNNBiLSTM] Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"[CNNBiLSTM] Model loaded from {path}")
        return self


if __name__ == "__main__":
    print("Testing CNNBiLSTM...")
    trainer = CNNBiLSTMTrainer(num_features=41, num_classes=5)
    X_fake = np.random.randn(200, 41).astype(np.float32)
    y_fake = np.random.randint(0, 5, 200)
    trainer.train(X_fake, y_fake, epochs=5, verbose=True)
    acc = trainer.evaluate(X_fake, y_fake)
    print(f"  Train acc: {acc:.4f}")
    print("CNNBiLSTM: OK")
