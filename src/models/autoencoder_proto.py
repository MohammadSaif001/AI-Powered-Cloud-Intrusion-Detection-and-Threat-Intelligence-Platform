"""
src/models/autoencoder_proto.py
AI-Powered Cloud IDS | Autoencoder Zero-Day Detection Prototype
Numpy-only autoencoder using PCA for dimensionality reduction as encoding.
References:
  - Agrawal et al. (2022): 'Deep Denoising Autoencoder achieves 94.7% zero-day detection,
    3.2% FPR on cloud-specific AWS traffic via reconstruction error thresholding'
  - Mekdad et al. (2022): 'VAE reconstruction error for unsupervised zero-day detection'
  - Synopsis Section 7.2.1: Autoencoder role — zero-day/novel attack detection
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class AutoencoderProto:
    """
    PCA-based autoencoder prototype for zero-day anomaly detection.
    Encoder: PCA compression (dimensionality reduction)
    Decoder: PCA reconstruction (inverse transform)
    Anomaly score: Mean Squared Error of reconstruction

    This is the Phase 1 prototype — full deep autoencoder (TensorFlow/PyTorch)
    will be implemented in Phase 2.
    Reference: Agrawal et al. (2022) reconstruction error thresholding.
    """

    def __init__(self, encoding_dim: int = 10, threshold_sigma: float = 3.0):
        """
        Args:
            encoding_dim: Latent space dimensionality (bottleneck)
            threshold_sigma: Threshold = mean + sigma * std of normal reconstruction errors
                             Reference: Mekdad et al. (2022) 3-sigma rule
        """
        self.encoding_dim = encoding_dim
        self.threshold_sigma = threshold_sigma
        self.pca = PCA(n_components=encoding_dim, random_state=42)
        self.scaler = MinMaxScaler()
        self.threshold_ = None
        self._fitted = False

    def fit(self, X_normal: np.ndarray) -> 'AutoencoderProto':
        """
        Train autoencoder on NORMAL traffic only.
        Reference: Agrawal et al. (2022) — trained on clean AWS baseline traffic.
        """
        X_scaled = self.scaler.fit_transform(X_normal)
        self.pca.fit(X_scaled)
        # Compute reconstruction error on training set to set threshold
        errors = self._reconstruction_errors(X_scaled)
        # Threshold = mean + sigma * std (3-sigma rule per Mekdad et al. 2022)
        self.threshold_ = errors.mean() + self.threshold_sigma * errors.std()
        self._fitted = True
        print(f"    [AutoEncoder] Trained | encoding_dim={self.encoding_dim} "
              f"| threshold={self.threshold_:.6f}")
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode then decode input through the PCA bottleneck."""
        X_scaled = self.scaler.transform(X)
        X_enc = self.pca.transform(X_scaled)   # encode (compress)
        X_rec = self.pca.inverse_transform(X_enc)  # decode (reconstruct)
        return X_rec

    def _reconstruction_errors(self, X_scaled: np.ndarray) -> np.ndarray:
        """MSE reconstruction error per sample."""
        X_enc = self.pca.transform(X_scaled)
        X_rec = self.pca.inverse_transform(X_enc)
        errors = np.mean((X_scaled - X_rec) ** 2, axis=1)
        return errors

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return MSE reconstruction error for each sample."""
        if not self._fitted:
            raise RuntimeError("AutoencoderProto not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self._reconstruction_errors(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary prediction: 1 = anomaly (potential zero-day), 0 = normal.
        Reference: Agrawal et al. (2022) threshold-based detection.
        """
        scores = self.anomaly_score(X)
        return (scores > self.threshold_).astype(int)

    def evaluate(self, X_normal: np.ndarray, X_attack: np.ndarray) -> dict:
        """
        Evaluate zero-day detection performance.
        Target: >=94.7% detection rate, <=3.2% FPR (Agrawal et al. 2022).
        """
        normal_scores = self.anomaly_score(X_normal)
        attack_scores = self.anomaly_score(X_attack)

        # True Positives: attack samples detected as anomaly
        tp = (attack_scores > self.threshold_).sum()
        fn = (attack_scores <= self.threshold_).sum()
        # True Negatives: normal samples correctly passed
        tn = (normal_scores <= self.threshold_).sum()
        fp = (normal_scores > self.threshold_).sum()

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * precision * detection_rate / (precision + detection_rate)
              if (precision + detection_rate) > 0 else 0)

        return {
            'detection_rate': round(detection_rate * 100, 2),
            'false_positive_rate': round(fpr * 100, 2),
            'precision': round(precision * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'threshold': round(self.threshold_, 6),
            'target_detection_rate': 94.7,  # Agrawal et al. (2022)
            'target_fpr': 3.2,              # Agrawal et al. (2022)
        }

    def variance_explained(self) -> float:
        """Fraction of variance captured by the encoding (PCA explained variance)."""
        return float(self.pca.explained_variance_ratio_.sum())
