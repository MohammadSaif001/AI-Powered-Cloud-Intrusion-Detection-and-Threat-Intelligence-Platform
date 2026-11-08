"""
Confidence-Based Model Router
==============================
Based on Sajid et al. (2024): Two-stage confidence routing achieves 99.3%
accuracy at 43% lower computational cost than a full DNN on all traffic.

How it works:
  Stage 1 (FAST): RandomForest handles 80-90% of traffic with high confidence.
                  If RF says "Normal with 95% confidence" → done. Fast path.
  Stage 2 (DEEP): The remaining 10-20% of ambiguous flows → sent to Transformer
                  or CNN-BiLSTM for deeper analysis.
  Ensemble fusion: When both deep models run, their probabilities are averaged.

Each flow gets:
  - prediction: integer class label (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)
  - confidence: float 0-1 (how sure the system is)
  - model_used: which model(s) made the decision
  - probabilities: full class probability vector
"""

import numpy as np
import joblib
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    """
    Result from the confidence router for a single network flow.

    Fields:
        prediction:    integer class (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)
        confidence:    probability of the predicted class (0.0 to 1.0)
        model_used:    name of model(s) that made this decision
        probabilities: full probability vector [p_normal, p_dos, p_probe, p_r2l, p_u2r]
        attack_label:  human-readable class name
    """
    prediction: int
    confidence: float
    model_used: str
    probabilities: np.ndarray
    attack_label: str = ""

    def __post_init__(self):
        CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
        if not self.attack_label:
            self.attack_label = CLASS_NAMES.get(self.prediction, "Unknown")

    def is_attack(self) -> bool:
        """Returns True if this flow is classified as any attack type."""
        return self.prediction != 0

    def severity(self) -> str:
        """
        Returns severity level based on attack type.
        U2R (privilege escalation) is highest risk.
        """
        severity_map = {0: "none", 1: "medium", 2: "low", 3: "high", 4: "critical"}
        return severity_map.get(self.prediction, "unknown")

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "attack_label": self.attack_label,
            "confidence": round(self.confidence, 4),
            "model_used": self.model_used,
            "probabilities": {
                "Normal": round(float(self.probabilities[0]), 4),
                "DoS": round(float(self.probabilities[1]), 4),
                "Probe": round(float(self.probabilities[2]), 4),
                "R2L": round(float(self.probabilities[3]), 4),
                "U2R": round(float(self.probabilities[4]), 4),
            },
            "is_attack": self.is_attack(),
            "severity": self.severity()
        }


class ConfidenceRouter:
    """
    Two-stage confidence-based model router.

    Usage:
        router = ConfidenceRouter(high_threshold=0.85, low_threshold=0.60)
        router.set_fast_model(rf_model)         # Stage 1: fast RandomForest
        router.set_deep_models(transformer, cnn_bilstm)  # Stage 2: deep models
        result = router.predict(X_single_flow)
    """

    CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}

    def __init__(
        self,
        high_threshold: float = 0.85,  # above this → use fast model result directly
        low_threshold: float = 0.60,   # below this → escalate to ensemble of deep models
        verbose: bool = False
    ):
        """
        Args:
            high_threshold: if fast model confidence >= this, accept result immediately
            low_threshold:  if fast model confidence < this, send to both deep models
            Between thresholds: send to one deep model (transformer)
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.verbose = verbose

        self.fast_model = None      # sklearn RandomForest
        self.transformer = None     # TransformerIDSTrainer
        self.cnn_bilstm = None      # CNNBiLSTMTrainer

        # Statistics for monitoring
        self.stats = {
            "total": 0,
            "fast_path": 0,     # resolved by fast model alone
            "medium_path": 0,   # resolved by one deep model
            "deep_path": 0,     # resolved by full ensemble
        }

    def set_fast_model(self, model):
        """
        Set the Stage 1 fast model (sklearn RandomForest or similar).
        Must have predict_proba(X) method returning (n_samples, n_classes).
        """
        self.fast_model = model

    def set_deep_models(self, transformer=None, cnn_bilstm=None):
        """
        Set Stage 2 deep models.
        Each must have predict_proba(X) method returning (n_samples, n_classes).
        """
        self.transformer = transformer
        self.cnn_bilstm = cnn_bilstm

    def predict_one(self, x: np.ndarray) -> DetectionResult:
        """
        Route a SINGLE flow through the confidence pipeline.

        Args:
            x: numpy array of shape (num_features,) — a single network flow
        Returns:
            DetectionResult with prediction, confidence, model info
        """
        self.stats["total"] += 1
        x2d = x.reshape(1, -1)  # make it (1, num_features) for model

        # ---- Stage 1: Fast Model ----
        if self.fast_model is not None:
            fast_proba = self.fast_model.predict_proba(x2d)[0]  # shape (n_classes,)
            fast_conf = float(fast_proba.max())
            fast_pred = int(fast_proba.argmax())

            if fast_conf >= self.high_threshold:
                # HIGH CONFIDENCE: accept fast model result directly
                self.stats["fast_path"] += 1
                if self.verbose:
                    print(f"  [Router] FAST path: {self.CLASS_NAMES[fast_pred]} (conf={fast_conf:.3f})")
                return DetectionResult(
                    prediction=fast_pred,
                    confidence=fast_conf,
                    model_used="RandomForest",
                    probabilities=fast_proba
                )

            elif fast_conf >= self.low_threshold:
                # MEDIUM CONFIDENCE: use transformer for verification
                if self.transformer is not None:
                    deep_proba = self.transformer.predict_proba(x2d)[0]
                    # Average fast + transformer probabilities
                    fused_proba = (fast_proba + deep_proba) / 2
                    pred = int(fused_proba.argmax())
                    conf = float(fused_proba.max())
                    self.stats["medium_path"] += 1
                    if self.verbose:
                        print(f"  [Router] MEDIUM path: {self.CLASS_NAMES[pred]} (conf={conf:.3f})")
                    return DetectionResult(
                        prediction=pred,
                        confidence=conf,
                        model_used="RF+Transformer",
                        probabilities=fused_proba
                    )

        # ---- Stage 2: Full Ensemble (low confidence or no fast model) ----
        probas = []
        model_names = []

        if self.fast_model is not None:
            fast_proba = self.fast_model.predict_proba(x2d)[0]
            probas.append(fast_proba)
            model_names.append("RF")

        if self.transformer is not None:
            t_proba = self.transformer.predict_proba(x2d)[0]
            probas.append(t_proba)
            model_names.append("Transformer")

        if self.cnn_bilstm is not None:
            c_proba = self.cnn_bilstm.predict_proba(x2d)[0]
            probas.append(c_proba)
            model_names.append("CNNBiLSTM")

        if not probas:
            raise RuntimeError("No models loaded in ConfidenceRouter. Call set_fast_model() or set_deep_models() first.")

        # Ensemble: simple average of all available probabilities
        fused_proba = np.mean(probas, axis=0)
        pred = int(fused_proba.argmax())
        conf = float(fused_proba.max())
        self.stats["deep_path"] += 1

        if self.verbose:
            print(f"  [Router] DEEP path ({'+'.join(model_names)}): {self.CLASS_NAMES[pred]} (conf={conf:.3f})")

        return DetectionResult(
            prediction=pred,
            confidence=conf,
            model_used="+".join(model_names),
            probabilities=fused_proba
        )

    def predict_batch(self, X: np.ndarray) -> list:
        """
        Route a batch of flows. Returns list of DetectionResult.
        Args:
            X: numpy array of shape (n_samples, n_features)
        """
        return [self.predict_one(X[i]) for i in range(len(X))]

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Convenience method: returns integer label array for batch X."""
        results = self.predict_batch(X)
        return np.array([r.prediction for r in results])

    def routing_stats(self) -> dict:
        """Returns routing statistics showing how much traffic went through each path."""
        total = max(self.stats["total"], 1)
        return {
            "total_flows": self.stats["total"],
            "fast_path_pct": round(100 * self.stats["fast_path"] / total, 1),
            "medium_path_pct": round(100 * self.stats["medium_path"] / total, 1),
            "deep_path_pct": round(100 * self.stats["deep_path"] / total, 1),
        }

    def reset_stats(self):
        self.stats = {"total": 0, "fast_path": 0, "medium_path": 0, "deep_path": 0}


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    print("Testing ConfidenceRouter...")
    # Create dummy models
    X_fake = np.random.randn(500, 41).astype(np.float32)
    y_fake = np.random.randint(0, 5, 500)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_fake, y_fake)

    router = ConfidenceRouter(high_threshold=0.85, low_threshold=0.60, verbose=True)
    router.set_fast_model(rf)

    # Test on 5 samples
    test_X = X_fake[:5]
    for i in range(5):
        result = router.predict_one(test_X[i])
        print(f"  Flow {i}: {result.to_dict()}")

    print("\nRouting stats:", router.routing_stats())
    print("ConfidenceRouter: OK")
