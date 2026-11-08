"""
src/xai/shap_analysis.py
AI-Powered Cloud IDS | SHAP Feature Importance Analysis
References:
  - Arreche et al. (2024): 'SHAP TreeExplainer achieves >98% F1 with high explanation fidelity'
  - Holla et al. (2025): 'SHAP identifies protocol-level features most susceptible to perturbation'
  - Chen et al. (2023): 'SHAP reduces analyst MTTI by 47%'
  - Karimipour et al. (2023): 'SHAP-based feature selection: 18/80 features, 99.9% accuracy'
"""
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPAnalyzer:
    """
    SHAP-based feature importance and adversarial vulnerability analysis.
    Falls back to sklearn feature_importances_ if shap is not installed.
    """

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None

    def compute_shap_values(self, X: np.ndarray,
                             max_samples: int = 1000) -> np.ndarray:
        """
        Compute SHAP values using TreeExplainer (for RF/XGBoost/GBM).
        Reference: Arreche et al. (2024) TreeExplainer at <350ms latency.
        """
        if not SHAP_AVAILABLE:
            print("    [!] shap not installed — using feature_importances_ fallback")
            return self._sklearn_importance_fallback(X)

        # Sub-sample for speed (Karimipour et al. 2023 approach)
        n = min(len(X), max_samples)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values

        self.explainer = shap.TreeExplainer(self.model)
        sv = self.explainer.shap_values(X_sample)
        # For multi-class: average absolute SHAP over all classes
        if isinstance(sv, list):
            sv = np.abs(np.array(sv)).mean(axis=0)
        else:
            sv = np.abs(sv)
        self.shap_values = sv
        return sv

    def _sklearn_importance_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback: use tree-based feature_importances_ as SHAP proxy."""
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            n = min(len(X), 1000)
            return np.tile(imp, (n, 1))
        return np.zeros((len(X), len(self.feature_names)))

    def feature_importance_df(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return DataFrame of features ranked by mean absolute SHAP value.
        Reference: Arreche et al. (2024) feature importance ranking.
        """
        sv = self.compute_shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP|': mean_abs,
        }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        return df

    def adversarial_vulnerability_ranking(self, X: np.ndarray) -> pd.DataFrame:
        """
        Rank features by SHAP variance — high variance features are more susceptible
        to adversarial perturbation (Holla et al. 2025 protocol feature selection).
        Features with HIGH mean importance AND HIGH variance = adversarially vulnerable.
        Features with HIGH mean importance AND LOW variance = adversarially stable.
        """
        sv = self.compute_shap_values(X)
        mean_abs = np.abs(sv).mean(axis=0)
        variance  = np.abs(sv).var(axis=0)
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean |SHAP|': mean_abs,
            'SHAP Variance': variance,
            'Vulnerability Score': mean_abs * variance,  # High both = very vulnerable
        }).sort_values('Vulnerability Score', ascending=False).reset_index(drop=True)
        df['Risk'] = pd.cut(df['Vulnerability Score'],
                            bins=[-1, df['Vulnerability Score'].quantile(0.33),
                                  df['Vulnerability Score'].quantile(0.67),
                                  float('inf')],
                            labels=['Low', 'Medium', 'High'])
        return df

    def select_stable_features(self, X: np.ndarray,
                                top_k: int = 20) -> list:
        """
        Select top-k features with high importance but LOW adversarial variance.
        Reference: Karimipour et al. (2023): 18/80 stable features -> 99.9% accuracy.
                   Holla et al. (2025): stable feature selection restores accuracy to 88%.
        """
        df = self.adversarial_vulnerability_ranking(X)
        # Score = importance / (variance + 1e-8) — maximize importance, minimize variance
        df['Stability Score'] = df['Mean |SHAP|'] / (df['SHAP Variance'] + 1e-8)
        stable = df.nlargest(top_k, 'Stability Score')['Feature'].tolist()
        return stable
