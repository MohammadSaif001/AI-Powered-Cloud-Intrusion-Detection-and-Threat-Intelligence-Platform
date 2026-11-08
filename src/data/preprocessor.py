"""
src/data/preprocessor.py
AI-Powered Cloud IDS | Data Preprocessing Module
References:
  - Arreche et al. (2024): SHAP+XAI preprocessing on NSL-KDD, CICIDS2017, UNSW-NB15
  - Holla et al. (2025): Min-Max normalization for adversarial robustness analysis
  - Kim et al. (2024) Hi-MLIC: Consolidated CM-format preprocessing pipeline
  - Tang et al. (2022): Class imbalance quantification before CGAN augmentation
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif

class Preprocessor:
    """Complete data preprocessing pipeline for cloud IDS benchmark datasets."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_enc = LabelEncoder()
        self._scaler_fitted = False
        self._label_enc_fitted = False

    # ── Cleaning ────────────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove infinite and NaN values.
        Critical for CIC-IDS2017 and CIC-IDS2018 which contain division-by-zero
        Inf values in flow duration-derived features.
        Reference: Synopsis Aug 13-14 preprocessing step; Mekdad et al. (2023).
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    # ── Normalization ────────────────────────────────────────────────────
    def normalize(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply Min-Max normalization: x_norm = (x - x_min) / (x_max - x_min).
        Reference: Synopsis Section 8.2 Aug 13-14; Arreche et al. (2024);
                   Holla et al. (2025) — normalization before adversarial evaluation.
        """
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if fit or not self._scaler_fitted:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            self._scaler_fitted = True
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X

    def normalize_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization using already-fitted scaler (for test sets)."""
        if not self._scaler_fitted:
            raise RuntimeError("Scaler not fitted. Call normalize() with fit=True first.")
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        return X

    # ── Label Encoding ───────────────────────────────────────────────────
    def encode_labels(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode string attack-category labels as integers.
        Reference: Arreche et al. (2024) multi-class label preparation.
        """
        if fit or not self._label_enc_fitted:
            encoded = self.label_enc.fit_transform(y.astype(str))
            self._label_enc_fitted = True
        else:
            encoded = self.label_enc.transform(y.astype(str))
        return encoded

    def decode_labels(self, y_enc: np.ndarray) -> np.ndarray:
        """Reverse label encoding back to string class names."""
        return self.label_enc.inverse_transform(y_enc)

    # ── Feature Selection ────────────────────────────────────────────────
    def select_features_mutual_info(self, X: pd.DataFrame, y: np.ndarray,
                                     top_k: int = 20,
                                     max_samples: int = 10000) -> tuple:
        """
        Rank features by Mutual Information score and return top-k.
        Sub-sampling to max_samples for computational efficiency.
        Reference: Synopsis Aug 11-12 EDA; Arreche et al. (2024) feature analysis;
                   Karimipour et al. (2023) SHAP feature selection baseline.
        Returns: (selected_columns: list, mi_series: pd.Series sorted desc)
        """
        X_num = X.select_dtypes(include=[np.number])
        n = len(X_num)
        if n > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, size=max_samples, replace=False)
            X_s, y_s = X_num.iloc[idx], y[idx]
        else:
            X_s, y_s = X_num, y
        mi = mutual_info_classif(X_s, y_s, random_state=42)
        mi_series = pd.Series(mi, index=X_num.columns).sort_values(ascending=False)
        top_cols = mi_series.head(top_k).index.tolist()
        return top_cols, mi_series

    # ── Class Distribution ────────────────────────────────────────────────
    def get_class_distribution(self, y: pd.Series) -> pd.DataFrame:
        """
        Compute count + percentage for each attack category.
        Reference: Synopsis Aug 11-12 — class distribution confirms 1:5000 imbalance
                   for U2R attacks motivating CGAN augmentation (Tang et al. 2022).
        """
        counts = y.value_counts()
        pct = (counts / len(y) * 100).round(4)
        df = pd.DataFrame({'Count': counts, 'Percentage (%)': pct})
        df.index.name = y.name or 'label'
        return df

    # ── CICIDS2017 Specific ───────────────────────────────────────────────
    def clean_cicids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CICIDS2017-specific cleaning: handles division-by-zero Inf values
        in Flow Duration, Flow Bytes/s, Flow Packets/s columns.
        Reference: Synopsis Aug 13-14; Djallel et al. (2023) CIC-IDS2018 preprocessing.
        """
        df = df.copy()
        # Replace inf/-inf with NaN then drop
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        before = len(df)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)
        print(f"    [CIC-IDS] Removed {removed:,} rows with Inf/NaN "
              f"({removed/before*100:.2f}% of {before:,} total)")
        return df

    # ── Imbalance Report ─────────────────────────────────────────────────
    def imbalance_report(self, y: pd.Series) -> dict:
        """
        Return imbalance ratio (max_class / min_class) and flag severe imbalance.
        Reference: Tang et al. (2022) — 1:100 to 1:5000 ratio in benchmark datasets.
        """
        counts = y.value_counts()
        ratio = counts.max() / counts.min()
        severe = ratio > 100
        return {
            'max_class': counts.idxmax(),
            'min_class': counts.idxmin(),
            'ratio': round(ratio, 2),
            'severe_imbalance': severe,
            'recommendation': 'CGAN+WGAN-GP augmentation required' if severe else 'Standard SMOTE sufficient'
        }


    def fit_transform(self, df: pd.DataFrame, target_col: str = 'attack_category') -> tuple:
        """Cleans, normalizes numerical features, encodes labels, and returns (X_norm, y_enc)."""
        df_clean = self.clean(df)
        label_col = target_col if target_col in df_clean.columns else ('label' if 'label' in df_clean.columns else None)
        
        if label_col is not None:
            y = df_clean[label_col]
            y_enc = self.encode_labels(y, fit=True)
            X = df_clean.drop(columns=[col for col in [label_col, 'label', 'attack_category', 'difficulty_level'] if col in df_clean.columns])
        else:
            y_enc = np.zeros(len(df_clean), dtype=int)
            X = df_clean

        X_norm = self.normalize(X, fit=True)
        numeric_cols = X_norm.select_dtypes(include=[np.number]).columns
        return X_norm[numeric_cols].values.astype(np.float32), y_enc

    def transform(self, df: pd.DataFrame, target_col: str = 'attack_category') -> tuple:
        """Applies existing fitted scaler and label encoder to new data."""
        df_clean = self.clean(df)
        label_col = target_col if target_col in df_clean.columns else ('label' if 'label' in df_clean.columns else None)
        
        if label_col is not None:
            y = df_clean[label_col]
            y_enc = self.encode_labels(y, fit=False)
            X = df_clean.drop(columns=[col for col in [label_col, 'label', 'attack_category', 'difficulty_level'] if col in df_clean.columns])
        else:
            y_enc = np.zeros(len(df_clean), dtype=int)
            X = df_clean

        X_norm = self.normalize(X, fit=False)
        numeric_cols = X_norm.select_dtypes(include=[np.number]).columns
        return X_norm[numeric_cols].values.astype(np.float32), y_enc
