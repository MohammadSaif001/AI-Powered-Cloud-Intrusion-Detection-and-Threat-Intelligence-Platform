"""
src/models/baseline_ml.py
AI-Powered Cloud IDS | Baseline ML Classifier Module
Implements RF, XGBoost, Decision Tree baselines on NSL-KDD.
References:
  - Arreche et al. (2024): 'RF, XGBoost, and DNN achieve >98% accuracy on NSL-KDD'
  - Attou et al. (2023): RF, DT, SVM, KNN baselines for concept drift analysis
  - Kim et al. (2024) Hi-MLIC: binary classification layer uses RF-style decision tree
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class BaselineMLSuite:
    """
    Suite of classical ML baseline classifiers.
    Reference: Arreche et al. (2024) validated RF (98.7%), XGBoost (99.1%),
               DNN (98.3%) on NSL-KDD with SHAP integration.
    """

    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=20,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=15, class_weight='balanced', random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=42
            ),
        }
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1
            )
        self.results = {}
        self._trained = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              model_names: list = None) -> dict:
        """
        Train all (or specified) baseline models.
        Returns dict of {model_name: trained_model}.
        """
        if model_names is None:
            model_names = list(self.models.keys())
        for name in model_names:
            if name not in self.models:
                continue
            model = self.models[name]
            model.fit(X_train, y_train)
            self._trained[name] = model
            print(f"    [+] Trained: {name}")
        return self._trained

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 class_names: list = None) -> dict:
        """
        Evaluate all trained models. Returns per-model metrics dict.
        Reference: Arreche et al. (2024) evaluation protocol.
        """
        for name, model in self._trained.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm   = confusion_matrix(y_test, y_pred)
            self.results[name] = {
                'accuracy': acc, 'precision': prec,
                'recall': rec, 'f1': f1, 'confusion_matrix': cm,
            }
        return self.results

    def get_feature_importances(self, feature_names: list) -> pd.DataFrame:
        """
        Return feature importances for tree-based models.
        Reference: Arreche et al. (2024) feature importance baseline
                   (compared with SHAP values for consistency).
        """
        rows = []
        for name, model in self._trained.items():
            if hasattr(model, 'feature_importances_'):
                for feat, imp in zip(feature_names, model.feature_importances_):
                    rows.append({'Model': name, 'Feature': feat, 'Importance': imp})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values('Importance', ascending=False)

    def summary_table(self) -> pd.DataFrame:
        """Return clean summary DataFrame of all model metrics."""
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name,
                'Accuracy (%)': round(metrics['accuracy'] * 100, 2),
                'Precision (%)': round(metrics['precision'] * 100, 2),
                'Recall (%)': round(metrics['recall'] * 100, 2),
                'F1-Score (%)': round(metrics['f1'] * 100, 2),
            })
        return pd.DataFrame(rows).sort_values('F1-Score (%)', ascending=False)
