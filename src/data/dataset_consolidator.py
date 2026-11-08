"""
src/data/dataset_consolidator.py
AI-Powered Cloud IDS | Dataset Consolidation to CM-Format
Consolidates NSL-KDD + CIC-IDS2017 + UNSW-NB15 into unified CM-format.
Reference: Kim et al. (2024) Hi-MLIC 'datasets consolidated into CM-CIC-IDS2017
           and CM-UNSW-NB15 for cross-dataset evaluation gap elimination'.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Universal attack-category taxonomy (5 groups, Hi-MLIC level-2)
# Reference: Kim et al. (2024) Table I — Hierarchical Attack Taxonomy
UNIVERSAL_TAXONOMY = {
    # NSL-KDD labels
    'normal':  'Normal',
    'dos':     'DoS',
    'probe':   'Probe',
    'r2l':     'R2L',
    'u2r':     'U2R',
    # CICIDS2017 labels
    'BENIGN':                   'Normal',
    'DoS Hulk':                 'DoS',
    'DoS GoldenEye':            'DoS',
    'DoS slowloris':            'DoS',
    'DoS Slowhttptest':         'DoS',
    'DDoS':                     'DoS',
    'PortScan':                 'Probe',
    'FTP-Patator':              'R2L',
    'SSH-Patator':              'R2L',
    'Web Attack Brute':         'R2L',
    'Web Attack XSS':           'R2L',
    'Web Attack Sql':           'R2L',
    'Infiltration':             'U2R',
    'Bot':                      'DoS',
    'Heartbleed':               'U2R',
    # UNSW-NB15 labels
    'Normal':   'Normal',
    'Generic':  'DoS',
    'Exploits': 'U2R',
    'Fuzzers':  'Probe',
    'DoS':      'DoS',
    'Recon':    'Probe',
    'Analysis': 'Probe',
    'Backdoor': 'U2R',
    'Shellcode':'U2R',
    'Worms':    'DoS',
}

# Shared 20-feature CM-format (features present in all 3 datasets)
# Reference: Kim et al. (2024) — 20 common features selected for CM consolidation
CM_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'count', 'srv_count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
]


class DatasetConsolidator:
    """
    Consolidates multiple benchmark datasets into unified CM-format.
    Reference: Kim et al. (2024) Hi-MLIC approach for cross-dataset generalization.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def map_labels(self, y: pd.Series) -> pd.Series:
        """Map dataset-specific labels to universal 5-category taxonomy."""
        return y.map(UNIVERSAL_TAXONOMY).fillna('Unknown')

    def consolidate(self, dfs: list, label_cols: list,
                    feature_map: dict = None) -> pd.DataFrame:
        """
        Merge multiple DataFrames into CM-format.
        Args:
            dfs: list of DataFrames (one per dataset)
            label_cols: list of label column names for each df
            feature_map: dict mapping each df's column names to CM_FEATURES names
        Returns:
            Consolidated DataFrame with CM_FEATURES + 'attack_category' column.
        """
        consolidated_parts = []
        for i, (df, lbl_col) in enumerate(zip(dfs, label_cols)):
            part = pd.DataFrame()
            fmap = feature_map[i] if feature_map else {}
            for cm_feat in CM_FEATURES:
                src_feat = fmap.get(cm_feat, cm_feat)
                if src_feat in df.columns:
                    part[cm_feat] = df[src_feat].values
                else:
                    part[cm_feat] = 0.0  # Zero-fill missing features
            part['attack_category'] = self.map_labels(df[lbl_col].reset_index(drop=True))
            consolidated_parts.append(part)
        result = pd.concat(consolidated_parts, ignore_index=True)
        return result

    def normalize_cm(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Min-Max normalize all CM feature columns."""
        df = df.copy()
        feat_cols = [c for c in CM_FEATURES if c in df.columns]
        if fit:
            df[feat_cols] = self.scaler.fit_transform(df[feat_cols])
        else:
            df[feat_cols] = self.scaler.transform(df[feat_cols])
        return df

    def get_cm_stats(self, df: pd.DataFrame) -> dict:
        """
        Return consolidated dataset statistics.
        Reference: Kim et al. (2024) Table III — CM dataset statistics.
        """
        total = len(df)
        dist = df['attack_category'].value_counts()
        ratio = dist.max() / dist.min() if dist.min() > 0 else float('inf')
        return {
            'total_samples': total,
            'n_classes': len(dist),
            'class_distribution': dist.to_dict(),
            'imbalance_ratio': round(ratio, 2),
            'n_features': len([c for c in CM_FEATURES if c in df.columns]),
        }
