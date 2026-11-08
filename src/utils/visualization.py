"""
src/utils/visualization.py
AI-Powered Cloud IDS | EDA and Model Visualization Module
References:
  - Holla et al. (2025): Adversarial accuracy degradation visualization
  - Arreche et al. (2024): Multi-model accuracy comparison charts
  - Tang et al. (2022): Class imbalance visualization motivating CGAN
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Class Distribution ───────────────────────────────────────────────────
def plot_class_distribution(dist_df: pd.DataFrame, title: str = 'Class Distribution',
                             save_path: str = None) -> plt.Figure:
    """
    Bar chart of attack-category counts with log scale for rare classes.
    Reference: Synopsis Aug 11-12 EDA; Tang et al. (2022) imbalance ratio visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#d62728' if idx != 'normal' else '#2ca02c'
              for idx in dist_df.index]
    sns.barplot(x=dist_df.index, y=dist_df['Count'], hue=dist_df.index,
                palette=colors, ax=ax, legend=False)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Attack Category', fontsize=11)
    ax.set_ylabel('Sample Count (log scale)', fontsize=11)
    for i, (idx, row) in enumerate(dist_df.iterrows()):
        ax.text(i, row['Count'] * 1.3, f"{row['Percentage (%)']:.2f}%",
                ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Correlation Heatmap ──────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, top_features: int = 12,
                              title: str = 'Feature Correlation Heatmap',
                              save_path: str = None) -> plt.Figure:
    """
    Pearson correlation heatmap of top numeric features.
    Reference: Synopsis Aug 11-12 — 'plot feature correlation heatmaps'.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > top_features:
        variances = numeric.var().nlargest(top_features).index
        numeric = numeric[variances]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
                annot=True, fmt='.2f', linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Mutual Information Bar ───────────────────────────────────────────────
def plot_mutual_information(mi_series: pd.Series, top_k: int = 15,
                             title: str = 'Feature Mutual Information Scores',
                             save_path: str = None) -> plt.Figure:
    """
    Horizontal bar chart of top-k mutual information discriminative features.
    Reference: Synopsis Aug 11-12; Arreche et al. (2024) feature importance.
    """
    top_mi = mi_series.head(top_k)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('crest', len(top_mi))
    bars = ax.barh(range(len(top_mi)), top_mi.values, color=colors)
    ax.set_yticks(range(len(top_mi)))
    ax.set_yticklabels(top_mi.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mutual Information Score (bits)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, top_mi.values)):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Adversarial Vulnerability Gap ────────────────────────────────────────
def plot_adversarial_vulnerability_gap(save_path: str = None) -> plt.Figure:
    """
    Accuracy comparison: pre-adversarial, under attack, post-hardening.
    Values from Holla et al. (2025, IEEE Access) Table 3.
    Reference: Gap 1 — Adversarial Brittleness.
    """
    attacks = ['No Attack\n(Baseline)', 'FGSM', 'PGD', 'DeepFool', 'Square\n(Black-Box)']
    pre = [85, 60, 55, 50, 62]
    post = [85, 88, 85, 82, 84]
    x = np.arange(len(attacks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, pre, w, label='Pre-Hardening (Holla et al. 2025)', color='#d62728', alpha=0.85)
    b2 = ax.bar(x + w/2, post, w, label='Post-Adversarial Training (Holla et al. 2025)', color='#2ca02c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(attacks, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Detection Accuracy (%)', fontsize=11)
    ax.set_title('Gap 1: Adversarial Brittleness — Pre vs Post Hardening\n'
                 '(Holla, Polepalli & Sasikumar, IEEE Access 2025)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=88, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target >=88%')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height()}%', ha='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Confusion Matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                           title: str = 'Confusion Matrix',
                           save_path: str = None) -> plt.Figure:
    """
    Normalized confusion matrix heatmap.
    Reference: Arreche et al. (2024) multi-class evaluation; Kim et al. (2024) Hi-MLIC.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Baseline Model Comparison ────────────────────────────────────────────
def plot_model_comparison(results: dict, metric: str = 'accuracy',
                           title: str = 'Baseline Model Comparison',
                           save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing multiple models on a single metric.
    Reference: Arreche et al. (2024) RF vs XGBoost vs DNN comparison.
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) * 100 for m in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette('Set2', len(models))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

# ── Reconstruction Error (Autoencoder) ───────────────────────────────────
def plot_reconstruction_error(normal_errors: np.ndarray, attack_errors: np.ndarray,
                               threshold: float = None,
                               title: str = 'Autoencoder Reconstruction Error Distribution',
                               save_path: str = None) -> plt.Figure:
    """
    KDE plot of reconstruction errors for normal vs attack traffic.
    Reference: Agrawal et al. (2022) — threshold-based zero-day detection (94.7% rate).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(normal_errors, ax=ax, color='#2ca02c', fill=True, alpha=0.4, label='Normal Traffic')
    sns.kdeplot(attack_errors, ax=ax, color='#d62728', fill=True, alpha=0.4, label='Attack Traffic')
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Detection Threshold = {threshold:.4f}')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Saved figure: {save_path}")
    return fig

import os
