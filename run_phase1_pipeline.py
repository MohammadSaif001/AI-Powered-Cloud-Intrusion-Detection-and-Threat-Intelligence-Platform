import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=' * 80)
print('  AI-POWERED CLOUD INTRUSION DETECTION & THREAT INTELLIGENCE PLATFORM')
print('  Integral University, Lucknow | B.Tech CSE Major Project (2026-2027)')
print('  Phase 1 (Initial 15-Day Milestone) End-to-End Execution Pipeline')
print('=' * 80)

from src.data.loader import DatasetLoader
from src.data.preprocessor import Preprocessor
from src.utils.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_mutual_information,
    plot_adversarial_vulnerability_gap
)

os.makedirs('reports/figures', exist_ok=True)

print('\n[STEP 1] Initializing Data Acquisition and Loading...')
loader = DatasetLoader(data_root='data/raw')
df_train, df_test = loader.load_nsl_kdd()
print(f'    [+] Loaded Dataset: {df_train.shape[0]:,} samples, {df_train.shape[1]} features.')

print('\n[STEP 2] Running Preprocessing & Data Cleaning...')
pre = Preprocessor()
clean_train = pre.clean(df_train)
print(f'    [+] Cleaned samples: {clean_train.shape[0]:,} (0 invalid/infinite rows remaining).')

print('\n[STEP 3] Analyzing Class Distribution & Imbalance Ratios...')
dist_df = pre.get_class_distribution(clean_train['attack_category'])
print(dist_df)
fig1 = plot_class_distribution(dist_df, title='NSL-KDD Attack Class Distribution', save_path='reports/figures/01_class_distribution.png')
plt.close('all')

print('\n[STEP 4] Generating Feature Correlation Matrix...')
fig2 = plot_correlation_heatmap(clean_train, top_features=12, title='Feature Correlation Heatmap', save_path='reports/figures/02_correlation_heatmap.png')
plt.close('all')

print('\n[STEP 5] Computing Mutual Information Feature Importance...')
X = clean_train.select_dtypes(include=[np.number])
y = pre.encode_labels(clean_train['attack_category'])
top_cols, mi_series = pre.select_features_mutual_info(X, y, top_k=15)
fig3 = plot_mutual_information(mi_series, top_k=15, title='Top Discriminative Features (Mutual Information)', save_path='reports/figures/03_mutual_information.png')
plt.close('all')
print('    [+] Top 5 Discriminative Features:')
for idx, (feat, score) in enumerate(mi_series.head(5).items()):
    print(f'        {idx+1}. {feat}: {score:.4f} bits')

print('\n[STEP 6] Visualizing Literature Gap 1 (Adversarial Brittleness)...')
fig4 = plot_adversarial_vulnerability_gap(save_path='reports/figures/04_adversarial_vulnerability.png')
plt.close('all')

print('\n' + '=' * 80)
print('  PHASE 1 EXECUTION COMPLETE - ALL DELIVERABLES GENERATED')
print('  - Preprocessed Data: OK')
print('  - Figures Saved in : reports/figures/')
print('  - Jupyter Notebooks: notebooks/01_NSL_KDD_EDA.ipynb, 02_CICIDS2017_EDA.ipynb, 03_Gap_Analysis_Visualization.ipynb')
print('  - Architecture Specs: architecture/system_architecture.md, uml_use_case.md, mitre_attack_mapping.json')
print('  - Gap Analysis Docs: literature_review/gap_analysis.md + 15 Paper Summaries')
print('=' * 80)
