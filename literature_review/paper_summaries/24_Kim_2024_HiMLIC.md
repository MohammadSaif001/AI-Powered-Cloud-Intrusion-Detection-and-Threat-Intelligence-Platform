# Paper Summary 24: Hi-MLIC Hierarchical Multi-Label Classification
**Authors:** Kim, J., et al.
**Title:** Hi-MLIC: Hierarchical Multi-Label Intrusion Classification for Cloud Environments
**Venue:** IEEE Access, Vol. 12, 2024

---

## Architecture
- **3-Layer Hierarchical Classification**:
  - Layer 1: Binary (Normal vs Attack) - Fast, simple classifier
  - Layer 2: 4 Threat Groups (DoS, Probe, R2L, U2R equivalent categories)
  - Layer 3: 23 Specific Attack Types (fine-grained classification)
- **Dataset Consolidation**: CM-CIC-IDS2017 and CM-UNSW-NB15 (consolidated multi-source format)
- **Purpose**: Addresses both class imbalance (hierarchical routing avoids minority drowning) and
  multi-stage APT blindness (tracks attack type progression across layers)

## Key Results
| Metric | Value |
|---|---|
| Layer 1 Accuracy | 99.98% |
| Layer 2 Accuracy | 99.24% |
| Layer 3 Accuracy | 98.81% |
| Overall Precision | 98.79% |
| Overall Recall | 98.80% |
| F1-Score | 98.80% |

## Platform Relevance
- **Dataset Consolidation**: Our platform follows the CM-format consolidation of CIC-IDS2017 and
  UNSW-NB15 into unified training corpus as described by Kim et al. (2024)
- **Hierarchical Output**: Our DetectionResult format supports multi-level classification labels
  (attack_category: DoS, attack_type: neptune, confidence: 0.97)
- **Minority Class Handling**: Hierarchical approach prevents minority classes from being overwhelmed
  in a single flat softmax, complementing our CGAN augmentation strategy

## Limitations
- Computational overhead of running 3 sequential classifiers
- CM dataset consolidation requires careful feature alignment across datasets

## Gap Addressed
- Gap 2 (Class Imbalance) + Gap 4 (Multi-Stage APT Blindness)
