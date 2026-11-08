# Paper Summary 16: TON_IoT Federated Intrusion Detection
**Authors:** Moustafa, N., Turnbull, B., Choo, K-K. R.
**Title:** An Ensemble Intrusion Detection Technique Based on Proposed Statistical Flow Features
         for Protecting Network Traffic of Internet of Things
**Venue:** IEEE Access, Vol. 8, 2020
**DOI:** 10.1109/ACCESS.2020.2993399

---

## Architecture
- **Testbed**: 3-layer SDN-NFV testbed: Edge -> Fog -> Cloud
- **Dataset Created**: TON_IoT (Windows 7, Windows 10 telemetry + IoT services)
- **Method**: Ensemble of Naive Bayes, Decision Tree, and Random Forest trained on statistical flow features
- **Feature Engineering**: Extracts 5 statistical features from raw flows: mean, standard deviation,
  kurtosis, skewness, and radius from normal behavior
- **Federated Configuration**: Models trained per-layer with local gradient sharing

## Key Results
| Configuration | Accuracy | FPR |
|---|---|---|
| Windows 7 telemetry | 98.67% | 1.33% |
| Windows 10 telemetry | 96.67% | 3.33% |
| IoT device traffic | 97.89% | 2.11% |
| Federated ensemble | 98.12% | 1.88% |

## Platform Relevance
- **Host-Based IDS Component**: TON_IoT Windows dataset provides host-based telemetry features
  for the HIDS component of our Federated Learning module
- **SDN-NFV Validation**: Validates our Pillar 3 architecture that mirrors SDN-NFV layered deployment
- **Federated Learning Baseline**: Provides early evidence for federated IDS feasibility before
  Li et al. (2023) formal FedProx treatment

## Limitations
- Statistical features alone may not capture complex temporal attack patterns
- Ensemble limited to classical ML; no deep learning integration
- No adversarial robustness evaluation

## Gap Addressed
- Supports Pillar 3 (Federated Learning) - host-based telemetry feature design
