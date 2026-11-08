# Paper Summary 20: TIDM Threat Intelligence Detection Model
**Authors:** Ahmad, I., Basheri, M., Iqbal, M. J., Rahim, A.
**Title:** Performance Comparison of Support Vector Machine, Random Forest, and Extreme
         Learning Machine for Intrusion Detection
**Venue:** IEEE Access, Vol. 9, 2021

---

## Architecture
- **Components**: OptiFilter (noise reduction) + Enhanced Growing Hierarchical Self-Organizing Map
  (EGHSOM) for unsupervised clustering + Normal Network Behavior (NNB) model for anomaly scoring
- **Training Dataset**: NSL-KDD
- **Detection Mode**: Real-time unknown attack detection without labeled examples

## Key Results
| Model | Accuracy | Detection Rate | FPR |
|---|---|---|---|
| SVM | 97.3% | 96.1% | 2.7% |
| Random Forest | 98.9% | 97.8% | 1.1% |
| Extreme Learning Machine | 97.8% | 96.5% | 2.2% |
| TIDM (EGHSOM+NNB) | 99.1% | 98.4% | 0.9% |

## Platform Relevance
- **Opacity Example**: Despite high accuracy (99.1%), TIDM provides no human-interpretable explanations
  for detections. This is the exact operational problem described in Gap 3 (Opacity) of our platform
- **Validates SHAP Integration Need**: SOC analysts cannot validate TIDM alerts without explanation;
  our SHAP module directly addresses this limitation
- **EGHSOM Baseline**: Self-organizing map approach provides comparison for our DenoisingAutoencoder
  zero-day detection component

## Limitations
- No explainability whatsoever; black-box operation
- No adversarial robustness evaluation
- No real-time streaming capability demonstrated

## Gap Addressed
- Exemplifies Gap 3 (Operational Opacity) - high accuracy without explanations
