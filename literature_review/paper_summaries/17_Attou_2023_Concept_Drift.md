# Paper Summary 17: Concept Drift in Cloud IDS
**Authors:** Attou, H., Mounjid, A., Guzzo, A., Fortino, G., Benazzouz, D.
**Title:** Cloud-Based Intrusion Detection Approach Using Machine Learning Techniques
**Venue:** IEEE Open Journal of the Communications Society, Vol. 4, 2023
**DOI:** 10.1109/OJCOMS.2023.3256127

---

## Architecture
- **Models Evaluated**: Random Forest (RF), Decision Tree (DT), SVM, K-Nearest Neighbors (KNN)
- **Dataset**: NSL-KDD, CIC-IDS2017
- **Evaluation Protocol**: Time-based train/test split to simulate temporal concept drift
- **Key Experiment**: Models trained on 2016 traffic, tested on 2017 traffic to measure degradation

## Key Results
| Model | 2016 Accuracy | 2017 Accuracy | FPR Increase |
|---|---|---|---|
| Random Forest | 99.2% | 93.8% | +5.4% |
| Decision Tree | 98.7% | 91.2% | +7.5% |
| SVM | 97.4% | 89.1% | +8.3% |
| KNN | 98.1% | 90.3% | +7.8% |

**Key Finding**: Without retraining, static classifiers experience 5-8% FPR increase as attack patterns
evolve over 12 months. This is the RISK-T02 (Concept Drift) identified in our risk analysis.

## Platform Relevance
- **Concept Drift Risk Validation**: Provides empirical data for RISK-T02 severity assessment
- **Motivates FL Continual Learning**: Demonstrates that periodic retraining (via Federated Learning)
  is essential to maintain IDS effectiveness over time
- **Baseline Comparison**: Provides degrading static classifier baselines against which our adaptive
  DRL AIRS and FL continual learning can be compared in Phase 3

## Limitations
- Only classical ML models tested (no deep learning adaptation capability demonstrated)
- Single cloud environment; no multi-tenant drift patterns studied

## Gap Addressed
- Provides empirical evidence for Concept Drift risk (RISK-T02)
