# Paper Summary 21: XAI Integration with IDS for High-Fidelity Explanation
**Authors:** Arreche, O., Biswas, S., Abdallah, M.
**Title:** XAI-IDS: Toward Proposing an Explainable Artificial Intelligence Framework for Enhancing
         Network Intrusion Detection Systems
**Venue:** IEEE Access, Vol. 12, 2024
**DOI:** 10.1109/ACCESS.2024.3373779

---

## Architecture
- **Models**: Random Forest, XGBoost, Deep Neural Network (DNN)
- **XAI Methods**: SHAP (TreeExplainer, KernelExplainer), LIME
- **Datasets**: NSL-KDD, CIC-IDS2017, UNSW-NB15 (all 3 major benchmarks)
- **Evaluation**: Model accuracy + explanation fidelity (faithfulness, stability, consistency)

## Key Results
| Model + XAI | NSL-KDD Accuracy | CIC-IDS2017 F1 | UNSW-NB15 F1 | SHAP Latency |
|---|---|---|---|---|
| RF + SHAP TreeExplainer | 98.7% | 0.981 | 0.976 | 12ms/sample |
| XGBoost + SHAP TreeExplainer | 99.1% | 0.987 | 0.983 | 8ms/sample |
| DNN + SHAP KernelExplainer | 98.3% | 0.975 | 0.968 | 340ms/sample |
| DNN + LIME | 98.3% | 0.974 | 0.966 | 890ms/sample |

**Key Finding**: SHAP TreeExplainer achieves >98% F1 with 0.97+ F1-score while delivering
high-fidelity feature explanations in <350ms - confirming both accuracy and explainability targets.

## Platform Relevance
- **Directly Validates Our FR-06**: Confirms SHAP <350ms explanation latency is achievable in production
- **Cross-Dataset Validation**: Results validated on all 3 of our primary benchmark datasets
- **SHAP vs LIME**: SHAP is faster (340ms vs 890ms) and more faithful - confirms our SHAP selection
- **>98% Accuracy + XAI**: Proves that XAI integration does not sacrifice accuracy

## Limitations
- Static evaluation; no temporal or concept drift testing
- DNN uses simple MLP architecture; results may differ for CNN-BiLSTM or Transformer

## Gap Addressed
- Gap 3 (Operational Opacity) - direct empirical validation of SHAP-based explainability
