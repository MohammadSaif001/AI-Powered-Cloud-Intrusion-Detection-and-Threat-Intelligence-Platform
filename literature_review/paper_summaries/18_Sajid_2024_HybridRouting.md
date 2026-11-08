# Paper Summary 18: Confidence-Based Hybrid Model Routing
**Authors:** Sajid, M. S. I., et al.
**Title:** A Two-Stage Hybrid Intrusion Detection System Using RF and Deep CNN
         with SHAP Explainability
**Venue:** IEEE Access, Vol. 12, 2024

---

## Architecture
- **Stage 1**: Random Forest (RF) as fast triage classifier
  - Routes 80-90% of traffic (high-confidence benign + obvious attacks) directly
  - Computational cost: very low (rule-based tree traversal)
- **Stage 2**: Deep CNN handles the remaining 10-20% ambiguous cases
  - More computationally expensive but applied only to uncertain flows
- **SHAP Integration**: All Stage 2 classifications accompanied by SHAP feature attribution
- **Dataset**: CIC-IDS2018

## Key Results
| Metric | Value |
|---|---|
| Overall Accuracy | 99.3% |
| Computational Saving | 43% vs full DNN on all flows |
| Detection Latency (Stage 1) | <1ms (RF inference) |
| Detection Latency (Stage 2) | 8-12ms (CNN inference) |
| SHAP Explanation Latency | <350ms |

## Platform Relevance
- **Confidence Router Design**: Directly inspires our Confidence-Based Router component that dispatches
  flows to Transformer, CNN-BiLSTM, GNN, or Autoencoder based on traffic characteristics
- **43% Cost Savings**: Validates the computational efficiency of tiered routing without accuracy penalty
- **SHAP Integration**: Confirms that SHAP is practical for production IDS deployment within 350ms budget

## Limitations
- Two-stage architecture; our platform extends to 6-way routing across specialized models
- CNN Stage 2 lacks temporal modeling capability (no LSTM/Transformer)
- No adversarial robustness evaluation

## Gap Addressed
- Validates confidence-based routing architecture (Pillar 2)
- Confirms SHAP explainability feasibility at IDS throughput (Gap 3: Opacity)
