# Paper Summary 29: Dynamic GNN+GAT for APT Detection
**Authors:** Zhou, X., et al.
**Title:** Hierarchical Intrusion Detection System Using Graph Neural Network
**Venue:** IEEE Transactions on Information Forensics and Security, Vol. 18, 2023
**DOI:** 10.1109/TIFS.2023.3256121

---

## Architecture
- **Dynamic Graph**: Traffic topology updated continuously as new network connections form/close
  - Nodes: network endpoints (IPs, services, containers)
  - Edges: active connections with weight = traffic volume
- **Graph Attention Network (GAT)**: Learns to focus on the most suspicious neighbor nodes
  during graph convolution
- **Hierarchical Output**: Classifies at both node-level (is this endpoint compromised?) and
  subgraph-level (is this subgraph an APT stage?)
- **Temporal Modeling**: Maintains a rolling 5-minute graph snapshot for lateral movement tracking
- **Dataset**: DARPA Transparent Computing (TC) dataset

## Key Results
| Metric | Value |
|---|---|
| APT Detection Rate | 97.8% |
| False Positive Rate | 2.1% |
| Lateral Movement Detection | 97.8% |
| Memory Scaling Issue | Noted for >1000 nodes |
| Training Time (DARPA TC) | 4.2 hours (V100 GPU) |

## Platform Relevance
- **Primary GNN Reference**: Our Dynamic GNN+GAT is inspired by and directly implements this architecture
- **Complementary with Wang et al. 2024**: While Wang focuses on provenance graphs from audit logs,
  Zhou focuses on real-time traffic topology graphs; our platform incorporates both approaches
- **Memory Risk Validation**: Confirms RISK-T05 (GNN Memory Scaling Risk) for large cloud deployments;
  justifies our graph sampling and sliding window mitigation strategies

## Limitations
- Memory scales poorly with large cloud deployments (>1000 nodes)
- Less effective for volumetric flooding attacks where topology is irrelevant (DDoS)
- Requires significant GPU memory for training

## Gap Addressed
- Gap 4 (Multi-Stage APT Blindness) - real-time traffic topology approach
