# Paper Summary 27: GCN+GAT for Multi-Stage APT Detection via Provenance Graphs
**Authors:** Wang, Q., et al.
**Title:** APT Detection and Attribution Using Provenance Graphs and Graph Neural Networks
**Venue:** IEEE Transactions on Information Forensics and Security, Vol. 19, 2024
**DOI:** 10.1109/TIFS.2024.3356721

---

## Architecture
- **Provenance Graph Construction**: Audit logs parsed to construct directed graphs where:
  - Nodes: processes, files, network connections, system calls
  - Edges: causality relationships (process spawned, file written, network connection made)
- **GCN Layer**: Graph Convolutional Network learns structural patterns of APT attack chains
- **GAT Layer**: Graph Attention Network weights attention over neighboring nodes in graph
- **Multi-Stage Detection**: Tracks APT across Initial Access -> Execution -> Persistence ->
  Lateral Movement -> Collection -> Exfiltration stages
- **Dataset**: DARPA Transparent Computing (TC) dataset

## Key Results
| Metric | Value |
|---|---|
| APT Detection Accuracy | 94.3% |
| False Positive Rate | 2.8% |
| Mean Time to Detect (MTTD) | 23 minutes (from first stage) |
| Lateral Movement Detection | 97.1% |
| Multi-Stage Chain Reconstruction | 89.4% complete chains |

## Platform Relevance
- **Gap 4 Solution**: Directly implements our solution to Multi-Stage APT Blindness
- **GNN+GAT Component**: Wang et al. (2024) + Zhou et al. (2023) together validate our Dynamic
  GNN+GAT implementation choice for APT detection (FR-04 GNN row)
- **Provenance Graph Preprocessing**: Our data pipeline includes provenance graph construction from
  CloudTrail/Azure Activity Logs for GNN input (FR-02 feature engineering)
- **DARPA TC Dataset**: Target dataset for our GNN+GAT module evaluation in Phase 3

## Limitations
- DARPA TC dataset is controlled; real cloud environments have noisier audit logs
- 23-minute MTTD means APT may have progressed significantly before detection

## Gap Addressed
- Gap 4 (Multi-Stage APT Blindness) - primary empirical validation
