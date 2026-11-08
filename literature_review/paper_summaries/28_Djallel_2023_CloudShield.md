# Paper Summary 28: CloudShield CNN-BiLSTM Architecture
**Authors:** Djallel, H., Aissa, M., Kouba, T., Bouchair, A., Kamel, N., Lakas, A.
**Title:** Real-Time Intrusion Detection System Using Deep Learning for Cloud Microservices
**Venue:** IEEE Access, 2023

---

## Architecture
- **CNN Component**: 1D Convolutional layers extract local spatial patterns within feature windows
  - Captures correlations between neighboring flow features (e.g., packet rate with byte rate)
  - Reduces feature dimensionality while preserving local correlation structure
- **BiLSTM Component**: Bidirectional LSTM captures temporal dependencies in traffic sequences
  - Forward LSTM: captures attack onset patterns
  - Backward LSTM: captures attack termination/cleanup patterns
- **Kafka Integration**: Real-time streaming ingestion via Apache Kafka at 10 Gbps
- **Kubernetes Deployment**: Each model deployed as independent pod with HPA
- **Dataset**: CSE-CIC-IDS2018 (10 days, 7 attack categories, 80+ features)

## Key Results
| Metric | Value |
|---|---|
| Detection Accuracy | 99.1% |
| Detection Latency | 3.2ms per flow |
| Throughput | 10 Gbps |
| Precision | 98.9% |
| Recall | 99.3% |
| F1-Score | 99.1% |

**Key Finding**: CNN-BiLSTM in a Kubernetes microservices deployment with Kafka streaming achieves
production-grade throughput (10 Gbps) and sub-5ms detection latency simultaneously.

## Platform Relevance
- **Architecture Blueprint**: Our CNN-BiLSTM component directly follows CloudShield's architecture
- **Kafka + K8s Validation**: Confirms our Layer 1-2 architecture (Kafka streaming + Kubernetes pods)
  is sufficient for 10 Gbps target throughput
- **Latency Target**: Our FR-04 latency target of <3.2ms for CNN-BiLSTM is based on this paper

## Limitations
- BiLSTM's bidirectional design requires buffering future traffic windows, introducing inherent
  latency in streaming scenarios (trade-off vs. unidirectional LSTM)
- Requires careful Kafka partition sizing to avoid latency spikes at 10 Gbps

## Gap Addressed
- Validates Pillar 2 (Multi-Model Detection) CNN-BiLSTM streaming component
