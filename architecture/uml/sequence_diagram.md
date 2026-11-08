# UML Sequence Diagram
## AI-Powered Cloud Intrusion Detection Platform
### Primary Sequence: End-to-End Intrusion Detection with Explanation and Automated Response

```mermaid
sequenceDiagram
    participant NET as Cloud Network Traffic
    participant KAFKA as Apache Kafka Broker
    participant FLINK as Apache Flink Feature Engine
    participant ROUTER as Confidence Router
    participant TRANS as Transformer Encoder
    participant CNN as CNN-BiLSTM
    participant GNN as Dynamic GNN+GAT
    participant AUTO as Autoencoder (Zero-Day)
    participant ALERT as Alert Manager
    participant SHAP as SHAP Explainer
    participant DASH as SOC Dashboard
    participant DRL as DRL AIRS Agent
    participant CLOUD as Cloud Provider API
    participant STIX as STIX/TAXII Server

    NET->>KAFKA: Raw network flow packet (VPC/eBPF)
    KAFKA->>FLINK: Ordered durable flow event stream
    FLINK->>FLINK: Extract 80+ statistical features
    FLINK->>FLINK: Construct traffic topology graph (for GNN)
    FLINK->>ROUTER: Feature vector + traffic metadata

    Note over ROUTER: Confidence-based routing decision
    ROUTER->>TRANS: High-dimensional multi-feature flow
    ROUTER->>CNN: Temporal streaming traffic window
    ROUTER->>GNN: Multi-hop traffic topology subgraph

    TRANS->>ALERT: {label: DoS, confidence: 0.97, features: [...]}
    CNN->>ALERT: {label: Probe, confidence: 0.89, features: [...]}
    GNN->>ALERT: {label: APT_LateralMove, confidence: 0.94, edges: [...]}

    Note over ALERT: Deduplication + correlation + prioritization
    ALERT->>ALERT: Correlate DoS + Probe as multi-stage attack
    ALERT->>SHAP: Request explanation for HIGH severity alert

    SHAP->>TRANS: TreeExplainer.shap_values(alert_features)
    TRANS-->>SHAP: SHAP values array [f1: 0.47, f2: -0.23, ...]
    SHAP->>ALERT: Top-10 features + plain-language summary (<350ms)

    ALERT->>DASH: Push enriched alert (detection + SHAP + severity)
    DASH->>DASH: Render SHAP force plot + feature ranking

    ALERT->>DRL: Trigger AIRS with threat context state
    DRL->>DRL: Q-network forward pass: select_action(state)
    DRL->>CLOUD: Execute: block_source_ip(attacker_ip)
    CLOUD-->>DRL: Mitigation confirmation + status

    DRL->>ALERT: Log mitigation action + reward signal
    ALERT->>STIX: Format IOC as STIX 2.1 bundle
    STIX->>STIX: Publish via TAXII 2.1 to participating organizations

    Note over NET,STIX: End-to-end detection-to-sharing latency: <10 minutes
```

---

### Secondary Sequence: Federated Learning Round

```mermaid
sequenceDiagram
    participant S as Federated Server
    participant C1 as Cloud Tenant 1 (Finance)
    participant C2 as Cloud Tenant 2 (Healthcare)
    participant C3 as Cloud Tenant 3 (E-Commerce)
    participant DP as Differential Privacy Module
    participant ML as MLflow Registry

    S->>C1: Broadcast global model weights (round N)
    S->>C2: Broadcast global model weights (round N)
    S->>C3: Broadcast global model weights (round N)

    C1->>C1: Local training on Finance traffic (non-IID)
    C2->>C2: Local training on Healthcare traffic (non-IID)
    C3->>C3: Local training on E-Commerce traffic (non-IID)

    C1->>DP: Submit local gradient update + FedProx term
    C2->>DP: Submit local gradient update + FedProx term
    C3->>DP: Submit local gradient update + FedProx term

    DP->>DP: Clip gradient norms (max_norm = 1.0)
    DP->>DP: Add Gaussian noise (epsilon=0.5, delta=1e-5)

    DP->>S: Noisy gradient from Tenant 1
    DP->>S: Noisy gradient from Tenant 2
    DP->>S: Noisy gradient from Tenant 3

    S->>S: FedAvg aggregation: weighted average of gradients
    S->>S: Update global model weights
    S->>ML: Log round N metrics (accuracy, loss, epsilon_spent)

    Note over S: Continue until convergence or 100 rounds max
    Note over C1,C3: Target: <2% accuracy loss vs centralized (Li et al. 2023)
```
