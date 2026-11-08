# UML Deployment Diagram
## AI-Powered Cloud Intrusion Detection Platform
### Kubernetes Cloud-Native Deployment Architecture

```mermaid
graph TB
    subgraph Internet["External Traffic Sources"]
        VPC[VPC Flow Logs]
        EBPF[eBPF Kernel Probes]
        ISTIO[Istio Service Mesh Traces]
        CT[CloudTrail / Azure Monitor / GCP Audit Logs]
    end

    subgraph K8s["Kubernetes Cluster (AWS EKS / Azure AKS)"]
        subgraph Ingestion["Namespace: ingestion"]
            K1[Kafka Broker Pod 1]
            K2[Kafka Broker Pod 2]
            K3[Kafka Broker Pod 3]
            ZK[Zookeeper Pod]
        end

        subgraph Features["Namespace: feature-engineering"]
            FL1[Flink JobManager Pod]
            FL2[Flink TaskManager Pod 1]
            FL3[Flink TaskManager Pod 2]
        end

        subgraph Detection["Namespace: detection-engine"]
            T1[Transformer Encoder Pod]
            C1[CNN-BiLSTM Pod]
            G1[GNN-GAT Pod]
            A1[Autoencoder Pod]
            GBM[SDN-GBM Pod]
            LB[LogBERT Pod]
            CR[Confidence Router Pod]
        end

        subgraph XAI["Namespace: xai-explainability"]
            SHAP1[SHAP Explainer Pod]
            AM[Alert Manager Pod]
        end

        subgraph Response["Namespace: adaptive-response"]
            DRL1[DRL AIRS Agent Pod]
            TI1[Threat Intel / TAXII Pod]
            FL_S[Federated Learning Server Pod]
        end

        subgraph Storage["Namespace: storage"]
            INFLUX[InfluxDB StatefulSet - Time Series]
            PG[PostgreSQL StatefulSet - Alerts]
            REDIS[Redis Pod - Caching]
            S3[S3/Blob/GCS - Model Artifacts]
        end

        subgraph Dashboard["Namespace: dashboard"]
            API[FastAPI REST Pod]
            REACT[React.js Frontend Pod]
            GRAFANA[Grafana Pod]
            PROM[Prometheus Pod]
        end
    end

    subgraph ML_Dev["Developer Machine (Local)"]
        JUPYTER[Jupyter Notebook Server]
        MLFLOW[MLflow Tracking Server]
        COLAB[Google Colab - GPU Training]
    end

    VPC --> K1
    EBPF --> K1
    ISTIO --> K2
    CT --> K3

    K1 --> FL1
    K2 --> FL1
    K3 --> FL1

    FL1 --> CR
    CR --> T1
    CR --> C1
    CR --> G1
    CR --> A1
    CR --> GBM
    CR --> LB

    T1 --> AM
    C1 --> AM
    G1 --> AM
    A1 --> AM
    GBM --> AM
    LB --> AM

    AM --> SHAP1
    AM --> DRL1
    AM --> INFLUX
    AM --> PG

    SHAP1 --> API
    DRL1 --> API
    TI1 --> API

    API --> REACT
    GRAFANA --> PROM
    REDIS --> API

    FL_S --> T1
    FL_S --> C1

    MLFLOW --> S3
    COLAB --> S3
    JUPYTER --> MLFLOW
```

---

### Node Specifications (Zero-Cost Strategy)

| Node / Service | Development Phase | Production Phase (Student Cloud) |
|---|---|---|
| **Kafka Cluster** | Local Docker Compose | AWS MSK (free tier) / Azure Event Hubs |
| **Flink** | Local single-node | AWS Kinesis / Azure Stream Analytics |
| **Detection Models** | Google Colab T4 GPU | AWS EKS (Educate credits) |
| **InfluxDB** | Local Docker | InfluxDB Cloud Free Tier |
| **PostgreSQL** | Local Docker | AWS RDS Free Tier |
| **MLflow** | Local (port 5000) | Self-hosted on EC2 t2.micro (free tier) |
| **Grafana** | Local Docker | Grafana Cloud Free Tier |
| **Total Cost** | **Rs 0** | **Rs 0 (student credits)** |
