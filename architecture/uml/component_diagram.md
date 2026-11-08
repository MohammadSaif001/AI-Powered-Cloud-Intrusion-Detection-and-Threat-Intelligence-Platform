# UML Component Diagram
## AI-Powered Cloud Intrusion Detection Platform
### Platform Component and Interface Specification

```mermaid
graph LR
    subgraph DataLayer["<<subsystem>> Data Layer"]
        DL["<<component>> DatasetLoader
[src/data/loader.py]"]
        PP["<<component>> Preprocessor
[src/data/preprocessor.py]"]
        DL -->|"load_nsl_kdd()
load_cicids2017()"| PP
    end

    subgraph FeatureLayer["<<subsystem>> Feature Engineering"]
        FE["<<component>> FlinkFeatureExtractor
[src/features/flink_extractor.py]"]
        GC["<<component>> GraphConstructor
[src/features/graph_builder.py]"]
        D3["<<component>> Drain3Parser
[src/features/drain_parser.py]"]
        PP -->|"cleaned_df"| FE
        FE -->|"topology_flows"| GC
        FE -->|"raw_log_lines"| D3
    end

    subgraph RoutingLayer["<<subsystem>> Confidence Routing"]
        CR["<<component>> ConfidenceRouter
[src/routing/router.py]"]
        FE -->|"feature_vectors"| CR
        GC -->|"graph_data"| CR
        D3 -->|"log_templates"| CR
    end

    subgraph DetectionLayer["<<subsystem>> Multi-Model Detection Engine"]
        TE["<<component>> TransformerEncoder
[src/models/transformer.py]"]
        CB["<<component>> CNNBiLSTM
[src/models/cnn_bilstm.py]"]
        GN["<<component>> DynamicGNN
[src/models/gnn_gat.py]"]
        AE["<<component>> Autoencoder
[src/models/autoencoder.py]"]
        GB["<<component>> SDNGradBoost
[src/models/sdn_gbm.py]"]
        LB["<<component>> LogBERT
[src/models/logbert.py]"]

        CR -->|"flow_features"| TE
        CR -->|"temporal_window"| CB
        CR -->|"graph_snapshot"| GN
        CR -->|"all_flows"| AE
        CR -->|"ddos_flows"| GB
        CR -->|"log_template_seq"| LB
    end

    subgraph XAILayer["<<subsystem>> XAI and Alert Management"]
        SH["<<component>> SHAPExplainer
[src/xai/shap_explainer.py]"]
        AM["<<component>> AlertManager
[src/alerts/alert_manager.py]"]
        TE -->|"DetectionResult"| AM
        CB -->|"DetectionResult"| AM
        GN -->|"DetectionResult"| AM
        AE -->|"DetectionResult"| AM
        GB -->|"DetectionResult"| AM
        LB -->|"DetectionResult"| AM
        AM -->|"HighSeverityAlert"| SH
        SH -->|"SHAPReport"| AM
    end

    subgraph ResponseLayer["<<subsystem>> Adaptive Response"]
        DR["<<component>> DRLAIRSAgent
[src/response/drl_airs.py]"]
        TI["<<component>> ThreatIntelModule
[src/threat_intel/stix_module.py]"]
        FL["<<component>> FederatedServer
[src/federated/fed_server.py]"]
        AM -->|"ThreatContext"| DR
        AM -->|"AlertPayload"| TI
    end

    subgraph APILayer["<<subsystem>> API and Dashboard"]
        API["<<component>> FastAPI REST
[api/main.py]"]
        DASH["<<component>> SOC Dashboard
[dashboard/src/App.jsx]"]
        AM -->|"EnrichedAlert"| API
        SH -->|"SHAPReport"| API
        DR -->|"MitigationStatus"| API
        API -->|"WebSocket + REST"| DASH
    end

    subgraph StorageLayer["<<subsystem>> Storage"]
        IDB["<<component>> InfluxDB
Time-series flows"]
        PDB["<<component>> PostgreSQL
Alert records"]
        RDB["<<component>> Redis
Cache + PubSub"]
        AM -->|"alert_record"| PDB
        FE -->|"flow_timeseries"| IDB
        API -->|"cache_read_write"| RDB
    end
```

---

### Interface Specifications

| Interface | Provider | Consumer | Data Format |
|---|---|---|---|
| `load_nsl_kdd()` | DatasetLoader | Preprocessor | `(DataFrame, DataFrame)` |
| `clean(df)` | Preprocessor | FeatureExtractor | `DataFrame` (no inf/NaN) |
| `feature_vectors` | FlinkExtractor | ConfidenceRouter | `ndarray [n_flows, 80]` |
| `DetectionResult` | All model components | AlertManager | `{label, confidence, features, timestamp}` |
| `HighSeverityAlert` | AlertManager | SHAPExplainer | `{alert_id, features, model_type}` |
| `SHAPReport` | SHAPExplainer | AlertManager | `{top_features, shap_values, summary_text}` |
| `ThreatContext` | AlertManager | DRLAIRSAgent | `{attack_type, severity, source, service}` |
| `AlertPayload` | AlertManager | ThreatIntelModule | `{ioc_candidates, alert_text, timestamp}` |
| `POST /api/v1/detect` | FastAPI | External | JSON: `{features: [...]}` |
| `GET /api/v1/alerts` | FastAPI | SOC Dashboard | JSON: `[{alert_id, severity, shap_url}]` |
