# UML Class Diagram
## AI-Powered Cloud Intrusion Detection Platform

```mermaid
classDiagram
    class DatasetLoader {
        +String data_root
        +load_nsl_kdd() tuple
        +load_cicids2017() tuple
        +load_unsw_nb15() tuple
        +load_cic_ids2018() tuple
        -_validate_checksum(path) bool
    }

    class Preprocessor {
        +MinMaxScaler scaler
        +LabelEncoder label_enc
        +clean(df) DataFrame
        +normalize(X) DataFrame
        +encode_labels(y) ndarray
        +select_features_mutual_info(X, y, top_k) tuple
        +get_class_distribution(y) DataFrame
    }

    class ConfidenceRouter {
        +float threshold_high
        +float threshold_medium
        +route(features, traffic_type) str
        +get_model_confidence(features, model_id) float
    }

    class TransformerEncoder {
        +int d_model
        +int n_heads
        +int n_layers
        +train(X, y, epochs) dict
        +predict(X) ndarray
        +predict_proba(X) ndarray
    }

    class CNNBiLSTM {
        +int cnn_filters
        +int lstm_units
        +float dropout_rate
        +train(X, y, epochs) dict
        +predict(X) ndarray
        +stream_predict(flow) dict
    }

    class DynamicGNN {
        +int hidden_dim
        +int num_gat_heads
        +build_graph(flows) Graph
        +train(graphs, labels, epochs) dict
        +detect_apt(graph) dict
    }

    class SDNGradientBoosting {
        +int n_estimators
        +float learning_rate
        +train(X, y) dict
        +predict(X) ndarray
        +get_openflow_rule(prediction) str
    }

    class DenoisingAutoencoder {
        +int encoding_dim
        +float threshold
        +train(X_normal, epochs) dict
        +reconstruct(X) ndarray
        +detect_anomaly(X) ndarray
    }

    class LogBERT {
        +str bert_model_name
        +Drain3 drain_parser
        +parse_logs(raw_logs) list
        +fine_tune(log_sequences) dict
        +detect_anomaly(log_sequence) dict
    }

    class SHAPExplainer {
        +explain_tree(model, X) dict
        +explain_deep(model, X) dict
        +generate_summary(shap_values, feature_names) str
        +get_top_features(shap_values, k) list
    }

    class FederatedServer {
        +int num_clients
        +float proximal_mu
        +float dp_epsilon
        +aggregate_gradients(client_updates) ndarray
        +apply_differential_privacy(gradients) ndarray
        +run_round(round_num) dict
    }

    class DRLAIRSAgent {
        +int state_dim
        +int action_dim
        +float epsilon
        +ReplayBuffer memory
        +select_action(state) int
        +train_step() float
        +execute_mitigation(action, context) dict
    }

    class ThreatIntelModule {
        +extract_iocs(alert_text) list
        +format_stix_bundle(iocs) Bundle
        +publish_taxii(bundle) bool
        +query_feed(indicator_type) list
    }

    class AlertManager {
        +InfluxDB time_series_db
        +deduplicate(alerts) list
        +correlate(alerts, window_sec) list
        +prioritize(alerts) list
        +dispatch_to_shap(alert) dict
    }

    DatasetLoader --> Preprocessor : provides data to
    Preprocessor --> ConfidenceRouter : provides features to
    ConfidenceRouter --> TransformerEncoder : routes flows to
    ConfidenceRouter --> CNNBiLSTM : routes flows to
    ConfidenceRouter --> DynamicGNN : routes flows to
    ConfidenceRouter --> SDNGradientBoosting : routes flows to
    ConfidenceRouter --> DenoisingAutoencoder : routes flows to
    ConfidenceRouter --> LogBERT : routes logs to
    TransformerEncoder --> AlertManager : sends detections to
    CNNBiLSTM --> AlertManager : sends detections to
    DynamicGNN --> AlertManager : sends detections to
    SDNGradientBoosting --> AlertManager : sends detections to
    DenoisingAutoencoder --> AlertManager : sends detections to
    LogBERT --> AlertManager : sends detections to
    AlertManager --> SHAPExplainer : requests explanation from
    AlertManager --> DRLAIRSAgent : triggers response via
    AlertManager --> ThreatIntelModule : shares IOCs via
    FederatedServer --> TransformerEncoder : aggregates gradients from
    FederatedServer --> CNNBiLSTM : aggregates gradients from
```

---

## Class Responsibilities Summary

| Class | Responsibility | Pattern |
|---|---|---|
| `DatasetLoader` | Unified interface for all 4 benchmark datasets | Facade |
| `Preprocessor` | Cleaning, normalization, feature selection | Strategy |
| `ConfidenceRouter` | Traffic-to-model dispatch decisions | Chain of Responsibility |
| `TransformerEncoder` | High-dimensional feature dependency classification | ML Model |
| `CNNBiLSTM` | Spatial-temporal streaming traffic detection | ML Model |
| `DynamicGNN` | Topology-aware APT/lateral movement detection | ML Model |
| `SDNGradientBoosting` | Line-rate volumetric DDoS detection | ML Model |
| `DenoisingAutoencoder` | Unsupervised zero-day anomaly detection | ML Model |
| `LogBERT` | Cloud audit log sequence anomaly detection | ML Model |
| `SHAPExplainer` | Feature attribution for all high-severity alerts | Decorator |
| `FederatedServer` | Gradient aggregation with FedProx + DP | Aggregator |
| `DRLAIRSAgent` | Adaptive automated threat mitigation policy | Agent |
| `ThreatIntelModule` | STIX/TAXII IOC extraction and sharing | Publisher |
| `AlertManager` | Alert deduplication, correlation, and dispatch | Mediator |
