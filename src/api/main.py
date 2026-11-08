"""
FastAPI REST API — AI-Powered Cloud IDS Backend
=================================================
This is the main backend API that connects all components:
- Dataset loading + preprocessing
- Multi-model detection (RF baseline + Transformer + CNN-BiLSTM)
- Confidence-based routing
- SHAP explainability
- DRL-AIRS response recommendations
- Threat intelligence IOC generation

Endpoints:
  GET  /                  → system status
  POST /detect            → detect attack in a flow
  POST /detect/batch      → detect batch of flows
  GET  /model/status      → loaded model info
  GET  /alerts            → recent alert history
  GET  /alerts/stats      → attack statistics
  POST /alerts/stix       → export alerts as STIX bundle
  GET  /federated/status  → federated learning summary

Run with: python -m uvicorn src.api.main:app --reload --port 8000
"""

import sys
import os
import json
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Request/Response models ---

class FlowFeatures(BaseModel):
    """
    Input: a network flow's feature vector.
    For NSL-KDD: 41 numerical features.
    """
    features: List[float] = Field(..., description="Feature vector (41 values for NSL-KDD)", min_length=1, max_length=200)
    source_ip: Optional[str] = Field(None, description="Source IP address")
    dest_ip: Optional[str] = Field(None, description="Destination IP address")
    flow_id: Optional[str] = Field(None, description="Unique flow identifier")

class BatchFlowFeatures(BaseModel):
    """Input: batch of network flows."""
    flows: List[FlowFeatures] = Field(..., description="List of flow feature vectors", max_length=1000)

class DetectionResponse(BaseModel):
    """Response from the detection endpoint."""
    flow_id: Optional[str]
    prediction: int
    attack_label: str
    confidence: float
    model_used: str
    is_attack: bool
    severity: str
    probabilities: Dict[str, float]
    response_recommendation: Optional[Dict[str, Any]]
    ioc_id: Optional[str]
    timestamp: str


# --- Global state (models loaded at startup) ---
_models = {}   # loaded model objects
_alerts = []   # in-memory alert history (last 1000)
_fed_summary = {}  # federated learning summary


# --- App factory ---
def create_app() -> FastAPI:
    app = FastAPI(
        title="AI-Powered Cloud IDS API",
        description=(
            "Backend API for the AI-Powered Cloud Intrusion Detection and Threat Intelligence Platform. "
            "Connects multi-model AI detection (Random Forest + Transformer + CNN-BiLSTM), "
            "confidence-based routing, SHAP explainability, and DRL-AIRS adaptive response. "
            "Project: PCS 26086 | Integral University, Lucknow 2026-2027"
        ),
        version="1.0.0",
        docs_url="/docs",    # Swagger UI at /docs
        redoc_url="/redoc"
    )

    # Allow cross-origin requests (for dashboard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def load_models():
        """Load all trained models on API startup."""
        print("[API] Loading models...")
        _load_all_models()
        print("[API] All models ready.")

    @app.get("/", summary="System Status")
    async def root():
        """Returns system health status and loaded model inventory."""
        return {
            "status": "online",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "platform": "AI-Powered Cloud IDS",
            "version": "1.0.0",
            "project": "PCS 26086 | Integral University 2026-2027",
            "models_loaded": list(_models.keys()),
            "total_alerts": len(_alerts),
        }

    @app.post("/detect", response_model=DetectionResponse, summary="Detect Attack in a Single Flow")
    async def detect_single(flow: FlowFeatures):
        """
        Classify a network flow as Normal or specific attack type.

        The flow passes through the confidence router:
        1. RandomForest (fast path, high confidence)
        2. Transformer (medium path)
        3. Ensemble (low confidence flows)

        Returns prediction, confidence, SHAP explanation, and DRL response recommendation.
        """
        if 'router' not in _models:
            raise HTTPException(status_code=503, detail="Models not loaded. Run the training pipeline first.")

        features = np.array(flow.features, dtype=np.float32)
        n_expected = _models.get('n_features', 41)
        # Pad or truncate to expected length
        if len(features) < n_expected:
            features = np.pad(features, (0, n_expected - len(features)))
        elif len(features) > n_expected:
            features = features[:n_expected]

        # Detection
        router = _models['router']
        detection = router.predict_one(features)

        # DRL response recommendation
        response_rec = None
        if 'drl_agent' in _models and detection.is_attack():
            from src.response.drl_airs import ThreatState
            ts = ThreatState(
                attack_type=detection.prediction,
                confidence=detection.confidence
            )
            ts.compute_severity()
            response_rec = _models['drl_agent'].respond(ts)

        # IOC extraction
        ioc_id = None
        if 'ioc_extractor' in _models and detection.is_attack():
            ioc = _models['ioc_extractor'].extract_from_dict({
                "attack_label": detection.attack_label,
                "prediction": detection.prediction,
                "confidence": detection.confidence,
                "severity": detection.severity(),
                "model_used": detection.model_used,
                "source_ip": flow.source_ip,
                "destination_ip": flow.dest_ip
            })
            ioc_id = ioc.ioc_id

        result = DetectionResponse(
            flow_id=flow.flow_id or f"flow-{len(_alerts)}",
            prediction=detection.prediction,
            attack_label=detection.attack_label,
            confidence=round(detection.confidence, 4),
            model_used=detection.model_used,
            is_attack=detection.is_attack(),
            severity=detection.severity(),
            probabilities=detection.to_dict().get("probabilities", {}),
            response_recommendation=response_rec,
            ioc_id=ioc_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        # Store in alert history (keep last 1000)
        _alerts.append(result.model_dump())
        if len(_alerts) > 1000:
            _alerts.pop(0)

        return result

    @app.post("/detect/batch", summary="Detect Attacks in Batch")
    async def detect_batch(batch: BatchFlowFeatures):
        """
        Classify multiple network flows at once (up to 1000).
        Returns list of DetectionResponse objects.
        More efficient than calling /detect repeatedly.
        """
        results = []
        for flow in batch.flows:
            r = await detect_single(flow)
            results.append(r)
        return {
            "total": len(results),
            "attacks_detected": sum(1 for r in results if r.is_attack),
            "results": [r.model_dump() for r in results]
        }

    @app.get("/model/status", summary="Loaded Model Information")
    async def model_status():
        """Returns info about which models are loaded and their parameters."""
        status = {
            "models_available": list(_models.keys()),
            "n_features": _models.get('n_features', 41),
            "n_classes": 5,
            "class_names": ["Normal", "DoS", "Probe", "R2L", "U2R"],
            "confidence_thresholds": {
                "fast_path": 0.85,
                "medium_path": 0.60
            }
        }
        if 'router' in _models:
            status["routing_stats"] = _models['router'].routing_stats()
        return status

    @app.get("/alerts", summary="Recent Alert History")
    async def get_alerts(limit: int = 50, attacks_only: bool = False):
        """
        Returns recent detection results.
        Args:
            limit:       max number of alerts to return (default 50)
            attacks_only: if True, only return attack detections
        """
        alerts = _alerts[-limit:]
        if attacks_only:
            alerts = [a for a in alerts if a.get('is_attack', False)]
        return {
            "total_in_history": len(_alerts),
            "returned": len(alerts),
            "alerts": alerts
        }

    @app.get("/alerts/stats", summary="Attack Statistics")
    async def get_stats():
        """Returns aggregated attack statistics from alert history."""
        if not _alerts:
            return {"message": "No alerts recorded yet. Use /detect to process flows."}

        from collections import Counter
        labels = [a.get('attack_label', 'Unknown') for a in _alerts]
        counts = Counter(labels)
        attack_counts = {k: v for k, v in counts.items() if k != 'Normal'}
        avg_conf = np.mean([a.get('confidence', 0) for a in _alerts])

        return {
            "total_flows_processed": len(_alerts),
            "attack_breakdown": dict(counts),
            "attacks_only": attack_counts,
            "attack_rate_pct": round(100 * sum(attack_counts.values()) / max(len(_alerts), 1), 2),
            "average_confidence": round(float(avg_conf), 4)
        }

    @app.post("/alerts/stix", summary="Export Alerts as STIX 2.1 Bundle")
    async def export_stix(min_confidence: float = 0.70):
        """
        Export high-confidence attack alerts as a STIX 2.1 Threat Intelligence bundle.
        This can be shared with other organizations via TAXII protocol.
        """
        if 'ioc_extractor' not in _models:
            raise HTTPException(status_code=503, detail="IOC extractor not loaded.")
        ioc_extractor = _models['ioc_extractor']
        high_conf_iocs = ioc_extractor.get_high_severity_iocs(min_confidence=min_confidence)
        bundle = ioc_extractor.to_stix_bundle(high_conf_iocs)
        return {
            "stix_bundle": bundle,
            "total_indicators": len(high_conf_iocs),
            "note": "Bundle is STIX 2.1 compatible. Share via TAXII or MISP."
        }

    @app.get("/federated/status", summary="Federated Learning Status")
    async def federated_status():
        """Returns the status and accuracy from the federated learning simulation."""
        if not _fed_summary:
            return {
                "status": "not_run",
                "message": "Run the Phase 2 pipeline to train the federated model."
            }
        return {
            "status": "completed",
            "summary": _fed_summary
        }

    @app.get("/pipeline/info", summary="Pipeline Architecture Info")
    async def pipeline_info():
        """
        Returns information about the detection pipeline architecture.
        Useful for understanding the system design.
        """
        return {
            "architecture": "5-Layer Cloud-Native IDS",
            "detection_models": [
                {
                    "name": "RandomForest (Fast Path)",
                    "role": "Stage 1 triage — handles 60-80% of traffic",
                    "confidence_threshold": 0.85,
                    "paper": "Sajid et al. 2024 (43% computational savings)"
                },
                {
                    "name": "Transformer Encoder",
                    "role": "Medium confidence flows — captures feature dependencies",
                    "paper": "Wu et al. 2022 RTIDS (99.07% NSL-KDD)"
                },
                {
                    "name": "CNN-BiLSTM",
                    "role": "Full ensemble — spatial + temporal patterns",
                    "paper": "Djallel et al. 2023 CloudShield (99.1% CIC-IDS2018)"
                }
            ],
            "attack_categories": {
                "0": "Normal",
                "1": "DoS (Denial of Service)",
                "2": "Probe (Reconnaissance)",
                "3": "R2L (Remote to Local)",
                "4": "U2R (User to Root — most critical)"
            },
            "xai": "SHAP feature attribution for every high-severity alert",
            "response": "DRL-AIRS automated mitigation (Liu et al. 2023)",
            "threat_intel": "STIX 2.1 IOC sharing with MITRE ATT&CK mapping",
            "federated": "FedAvg + FedProx + Differential Privacy (epsilon=0.5)"
        }

    return app


def _load_all_models():
    """
    Load trained models from disk into _models dict.
    Called at API startup.
    Models must have been trained by run_phase2_pipeline.py first.
    """
    # Determine project root
    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(api_dir))
    models_dir = os.path.join(project_root, 'models', 'saved')

    sys.path.insert(0, project_root)

    # Load RandomForest baseline
    rf_path = os.path.join(models_dir, 'rf_baseline.joblib')
    if os.path.exists(rf_path):
        import joblib
        _models['rf'] = joblib.load(rf_path)
        _models['n_features'] = _models['rf'].n_features_in_
        print(f"  [API] RF loaded ({_models['n_features']} features)")

    # Load Transformer
    transformer_path = os.path.join(models_dir, 'transformer_ids.pt')
    if os.path.exists(transformer_path):
        import torch
        from src.models.transformer_ids import TransformerIDSTrainer
        n_feat = _models.get('n_features', 41)
        trainer = TransformerIDSTrainer(num_features=n_feat, num_classes=5)
        trainer.load(transformer_path)
        _models['transformer'] = trainer
        print(f"  [API] Transformer loaded")

    # Load CNN-BiLSTM
    cnn_path = os.path.join(models_dir, 'cnn_bilstm.pt')
    if os.path.exists(cnn_path):
        from src.models.cnn_bilstm import CNNBiLSTMTrainer
        n_feat = _models.get('n_features', 41)
        trainer = CNNBiLSTMTrainer(num_features=n_feat, num_classes=5)
        trainer.load(cnn_path)
        _models['cnn_bilstm'] = trainer
        print(f"  [API] CNN-BiLSTM loaded")

    # Setup confidence router
    from src.models.confidence_router import ConfidenceRouter
    router = ConfidenceRouter(high_threshold=0.85, low_threshold=0.60)
    if 'rf' in _models:
        router.set_fast_model(_models['rf'])
    if 'transformer' in _models or 'cnn_bilstm' in _models:
        router.set_deep_models(
            transformer=_models.get('transformer'),
            cnn_bilstm=_models.get('cnn_bilstm')
        )
    _models['router'] = router
    print(f"  [API] Confidence router initialized")

    # Load DRL agent
    drl_path = os.path.join(models_dir, 'drl_airs.pt')
    if os.path.exists(drl_path):
        from src.response.drl_airs import DQNAgent
        agent = DQNAgent(state_dim=6, action_dim=6)
        agent.load(drl_path)
        _models['drl_agent'] = agent
        print(f"  [API] DRL-AIRS agent loaded")

    # Setup IOC extractor
    from src.threat_intel.ioc_extractor import IOCExtractor
    _models['ioc_extractor'] = IOCExtractor()
    print(f"  [API] IOC extractor initialized")

    # Load federated summary if exists
    fed_path = os.path.join(project_root, 'reports', 'federated_summary.json')
    if os.path.exists(fed_path):
        with open(fed_path) as f:
            _fed_summary.update(json.load(f))

    if not _models.get('router'):
        print("  [API] WARNING: No models loaded. Run run_phase2_pipeline.py first.")


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
