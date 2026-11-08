"""
Threat Intelligence — IOC Extractor and STIX 2.1 Formatter
===========================================================
Based on Samara et al. (2022, IEEE Access)

What is Threat Intelligence?
  When our IDS detects an attack, it generates an Indicator of Compromise (IOC):
  specific data points that identify the attack:
    - IP addresses used by attackers
    - Attack type (DoS, U2R, etc.)
    - Timestamp of detection
    - Confidence score
    - Which features triggered the alert

  These IOCs can be SHARED with other organizations using standardized formats
  so they can defend against the same attackers.

STIX 2.1 (Structured Threat Information Expression):
  Industry-standard JSON format for sharing threat data.
  Every IOC becomes a STIX "Indicator" or "Attack Pattern" object.
  TAXII (Trusted Automated Exchange of Intelligence Information) is the
  protocol for transmitting STIX bundles between organizations.

Samara et al. (2022): threat indicator propagation dropped from HOURS
to 8.3 minutes using automated BERT-based IOC extraction + blockchain.
We implement the IOC extraction part using pattern matching + classification.

MITRE ATT&CK Cloud Matrix:
  Every attack type maps to specific MITRE ATT&CK technique IDs.
  This provides a common language for threat actors and techniques.
"""

import json
import uuid
import datetime
import re
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


# ============================================================
# MITRE ATT&CK Cloud Matrix Mapping
# Maps our IDS attack categories → MITRE ATT&CK technique IDs
# ============================================================
MITRE_ATTACK_MAPPING = {
    "Normal": None,
    "DoS": {
        "tactic": "Impact",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service",
        "sub_techniques": ["T1498.001", "T1498.002"],
        "cloud_tactic": "TA0040"
    },
    "Probe": {
        "tactic": "Discovery",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "sub_techniques": ["T1526"],
        "cloud_tactic": "TA0007"
    },
    "R2L": {
        "tactic": "Initial Access",
        "technique_id": "T1110",
        "technique_name": "Brute Force",
        "sub_techniques": ["T1110.001", "T1110.003"],
        "cloud_tactic": "TA0001"
    },
    "U2R": {
        "tactic": "Privilege Escalation",
        "technique_id": "T1548",
        "technique_name": "Abuse Elevation Control Mechanism",
        "sub_techniques": ["T1548.003"],
        "cloud_tactic": "TA0004"
    }
}

CLASS_NAMES = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}


@dataclass
class IOC:
    """
    Indicator of Compromise — the core threat data unit.

    Each IOC represents one detected attack event with all context needed
    for sharing and response.
    """
    ioc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    attack_type: str = "Unknown"
    attack_class: int = 0
    confidence: float = 0.0
    severity: str = "low"
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    model_used: str = ""
    top_features: Dict[str, float] = field(default_factory=dict)
    mitre_technique: Optional[str] = None
    mitre_tactic: Optional[str] = None
    raw_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_stix_indicator(self) -> dict:
        """
        Convert IOC to STIX 2.1 Indicator object.
        https://docs.oasis-open.org/cti/stix/v2.1/

        STIX Indicator represents a detectable pattern that
        indicates the presence of malicious or suspicious activity.
        """
        mitre_info = MITRE_ATTACK_MAPPING.get(self.attack_type, {}) or {}
        technique_id = mitre_info.get("technique_id", "T0000")
        technique_name = mitre_info.get("technique_name", "Unknown Technique")
        tactic = mitre_info.get("tactic", "Unknown")

        return {
            "type": "indicator",
            "spec_version": "2.1",
            "id": f"indicator--{self.ioc_id}",
            "created": self.timestamp,
            "modified": self.timestamp,
            "name": f"{self.attack_type} Intrusion Detected",
            "description": (
                f"AI-IDS detected {self.attack_type} attack with {self.confidence:.1%} confidence. "
                f"Model: {self.model_used}. MITRE {technique_id}: {technique_name}."
            ),
            "indicator_types": [self._get_stix_indicator_type()],
            "pattern": self._build_pattern(),
            "pattern_type": "stix",
            "valid_from": self.timestamp,
            "confidence": int(self.confidence * 100),
            "labels": [self.attack_type.lower(), f"severity-{self.severity}"],
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "url": f"https://attack.mitre.org/techniques/{technique_id}/",
                    "external_id": technique_id
                }
            ],
            "custom": {
                "x_ai_confidence": self.confidence,
                "x_attack_class": self.attack_class,
                "x_top_features": self.top_features,
                "x_model_used": self.model_used,
                "x_mitre_tactic": tactic
            }
        }

    def _get_stix_indicator_type(self) -> str:
        """Map attack type to STIX indicator type vocabulary."""
        mapping = {
            "DoS": "malicious-activity",
            "Probe": "anomalous-activity",
            "R2L": "compromised",
            "U2R": "malicious-activity"
        }
        return mapping.get(self.attack_type, "anomalous-activity")

    def _build_pattern(self) -> str:
        """Build a simple STIX pattern expression."""
        if self.source_ip:
            return f"[network-traffic:src_ref.type = 'ipv4-addr' AND network-traffic:src_ref.value = '{self.source_ip}']"
        return f"[network-traffic:dst_port > 0 AND x-ids-alert:attack_type = '{self.attack_type}']"


class IOCExtractor:
    """
    Extracts IOCs from IDS detection results and formats them for sharing.

    Integrates with the confidence router output and SHAP explanations
    to produce rich, actionable IOC records.
    """

    def __init__(self, feature_names: List[str] = None):
        """
        Args:
            feature_names: list of feature names (e.g., NSL-KDD column names)
                           Used to label feature contributions in IOCs
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(41)]
        self.ioc_history: List[IOC] = []

    def extract_from_detection(
        self,
        detection_result,          # ConfidenceRouter DetectionResult
        feature_values: np.ndarray,  # original flow feature values
        feature_importances: np.ndarray = None,  # SHAP values or importances
        source_ip: str = None,
        dest_ip: str = None
    ) -> IOC:
        """
        Create an IOC from a detection result.

        Args:
            detection_result:    DetectionResult from ConfidenceRouter
            feature_values:      (n_features,) array of flow feature values
            feature_importances: (n_features,) array of SHAP/importance scores
            source_ip:           source IP address (if available)
            dest_ip:             destination IP address (if available)
        Returns:
            IOC object ready for sharing
        """
        attack_type = CLASS_NAMES.get(detection_result.prediction, "Unknown")
        severity = detection_result.severity()
        mitre = MITRE_ATTACK_MAPPING.get(attack_type, {}) or {}

        # Get top features if importances provided
        top_features = {}
        if feature_importances is not None:
            top_idx = np.argsort(np.abs(feature_importances))[-5:][::-1]
            for i in top_idx:
                fname = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                top_features[fname] = round(float(feature_importances[i]), 4)

        ioc = IOC(
            attack_type=attack_type,
            attack_class=detection_result.prediction,
            confidence=detection_result.confidence,
            severity=severity,
            source_ip=source_ip,
            destination_ip=dest_ip,
            model_used=detection_result.model_used,
            top_features=top_features,
            mitre_technique=mitre.get("technique_id"),
            mitre_tactic=mitre.get("tactic"),
            raw_probabilities=detection_result.to_dict().get("probabilities", {})
        )

        self.ioc_history.append(ioc)
        return ioc

    def extract_from_dict(self, alert_dict: dict) -> IOC:
        """
        Create an IOC from a plain alert dictionary.
        Useful when receiving alerts from external systems.
        """
        attack_type = alert_dict.get("attack_label", "Unknown")
        mitre = MITRE_ATTACK_MAPPING.get(attack_type, {}) or {}

        ioc = IOC(
            attack_type=attack_type,
            attack_class=alert_dict.get("prediction", 0),
            confidence=alert_dict.get("confidence", 0.0),
            severity=alert_dict.get("severity", "low"),
            source_ip=alert_dict.get("source_ip"),
            destination_ip=alert_dict.get("destination_ip"),
            model_used=alert_dict.get("model_used", ""),
            mitre_technique=mitre.get("technique_id"),
            mitre_tactic=mitre.get("tactic")
        )
        self.ioc_history.append(ioc)
        return ioc

    def to_stix_bundle(self, iocs: List[IOC] = None) -> dict:
        """
        Package multiple IOCs into a STIX 2.1 Bundle for sharing.

        A STIX Bundle is the top-level container transmitted via TAXII protocol.
        Organizations receiving this bundle can immediately ingest these
        indicators into their own security tooling (MISP, Splunk, etc.)

        Args:
            iocs: list of IOCs to bundle (default: all in history)
        Returns:
            STIX 2.1 Bundle as Python dict (JSON-serializable)
        """
        if iocs is None:
            iocs = self.ioc_history

        indicators = [ioc.to_stix_indicator() for ioc in iocs]

        # Add an Identity object identifying our platform
        identity = {
            "type": "identity",
            "spec_version": "2.1",
            "id": f"identity--{uuid.uuid4()}",
            "created": datetime.datetime.utcnow().isoformat() + "Z",
            "modified": datetime.datetime.utcnow().isoformat() + "Z",
            "name": "AI-Cloud-IDS Platform",
            "identity_class": "system",
            "description": "AI-Powered Cloud Intrusion Detection and Threat Intelligence Platform — Integral University"
        }

        bundle = {
            "type": "bundle",
            "id": f"bundle--{uuid.uuid4()}",
            "objects": [identity] + indicators
        }
        return bundle

    def save_stix_bundle(self, filepath: str, iocs: List[IOC] = None):
        """Save STIX bundle to JSON file."""
        bundle = self.to_stix_bundle(iocs)
        with open(filepath, 'w') as f:
            json.dump(bundle, f, indent=2)
        print(f"[ThreatIntel] STIX bundle saved to {filepath} ({len(bundle['objects'])-1} indicators)")

    def get_attack_summary(self) -> dict:
        """
        Returns a summary of all detected attacks by type.
        Useful for SOC dashboard statistics.
        """
        if not self.ioc_history:
            return {}

        summary = {}
        for ioc in self.ioc_history:
            t = ioc.attack_type
            if t not in summary:
                summary[t] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "mitre_technique": ioc.mitre_technique,
                    "mitre_tactic": ioc.mitre_tactic,
                    "severity": ioc.severity
                }
            summary[t]["count"] += 1
            summary[t]["avg_confidence"] = (
                (summary[t]["avg_confidence"] * (summary[t]["count"] - 1) + ioc.confidence)
                / summary[t]["count"]
            )

        return summary

    def get_high_severity_iocs(self, min_confidence: float = 0.75) -> List[IOC]:
        """Return only high-confidence attack IOCs (excludes Normal)."""
        return [
            ioc for ioc in self.ioc_history
            if ioc.attack_type != "Normal" and ioc.confidence >= min_confidence
        ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.models.confidence_router import DetectionResult

    print("Testing IOC Extractor...")

    # Simulate a detection result
    dr = DetectionResult(
        prediction=4,  # U2R
        confidence=0.93,
        model_used="RF+Transformer",
        probabilities=np.array([0.02, 0.03, 0.01, 0.01, 0.93])
    )

    extractor = IOCExtractor(feature_names=[f"nsl_feat_{i}" for i in range(41)])
    fake_features = np.random.randn(41)
    fake_importances = np.random.randn(41)

    ioc = extractor.extract_from_detection(
        dr, fake_features, fake_importances,
        source_ip="192.168.1.100", dest_ip="10.0.0.5"
    )
    print(f"  IOC created: {ioc.ioc_id[:8]}... | Type: {ioc.attack_type} | Severity: {ioc.severity}")

    # Add more IOCs
    for cls in [1, 2, 3]:
        dr2 = DetectionResult(
            prediction=cls, confidence=0.8,
            model_used="RandomForest",
            probabilities=np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        )
        extractor.extract_from_detection(dr2, np.random.randn(41))

    # STIX bundle
    bundle = extractor.to_stix_bundle()
    print(f"  STIX bundle: {len(bundle['objects'])} objects")
    print(f"  Attack summary: {extractor.get_attack_summary()}")
    print("IOC Extractor: OK")
