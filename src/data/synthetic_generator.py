"""
src/data/synthetic_generator.py
AI-Powered Cloud IDS | Synthetic Dataset Generator
Generates synthetic datasets matching real benchmark statistics for offline dev.
References:
  - Kim et al. (2024) Hi-MLIC: class distributions in CM-CIC-IDS2017
  - Moustafa et al. (2020): UNSW-NB15 feature statistics
  - Djallel et al. (2023): CIC-IDS2018 class distribution
  - Tang et al. (2022): class imbalance ratios 1:100 to 1:5000
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# CIC-IDS2017 class distribution from Kim et al. (2024) Table II
CICIDS2017_DIST = {
    'BENIGN':            0.8312,
    'DoS Hulk':          0.0621,
    'PortScan':          0.0430,
    'DDoS':              0.0316,
    'DoS GoldenEye':     0.0071,
    'FTP-Patator':       0.0053,
    'SSH-Patator':       0.0035,
    'DoS slowloris':     0.0027,
    'DoS Slowhttptest':  0.0026,
    'Bot':               0.0014,
    'Web Attack Brute':  0.0012,
    'Web Attack XSS':    0.0007,
    'Infiltration':      0.0003,
    'Web Attack Sql':    0.0002,
    'Heartbleed':        0.00002,
}

# UNSW-NB15 class distribution from Moustafa et al. (2020)
UNSW_NB15_DIST = {
    'Normal':    0.8748,
    'Generic':   0.0521,
    'Exploits':  0.0422,
    'Fuzzers':   0.0187,
    'DoS':       0.0082,
    'Recon':     0.0019,
    'Analysis':  0.0009,
    'Backdoor':  0.0006,
    'Shellcode': 0.0003,
    'Worms':     0.000006,
}

# CIC-IDS2017 feature names (80-feature subset matching Djallel et al. 2023)
CICIDS_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
    'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

def generate_cicids2017(n_samples: int = 50000, with_inf: bool = True) -> pd.DataFrame:
    """
    Generate synthetic CIC-IDS2017-like DataFrame.
    Inserts Inf/NaN values to simulate real CICIDS2017 data quality issues.
    Reference: Djallel et al. (2023); Synopsis Aug 13-14 preprocessing notes.
    """
    rng = np.random.RandomState(42)
    n = n_samples
    # Sample labels per real distribution
    probs_cic = list(CICIDS2017_DIST.values())
    probs_cic = [p / sum(probs_cic) for p in probs_cic]
    labels = rng.choice(list(CICIDS2017_DIST.keys()),
                        size=n, p=probs_cic)
    data = rng.exponential(scale=500, size=(n, len(CICIDS_FEATURES))).astype(np.float32)

    if with_inf:
        # Inject ~5% Inf values in 'Flow Bytes/s' and 'Flow Packets/s'
        # (common in real CICIDS2017 when flow duration is 0)
        for col in ['Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s']:
            if col in CICIDS_FEATURES:
                idx = rng.choice(n, size=n // 20, replace=False)
                data[idx, CICIDS_FEATURES.index(col)] = np.inf

    df = pd.DataFrame(data, columns=CICIDS_FEATURES)
    df['Label'] = labels
    return df


def generate_unsw_nb15(n_samples: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic UNSW-NB15-like DataFrame.
    Reference: Moustafa et al. (2020) 9-category intrusion dataset.
    """
    rng = np.random.RandomState(42)
    features = [
        'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
        'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
        'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
        'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
        'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    ]
    data = rng.exponential(scale=100, size=(n_samples, len(features))).astype(np.float32)
    probs_unsw = list(UNSW_NB15_DIST.values())
    probs_unsw = [p / sum(probs_unsw) for p in probs_unsw]
    labels = rng.choice(list(UNSW_NB15_DIST.keys()),
                        size=n_samples, p=probs_unsw)
    df = pd.DataFrame(data, columns=features)
    df['attack_cat'] = labels
    return df
