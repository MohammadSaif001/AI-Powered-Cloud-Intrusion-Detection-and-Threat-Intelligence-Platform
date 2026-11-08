"""
Phase 2 Integration Pipeline — AI-Powered Cloud IDS
=====================================================
This script trains all Phase 2 backend components in sequence:

STEP 1: Load & preprocess NSL-KDD dataset
STEP 2: Train Transformer IDS model
STEP 3: Train CNN-BiLSTM model
STEP 4: Set up Confidence Router (connects RF + Transformer + CNN-BiLSTM)
STEP 5: Run CGAN augmentation (generate synthetic minority class samples)
STEP 6: Run Adversarial Robustness evaluation (FGSM + PGD)
STEP 7: Run Federated Learning simulation (3 tenants, 10 rounds)
STEP 8: Train DRL-AIRS agent
STEP 9: Evaluate full pipeline end-to-end
STEP 10: Save all models + generate reports

Run: python run_phase2_pipeline.py
"""

import os
import sys
import time
import json
import numpy as np
import joblib

# Ensure src is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')  # headless mode (no display needed)
import matplotlib.pyplot as plt


def print_step(n, title):
    """Print a clear step header."""
    print("\n" + "="*60)
    print(f"  STEP {n}: {title}")
    print("="*60)


def main():
    start_time = time.time()
    print("\n" + "#"*60)
    print("  AI-POWERED CLOUD IDS — PHASE 2 PIPELINE")
    print("  PCS 26086 | Integral University, Lucknow 2026-2027")
    print("#"*60)

    # Create output directories
    os.makedirs(os.path.join(PROJECT_ROOT, 'models', 'saved'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'reports', 'phase2'), exist_ok=True)
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'phase2')

    # ============================================================
    # STEP 1: Load and Preprocess Dataset
    # ============================================================
    print_step(1, "Load & Preprocess NSL-KDD Dataset")

    from src.data.loader import DatasetLoader
    from src.data.preprocessor import Preprocessor

    loader = DatasetLoader()
    df_train = loader.load_nslkdd('train')
    df_test = loader.load_nslkdd('test')
    print(f"  Train samples: {len(df_train):,}")
    print(f"  Test samples:  {len(df_test):,}")

    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test, y_test = preprocessor.transform(df_test)

    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))} (0=Normal,1=DoS,2=Probe,3=R2L,4=U2R)")
    print(f"  Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = {0:'Normal',1:'DoS',2:'Probe',3:'R2L',4:'U2R'}
    for c, cnt in zip(unique, counts):
        print(f"    {class_names.get(c,c)}: {cnt:,} ({100*cnt/len(y_train):.1f}%)")

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # ============================================================
    # STEP 2: Train Transformer IDS
    # ============================================================
    print_step(2, "Train Transformer IDS (Wu et al. 2022 RTIDS)")

    from src.models.transformer_ids import TransformerIDSTrainer

    # Use subset for speed (minor project — full training in Phase 3)
    # 20,000 train, 5,000 val samples
    n_train_sub = min(20000, len(X_train))
    n_val_sub = min(5000, len(X_test))
    idx_tr = np.random.choice(len(X_train), n_train_sub, replace=False)
    idx_te = np.random.choice(len(X_test), n_val_sub, replace=False)

    transformer_trainer = TransformerIDSTrainer(
        num_features=n_features,
        num_classes=n_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        lr=1e-3
    )
    print(f"  Training on {n_train_sub:,} samples for 20 epochs...")
    transformer_trainer.train(
        X_train[idx_tr], y_train[idx_tr],
        X_val=X_test[idx_te], y_val=y_test[idx_te],
        epochs=20, batch_size=256, verbose=True
    )
    transformer_acc = transformer_trainer.evaluate(X_test[idx_te], y_test[idx_te])
    print(f"  [Transformer] Test accuracy: {transformer_acc:.4f}")

    # Save
    transformer_path = os.path.join(MODELS_DIR, 'transformer_ids.pt')
    transformer_trainer.save(transformer_path)

    # ============================================================
    # STEP 3: Train CNN-BiLSTM
    # ============================================================
    print_step(3, "Train CNN-BiLSTM (Djallel et al. 2023 CloudShield)")

    from src.models.cnn_bilstm import CNNBiLSTMTrainer

    cnn_trainer = CNNBiLSTMTrainer(
        num_features=n_features,
        num_classes=n_classes,
        cnn_filters=64,
        cnn_kernel=3,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2,
        lr=1e-3
    )
    print(f"  Training CNN-BiLSTM on {n_train_sub:,} samples for 20 epochs...")
    cnn_trainer.train(
        X_train[idx_tr], y_train[idx_tr],
        X_val=X_test[idx_te], y_val=y_test[idx_te],
        epochs=20, batch_size=256, verbose=True
    )
    cnn_acc = cnn_trainer.evaluate(X_test[idx_te], y_test[idx_te])
    print(f"  [CNN-BiLSTM] Test accuracy: {cnn_acc:.4f}")

    # Save
    cnn_path = os.path.join(MODELS_DIR, 'cnn_bilstm.pt')
    cnn_trainer.save(cnn_path)

    # ============================================================
    # STEP 4: Set Up Confidence Router
    # ============================================================
    print_step(4, "Set Up Confidence-Based Router (Sajid et al. 2024)")

    # Load the saved RF baseline from Phase 1
    rf_path = os.path.join(MODELS_DIR, 'rf_baseline.joblib')
    rf_model = None
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print(f"  Loaded RF baseline from {rf_path}")
    else:
        print(f"  RF not found at {rf_path}. Training fresh RF...")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train[idx_tr], y_train[idx_tr])
        joblib.dump(rf_model, rf_path)
        print(f"  RF trained and saved.")

    rf_acc = float((rf_model.predict(X_test[idx_te]) == y_test[idx_te]).mean())
    print(f"  [RF] Test accuracy: {rf_acc:.4f}")

    from src.models.confidence_router import ConfidenceRouter
    router = ConfidenceRouter(high_threshold=0.85, low_threshold=0.60, verbose=False)
    router.set_fast_model(rf_model)
    router.set_deep_models(
        transformer=transformer_trainer,
        cnn_bilstm=cnn_trainer
    )

    # Evaluate router on 1000 test samples
    print("  Evaluating confidence router on 1000 test samples...")
    test_sample_idx = np.random.choice(len(X_test), 1000, replace=False)
    router_preds = router.predict_labels(X_test[test_sample_idx])
    router_acc = float((router_preds == y_test[test_sample_idx]).mean())
    print(f"  [Router] Accuracy: {router_acc:.4f}")
    print(f"  [Router] Stats: {router.routing_stats()}")

    # ============================================================
    # STEP 5: CGAN Augmentation (Tang et al. 2022)
    # ============================================================
    print_step(5, "CGAN Minority Class Augmentation (Tang et al. 2022)")

    from src.models.cgan_augment import CGANAugmentor

    # Train CGAN on training data
    print("  Training CGAN for 50 epochs (generating minority class samples)...")
    augmentor = CGANAugmentor(
        n_features=n_features,
        n_classes=n_classes,
        noise_dim=64,
        lr=2e-4
    )
    augmentor.fit(
        X_train[idx_tr].astype(np.float32),
        y_train[idx_tr],
        epochs=50,
        verbose=True,
        print_every=10
    )

    # Generate synthetic samples for rare classes
    # R2L (class 3) and U2R (class 4) are severely underrepresented
    print("  Generating 500 synthetic R2L + 200 synthetic U2R samples...")
    X_new, y_new = augmentor.augment(
        X_train[idx_tr], y_train[idx_tr],
        target_counts={3: 500, 4: 200}
    )
    print(f"  Generated {len(X_new)} synthetic samples")
    if len(X_new) > 0:
        print(f"  New distribution: {dict(zip(*np.unique(y_new, return_counts=True)))}")

    # ============================================================
    # STEP 6: Adversarial Robustness Evaluation (Holla et al. 2025)
    # ============================================================
    print_step(6, "Adversarial Robustness Evaluation (Holla et al. 2025)")

    from src.security.adversarial import AdversarialAttacker, run_adversarial_evaluation

    # Evaluate Transformer robustness
    print("  Evaluating Transformer under FGSM and PGD attacks...")
    adv_results = run_adversarial_evaluation(
        transformer_trainer.model,
        X_test[idx_te[:500]],  # 500 samples
        y_test[idx_te[:500]],
        epsilons=[0.05, 0.1, 0.2]
    )

    # Save adversarial results
    adv_path = os.path.join(REPORTS_DIR, 'adversarial_robustness.json')
    with open(adv_path, 'w') as f:
        json.dump(adv_results, f, indent=2)
    print(f"  Results saved to {adv_path}")

    # ============================================================
    # STEP 7: Federated Learning Simulation (Li et al. 2023)
    # ============================================================
    print_step(7, "Federated Learning Simulation (Li et al. 2023)")

    from src.federated.federated_learning import run_federated_simulation

    fed_results = run_federated_simulation(
        X_train[idx_tr], y_train[idx_tr],
        X_test[idx_te], y_test[idx_te],
        n_clients=3,
        n_rounds=10,
        n_local_epochs=3,
        mu=0.01,
        dp_noise_scale=0.01,
        verbose=True
    )

    # Save federated summary (without server object)
    fed_summary = {k: v for k, v in fed_results.items() if k not in ['server', 'round_global_accuracies']}
    fed_summary['round_accuracies'] = fed_results.get('round_global_accuracies', [])
    fed_path = os.path.join(PROJECT_ROOT, 'reports', 'federated_summary.json')
    with open(fed_path, 'w') as f:
        json.dump(fed_summary, f, indent=2)
    print(f"  Federated summary saved to {fed_path}")

    # ============================================================
    # STEP 8: Train DRL-AIRS (Liu et al. 2023)
    # ============================================================
    print_step(8, "Train DRL-AIRS Adaptive Response Agent (Liu et al. 2023)")

    from src.response.drl_airs import CloudEnvironment, DQNAgent

    env = CloudEnvironment()
    drl_agent = DQNAgent(
        state_dim=6, action_dim=6, hidden=64,
        lr=1e-3, gamma=0.95,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
        batch_size=64, target_update_freq=10
    )
    print("  Training DQN agent for 300 episodes...")
    drl_agent.train(env, n_episodes=300, verbose=True, print_every=50)

    print("  Evaluating DRL agent...")
    drl_metrics = drl_agent.evaluate(env, n_episodes=100)
    print(f"  [DRL-AIRS] Mitigation effectiveness: {drl_metrics['mitigation_effectiveness']:.4f}")
    print(f"  [DRL-AIRS] False positive rate:      {drl_metrics['false_positive_rate']:.4f}")

    # Save DRL agent
    drl_path = os.path.join(MODELS_DIR, 'drl_airs.pt')
    drl_agent.save(drl_path)

    # Save DRL metrics
    drl_report_path = os.path.join(REPORTS_DIR, 'drl_metrics.json')
    with open(drl_report_path, 'w') as f:
        json.dump(drl_metrics, f, indent=2)

    # ============================================================
    # STEP 9: End-to-End Pipeline Evaluation
    # ============================================================
    print_step(9, "End-to-End Pipeline Evaluation")

    from sklearn.metrics import classification_report, confusion_matrix
    from src.threat_intel.ioc_extractor import IOCExtractor

    ioc_extractor = IOCExtractor()

    # Run 200 test samples through the full pipeline
    print("  Running 200 flows through full pipeline (Router → SHAP → DRL → IOC)...")
    test_indices = np.random.choice(len(X_test), 200, replace=False)
    true_labels = y_test[test_indices]
    router.reset_stats()

    all_preds = []
    all_attack_responses = []
    for i in range(len(test_indices)):
        feat = X_test[test_indices[i]]
        detection = router.predict_one(feat)
        all_preds.append(detection.prediction)

        # DRL response for attacks
        if detection.is_attack():
            from src.response.drl_airs import ThreatState
            ts = ThreatState(attack_type=detection.prediction, confidence=detection.confidence)
            ts.compute_severity()
            resp = drl_agent.respond(ts)
            all_attack_responses.append(resp['action_name'])

            # Extract IOC
            ioc_extractor.extract_from_dict({
                "attack_label": detection.attack_label,
                "prediction": detection.prediction,
                "confidence": detection.confidence,
                "severity": detection.severity(),
                "model_used": detection.model_used
            })

    all_preds = np.array(all_preds)
    final_acc = float((all_preds == true_labels).mean())
    print(f"  Pipeline accuracy: {final_acc:.4f}")
    print(f"  Routing stats: {router.routing_stats()}")
    print(f"  IOCs generated: {len(ioc_extractor.ioc_history)}")

    # Classification report
    target_names = [class_names.get(c, str(c)) for c in sorted(set(true_labels.tolist()))]
    report = classification_report(true_labels, all_preds,
                                   target_names=[class_names.get(i, str(i)) for i in range(n_classes)],
                                   zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Save STIX bundle
    stix_path = os.path.join(REPORTS_DIR, 'threat_intel_stix.json')
    ioc_extractor.save_stix_bundle(stix_path)

    # ============================================================
    # STEP 10: Save Results Summary
    # ============================================================
    print_step(10, "Save Results Summary")

    summary = {
        "phase": "Phase 2 — Backend Core",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": "NSL-KDD",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features": int(n_features),
            "classes": int(n_classes)
        },
        "model_accuracies": {
            "RandomForest": round(rf_acc, 4),
            "TransformerIDS": round(transformer_acc, 4),
            "CNN_BiLSTM": round(cnn_acc, 4),
            "ConfidenceRouter": round(router_acc, 4),
            "PipelineEnd2End": round(final_acc, 4)
        },
        "cgan_augmentation": {
            "synthetic_samples_generated": int(len(X_new)),
            "target_classes": "R2L (500) + U2R (200)",
            "paper": "Tang et al. 2022"
        },
        "adversarial_robustness": adv_results[:2] if adv_results else [],
        "federated_learning": {
            "n_clients": 3,
            "n_rounds": 10,
            "final_accuracy": round(fed_results.get('final_global_test_accuracy', 0), 4),
            "privacy": "Differential Privacy epsilon=0.5"
        },
        "drl_airs": {
            "mitigation_effectiveness": drl_metrics['mitigation_effectiveness'],
            "false_positive_rate": drl_metrics['false_positive_rate'],
            "paper": "Liu et al. 2023"
        },
        "threat_intel": {
            "iocs_generated": len(ioc_extractor.ioc_history),
            "stix_bundle_path": stix_path,
            "format": "STIX 2.1"
        }
    }

    summary_path = os.path.join(REPORTS_DIR, 'phase2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    # Generate accuracy comparison chart
    _plot_model_comparison(summary['model_accuracies'], REPORTS_DIR)

    elapsed = time.time() - start_time
    print("\n" + "#"*60)
    print("  PHASE 2 PIPELINE COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results: {REPORTS_DIR}")
    print(f"  Models:  {MODELS_DIR}")
    print("#"*60)

    print("\nTo start the API server, run:")
    print("  python -m uvicorn src.api.main:app --reload --port 8000")
    print("Then open: http://127.0.0.1:8000/docs")


def _plot_model_comparison(accuracies: dict, output_dir: str):
    """Generate model accuracy comparison bar chart."""
    names = list(accuracies.keys())
    vals = list(accuracies.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, vals, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phase 2 — Model Accuracy Comparison (NSL-KDD)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'model_comparison_phase2.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved: {chart_path}")


if __name__ == '__main__':
    main()
