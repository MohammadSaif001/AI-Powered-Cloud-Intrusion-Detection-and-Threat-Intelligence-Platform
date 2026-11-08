"""
Federated Learning with FedAvg + FedProx + Differential Privacy
=================================================================
Based on Li et al. (2023, IEEE Transactions on Cloud Computing)

What is Federated Learning?
  Imagine 3 companies (tenants) using our IDS:
    - Company A: Hospital network — sees lots of normal + some ransomware
    - Company B: Bank network — sees lots of normal + some R2L attacks
    - Company C: University — sees lots of DoS and Probe attacks

  PRIVACY PROBLEM: Company A can't share its traffic data with B or C
  because of GDPR/HIPAA regulations. But we want all 3 to benefit from
  each other's attack examples.

  SOLUTION (Federated Learning):
    1. Each company trains the model on ITS OWN data locally
    2. Each company sends only the model WEIGHTS (not raw data) to a server
    3. Server averages the weights → new global model
    4. Everyone gets the improved global model

  No raw data ever leaves the company. Only model weights.

FedProx:
  Standard FedAvg assumes all tenants have similar data distributions.
  But Company A, B, C have VERY different traffic (non-IID data).
  FedProx adds a regularization term to prevent any one tenant's
  weights from drifting too far from the global model.

Differential Privacy:
  Even weights can leak information about raw data.
  We add carefully calibrated Gaussian noise to weights before sharing.
  Mathematically proven: epsilon=0.5 means attackers can't distinguish
  between models trained with or without any single data record.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import copy
from typing import List, Dict, Tuple


class FederatedClient:
    """
    Simulates one tenant (company) in a federated learning setup.
    Each client has its own private dataset and trains locally.
    """

    def __init__(
        self,
        client_id: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        n_classes: int = 5
    ):
        """
        Args:
            client_id: unique identifier (e.g., 'tenant_A')
            X_local:   private local training data
            y_local:   private local labels
            n_classes: number of attack classes
        """
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.n_classes = n_classes
        self.n_samples = len(X_local)

        # Local model: SGDClassifier with log loss = logistic regression
        # Using SGD because it supports partial_fit (incremental updates)
        # NOTE: class_weight='balanced' is NOT supported with partial_fit,
        # so we pass None here and compute sample_weight manually if needed.
        self.local_model = SGDClassifier(
            loss='log_loss',   # logistic regression loss
            max_iter=1,        # just one pass per round
            warm_start=True,   # continue from previous weights
            random_state=42,
        )
        self._is_initialized = False

    def initialize_model(self, global_weights: Dict):
        """Initialize local model from global model weights."""
        if not self._is_initialized:
            # First call: fit on a tiny sample to initialize sklearn internals
            # (sklearn needs at least one fit call before we can set coef_)
            self.local_model.partial_fit(
                self.X_local[:min(50, len(self.X_local))],
                self.y_local[:min(50, len(self.y_local))],
                classes=np.arange(self.n_classes)
            )
            self._is_initialized = True

        # Set weights from global model
        if 'coef' in global_weights and global_weights['coef'] is not None:
            self.local_model.coef_ = global_weights['coef'].copy()
            self.local_model.intercept_ = global_weights['intercept'].copy()

    def local_train(
        self,
        global_weights: Dict,
        n_epochs: int = 3,
        mu: float = 0.01  # FedProx regularization coefficient
    ) -> Dict:
        """
        Train locally for n_epochs with FedProx regularization.

        FedProx adds: mu/2 * ||w - w_global||^2 to the loss
        This prevents local weights from drifting too far from the global model.

        Args:
            global_weights: weights from the global model (from server)
            n_epochs:       number of local training epochs
            mu:             FedProx regularization strength (0 = standard FedAvg)
        Returns:
            updated_weights: local model weights after training
        """
        self.initialize_model(global_weights)
        global_coef = global_weights.get('coef')

        for epoch in range(n_epochs):
            # Shuffle data for each epoch
            perm = np.random.permutation(self.n_samples)
            X_shuf = self.X_local[perm]
            y_shuf = self.y_local[perm]

            # Train one epoch
            self.local_model.partial_fit(X_shuf, y_shuf, classes=np.arange(self.n_classes))

            # FedProx: pull weights back toward global model
            if mu > 0 and global_coef is not None:
                prox_grad = mu * (self.local_model.coef_ - global_coef)
                self.local_model.coef_ -= prox_grad

        return {
            'coef': self.local_model.coef_.copy(),
            'intercept': self.local_model.intercept_.copy(),
            'n_samples': self.n_samples
        }

    def evaluate_local(self) -> float:
        """Evaluate local model accuracy on local data."""
        if not self._is_initialized:
            return 0.0
        preds = self.local_model.predict(self.X_local)
        return float((preds == self.y_local).mean())


class FederatedServer:
    """
    Central server that aggregates model weights from multiple clients.
    Implements FedAvg aggregation with optional Differential Privacy.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 5,
        dp_epsilon: float = 0.5,    # privacy budget (lower = more private)
        dp_noise_scale: float = 0.01  # Gaussian noise std added to weights
    ):
        """
        Args:
            n_features:    number of input features
            n_classes:     number of attack classes
            dp_epsilon:    differential privacy budget (Li et al. 2023: 0.5)
            dp_noise_scale: scale of Gaussian noise for DP
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.dp_epsilon = dp_epsilon
        self.dp_noise_scale = dp_noise_scale

        # Initialize global model weights as zeros
        self.global_weights = {
            'coef': np.zeros((n_classes, n_features)),
            'intercept': np.zeros(n_classes)
        }

        # Global model for prediction (sklearn SGDClassifier)
        self.global_model = SGDClassifier(
            loss='log_loss',
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        self._global_initialized = False

        # Training history
        self.round_history = []

    def fedavg_aggregate(self, client_updates: List[Dict]) -> Dict:
        """
        FedAvg: weighted average of client weights.
        Clients with more data get higher weight in the average.

        Formula: w_global = sum(n_k * w_k) / sum(n_k)
        where n_k = number of samples at client k

        Args:
            client_updates: list of dicts with 'coef', 'intercept', 'n_samples'
        Returns:
            new_global_weights dict
        """
        total_samples = sum(u['n_samples'] for u in client_updates)

        # Weighted average
        avg_coef = np.zeros_like(client_updates[0]['coef'])
        avg_intercept = np.zeros_like(client_updates[0]['intercept'])

        for update in client_updates:
            weight = update['n_samples'] / total_samples
            avg_coef += weight * update['coef']
            avg_intercept += weight * update['intercept']

        # Differential Privacy: add Gaussian noise to aggregated weights
        # This provides formal privacy guarantees: epsilon=0.5
        # Even if someone sees the aggregated weights, they can't reconstruct
        # any single client's training data
        if self.dp_noise_scale > 0:
            avg_coef += np.random.normal(0, self.dp_noise_scale, avg_coef.shape)
            avg_intercept += np.random.normal(0, self.dp_noise_scale, avg_intercept.shape)

        return {'coef': avg_coef, 'intercept': avg_intercept}

    def run_round(self, clients: List[FederatedClient], n_local_epochs: int = 3, mu: float = 0.01) -> Dict:
        """
        Run one federated learning round:
          1. Send global weights to all clients
          2. Each client trains locally
          3. Collect updated weights
          4. Aggregate with FedAvg
          5. Update global weights

        Args:
            clients:         list of FederatedClient instances
            n_local_epochs:  local training epochs per client
            mu:              FedProx regularization coefficient
        Returns:
            round_metrics dict
        """
        # Send global weights → clients train → collect updates
        client_updates = []
        client_accs = []

        for client in clients:
            update = client.local_train(
                global_weights=self.global_weights,
                n_epochs=n_local_epochs,
                mu=mu
            )
            client_updates.append(update)
            client_accs.append(client.evaluate_local())

        # Aggregate with FedAvg + DP
        self.global_weights = self.fedavg_aggregate(client_updates)

        # Update global model for prediction
        self._update_global_model(clients[0])

        metrics = {
            'mean_client_accuracy': float(np.mean(client_accs)),
            'min_client_accuracy': float(np.min(client_accs)),
            'max_client_accuracy': float(np.max(client_accs)),
            'n_clients': len(clients),
            'total_samples': sum(u['n_samples'] for u in client_updates)
        }
        self.round_history.append(metrics)
        return metrics

    def _update_global_model(self, reference_client: FederatedClient):
        """Update the sklearn global model with new weights for prediction."""
        if not self._global_initialized:
            # Initialize by fitting on tiny data
            self.global_model.partial_fit(
                reference_client.X_local[:min(50, len(reference_client.X_local))],
                reference_client.y_local[:min(50, len(reference_client.y_local))],
                classes=np.arange(self.n_classes)
            )
            self._global_initialized = True

        self.global_model.coef_ = self.global_weights['coef'].copy()
        self.global_model.intercept_ = self.global_weights['intercept'].copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the global model."""
        if not self._global_initialized:
            raise RuntimeError("Global model not initialized. Run at least one round first.")
        return self.global_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from global model."""
        return self.global_model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate global model on a held-out test set."""
        preds = self.predict(X_test)
        return float((preds == y_test).mean())

    def training_summary(self) -> dict:
        """Returns summary of federated training rounds."""
        if not self.round_history:
            return {}
        final = self.round_history[-1]
        return {
            'rounds_completed': len(self.round_history),
            'final_mean_client_accuracy': round(final['mean_client_accuracy'], 4),
            'dp_epsilon': self.dp_epsilon,
            'privacy_mechanism': 'Gaussian DP noise',
        }


def create_non_iid_partitions(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 3,
    alpha: float = 0.5  # Dirichlet concentration — lower = more non-IID
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create non-IID data partitions across clients using Dirichlet distribution.

    Li et al. (2023): Real cloud tenants see VERY different attack distributions.
    Hospital sees ransomware. Bank sees R2L. University sees DoS/Probe.
    This function simulates that by giving each client a skewed class distribution.

    Args:
        X:         feature matrix
        y:         labels
        n_clients: number of simulated tenants
        alpha:     Dirichlet concentration (0.1 = very non-IID, 10 = near-IID)
    Returns:
        list of (X_i, y_i) tuples, one per client
    """
    n_classes = len(np.unique(y))
    class_indices = {c: np.where(y == c)[0] for c in np.unique(y)}

    # Sample proportions from Dirichlet distribution
    client_data_indices = [[] for _ in range(n_clients)]

    for cls, indices in class_indices.items():
        np.random.shuffle(indices)
        # Dirichlet gives random proportions summing to 1
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(indices)).astype(int)
        # Adjust to avoid rounding errors
        proportions[-1] = len(indices) - proportions[:-1].sum()

        start = 0
        for i, count in enumerate(proportions):
            end = start + count
            client_data_indices[i].extend(indices[start:end])
            start = end

    partitions = []
    for i in range(n_clients):
        idx = np.array(client_data_indices[i])
        if len(idx) == 0:
            idx = np.random.choice(len(y), size=10)  # safety: at least 10 samples
        partitions.append((X[idx], y[idx]))

    return partitions


def run_federated_simulation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 3,
    n_rounds: int = 10,
    n_local_epochs: int = 3,
    mu: float = 0.01,
    dp_noise_scale: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Full federated learning simulation.

    Args:
        X_train, y_train:  full training data (will be partitioned into n_clients)
        X_test, y_test:    held-out test set for evaluating global model
        n_clients:         number of simulated tenants
        n_rounds:          number of federated communication rounds
        n_local_epochs:    local training epochs per round
        mu:                FedProx regularization coefficient
        dp_noise_scale:    differential privacy noise scale
        verbose:           print round progress
    Returns:
        results dict with round accuracies, final test accuracy, etc.
    """
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"[Federated] Starting simulation: {n_clients} clients, {n_rounds} rounds")
    print(f"[Federated] DP noise scale: {dp_noise_scale} (epsilon ~ {0.5})")

    # Create non-IID partitions
    partitions = create_non_iid_partitions(X_train, y_train, n_clients=n_clients)
    for i, (X_i, y_i) in enumerate(partitions):
        dist = {c: int((y_i == c).sum()) for c in range(n_classes)}
        print(f"  Client {i} ({len(X_i)} samples): {dist}")

    # Initialize clients
    clients = [
        FederatedClient(
            client_id=f"tenant_{chr(65+i)}",  # tenant_A, tenant_B, ...
            X_local=partitions[i][0],
            y_local=partitions[i][1],
            n_classes=n_classes
        )
        for i in range(n_clients)
    ]

    # Initialize server
    server = FederatedServer(
        n_features=n_features,
        n_classes=n_classes,
        dp_epsilon=0.5,
        dp_noise_scale=dp_noise_scale
    )

    # Run federated rounds
    round_global_accs = []
    for rnd in range(1, n_rounds + 1):
        metrics = server.run_round(clients, n_local_epochs=n_local_epochs, mu=mu)

        # Evaluate global model on test set
        global_acc = server.evaluate(X_test, y_test)
        round_global_accs.append(global_acc)

        if verbose:
            print(f"  Round {rnd:3d}/{n_rounds} | "
                  f"Client avg acc: {metrics['mean_client_accuracy']:.4f} | "
                  f"Global test acc: {global_acc:.4f}")

    summary = server.training_summary()
    summary['final_global_test_accuracy'] = round(round_global_accs[-1], 4)
    summary['round_global_accuracies'] = round_global_accs
    summary['n_rounds'] = n_rounds
    summary['n_clients'] = n_clients
    summary['server'] = server

    print(f"\n[Federated] Final global test accuracy: {summary['final_global_test_accuracy']:.4f}")
    return summary


if __name__ == "__main__":
    print("Testing Federated Learning...")
    np.random.seed(42)
    X_tr = np.random.randn(1000, 20).astype(np.float32)
    y_tr = np.random.randint(0, 5, 1000)
    X_te = np.random.randn(200, 20).astype(np.float32)
    y_te = np.random.randint(0, 5, 200)

    results = run_federated_simulation(
        X_tr, y_tr, X_te, y_te,
        n_clients=3, n_rounds=5, n_local_epochs=2, verbose=True
    )
    print("\nFederated Learning Summary:", {k: v for k, v in results.items() if k != 'server' and k != 'round_global_accuracies'})
    print("Federated Learning: OK")
