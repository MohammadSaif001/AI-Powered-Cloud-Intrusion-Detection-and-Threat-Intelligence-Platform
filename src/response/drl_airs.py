"""
Deep Reinforcement Learning — Adaptive Intrusion Response System (AIRS)
========================================================================
Based on Liu et al. (2023, IEEE Transactions on Cloud Computing)

What is DRL-AIRS?
  When the IDS detects an attack, a human SOC analyst would normally decide:
  "Should I block this IP? Isolate the VM? Or just alert and monitor?"
  Wrong decisions waste resources (false block = good traffic disrupted)
  or miss threats (false allow = attack proceeds).

  DRL-AIRS trains an AI agent to make these decisions AUTOMATICALLY:
  - STATE: what threat is happening, its severity, its confidence score
  - ACTIONS: block_ip, isolate_vm, throttle, alert_soc, revoke_creds, noop
  - REWARD: +10 if mitigation stops the attack, -5 if good traffic disrupted
  - GOAL: maximize total reward = stop attacks while minimizing disruption

  Liu et al. (2023): DQN achieves 91.3% mitigation effectiveness,
  23% fewer false responses than rule-based systems.

DQN (Deep Q-Network):
  Maintains a Q-table: Q(state, action) = expected future reward
  Uses a neural network (Q-network) to approximate Q-values
  Trains by: Bellman equation, experience replay (learns from past decisions)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# Action space for the AIRS
# ============================================================
class Actions:
    NOOP = 0           # No action — monitor only (low severity)
    ALERT_SOC = 1      # Send alert to Security Operations Center
    THROTTLE = 2       # Rate-limit suspicious IP/flow
    BLOCK_IP = 3       # Block the source IP at firewall
    ISOLATE_VM = 4     # Isolate the victim VM from network
    REVOKE_CREDS = 5   # Revoke cloud credentials (most aggressive)

    ALL = [NOOP, ALERT_SOC, THROTTLE, BLOCK_IP, ISOLATE_VM, REVOKE_CREDS]
    NAMES = {
        0: "NOOP",
        1: "ALERT_SOC",
        2: "THROTTLE",
        3: "BLOCK_IP",
        4: "ISOLATE_VM",
        5: "REVOKE_CREDS"
    }
    N = 6


@dataclass
class ThreatState:
    """
    Represents the current threat context visible to the DRL agent.

    Features:
      - attack_type:    0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
      - confidence:     model confidence in this classification (0-1)
      - severity_score: 0-1 (computed from attack type + confidence)
      - flow_rate:      normalized packets/second (0-1)
      - duration:       how long this threat has been active (0-1)
      - repeated:       is this the same source as a previous alert? (0 or 1)
    """
    attack_type: int = 0
    confidence: float = 0.0
    severity_score: float = 0.0
    flow_rate: float = 0.0
    duration: float = 0.0
    repeated: float = 0.0

    # Severity weights per attack type (U2R is most critical)
    SEVERITY_WEIGHTS = {0: 0.0, 1: 0.4, 2: 0.3, 3: 0.7, 4: 1.0}

    def compute_severity(self):
        self.severity_score = self.SEVERITY_WEIGHTS.get(self.attack_type, 0) * self.confidence

    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network input."""
        return np.array([
            self.attack_type / 4.0,  # normalize to [0,1]
            self.confidence,
            self.severity_score,
            self.flow_rate,
            self.duration,
            self.repeated
        ], dtype=np.float32)

    @classmethod
    def from_detection_result(cls, detection_result, flow_rate=0.5, duration=0.3, repeated=0.0):
        """Create ThreatState from a ConfidenceRouter DetectionResult."""
        s = cls(
            attack_type=detection_result.prediction,
            confidence=detection_result.confidence,
            flow_rate=flow_rate,
            duration=duration,
            repeated=repeated
        )
        s.compute_severity()
        return s


class CloudEnvironment:
    """
    Simulated cloud environment for DRL training.

    This is the MDP (Markov Decision Process) environment:
    - Generates threat scenarios randomly
    - Receives actions from the DRL agent
    - Returns reward based on action effectiveness

    Liu et al. (2023): trained DQN on this type of simulation,
    then deployed against real threats.
    """

    # Optimal actions for each attack type (ground truth for reward shaping)
    OPTIMAL_ACTIONS = {
        0: Actions.NOOP,       # Normal traffic → no action
        1: Actions.THROTTLE,   # DoS → throttle/rate-limit
        2: Actions.ALERT_SOC,  # Probe → alert and monitor
        3: Actions.BLOCK_IP,   # R2L → block source IP
        4: Actions.ISOLATE_VM  # U2R → isolate the VM
    }

    def __init__(self):
        self.current_state = None
        self.step_count = 0
        self.max_steps = 50  # each episode has max 50 threat events

    def reset(self) -> np.ndarray:
        """Start a new episode with a fresh threat scenario."""
        self.step_count = 0
        self.current_state = self._generate_threat()
        return self.current_state.to_array()

    def step(self, action: int) -> tuple:
        """
        Take an action in the environment.

        Args:
            action: integer from Actions.*

        Returns:
            (next_state, reward, done, info)
        """
        self.step_count += 1
        reward = self._compute_reward(action, self.current_state)
        info = {
            'attack_type': self.current_state.attack_type,
            'action_taken': Actions.NAMES.get(action, str(action)),
            'reward': reward
        }

        # Generate next threat state
        self.current_state = self._generate_threat()
        next_state = self.current_state.to_array()
        done = (self.step_count >= self.max_steps)

        return next_state, reward, done, info

    def _generate_threat(self) -> ThreatState:
        """Generate a random threat scenario (simulating real cloud traffic mix)."""
        # Attack type distribution mirrors NSL-KDD: mostly Normal/DoS, rare U2R
        attack_probs = [0.5, 0.3, 0.1, 0.07, 0.03]  # Normal, DoS, Probe, R2L, U2R
        attack_type = int(np.random.choice(5, p=attack_probs))

        s = ThreatState(
            attack_type=attack_type,
            confidence=float(np.random.uniform(0.5, 1.0)),
            flow_rate=float(np.random.uniform(0.0, 1.0)),
            duration=float(np.random.uniform(0.0, 1.0)),
            repeated=float(np.random.choice([0, 1], p=[0.7, 0.3]))
        )
        s.compute_severity()
        return s

    def _compute_reward(self, action: int, state: ThreatState) -> float:
        """
        Reward function:
        - Correct action for real attack → +10
        - Action on Normal (false positive) → -5 (disrupts good traffic)
        - Noop on real attack → -8 (missed threat)
        - Action intensity matches severity → bonus
        - Overly aggressive on low-severity threat → penalty

        Liu et al. (2023): reward = mitigation_success - disruption_cost
        """
        optimal = self.OPTIMAL_ACTIONS.get(state.attack_type, Actions.NOOP)

        if state.attack_type == 0:  # Normal traffic
            if action == Actions.NOOP:
                return +2.0   # Correctly identified as normal, no disruption
            elif action == Actions.ALERT_SOC:
                return -1.0   # Minor false alert
            else:
                return -5.0   # Unnecessary blocking/isolation = disruption

        # Real attack scenario
        if action == optimal:
            # Perfect response — reward scaled by severity
            return +10.0 * state.severity_score + 2.0

        elif action == Actions.NOOP:
            # Missed attack — worst outcome
            return -8.0 * state.severity_score

        elif action > optimal:
            # Overkill (e.g., ISOLATE_VM for a simple probe) — partial penalty
            return +3.0 - 2.0 * (action - optimal)

        else:
            # Insufficient response (e.g., just ALERT for U2R)
            return +1.0 - 3.0 * (optimal - action) * state.severity_score


class DQNNetwork(nn.Module):
    """
    Deep Q-Network: maps state → Q-values for all actions.

    For state s and action a, Q(s,a) predicts the total future reward
    if we take action a in state s and act optimally from then on.
    Agent picks action with highest Q-value.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)  # one Q-value per action
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """
    Experience Replay Buffer.
    Stores (state, action, reward, next_state, done) transitions.
    Training samples random mini-batches to break correlation.
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with epsilon-greedy exploration and experience replay.
    Implements the core DRL algorithm for AIRS.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 6,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,         # discount factor (how much to value future rewards)
        epsilon: float = 1.0,         # exploration rate (1=always random, 0=always greedy)
        epsilon_min: float = 0.05,    # minimum exploration
        epsilon_decay: float = 0.995, # decay rate per episode
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10, # update target network every N episodes
        device: str = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Online network (trained every step)
        self.q_network = DQNNetwork(state_dim, action_dim, hidden).to(self.device)
        # Target network (updated less frequently for stability)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.episode_count = 0
        self.train_losses = []
        self.episode_rewards = []

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection:
        - With probability epsilon: random action (explore)
        - Otherwise: best Q-value action (exploit)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                return int(q_values.argmax().item())

    def train_step(self) -> Optional[float]:
        """Sample a batch from replay buffer and update Q-network."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        # Q(s, a) from online network
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q: Bellman equation
        # Q_target(s, a) = r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            next_q[dones_t] = 0.0
            target_q = rewards_t + self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, env: CloudEnvironment, n_episodes: int = 500, verbose: bool = True, print_every: int = 50):
        """
        Train the DQN agent in the cloud simulation environment.

        Args:
            env:          CloudEnvironment instance
            n_episodes:   number of training episodes
            verbose:      print progress
            print_every:  print every N episodes
        """
        print(f"[DRL-AIRS] Starting training: {n_episodes} episodes")
        for ep in range(1, n_episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

                loss = self.train_step()
                if loss is not None:
                    self.train_losses.append(loss)

            self.episode_rewards.append(ep_reward)
            self.episode_count += 1

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update target network
            if ep % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if verbose and ep % print_every == 0:
                recent_reward = np.mean(self.episode_rewards[-print_every:])
                print(f"  Episode {ep:4d}/{n_episodes} | "
                      f"Avg Reward: {recent_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        print(f"[DRL-AIRS] Training complete. Final avg reward: {np.mean(self.episode_rewards[-50:]):.2f}")

    def evaluate(self, env: CloudEnvironment, n_episodes: int = 100) -> dict:
        """
        Evaluate the trained agent's mitigation effectiveness.
        Returns metrics matching Liu et al. (2023) evaluation protocol.
        """
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # fully greedy during evaluation

        total_correct = 0
        total_steps = 0
        attack_steps = 0
        attack_correct = 0
        false_positives = 0  # blocked normal traffic

        for ep in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_steps += 1

                attack_type = info['attack_type']
                optimal = CloudEnvironment.OPTIMAL_ACTIONS.get(attack_type, Actions.NOOP)
                if action == optimal:
                    total_correct += 1

                if attack_type != 0:  # actual attack
                    attack_steps += 1
                    if action != Actions.NOOP:  # took some action
                        attack_correct += 1
                    if action in [Actions.BLOCK_IP, Actions.ISOLATE_VM, Actions.REVOKE_CREDS]:
                        # Also count these as "effectively mitigated"
                        pass
                else:  # normal traffic
                    if action not in [Actions.NOOP, Actions.ALERT_SOC]:
                        false_positives += 1

        self.epsilon = old_epsilon
        mitigation_rate = attack_correct / max(attack_steps, 1)
        fp_rate = false_positives / max(total_steps - attack_steps, 1)

        return {
            "mitigation_effectiveness": round(mitigation_rate, 4),
            "false_positive_rate": round(fp_rate, 4),
            "total_steps": total_steps,
            "attack_steps": attack_steps,
        }

    def respond(self, threat_state: ThreatState) -> dict:
        """
        Make a response decision for a detected threat.
        This is the production interface called by the detection pipeline.

        Args:
            threat_state: ThreatState from a DetectionResult
        Returns:
            response dict with action name and justification
        """
        state = threat_state.to_array()
        self.epsilon = 0.0  # always greedy in production
        action = self.select_action(state)
        action_name = Actions.NAMES.get(action, "NOOP")

        return {
            "action": action,
            "action_name": action_name,
            "attack_type": threat_state.attack_type,
            "severity": threat_state.severity_score,
            "justification": self._get_justification(action, threat_state)
        }

    def _get_justification(self, action: int, state: ThreatState) -> str:
        """Human-readable explanation for the chosen action."""
        attack_names = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
        attack = attack_names.get(state.attack_type, "Unknown")
        action_name = Actions.NAMES.get(action, str(action))
        return (f"Detected {attack} (severity={state.severity_score:.2f}, "
                f"confidence={state.confidence:.2f}). Action: {action_name}.")

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }, path)
        print(f"[DRL-AIRS] Agent saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"[DRL-AIRS] Agent loaded from {path}")
        return self


if __name__ == "__main__":
    print("Testing DRL-AIRS...")
    env = CloudEnvironment()
    agent = DQNAgent(state_dim=6, action_dim=6, hidden=32)
    agent.train(env, n_episodes=50, verbose=True, print_every=10)

    metrics = agent.evaluate(env, n_episodes=20)
    print(f"\nEvaluation results: {metrics}")

    # Test respond()
    state = ThreatState(attack_type=4, confidence=0.95)
    state.compute_severity()
    response = agent.respond(state)
    print(f"U2R response: {response}")
    print("DRL-AIRS: OK")
