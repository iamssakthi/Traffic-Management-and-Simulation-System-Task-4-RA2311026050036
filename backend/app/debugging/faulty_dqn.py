"""
Deliberately faulty Deep Q-Network (DQN) implementation.
Contains controllable defects:
- BUG-IMP-01: Target Network Gradient Leakage
- BUG-IMP-02: Missing Optimizer Gradient Zeroing
- BUG-ALG-04: Frozen Target Network Synchronization
- BUG-ALG-01: Terminal State Bootstrap Violation
"""
from typing import Dict, Any, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.debugging.faulty_replay_buffer import FaultyReplayBuffer

class FaultyQNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FaultyDQNAgent:
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 2,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_capacity: int = 20000,
        seed: Optional[int] = None,
        # Bug toggles
        bug_leak_gradients: bool = True,       # BUG-IMP-01
        bug_missing_zero_grad: bool = True,    # BUG-IMP-02
        bug_frozen_target: bool = False,       # BUG-ALG-04
        bug_terminal_bootstrap: bool = True,   # BUG-ALG-01
        bug_buffer_misalignment: bool = False, # BUG-IMP-03
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cpu")

        self.bug_leak_gradients = bug_leak_gradients
        self.bug_missing_zero_grad = bug_missing_zero_grad
        self.bug_frozen_target = bug_frozen_target
        self.bug_terminal_bootstrap = bug_terminal_bootstrap

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.policy_net = FaultyQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = FaultyQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()  # Unbounded MSE loss without Huber capping

        self.replay_buffer = FaultyReplayBuffer(
            capacity=buffer_capacity,
            seed=seed,
            enable_misalignment_bug=bug_buffer_misalignment
        )
        self.train_step_count: int = 0
        self.latest_loss: float = 0.0
        self.loss_history = []
        self.gradient_norm_history = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample(self.batch_size)
        b_states = b_states.to(self.device)
        b_actions = b_actions.to(self.device)
        b_rewards = b_rewards.to(self.device)
        b_next_states = b_next_states.to(self.device)
        b_dones = b_dones.to(self.device)

        q_eval = self.policy_net(b_states).gather(1, b_actions)

        # BUG-IMP-01: Omitting torch.no_grad() -> Target network computes autograd gradients!
        if self.bug_leak_gradients:
            q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)
        else:
            with torch.no_grad():
                q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)

        # BUG-ALG-01: Omitting terminal mask
        if self.bug_terminal_bootstrap:
            q_target = b_rewards + self.gamma * q_next
        else:
            q_target = b_rewards + (1.0 - b_dones) * self.gamma * q_next

        loss = self.criterion(q_eval, q_target)

        # BUG-IMP-02: Omitting optimizer.zero_grad() -> Unbounded gradient accumulation!
        if not self.bug_missing_zero_grad:
            self.optimizer.zero_grad()

        loss.backward()

        # Track gradient norm before step
        total_norm = 0.0
        for p in self.policy_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norm_history.append(float(total_norm))

        self.optimizer.step()

        self.train_step_count += 1
        loss_val = float(loss.item())
        self.latest_loss = loss_val
        self.loss_history.append(loss_val)

        # BUG-ALG-04: If frozen_target is True, synchronization is never executed
        if not self.bug_frozen_target:
            if self.train_step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

