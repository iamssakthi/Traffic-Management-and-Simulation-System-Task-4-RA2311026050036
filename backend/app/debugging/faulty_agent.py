"""
Deliberately faulty Tabular Q-Learning agent.
Contains controlled algorithmic and training defects:
- BUG-ALG-01: Terminal State Bootstrap Violation
- BUG-ALG-03: Premature Epsilon Collapse
- BUG-ALG-02: Inverted Congestion Reward Sign
"""
from typing import Dict, Any, List, Optional
import numpy as np

class FaultyQLearningAgent:
    def __init__(
        self,
        num_states: int = 288,
        num_actions: int = 2,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        # Controlled bug flags
        bug_terminal_bootstrap: bool = True,  # BUG-ALG-01
        bug_premature_decay: bool = True,      # BUG-ALG-03
        bug_inverted_reward: bool = False,     # BUG-ALG-02
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(seed)

        self.bug_terminal_bootstrap = bug_terminal_bootstrap
        self.bug_premature_decay = bug_premature_decay
        self.bug_inverted_reward = bug_inverted_reward

        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)
        self.training_steps: int = 0
        self.td_error_history: List[float] = []
        self.bug_triggers: Dict[str, int] = {
            "BUG-ALG-01": 0,
            "BUG-ALG-02": 0,
            "BUG-ALG-03": 0,
        }

    def select_action(self, state: int, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and self.rng.rand() < self.epsilon:
            return int(self.rng.choice(self.num_actions))

        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Q-Learning update with intentional defects.
        """
        effective_reward = reward
        if self.bug_inverted_reward:
            # BUG-ALG-02: Invert reward sign
            effective_reward = -effective_reward
            self.bug_triggers["BUG-ALG-02"] += 1

        old_val = self.q_table[state, action]

        if self.bug_terminal_bootstrap:
            # BUG-ALG-01: Omits terminal check! Bootstraps phantom value even when done=True!
            target = effective_reward + self.gamma * np.max(self.q_table[next_state])
            if done:
                self.bug_triggers["BUG-ALG-01"] += 1
        else:
            if done:
                target = effective_reward
            else:
                target = effective_reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - old_val
        self.q_table[state, action] = old_val + self.alpha * td_error
        self.training_steps += 1
        abs_err = float(abs(td_error))
        self.td_error_history.append(abs_err)
        return abs_err

    def decay_epsilon(self):
        """
        Exploration rate decay.
        If bug_premature_decay is True (BUG-ALG-03):
        Collapses aggressively by 0.70x down to 0.0 within 5-8 episodes.
        """
        if self.bug_premature_decay:
            # BUG-ALG-03: Premature Epsilon Collapse
            self.epsilon = self.epsilon * 0.70
            if self.epsilon < 0.02:
                self.epsilon = 0.0
            self.bug_triggers["BUG-ALG-03"] += 1
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

