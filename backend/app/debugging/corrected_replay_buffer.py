"""
Corrected, verified experience replay buffer.
Guarantees strict atomic transition alignment, uniform sampling, and zero index corruption.
"""
from typing import Tuple, Optional
import random
from collections import deque
import numpy as np
import torch

class CorrectedReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Append verified atomic transition tuple."""
        self.buffer.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(done)
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch with guaranteed atomic alignment (s, a, r, s', done).
        """
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

