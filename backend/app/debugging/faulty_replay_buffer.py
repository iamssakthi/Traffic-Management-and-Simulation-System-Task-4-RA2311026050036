"""
Deliberately faulty experience replay buffer containing controllable implementation bugs.
Demonstrates BUG-IMP-03: Off-by-one transition misalignment.
"""
from typing import Tuple, List, Optional
import random
from collections import deque
import numpy as np
import torch

class FaultyReplayBuffer:
    def __init__(
        self,
        capacity: int = 20000,
        seed: Optional[int] = None,
        enable_misalignment_bug: bool = True,
    ):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.enable_misalignment_bug = enable_misalignment_bug

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Append transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch.
        If enable_misalignment_bug is True (BUG-IMP-03):
        Corrupts alignment by shifting next_states by 1 offset index.
        """
        n = len(self.buffer)
        if n < batch_size:
            raise ValueError(f"Buffer size {n} < batch_size {batch_size}")

        sampled_indices = self.rng.sample(range(n), batch_size)
        raw_list = list(self.buffer)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in sampled_indices:
            s, a, r, s_prime, d = raw_list[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            dones.append(d)

            if self.enable_misalignment_bug:
                # BUG-IMP-03: Off-by-one index corruption!
                # Pairs current state with next state from a completely different transition!
                corrupted_idx = (idx + 1) % n
                next_states.append(raw_list[corrupted_idx][3])
            else:
                next_states.append(s_prime)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

