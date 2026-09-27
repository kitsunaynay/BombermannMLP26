from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass
class Transition:
    """One experience collected from the environment."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class ReplayBuffer:
    """Fixed-size replay memory for DQN training."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        transition = Transition(
            state=np.asarray(state, dtype=np.float16).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float16).copy(),
            done=bool(done),
            next_mask=np.asarray(next_mask, dtype=bool).copy(),
        )

        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions "
                f"from buffer of size {len(self.buffer)}"
            )

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
