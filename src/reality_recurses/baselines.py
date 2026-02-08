from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class RandomBaseline:
    action_dim: int
    seed: int = 123

    def __post_init__(self):
        self.rng = np.random.default_rng(int(self.seed))
        self.total_steps = 0
        self.total_information_gained = 0.0
        self.total_energy = 0.0

    def act(self, state: np.ndarray) -> np.ndarray:
        self.total_steps += 1
        return self.rng.uniform(low=-1.0, high=1.0, size=(self.action_dim,)).astype(np.float32)

    def learn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> dict[str, Any]:
        return {"stored": False}

@dataclass
class ZeroActionBaseline:
    action_dim: int

    def __post_init__(self):
        self.total_steps = 0
        self.total_information_gained = 0.0
        self.total_energy = 0.0

    def act(self, state: np.ndarray) -> np.ndarray:
        self.total_steps += 1
        return np.zeros((self.action_dim,), dtype=np.float32)

    def learn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> dict[str, Any]:
        return {"stored": False}
