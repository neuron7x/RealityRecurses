from __future__ import annotations

import numpy as np

class LinearTanhEnv:
    """Simple deterministic environment: next_state = tanh(A @ state + B @ action)."""

    def __init__(self, state_dim: int, action_dim: int, seed: int = 123):
        rng = np.random.default_rng(int(seed))
        self.A = rng.normal(size=(state_dim, state_dim)).astype(np.float32) * 0.05
        self.B = rng.normal(size=(state_dim, action_dim)).astype(np.float32) * 0.10

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = state.astype(np.float32, copy=False).reshape((-1,))
        a = action.astype(np.float32, copy=False).reshape((-1,))
        x = self.A @ s + self.B @ a
        return np.tanh(x).astype(np.float32)
