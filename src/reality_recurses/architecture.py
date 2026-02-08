"""
Fractal Architecture Engine
============================
Self-similar intelligence architecture where each processing node
contains a compressed replica of the whole system.

Principle: Intelligence is scale-invariant. The same sense→compress→model→act
loop operates at every level — from individual feature detection to
strategic planning. Each fractal layer processes at its natural timescale.

        ┌─────────────────────────────────────┐
        │          REALITY SCALER             │
        │  ┌───────────────────────────────┐  │
        │  │      CAUSAL MODELER           │  │
        │  │  ┌─────────────────────────┐  │  │
        │  │  │    COMPRESSOR           │  │  │
        │  │  │  ┌───────────────────┐  │  │  │
        │  │  │  │   SENSOR          │  │  │  │
        │  │  │  │  ┌─────────────┐  │  │  │  │
        │  │  │  │  │  NUCLEUS    │  │  │  │  │
        │  │  │  │  └─────────────┘  │  │  │  │
        │  │  │  └───────────────────┘  │  │  │
        │  │  └─────────────────────────┘  │  │
        │  └───────────────────────────────┘  │
        └─────────────────────────────────────┘
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class FractalScale(Enum):
    """Processing scales in the fractal hierarchy."""
    NUCLEUS = 0       # Atomic operations: bit-level, single features
    SENSOR = 1        # Feature extraction: patterns, edges, tokens
    COMPRESSOR = 2    # Abstraction: compression, invariant extraction
    MODELER = 3       # Causal modeling: prediction, world models
    SCALER = 4        # Strategic: planning, goal-directed behavior
    REALITY = 5       # Meta: self-modification, architecture evolution


@dataclass
class FractalState:
    """State of a single fractal node."""
    scale: FractalScale
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    energy_consumed: float = 0.0
    bits_processed: float = 0.0
    compression_ratio: float = 1.0
    information_gain: float = 0.0       # Bits of useful information extracted
    entropy_generated: float = 0.0
    cycle_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def efficiency(self) -> float:
        """Information gain per unit energy."""
        if self.energy_consumed <= 0:
            return float('inf')
        return self.information_gain / self.energy_consumed

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class FractalNode(ABC):
    """
    Abstract fractal processing node.
    
    Every node in the hierarchy implements the same interface:
    sense → compress → model → act → learn
    
    This self-similarity is what makes the architecture fractal.
    Each node can contain child nodes of the scale below.
    """

    def __init__(self, scale: FractalScale, max_children: int = 8,
                 energy_per_cycle: float = 1e-6):
        self.state = FractalState(scale=scale)
        self.children: list[FractalNode] = []
        self.max_children = max_children
        self.energy_per_cycle = energy_per_cycle
        self._input_buffer: list[np.ndarray] = []
        self._output_buffer: list[np.ndarray] = []
        self._learning_signal: float = 0.0

    @property
    def depth(self) -> int:
        """Depth of the fractal tree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def total_nodes(self) -> int:
        """Total nodes in the subtree."""
        return 1 + sum(c.total_nodes for c in self.children)

    def add_child(self, child: FractalNode) -> bool:
        """Add a child node if capacity allows."""
        if len(self.children) >= self.max_children:
            return False
        self.children.append(child)
        return True

    @abstractmethod
    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        """Transform raw input into this node's representation."""
        ...

    @abstractmethod
    def compress(self, sensed: np.ndarray) -> np.ndarray:
        """Compress representation, extracting invariants."""
        ...

    @abstractmethod
    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        """Build predictive model from compressed representation."""
        ...

    @abstractmethod
    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        """Generate action/output based on model."""
        ...

    def learn(self, prediction_error: float) -> None:
        """Update internal state based on prediction error."""
        self._learning_signal = prediction_error
        self.state.information_gain += max(0, -np.log2(max(1e-10, 1 - abs(prediction_error))))

    def cycle(self, raw_input: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Execute one full sense→compress→model→act cycle.
        This is the fundamental operation at every fractal scale.
        """
        # Thermodynamic accounting
        self.state.energy_consumed += self.energy_per_cycle
        self.state.cycle_count += 1
        self.state.last_active = time.time()

        # The fractal loop
        sensed = self.sense(raw_input)
        compressed = self.compress(sensed)
        prediction = self.model(compressed)
        action = self.act(prediction)

        # Track compression
        input_bits = raw_input.size * 32  # float32
        output_bits = compressed.size * 32
        self.state.bits_processed += input_bits
        if input_bits > 0:
            self.state.compression_ratio = output_bits / input_bits
        self.state.entropy_generated += max(0, input_bits - output_bits) * 0.01

        # Propagate to children
        if self.children:
            child_outputs = []
            for child in self.children:
                chunk = compressed  # Each child sees the full compressed repr
                child_out, _ = child.cycle(chunk)
                child_outputs.append(child_out)
            if child_outputs:
                action = np.mean(child_outputs, axis=0)

        self._output_buffer.append(action)
        return action, prediction

    def get_tree_report(self) -> dict[str, Any]:
        """Recursive report of the fractal tree."""
        return {
            "node_id": self.state.node_id,
            "scale": self.state.scale.name,
            "cycles": self.state.cycle_count,
            "bits_processed": self.state.bits_processed,
            "compression_ratio": self.state.compression_ratio,
            "information_gain": self.state.information_gain,
            "efficiency": self.state.efficiency,
            "children": [c.get_tree_report() for c in self.children],
            "total_nodes": self.total_nodes,
            "depth": self.depth,
        }


# ─── Concrete Fractal Nodes ───────────────────────────────────────────────────


class NucleusNode(FractalNode):
    """
    Scale 0: Atomic feature processing.
    Operates on individual values, detects basic patterns.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__(FractalScale.NUCLEUS, max_children=0, energy_per_cycle=1e-9)
        self.feature_dim = feature_dim
        self._weights = np.random.randn(feature_dim) * 0.01
        self._bias = 0.0
        self._running_mean = np.zeros(feature_dim)
        self._running_var = np.ones(feature_dim)
        self._momentum = 0.1

    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        x = raw_input.flatten()[:self.feature_dim]
        if len(x) < self.feature_dim:
            x = np.pad(x, (0, self.feature_dim - len(x)))
        # Update running statistics
        self._running_mean = (1 - self._momentum) * self._running_mean + self._momentum * x
        self._running_var = (1 - self._momentum) * self._running_var + self._momentum * (x - self._running_mean) ** 2
        return x

    def compress(self, sensed: np.ndarray) -> np.ndarray:
        # Normalize and threshold — extract only significant deviations
        normalized = (sensed - self._running_mean) / (np.sqrt(self._running_var) + 1e-8)
        # Sparse activation: keep only significant features
        mask = np.abs(normalized) > 1.0
        return normalized * mask

    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        activation = float(np.dot(compressed, self._weights) + self._bias)
        prediction = 1.0 / (1.0 + np.exp(-np.clip(activation, -10, 10)))
        return {"prediction": prediction, "activation": activation, "sparsity": float(np.mean(compressed == 0))}

    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        return np.array([model_output["prediction"]])

    def learn(self, prediction_error: float) -> None:
        super().learn(prediction_error)
        # Simple Hebbian-like update
        lr = 0.01 * abs(prediction_error)
        self._weights += lr * prediction_error * self._running_mean
        self._bias += lr * prediction_error


class SensorNode(FractalNode):
    """
    Scale 1: Feature extraction.
    Groups nucleus outputs into meaningful patterns.
    """

    def __init__(self, input_dim: int = 256, compressed_dim: int = 64, n_nuclei: int = 4):
        super().__init__(FractalScale.SENSOR, max_children=n_nuclei, energy_per_cycle=1e-7)
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        # Random projection for sensing
        self._projection = np.random.randn(input_dim, compressed_dim) / np.sqrt(input_dim)
        # Compression via learned basis
        self._basis = np.eye(compressed_dim)
        self._activation_history: list[float] = []

        # Create nucleus children
        for _ in range(n_nuclei):
            self.add_child(NucleusNode(feature_dim=compressed_dim))

    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        x = raw_input.flatten()[:self.input_dim]
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        return x @ self._projection

    def compress(self, sensed: np.ndarray) -> np.ndarray:
        # Keep top-k components by magnitude
        k = max(1, self.compressed_dim // 4)
        indices = np.argsort(np.abs(sensed))[-k:]
        sparse = np.zeros_like(sensed)
        sparse[indices] = sensed[indices]
        return sparse

    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        energy = float(np.sum(compressed ** 2))
        self._activation_history.append(energy)
        # Predict: is this input novel?
        if len(self._activation_history) > 10:
            mean_energy = np.mean(self._activation_history[-100:])
            novelty = abs(energy - mean_energy) / (mean_energy + 1e-8)
        else:
            novelty = 1.0
        return {"energy": energy, "novelty": novelty, "compressed": compressed}

    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        # Output is weighted by novelty — attend more to novel inputs
        weight = min(2.0, 0.5 + model_output["novelty"])
        return model_output["compressed"] * weight


class CompressorNode(FractalNode):
    """
    Scale 2: Abstraction layer.
    Extracts invariants across time, compresses temporal sequences.
    """

    def __init__(self, state_dim: int = 128, memory_length: int = 32, n_sensors: int = 4):
        super().__init__(FractalScale.COMPRESSOR, max_children=n_sensors, energy_per_cycle=1e-5)
        self.state_dim = state_dim
        self.memory_length = memory_length
        self._temporal_buffer = np.zeros((memory_length, state_dim))
        self._buffer_idx = 0
        self._covariance = np.eye(state_dim) * 0.01
        self._mean = np.zeros(state_dim)
        self._n_samples = 0

        for _ in range(n_sensors):
            self.add_child(SensorNode(input_dim=state_dim, compressed_dim=state_dim // 2))

    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        x = raw_input.flatten()[:self.state_dim]
        if len(x) < self.state_dim:
            x = np.pad(x, (0, self.state_dim - len(x)))
        # Store in temporal buffer
        self._temporal_buffer[self._buffer_idx % self.memory_length] = x
        self._buffer_idx += 1
        return x

    def compress(self, sensed: np.ndarray) -> np.ndarray:
        # Online PCA-like compression via running covariance
        self._n_samples += 1
        delta = sensed - self._mean
        self._mean += delta / self._n_samples
        delta2 = sensed - self._mean
        self._covariance += (np.outer(delta, delta2) - self._covariance) / self._n_samples

        # Extract principal direction
        if self._n_samples > 10:
            # Power iteration for top eigenvector (cheap)
            v = np.random.randn(self.state_dim)
            for _ in range(3):
                v = self._covariance @ v
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            # Project onto principal subspace
            projection = np.dot(sensed - self._mean, v)
            return np.array([projection, np.linalg.norm(sensed - self._mean)])
        return sensed[:2]

    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        # Temporal prediction: compare current to recent past
        if self._buffer_idx > 1:
            recent = self._temporal_buffer[max(0, self._buffer_idx - 2) % self.memory_length]
            temporal_delta = float(np.linalg.norm(
                self._temporal_buffer[(self._buffer_idx - 1) % self.memory_length] - recent
            ))
        else:
            temporal_delta = 0.0

        return {
            "principal_component": float(compressed[0]) if len(compressed) > 0 else 0.0,
            "deviation": float(compressed[1]) if len(compressed) > 1 else 0.0,
            "temporal_delta": temporal_delta,
            "buffer_fill": min(1.0, self._buffer_idx / self.memory_length),
        }

    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        return np.array([
            model_output["principal_component"],
            model_output["deviation"],
            model_output["temporal_delta"]
        ])


class CausalModelerNode(FractalNode):
    """
    Scale 3: Causal world model.
    Learns state-action-outcome triples and builds predictive models.
    """

    def __init__(self, world_dim: int = 64, action_dim: int = 16, n_compressors: int = 3):
        super().__init__(FractalScale.MODELER, max_children=n_compressors, energy_per_cycle=1e-3)
        self.world_dim = world_dim
        self.action_dim = action_dim
        # Simple linear world model: next_state = W @ [state, action]
        self._transition_matrix = np.random.randn(world_dim, world_dim + action_dim) * 0.01
        self._prediction_errors: list[float] = []
        self._causal_graph: dict[int, set[int]] = {}  # Variable → causes
        self._current_state = np.zeros(world_dim)
        self._last_action = np.zeros(action_dim)
        self._last_prediction = np.zeros(world_dim)

        for _ in range(n_compressors):
            self.add_child(CompressorNode(state_dim=world_dim))

    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        x = raw_input.flatten()[:self.world_dim]
        if len(x) < self.world_dim:
            x = np.pad(x, (0, self.world_dim - len(x)))
        self._current_state = x
        return x

    def compress(self, sensed: np.ndarray) -> np.ndarray:
        # Prediction error as compressed representation
        error = sensed - self._last_prediction
        self._prediction_errors.append(float(np.linalg.norm(error)))
        return error

    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        # Update causal graph based on prediction errors
        error_norm = float(np.linalg.norm(compressed))
        
        # Identify which dimensions have high prediction error → potential causal links
        high_error_dims = set(np.where(np.abs(compressed[:self.world_dim]) > 0.1)[0].tolist())
        
        for dim in high_error_dims:
            if dim not in self._causal_graph:
                self._causal_graph[dim] = set()
            # Hypothesize that high-error dims are causally influenced by action
            action_influences = set(range(self.action_dim))
            self._causal_graph[dim] |= action_influences

        # Make next prediction
        sa = np.concatenate([self._current_state, self._last_action])[:self.world_dim + self.action_dim]
        if len(sa) < self.world_dim + self.action_dim:
            sa = np.pad(sa, (0, self.world_dim + self.action_dim - len(sa)))
        self._last_prediction = self._transition_matrix @ sa

        return {
            "prediction_error": error_norm,
            "causal_edges": sum(len(v) for v in self._causal_graph.values()),
            "causal_variables": len(self._causal_graph),
            "mean_prediction_error": float(np.mean(self._prediction_errors[-100:])) if self._prediction_errors else 0.0,
            "next_state_prediction": self._last_prediction,
        }

    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        # Action selection: explore where prediction error is high
        pred = model_output["next_state_prediction"]
        # Information-seeking action: move toward high-uncertainty regions
        uncertainty = np.abs(pred)
        action = uncertainty[:self.action_dim] / (np.linalg.norm(uncertainty[:self.action_dim]) + 1e-8)
        self._last_action = action
        return action

    def learn(self, prediction_error: float) -> None:
        super().learn(prediction_error)
        # Update transition matrix via gradient
        lr = 0.001
        error = self._current_state - self._last_prediction
        sa = np.concatenate([self._current_state, self._last_action])[:self.world_dim + self.action_dim]
        if len(sa) < self.world_dim + self.action_dim:
            sa = np.pad(sa, (0, self.world_dim + self.action_dim - len(sa)))
        # Outer product update
        self._transition_matrix += lr * np.outer(error[:self.world_dim], sa)


class RealityScalerNode(FractalNode):
    """
    Scale 4: Strategic intelligence.
    Orchestrates all lower scales, manages scaling regime transitions,
    and optimizes information gain per unit energy.
    """

    def __init__(self, n_modelers: int = 2):
        super().__init__(FractalScale.SCALER, max_children=n_modelers, energy_per_cycle=1e-2)
        self._scaling_regime = "static"
        self._total_information_gain = 0.0
        self._total_energy = 0.0
        self._regime_history: list[dict] = []
        self._information_gain_rate: list[float] = []

        for _ in range(n_modelers):
            self.add_child(CausalModelerNode())

    def sense(self, raw_input: np.ndarray) -> np.ndarray:
        return raw_input

    def compress(self, sensed: np.ndarray) -> np.ndarray:
        # At the scaler level, compression is about regime detection
        return sensed

    def model(self, compressed: np.ndarray) -> dict[str, Any]:
        # Aggregate child information
        total_gain = sum(c.state.information_gain for c in self.children)
        total_energy = sum(c.state.energy_consumed for c in self.children)
        
        # Detect scaling regime
        if len(self._information_gain_rate) > 10:
            recent_rate = np.mean(self._information_gain_rate[-10:])
            old_rate = np.mean(self._information_gain_rate[:10]) if len(self._information_gain_rate) > 20 else recent_rate
            if old_rate > 0 and recent_rate / old_rate < 0.5:
                new_regime = "diminishing_returns"
            elif recent_rate > old_rate * 1.5:
                new_regime = "accelerating"
            else:
                new_regime = "linear"

            if new_regime != self._scaling_regime:
                self._regime_history.append({
                    "from": self._scaling_regime,
                    "to": new_regime,
                    "cycle": self.state.cycle_count,
                    "timestamp": time.time()
                })
                self._scaling_regime = new_regime

        self._total_information_gain = total_gain
        self._total_energy = total_energy

        gain_rate = total_gain / max(1, self.state.cycle_count)
        self._information_gain_rate.append(gain_rate)

        return {
            "scaling_regime": self._scaling_regime,
            "total_information_gain": total_gain,
            "total_energy": total_energy,
            "gain_per_cycle": gain_rate,
            "gain_per_energy": total_gain / max(1e-10, total_energy),
            "regime_transitions": len(self._regime_history),
        }

    def act(self, model_output: dict[str, Any]) -> np.ndarray:
        # Meta-action: resource allocation signal
        regime = model_output["scaling_regime"]
        if regime == "diminishing_returns":
            # Signal: need more physical interaction
            return np.array([0.0, 1.0])  # [static_weight, interaction_weight]
        elif regime == "accelerating":
            return np.array([0.5, 0.5])
        return np.array([0.3, 0.7])


def build_fractal_agent(depth: int = 4) -> FractalNode:
    """
    Build a complete fractal agent with the specified depth.
    Default depth=4 creates: Scaler → Modeler → Compressor → Sensor → Nucleus
    """
    if depth <= 0:
        return NucleusNode()
    elif depth == 1:
        return SensorNode()
    elif depth == 2:
        return CompressorNode()
    elif depth == 3:
        return CausalModelerNode()
    else:
        return RealityScalerNode()
