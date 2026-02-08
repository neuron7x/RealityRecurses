"""
Reality Scaler Agent
====================
The singular agent that scales the information reality of a language model
to the scales of physical reality.

This is the convergence point — the singularity where:
• Information physics governs every operation
• Fractal architecture processes at every scale
• Causal engine learns the structure of reality
• Divergent explorer generates and tests hypotheses
• Thermodynamic memory stores and compresses experience

    ╔══════════════════════════════════════════════════════╗
    ║              REALITY SCALER AGENT                    ║
    ║                                                      ║
    ║  Physical Reality ←──┐                               ║
    ║        │              │                               ║
    ║        ▼              │                               ║
    ║  ┌──────────┐   ┌────┴─────┐   ┌──────────────┐    ║
    ║  │ SENSORS  │──→│ FRACTAL  │──→│   CAUSAL     │    ║
    ║  │          │   │ PROCESSOR│   │   ENGINE     │    ║
    ║  └──────────┘   └──────────┘   └──────┬───────┘    ║
    ║                                        │             ║
    ║  ┌──────────────┐   ┌─────────────────▼──────────┐  ║
    ║  │ THERMODYNAMIC│←──│      DIVERGENT             │  ║
    ║  │    MEMORY    │   │      EXPLORER              │  ║
    ║  └──────┬───────┘   └─────────────────┬──────────┘  ║
    ║         │                              │             ║
    ║         ▼                              ▼             ║
    ║  ┌──────────────────────────────────────────────┐   ║
    ║  │            ACTION GENERATOR                   │   ║
    ║  │      (information-seeking behavior)           │   ║
    ║  └──────────────────────────┬───────────────────┘   ║
    ║                              │                       ║
    ║                              ▼                       ║
    ║                    Physical Reality                   ║
    ╚══════════════════════════════════════════════════════╝

Usage:
    agent = RealityScalerAgent.create(state_dim=64, action_dim=16)
    
    for step in range(1000):
        state = environment.observe()
        action = agent.act(state)
        next_state = environment.step(action)
        report = agent.learn(state, action, next_state)
"""

from __future__ import annotations

import json
import time
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from reality_recurses.information_physics import (
    LandauerEngine, BekensteinBound, EntropyBudget, ThermodynamicRegime
)
from reality_recurses.architecture import (
    RealityScalerNode, build_fractal_agent, FractalScale
)
from reality_recurses.thermodynamic_memory import ThermodynamicMemorySystem
from reality_recurses.causal_engine import CausalInferenceEngine
from reality_recurses.divergent_engine import DivergentExplorer


@dataclass
class AgentConfig:
    """Configuration for the Reality Scaler Agent."""
    state_dim: int = 64
    action_dim: int = 16
    
    # Physics
    temperature: float = 300.0
    energy_budget: float = 1.0
    overhead_factor: float = 1e6
    
    # Fractal
    fractal_depth: int = 4
    
    # Memory
    working_memory_capacity: int = 7
    episodic_memory_capacity: int = 1000
    semantic_memory_capacity: int = 500
    
    # Divergent
    n_hypothesis_populations: int = 3
    hypothesis_population_size: int = 15
    evolution_interval: int = 10
    
    # Causal
    causal_decay_rate: float = 0.001
    causal_pruning_threshold: float = 0.01
    
    # Scaling
    initial_entropy_budget: float = 1e6  # Bits
    regime_transition_threshold: float = 0.99

    # Determinism
    seed: Optional[int] = 123

    # Physical throughput bound (bits/act or bits/learn estimate)
    max_bits_per_tick: int = 10_000_000

    # Action bounds
    max_actions_per_tick: int = 1

    # Information gain gate
    min_information_gain_bits: float = 0.01


@dataclass
class ScalingMetrics:
    """Real-time metrics tracking the agent's scaling behavior."""
    total_steps: int = 0
    total_information_gained: float = 0.0
    total_energy_consumed: float = 0.0
    current_regime: str = "static"
    regime_transitions: int = 0
    
    # Per-step metrics
    information_rate: list[float] = field(default_factory=list)
    energy_rate: list[float] = field(default_factory=list)
    prediction_error_rate: list[float] = field(default_factory=list)
    compression_ratio_rate: list[float] = field(default_factory=list)

    @property
    def information_per_energy(self) -> float:
        if self.total_energy_consumed <= 0:
            return float('inf')
        return self.total_information_gained / self.total_energy_consumed

    @property
    def mean_prediction_error(self) -> float:
        if not self.prediction_error_rate:
            return float('inf')
        return float(np.mean(self.prediction_error_rate[-100:]))

    @property
    def is_scaling_efficiently(self) -> bool:
        """Check if information gain per energy is improving."""
        if len(self.information_rate) < 20:
            return True
        recent = np.mean(self.information_rate[-10:])
        older = np.mean(self.information_rate[-20:-10])
        return recent >= older * 0.9  # Allow 10% degradation


@dataclass
class DecisionEvent:
    """Append-only, structured trace of agent decisions and failures."""
    step: int
    phase: str  # act|learn|audit|error
    timestamp: float
    energy_consumed_J: float
    bits_processed: float
    bits_erased: float
    info_gained_bits: float
    details: dict[str, Any] = field(default_factory=dict)


class DecisionLog:
    """Bounded decision log to prevent silent failure."""

    def __init__(self, capacity: int = 20000):
        self.capacity = int(capacity)
        self.events: list[DecisionEvent] = []

    def append(self, ev: DecisionEvent) -> None:
        self.events.append(ev)
        if len(self.events) > self.capacity:
            self.events.pop(0)

    def to_jsonl(self) -> str:
        lines = []
        for e in self.events:
            lines.append(json.dumps({
                "step": e.step,
                "phase": e.phase,
                "timestamp": e.timestamp,
                "energy_consumed_J": e.energy_consumed_J,
                "bits_processed": e.bits_processed,
                "bits_erased": e.bits_erased,
                "info_gained_bits": e.info_gained_bits,
                "details": e.details,
            }, default=str))
        return "\n".join(lines) + ("\n" if lines else "")

class RealityScalerAgent:
    """
    The Reality Scaler: a singular agent that scales the information reality
    of a computational system to the scales of physical reality.
    
    This agent embodies the thesis that intelligence scales not through
    more parameters or data, but through deeper integration with the
    physical processes that generate information.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.metrics = ScalingMetrics()
        self.decision_log = DecisionLog(capacity=20000)

        # Determinism
        if config.seed is not None:
            np.random.seed(int(config.seed))
            random.seed(int(config.seed))

        # ── Physics Engine ──
        self.physics = LandauerEngine(
            temperature=config.temperature,
            energy_budget=config.energy_budget,
            overhead_factor=config.overhead_factor
        )
        self.entropy_budget = EntropyBudget(
            total_entropy_bits=config.initial_entropy_budget,
            regime="static"
        )
        
        # ── Fractal Processor ──
        self.fractal = build_fractal_agent(depth=config.fractal_depth)
        
        # ── Memory System ──
        self.memory = ThermodynamicMemorySystem(
            working_capacity=config.working_memory_capacity,
            episodic_capacity=config.episodic_memory_capacity,
            semantic_capacity=config.semantic_memory_capacity,
            physics=self.physics,
        )
        
        # ── Causal Engine ──
        self.causal = CausalInferenceEngine()
        
        # ── Divergent Explorer ──
        self.divergent = DivergentExplorer(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            n_populations=config.n_hypothesis_populations,
            pop_size=config.hypothesis_population_size
        )
        
        # ── Internal state ──
        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None
        self._step_count = 0
        self._created_at = time.time()

    def _log_event(self, phase: str, info_gained_bits: float = 0.0, **details) -> None:
        st = self.physics.state
        self.decision_log.append(DecisionEvent(
            step=int(self.metrics.total_steps),
            phase=str(phase),
            timestamp=time.time(),
            energy_consumed_J=float(st.energy_consumed),
            bits_processed=float(st.bits_processed),
            bits_erased=float(st.bits_erased),
            info_gained_bits=float(info_gained_bits),
            details=details,
        ))

    def self_audit(self) -> dict[str, Any]:
        """Fail-closed self-audit: returns invariant status and evidence."""
        st = self.physics.state
        mem = self.memory.get_report()
        eb = self.entropy_budget.get_state()

        invariants = {
            "energy_within_budget": bool(st.energy_consumed <= st.energy_budget + 1e-12),
            "memory_bounded_working": bool(mem["working_memory"]["size"] <= mem["working_memory"]["capacity"]),
            "memory_bounded_episodic": bool(mem["episodic_memory"]["size"] <= mem["episodic_memory"]["capacity"]),
            "memory_bounded_semantic": bool(mem["semantic_memory"]["size"] <= mem["semantic_memory"]["capacity"]),
            "entropy_accounting_sane": bool(eb["consumed_entropy_bits"] <= eb["total_entropy_bits"] + 1e-9),
        }
        ok = all(invariants.values())
        report = {"ok": ok, "invariants": invariants, "entropy_budget": eb, "memory": mem, "physics": self.physics.get_report()}
        self._log_event("audit", info_gained_bits=0.0, **report)
        return report

    @classmethod
    def create(cls, state_dim: int = 64, action_dim: int = 16,
               energy_budget: float = 1.0, **kwargs) -> RealityScalerAgent:
        """Factory method with sensible defaults."""
        config = AgentConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            energy_budget=energy_budget,
            **kwargs
        )
        return cls(config)

    # ─── Core Loop ─────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Observe state, process through all subsystems, generate action.
        The action is information-seeking: it targets maximum learning.
        """
        self._step_count += 1
        
        # 1. Thermodynamic check — can we afford to process?
        estimated_bits = int(state.size * 32)
        if estimated_bits > self.config.max_bits_per_tick:
            self._log_event('act', info_gained_bits=0.0, reason='THROUGHPUT_LIMIT', estimated_bits=estimated_bits)
            return np.zeros(self.config.action_dim)
        if not self.physics.can_afford(estimated_bits):
            # Energy depleted — return null action
            self._log_event('act', info_gained_bits=0.0, reason='ENERGY_DEPLETED', estimated_bits=estimated_bits)
            return np.zeros(self.config.action_dim)
        
        # 2. Fractal processing — multi-scale feature extraction
        fractal_output, fractal_prediction = self.fractal.cycle(state)
        
        # 3. Memory recall — what do we know about states like this?
        memories = self.memory.recall(state, memory_type="all", top_k=3)
        
        # 4. Divergent suggestion — which action maximizes learning?
        divergent_action = self.divergent.suggest_action(state)
        
        # 5. Causal reasoning — what do we expect to happen?
        if self._last_action is not None:
            causal_predictions = self.divergent.predict(state, divergent_action)
        
        # 6. Thermodynamic accounting
        self.physics.process_bits(estimated_bits, erasure_fraction=0.05, label="act")
        self._log_event('act', info_gained_bits=0.0, estimated_bits=estimated_bits)
        
        # Store for learning
        self._last_state = state.copy()
        self._last_action = divergent_action.copy()
        self._last_prediction = fractal_prediction
        
        return divergent_action

    def learn(self, state: np.ndarray, action: np.ndarray,
              next_state: np.ndarray) -> dict[str, Any]:
        """
        Learn from a state-action-outcome triple.
        This is where physical interaction becomes compressed knowledge.
        """
        try:
            self.metrics.total_steps += 1

            # 1. Compute prediction error
            if self._last_prediction is not None:
                pred_array = np.array([self._last_prediction.get("prediction", 0.0)])
                error = float(np.linalg.norm(next_state.flatten()[:len(pred_array)] - pred_array))
            else:
                error = float(np.linalg.norm(next_state))

            self.metrics.prediction_error_rate.append(error)

            # 2. Fractal learning
            self.fractal.learn(error)

            # 3. Causal learning — update causal graph
            pre_state = {"state": state}
            post_state = {"state": next_state}
            action_dict = {"action": action}
            causal_result = self.causal.process_interaction(action_dict, pre_state, post_state)

            # 4. Divergent learning — test all hypotheses
            divergent_result = self.divergent.explore(state, action, next_state)

            # 5. Memory — store experience
            memory_result = self.memory.process_experience(state, action, next_state)

            # 6. Entropy budget — consume and possibly transition
            denom = float(np.linalg.norm(next_state)) + 1e-8
            ratio = max(1e-10, error / denom)
            info_bits = float(max(0.0, -np.log2(ratio)))
            if info_bits < self.config.min_information_gain_bits:
                info_bits = 0.0
            self.entropy_budget.consume(info_bits)

            if self.entropy_budget.regime != "static":
                self.entropy_budget.inject_entropy(info_bits * 0.5, source="physical_interaction")

            # 7. Update metrics
            self.metrics.total_information_gained += info_bits
            self.metrics.total_energy_consumed = self.physics.state.energy_consumed
            self.metrics.information_rate.append(info_bits)
            self.metrics.energy_rate.append(self.physics.state.energy_consumed)
            self.metrics.compression_ratio_rate.append(self.fractal.state.compression_ratio)
            self.metrics.current_regime = self.entropy_budget.regime

            # Thermodynamic accounting
            estimated_bits = int((state.size + action.size + next_state.size) * 32)
            if estimated_bits > self.config.max_bits_per_tick:
                self._log_event("learn", info_gained_bits=info_bits, reason="THROUGHPUT_LIMIT", estimated_bits=estimated_bits)
                return self._build_step_report(error, {"graph_state": self.causal.graph.get_report()}, divergent_result, memory_result)

            self.physics.process_bits(estimated_bits, erasure_fraction=0.1, label="learn")

            self._log_event("learn", info_gained_bits=info_bits, error=error, estimated_bits=estimated_bits)
            return self._build_step_report(error, causal_result, divergent_result, memory_result)
        except Exception as e:
            self._log_event("error", info_gained_bits=0.0, where="learn", error=str(e), traceback=traceback.format_exc())
            raise
    def step(self, state: np.ndarray, environment_step_fn=None) -> dict[str, Any]:
        """
        Convenience: full act → step → learn cycle.
        If environment_step_fn is provided, uses it to get next_state.
        """
        action = self.act(state)
        
        if environment_step_fn is not None:
            next_state = environment_step_fn(action)
        else:
            # Self-simulation: predict next state from current model
            self.physics.process_bits(int(state.size * 8), erasure_fraction=0.02, label='internal_sim')
            self._log_event('act', info_gained_bits=0.0, reason='INTERNAL_SIMULATION')
            predictions = self.divergent.predict(state, action)
            if predictions:
                next_state = list(predictions.values())[0]
            else:
                next_state = state + np.random.randn(*state.shape) * 0.1
        
        return self.learn(state, action, next_state)

    # ─── Reports ───────────────────────────────────────────────────────────────

    def _build_step_report(self, error: float, causal_result: dict,
                           divergent_result: dict, memory_result: dict) -> dict[str, Any]:
        return {
            "step": self.metrics.total_steps,
            "prediction_error": error,
            "scaling_regime": self.metrics.current_regime,
            "information_gained": self.metrics.information_rate[-1] if self.metrics.information_rate else 0,
            "information_per_energy": self.metrics.information_per_energy,
            "is_scaling_efficiently": self.metrics.is_scaling_efficiently,
            "causal": {
                "edges": causal_result.get("graph_state", {}).get("n_edges", 0),
                "variables": causal_result.get("graph_state", {}).get("n_variables", 0),
            },
            "divergent": {
                k: v.get("mean_error", 0) if isinstance(v, dict) else v
                for k, v in divergent_result.items()
                if "evolution" not in k
            },
            "memory": memory_result,
        }

    def get_full_report(self) -> dict[str, Any]:
        """Comprehensive report of agent state."""
        return {
            "agent": {
                "steps": self.metrics.total_steps,
                "uptime_seconds": time.time() - self._created_at,
                "scaling_regime": self.metrics.current_regime,
                "total_information_gained": self.metrics.total_information_gained,
                "total_energy_consumed": self.metrics.total_energy_consumed,
                "information_per_energy": self.metrics.information_per_energy,
                "mean_prediction_error": self.metrics.mean_prediction_error,
                "is_scaling_efficiently": self.metrics.is_scaling_efficiently,
            },
            "physics": self.physics.get_report(),
            "entropy_budget": self.entropy_budget.get_state(),
            "fractal": self.fractal.get_tree_report(),
            "memory": self.memory.get_report(),
            "causal": self.causal.graph.get_report(),
            "divergent": self.divergent.get_report(),
        }

    def save_report(self, filepath: str):
        """Save full report to JSON."""
        report = self.get_full_report()
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, set):
                return list(obj)
            return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=convert)


# ─── Simulation Environment ───────────────────────────────────────────────────


class PhysicsSimEnvironment:
    """
    Simple physics simulation for testing the agent.
    Implements a nonlinear dynamical system with hidden causal structure.
    """

    def __init__(self, state_dim: int = 64, n_hidden_causes: int = 5):
        self.state_dim = state_dim
        self.n_hidden_causes = n_hidden_causes
        
        # Hidden causal structure
        self._causal_matrix = np.random.randn(state_dim, state_dim) * 0.1
        # Make it sparse — only some variables cause others
        mask = np.random.random((state_dim, state_dim)) > 0.9
        self._causal_matrix *= mask
        # Add self-dynamics (inertia)
        self._causal_matrix += np.eye(state_dim) * 0.8
        
        # Action influence
        self._action_matrix = np.random.randn(state_dim, 16) * 0.2
        
        # State
        self.state = np.random.randn(state_dim) * 0.5
        self._noise_level = 0.05
        self._step_count = 0

    def observe(self) -> np.ndarray:
        """Get current state with observation noise."""
        return self.state + np.random.randn(self.state_dim) * self._noise_level

    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one step: new_state = f(state, action) + noise."""
        action_padded = np.zeros(16)
        action_padded[:min(len(action), 16)] = action[:16]
        
        # Nonlinear dynamics
        self.state = np.tanh(
            self._causal_matrix @ self.state +
            self._action_matrix @ action_padded +
            np.random.randn(self.state_dim) * self._noise_level
        )
        self._step_count += 1
        return self.observe()


def run_simulation(n_steps: int = 200, state_dim: int = 64, action_dim: int = 16,
                   verbose: bool = True) -> dict[str, Any]:
    """
    Run the Reality Scaler agent in a simulated physics environment.
    """
    # Create agent and environment
    agent = RealityScalerAgent.create(
        state_dim=state_dim,
        action_dim=action_dim,
        energy_budget=10.0,
    )
    env = PhysicsSimEnvironment(state_dim=state_dim)

    if verbose:
        print("╔══════════════════════════════════════════════════════╗")
        print("║         REALITY SCALER — SIMULATION START           ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║  State dim: {state_dim:>4}  │  Action dim: {action_dim:>4}            ║")
        print(f"║  Steps: {n_steps:>7}  │  Fractal depth: 4             ║")
        print("╚══════════════════════════════════════════════════════╝")
        print()

    results = []
    for step in range(n_steps):
        state = env.observe()
        report = agent.step(state, environment_step_fn=env.step)
        results.append(report)

        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step+1:>5}/{n_steps} │ "
                  f"error: {report['prediction_error']:.4f} │ "
                  f"regime: {report['scaling_regime']:>20} │ "
                  f"info/energy: {report['information_per_energy']:.2f}")

    # Final report
    full_report = agent.get_full_report()
    
    if verbose:
        print()
        print("╔══════════════════════════════════════════════════════╗")
        print("║         REALITY SCALER — SIMULATION COMPLETE        ║")
        print("╠══════════════════════════════════════════════════════╣")
        a = full_report["agent"]
        print(f"║  Total steps:           {a['steps']:>10}                ║")
        print(f"║  Scaling regime:        {a['scaling_regime']:>20}    ║")
        print(f"║  Information gained:    {a['total_information_gained']:>10.2f} bits        ║")
        print(f"║  Mean pred error:       {a['mean_prediction_error']:>10.4f}              ║")
        print(f"║  Info/Energy:           {a['information_per_energy']:>10.4f}              ║")
        print(f"║  Scaling efficiently:   {str(a['is_scaling_efficiently']):>10}              ║")
        print("╠══════════════════════════════════════════════════════╣")
        c = full_report["causal"]
        print(f"║  Causal variables:      {c['n_variables']:>10}                ║")
        print(f"║  Causal edges:          {c['n_edges']:>10}                ║")
        d = full_report["divergent"]
        for domain, info in d["populations"].items():
            print(f"║  [{domain:>10}] gen={info['generation']:>3} "
                  f"fit={info['mean_fitness']:.3f} "
                  f"div={info['diversity']:.4f}     ║")
        m = full_report["memory"]
        print(f"║  Memory traces:         {m['total_traces']:>10}                ║")
        print(f"║  Consolidations:        {m['consolidations']:>10}                ║")
        print("╚══════════════════════════════════════════════════════╝")

    return full_report


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_simulation(n_steps=200, verbose=True)
