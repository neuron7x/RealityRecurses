"""
Causal Engine
=============
Learns causal structure from state-action-outcome triples.
Implements interventional reasoning and counterfactual inference.

Core insight: Physical interaction generates causal data that passive
observation cannot. An agent that acts in the world can distinguish
correlation from causation through intervention.

Causal Graph: G = (V, E) where
  V = observable variables
  E = directed causal edges (A → B means A causes B)
  
Each edge has:
  - strength: how strongly A influences B
  - confidence: how many interventions support this edge
  - latency: temporal delay between cause and effect
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class CausalEdge:
    """A directed causal relationship between two variables."""
    source: str
    target: str
    strength: float = 0.0
    confidence: float = 0.0
    latency: float = 0.0            # Time steps between cause and effect
    interventions: int = 0            # Number of interventions supporting this
    observations: int = 0             # Number of observations supporting this
    last_updated: float = field(default_factory=time.time)

    @property
    def total_evidence(self) -> int:
        return self.interventions + self.observations

    @property
    def intervention_ratio(self) -> float:
        """Fraction of evidence from interventions vs passive observation."""
        if self.total_evidence == 0:
            return 0.0
        return self.interventions / self.total_evidence

    def update_from_intervention(self, observed_strength: float):
        """Update edge based on interventional evidence (strongest form)."""
        self.interventions += 1
        lr = 1.0 / (1.0 + self.interventions)
        self.strength = (1 - lr) * self.strength + lr * observed_strength
        self.confidence = min(1.0, self.confidence + 0.1)
        self.last_updated = time.time()

    def update_from_observation(self, observed_correlation: float):
        """Update edge based on observational evidence (weaker form)."""
        self.observations += 1
        lr = 1.0 / (1.0 + self.total_evidence)
        # Observational evidence is weighted less
        self.strength = (1 - lr * 0.5) * self.strength + lr * 0.5 * observed_correlation
        self.confidence = min(0.8, self.confidence + 0.02)  # Cap at 0.8 for obs only
        self.last_updated = time.time()

    def decay(self, rate: float = 0.001):
        """Decay confidence over time (forgetting)."""
        self.confidence *= (1 - rate)


@dataclass
class CausalVariable:
    """A variable in the causal graph."""
    name: str
    dim: int = 1                      # Dimensionality
    current_value: Optional[np.ndarray] = None
    history: list[np.ndarray] = field(default_factory=list)
    max_history: int = 1000
    is_action: bool = False           # True if this is an action variable
    is_observable: bool = True

    def observe(self, value: np.ndarray):
        self.current_value = value
        self.history.append(value.copy())
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


class CausalGraph:
    """
    Dynamic causal graph that learns structure from interactions.
    
    The graph evolves as the agent interacts with its environment:
    1. Interventions (actions) create strong causal evidence
    2. Observations create weaker correlational evidence
    3. Edges are pruned when evidence decays
    4. New variables are added when novel features are detected
    """

    def __init__(self, decay_rate: float = 0.001, pruning_threshold: float = 0.01):
        self.variables: dict[str, CausalVariable] = {}
        self.edges: dict[tuple[str, str], CausalEdge] = {}
        self.decay_rate = decay_rate
        self.pruning_threshold = pruning_threshold
        self._intervention_log: list[dict] = []

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        """Graph density: actual edges / possible edges."""
        n = self.n_variables
        max_edges = n * (n - 1)
        return self.n_edges / max_edges if max_edges > 0 else 0.0

    def add_variable(self, name: str, dim: int = 1, is_action: bool = False) -> CausalVariable:
        """Register a new variable in the causal graph."""
        if name not in self.variables:
            self.variables[name] = CausalVariable(name=name, dim=dim, is_action=is_action)
        return self.variables[name]

    def observe(self, variable_name: str, value: np.ndarray):
        """Record an observation of a variable."""
        if variable_name not in self.variables:
            self.add_variable(variable_name, dim=value.shape[0] if value.ndim > 0 else 1)
        self.variables[variable_name].observe(value)

    def intervene(self, action_var: str, action_value: np.ndarray,
                  observed_effects: dict[str, np.ndarray]) -> dict[str, float]:
        """
        Record an intervention and its observed effects.
        This is the primary mechanism for learning causal structure.
        
        Args:
            action_var: Name of the variable being intervened on
            action_value: Value set by intervention
            observed_effects: {variable_name: observed_change}
            
        Returns:
            Estimated causal strengths for each affected variable
        """
        if action_var not in self.variables:
            self.add_variable(action_var, dim=action_value.shape[0] if action_value.ndim > 0 else 1,
                            is_action=True)

        self.variables[action_var].observe(action_value)
        strengths = {}

        for target_name, effect in observed_effects.items():
            if target_name not in self.variables:
                self.add_variable(target_name, dim=effect.shape[0] if effect.ndim > 0 else 1)
            self.variables[target_name].observe(effect)

            # Compute causal strength as normalized effect magnitude
            effect_magnitude = float(np.linalg.norm(effect))
            action_magnitude = float(np.linalg.norm(action_value)) + 1e-8
            strength = effect_magnitude / action_magnitude

            # Update or create edge
            edge_key = (action_var, target_name)
            if edge_key not in self.edges:
                self.edges[edge_key] = CausalEdge(source=action_var, target=target_name)
            self.edges[edge_key].update_from_intervention(strength)
            strengths[target_name] = strength

        self._intervention_log.append({
            "action": action_var,
            "effects": list(observed_effects.keys()),
            "strengths": strengths,
            "timestamp": time.time()
        })

        return strengths

    def do_intervention(self, action_var: str, action_value: np.ndarray,
                        observed_effects: dict[str, np.ndarray]) -> dict[str, float]:
        """Alias for `intervene`, matching do-operator naming."""
        return self.intervene(action_var, action_value, observed_effects)

    def do(self, action_var: str, action_value: np.ndarray,
           observed_effects: dict[str, np.ndarray]) -> dict[str, float]:
        """Short alias for do-intervention."""
        return self.intervene(action_var, action_value, observed_effects)

    def counterfactual_propagate(self, action_var: str, action_value: np.ndarray,
                       *, horizon: int = 1) -> dict[str, np.ndarray]:
        """Lightweight counterfactual estimate based on current edge strengths."""
        if action_var not in self.variables:
            return {}
        # Use linear propagation by strength; best-effort heuristic
        out: dict[str, np.ndarray] = {}
        for (src, tgt), edge in self.edges.items():
            if src != action_var:
                continue
            strength = float(edge.strength)
            if strength == 0.0:
                continue
            # scale action_value into target space (match dims)
            av = np.array(action_value, copy=False).flatten()
            td = self.variables[tgt].dim
            if av.size < td:
                pad = np.pad(av, (0, td - av.size))
                av2 = pad
            else:
                av2 = av[:td]
            out[tgt] = (strength * av2).reshape((td,))
        return out

    def update_correlations(self):
        """
        Update edges based on observed correlations between variable histories.
        Weaker than interventional evidence but still informative.
        """
        var_names = list(self.variables.keys())
        for i, name_a in enumerate(var_names):
            var_a = self.variables[name_a]
            if len(var_a.history) < 10:
                continue
            for j, name_b in enumerate(var_names):
                if i == j:
                    continue
                var_b = self.variables[name_b]
                if len(var_b.history) < 10:
                    continue

                # Compute lagged correlation
                min_len = min(len(var_a.history), len(var_b.history))
                hist_a = np.array([np.mean(h) for h in var_a.history[-min_len:]])
                hist_b = np.array([np.mean(h) for h in var_b.history[-min_len:]])

                if np.std(hist_a) < 1e-8 or np.std(hist_b) < 1e-8:
                    continue

                # Lagged correlation (A at t → B at t+1)
                if min_len > 2:
                    corr = float(np.corrcoef(hist_a[:-1], hist_b[1:])[0, 1])
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        edge_key = (name_a, name_b)
                        if edge_key not in self.edges:
                            self.edges[edge_key] = CausalEdge(source=name_a, target=name_b)
                        self.edges[edge_key].update_from_observation(abs(corr))

    def decay_and_prune(self):
        """Apply temporal decay and prune weak edges."""
        to_remove = []
        for key, edge in self.edges.items():
            edge.decay(self.decay_rate)
            if edge.confidence < self.pruning_threshold and edge.total_evidence > 5:
                to_remove.append(key)
        for key in to_remove:
            del self.edges[key]

    def get_causes(self, variable_name: str) -> list[CausalEdge]:
        """Get all variables that cause the given variable."""
        return [edge for (src, tgt), edge in self.edges.items() if tgt == variable_name]

    def get_effects(self, variable_name: str) -> list[CausalEdge]:
        """Get all variables caused by the given variable."""
        return [edge for (src, tgt), edge in self.edges.items() if src == variable_name]

    def counterfactual(self, intervention_var: str, intervention_value: np.ndarray,
                       target_var: str) -> Optional[float]:
        """
        Estimate: "What would target_var be if we set intervention_var to intervention_value?"
        Uses the causal graph for forward propagation.
        """
        edge_key = (intervention_var, target_var)
        if edge_key not in self.edges:
            return None
        
        edge = self.edges[edge_key]
        action_magnitude = float(np.linalg.norm(intervention_value))
        return edge.strength * action_magnitude

    def information_value_of_intervention(self, variable_name: str) -> float:
        """
        Estimate how much information we'd gain by intervening on this variable.
        High-value targets are those with many uncertain outgoing edges.
        """
        effects = self.get_effects(variable_name)
        if not effects:
            return 1.0  # Unknown → high exploration value
        
        # Value = sum of uncertainty in outgoing edges
        uncertainty = sum(1.0 - e.confidence for e in effects)
        return uncertainty / len(effects)

    def suggest_intervention(self) -> Optional[str]:
        """Suggest the most informative variable to intervene on."""
        action_vars = [name for name, var in self.variables.items() if var.is_action]
        if not action_vars:
            return None
        
        values = {name: self.information_value_of_intervention(name) for name in action_vars}
        return max(values, key=values.get)

    def get_report(self) -> dict[str, Any]:
        return {
            "n_variables": self.n_variables,
            "n_edges": self.n_edges,
            "density": self.density,
            "total_interventions": len(self._intervention_log),
            "action_variables": [n for n, v in self.variables.items() if v.is_action],
            "strongest_edges": sorted(
                [{"source": e.source, "target": e.target, "strength": e.strength,
                  "confidence": e.confidence, "interventions": e.interventions}
                 for e in self.edges.values()],
                key=lambda x: x["confidence"], reverse=True
            )[:10],
            "suggested_intervention": self.suggest_intervention(),
        }


class CausalInferenceEngine:
    """
    High-level causal inference: combines multiple causal graphs
    and performs multi-step reasoning.
    """

    def __init__(self):
        self.graph = CausalGraph()
        self._prediction_history: list[dict] = []

    def process_interaction(self, action: dict[str, np.ndarray],
                           pre_state: dict[str, np.ndarray],
                           post_state: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Process a complete interaction cycle:
        1. Record pre-state observations
        2. Record intervention (action)
        3. Compute effects (post - pre)
        4. Update causal graph
        5. Return analysis
        """
        # Record pre-state
        for name, value in pre_state.items():
            self.graph.observe(name, value)

        # Compute effects
        effects = {}
        for name, post_val in post_state.items():
            if name in pre_state:
                delta = post_val - pre_state[name]
                effects[name] = delta

        # Record intervention
        for action_name, action_value in action.items():
            strengths = self.graph.intervene(action_name, action_value, effects)

        # Update correlations and prune
        self.graph.update_correlations()
        self.graph.decay_and_prune()

        return {
            "effects_detected": len(effects),
            "graph_state": self.graph.get_report(),
        }

    def predict_effect(self, action_var: str, action_value: np.ndarray) -> dict[str, float]:
        """Predict effects of a hypothetical action using the causal model."""
        predictions = {}
        for edge in self.graph.get_effects(action_var):
            predicted = self.graph.counterfactual(action_var, action_value, edge.target)
            if predicted is not None:
                predictions[edge.target] = predicted
        return predictions

    def plan_information_gathering(self, n_steps: int = 5) -> list[str]:
        """Plan a sequence of interventions to maximize information gain."""
        plan = []
        for _ in range(n_steps):
            suggestion = self.graph.suggest_intervention()
            if suggestion and suggestion not in plan:
                plan.append(suggestion)
            elif self.graph.variables:
                # Explore random variable
                import random
                candidates = [n for n in self.graph.variables.keys() if n not in plan]
                if candidates:
                    plan.append(random.choice(candidates))
        return plan
