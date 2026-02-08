"""
Divergent Intelligence Engine
==============================
The agent as a divergent system — exploring multiple causal hypotheses
simultaneously, evolving them through interaction with reality.

Divergence principle: Intelligence scales not by finding ONE answer,
but by maintaining a population of competing hypotheses about reality
and selecting among them through physical interaction.

This is the antithesis of convergent LLM behavior (mode-seeking).
The divergent agent is an explorer of possibility space.

    ┌──────────────────────────────────────────────┐
    │              HYPOTHESIS SPACE                 │
    │                                               │
    │  H₁ ──→ test ──→ survive ──→ evolve          │
    │  H₂ ──→ test ──→ die                         │
    │  H₃ ──→ test ──→ survive ──→ mutate ──→ H₃'  │
    │  H₄ ──→ test ──→ survive ──→ merge(H₁)──→ H₅│
    │  ...                                          │
    │                                               │
    │  Selection pressure = prediction accuracy     │
    │  Mutation rate ∝ 1/confidence                 │
    │  Crossover = hypothesis merging               │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Hypothesis:
    """
    A single hypothesis about how some aspect of reality works.
    Encodes a predictive model that can be tested against observations.
    """
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # Model: y = W @ x + b
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    input_dim: int = 0
    output_dim: int = 0

    # Fitness tracking
    total_predictions: int = 0
    correct_predictions: int = 0
    cumulative_error: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_tested: float = field(default_factory=time.time)

    # Metadata
    domain: str = "general"         # What aspect of reality this models
    confidence: float = 0.5
    is_alive: bool = True

    @property
    def fitness(self) -> float:
        """Fitness = accuracy adjusted by confidence and age."""
        if self.total_predictions == 0:
            return 0.5  # Prior: uncertain hypotheses get neutral fitness
        accuracy = self.correct_predictions / self.total_predictions
        age_factor = min(1.0, self.total_predictions / 100)  # Trust older hypotheses more
        return accuracy * age_factor * self.confidence

    @property
    def mean_error(self) -> float:
        if self.total_predictions == 0:
            return float('inf')
        return self.cumulative_error / self.total_predictions

    @property
    def mutation_rate(self) -> float:
        """Mutation rate inversely proportional to confidence."""
        return max(0.01, 1.0 - self.confidence)

    def initialize(self, input_dim: int, output_dim: int):
        """Initialize model parameters."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim) * 0.1
        self.bias = np.zeros(output_dim)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make a prediction."""
        if self.weights is None:
            raise ValueError("Hypothesis not initialized")
        x_flat = x.flatten()[:self.input_dim]
        if len(x_flat) < self.input_dim:
            x_flat = np.pad(x_flat, (0, self.input_dim - len(x_flat)))
        return self.weights @ x_flat + self.bias

    def test(self, x: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        """
        Test hypothesis against reality.
        Returns prediction error.
        """
        y_pred = self.predict(x)
        y_true_flat = y_true.flatten()[:self.output_dim]
        if len(y_true_flat) < self.output_dim:
            y_true_flat = np.pad(y_true_flat, (0, self.output_dim - len(y_true_flat)))

        error = float(np.linalg.norm(y_pred - y_true_flat))
        self.total_predictions += 1
        self.cumulative_error += error
        self.last_tested = time.time()

        if error < threshold:
            self.correct_predictions += 1
            self.confidence = min(0.99, self.confidence + 0.01)
        else:
            self.confidence = max(0.01, self.confidence - 0.02)

        # Online gradient update
        lr = 0.01 * self.mutation_rate
        grad = y_pred - y_true_flat
        x_flat = x.flatten()[:self.input_dim]
        if len(x_flat) < self.input_dim:
            x_flat = np.pad(x_flat, (0, self.input_dim - len(x_flat)))
        self.weights -= lr * np.outer(grad, x_flat)
        self.bias -= lr * grad

        return error

    def mutate(self) -> Hypothesis:
        """Create a mutated copy of this hypothesis."""
        child = Hypothesis(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            generation=self.generation + 1,
            parent_ids=[self.hypothesis_id],
            domain=self.domain,
            confidence=0.5,
        )
        child.weights = self.weights.copy() + np.random.randn(*self.weights.shape) * self.mutation_rate * 0.1
        child.bias = self.bias.copy() + np.random.randn(*self.bias.shape) * self.mutation_rate * 0.1
        return child

    @staticmethod
    def crossover(h1: Hypothesis, h2: Hypothesis) -> Hypothesis:
        """Create offspring by merging two hypotheses."""
        assert h1.input_dim == h2.input_dim and h1.output_dim == h2.output_dim
        child = Hypothesis(
            input_dim=h1.input_dim,
            output_dim=h1.output_dim,
            generation=max(h1.generation, h2.generation) + 1,
            parent_ids=[h1.hypothesis_id, h2.hypothesis_id],
            domain=h1.domain,
            confidence=0.5,
        )
        # Weighted average by fitness
        total_fit = h1.fitness + h2.fitness + 1e-8
        w1 = h1.fitness / total_fit
        w2 = h2.fitness / total_fit
        child.weights = w1 * h1.weights + w2 * h2.weights
        child.bias = w1 * h1.bias + w2 * h2.bias
        return child


class HypothesisPopulation:
    """
    Evolutionary population of competing hypotheses.
    Implements selection, mutation, and crossover.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 population_size: int = 20, domain: str = "general"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.domain = domain
        self.generation = 0
        self.hypotheses: list[Hypothesis] = []
        self._fitness_history: list[float] = []
        self._diversity_history: list[float] = []

        # Initialize population
        for _ in range(population_size):
            h = Hypothesis(domain=domain)
            h.initialize(input_dim, output_dim)
            self.hypotheses.append(h)

    @property
    def best_hypothesis(self) -> Hypothesis:
        return max(self.hypotheses, key=lambda h: h.fitness)

    @property
    def mean_fitness(self) -> float:
        if not self.hypotheses:
            return 0.0
        return float(np.mean([h.fitness for h in self.hypotheses]))

    @property
    def diversity(self) -> float:
        """Measure diversity as variance in weight space."""
        if len(self.hypotheses) < 2:
            return 0.0
        weights = np.array([h.weights.flatten() for h in self.hypotheses if h.weights is not None])
        if len(weights) < 2:
            return 0.0
        return float(np.mean(np.var(weights, axis=0)))

    def test_all(self, x: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
        """Test all hypotheses against an observation."""
        errors = []
        for h in self.hypotheses:
            error = h.test(x, y_true, threshold)
            errors.append(error)
        return {
            "mean_error": float(np.mean(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "best_id": self.best_hypothesis.hypothesis_id,
        }

    def ensemble_predict(self, x: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction from all living hypotheses."""
        predictions = []
        weights = []
        for h in self.hypotheses:
            if h.is_alive and h.weights is not None:
                predictions.append(h.predict(x))
                weights.append(h.fitness + 1e-8)
        if not predictions:
            return np.zeros(self.output_dim)
        weights = np.array(weights)
        weights /= weights.sum()
        return sum(w * p for w, p in zip(weights, predictions))

    def evolve(self, elite_fraction: float = 0.2, mutation_rate: float = 0.3,
               crossover_rate: float = 0.2) -> dict[str, Any]:
        """
        Evolve the population through selection, mutation, and crossover.
        """
        self.generation += 1
        n = len(self.hypotheses)
        if n == 0:
            return {"generation": self.generation, "status": "empty"}

        # Sort by fitness
        self.hypotheses.sort(key=lambda h: h.fitness, reverse=True)

        # Record metrics
        self._fitness_history.append(self.mean_fitness)
        self._diversity_history.append(self.diversity)

        # Elite selection
        n_elite = max(2, int(n * elite_fraction))
        elites = self.hypotheses[:n_elite]

        new_population = list(elites)  # Elites survive

        # Mutation
        n_mutants = int(n * mutation_rate)
        for _ in range(n_mutants):
            parent = elites[np.random.randint(len(elites))]
            child = parent.mutate()
            new_population.append(child)

        # Crossover
        n_crossover = int(n * crossover_rate)
        for _ in range(n_crossover):
            if len(elites) >= 2:
                idx = np.random.choice(len(elites), size=2, replace=False)
                child = Hypothesis.crossover(elites[idx[0]], elites[idx[1]])
                new_population.append(child)

        # Fill remaining with random new hypotheses (immigration)
        while len(new_population) < self.population_size:
            h = Hypothesis(domain=self.domain, generation=self.generation)
            h.initialize(self.input_dim, self.output_dim)
            new_population.append(h)

        # Trim to population size
        self.hypotheses = new_population[:self.population_size]

        # Kill low-fitness old hypotheses
        for h in self.hypotheses:
            if h.total_predictions > 50 and h.fitness < 0.1:
                h.is_alive = False

        self.hypotheses = [h for h in self.hypotheses if h.is_alive]

        # Replenish if kills dropped us below target
        while len(self.hypotheses) < self.population_size:
            h = Hypothesis(domain=self.domain, generation=self.generation)
            h.initialize(self.input_dim, self.output_dim)
            self.hypotheses.append(h)

        return {
            "generation": self.generation,
            "population_size": len(self.hypotheses),
            "mean_fitness": self.mean_fitness,
            "best_fitness": self.best_hypothesis.fitness,
            "diversity": self.diversity,
            "n_elite": n_elite,
            "n_mutants": n_mutants,
            "n_crossover": n_crossover,
        }


class DivergentExplorer:
    """
    Top-level divergent intelligence agent.
    Maintains multiple hypothesis populations across domains
    and orchestrates exploration of reality.
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 16,
                 n_populations: int = 3, pop_size: int = 15):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.populations: dict[str, HypothesisPopulation] = {}

        # Default populations
        domains = ["dynamics", "reward", "novelty"][:n_populations]
        for domain in domains:
            self.populations[domain] = HypothesisPopulation(
                input_dim=state_dim + action_dim,
                output_dim=state_dim,
                population_size=pop_size,
                domain=domain
            )

        self._exploration_history: list[dict] = []
        self._total_interactions = 0

    def explore(self, state: np.ndarray, action: np.ndarray,
                next_state: np.ndarray) -> dict[str, Any]:
        """
        Process one interaction with reality.
        Tests all hypotheses and evolves populations.
        """
        self._total_interactions += 1
        x = np.concatenate([
            state.flatten()[:self.state_dim],
            action.flatten()[:self.action_dim]
        ])
        if len(x) < self.state_dim + self.action_dim:
            x = np.pad(x, (0, self.state_dim + self.action_dim - len(x)))
        y = next_state.flatten()[:self.state_dim]
        if len(y) < self.state_dim:
            y = np.pad(y, (0, self.state_dim - len(y)))

        results = {}
        for domain, pop in self.populations.items():
            test_result = pop.test_all(x, y)
            results[domain] = test_result

            # Evolve periodically
            if self._total_interactions % 10 == 0:
                evo_result = pop.evolve()
                results[f"{domain}_evolution"] = evo_result

        self._exploration_history.append({
            "interaction": self._total_interactions,
            "results": {k: v["mean_error"] for k, v in results.items() if "mean_error" in v},
            "timestamp": time.time()
        })

        return results

    def predict(self, state: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
        """Ensemble prediction from all populations."""
        x = np.concatenate([
            state.flatten()[:self.state_dim],
            action.flatten()[:self.action_dim]
        ])
        if len(x) < self.state_dim + self.action_dim:
            x = np.pad(x, (0, self.state_dim + self.action_dim - len(x)))
        
        predictions = {}
        for domain, pop in self.populations.items():
            predictions[domain] = pop.ensemble_predict(x)
        return predictions

    def suggest_action(self, state: np.ndarray) -> np.ndarray:
        """
        Suggest the most informative action — the one that maximizes
        expected disagreement between hypothesis populations.
        """
        best_action = None
        max_disagreement = -1

        for _ in range(10):  # Sample random actions
            action = np.random.randn(self.action_dim) * 0.5
            predictions = self.predict(state, action)
            
            # Disagreement = variance across population predictions
            pred_values = list(predictions.values())
            if len(pred_values) >= 2:
                stacked = np.stack(pred_values)
                disagreement = float(np.mean(np.var(stacked, axis=0)))
            else:
                disagreement = 0.0

            if disagreement > max_disagreement:
                max_disagreement = disagreement
                best_action = action

        return best_action if best_action is not None else np.random.randn(self.action_dim) * 0.5

    def get_report(self) -> dict[str, Any]:
        return {
            "total_interactions": self._total_interactions,
            "populations": {
                domain: {
                    "size": len(pop.hypotheses),
                    "generation": pop.generation,
                    "mean_fitness": pop.mean_fitness,
                    "best_fitness": pop.best_hypothesis.fitness if pop.hypotheses else 0,
                    "diversity": pop.diversity,
                }
                for domain, pop in self.populations.items()
            },
            "exploration_history_length": len(self._exploration_history),
        }
