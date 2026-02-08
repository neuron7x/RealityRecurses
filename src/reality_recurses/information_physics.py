"""
Information Physics Engine
==========================
Physical foundations for intelligence scaling.
Implements Landauer principle, Bekenstein bound, and thermodynamic
constraints on information processing.

Every computation has a physical cost. Every bit erased dissipates energy.
Intelligence scales to the limits imposed by the physics of information itself.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ─── Physical Constants ────────────────────────────────────────────────────────

BOLTZMANN_K = 1.380649e-23        # J/K
PLANCK_H = 6.62607015e-34         # J·s
SPEED_OF_LIGHT = 2.99792458e8     # m/s
LANDAUER_LIMIT_300K = BOLTZMANN_K * 300 * math.log(2)  # ~2.85e-21 J per bit


class ThermodynamicRegime(Enum):
    """Operating regimes for information processing systems."""
    REVERSIBLE = "reversible"          # Theoretical minimum dissipation
    NEAR_REVERSIBLE = "near_reversible"  # 10x Landauer limit
    EFFICIENT = "efficient"            # 100x Landauer limit
    STANDARD = "standard"              # Current silicon (~1e6 x Landauer)
    WASTEFUL = "wasteful"              # Unoptimized


@dataclass
class ThermodynamicState:
    """Thermodynamic state of an information-processing system."""
    temperature: float = 300.0           # Kelvin
    energy_budget: float = 1.0           # Joules available
    entropy_produced: float = 0.0        # Bits of entropy generated
    bits_processed: float = 0.0          # Total bits processed
    bits_erased: float = 0.0             # Irreversible erasures
    energy_consumed: float = 0.0         # Joules consumed
    timestamp: float = field(default_factory=time.time)

    @property
    def landauer_limit(self) -> float:
        """Minimum energy per bit erasure at current temperature."""
        return BOLTZMANN_K * self.temperature * math.log(2)

    @property
    def energy_remaining(self) -> float:
        return max(0.0, self.energy_budget - self.energy_consumed)

    @property
    def theoretical_bits_remaining(self) -> float:
        """Maximum bits that could still be erased given remaining energy."""
        if self.landauer_limit <= 0:
            return float('inf')
        return self.energy_remaining / self.landauer_limit

    @property
    def efficiency_ratio(self) -> float:
        """How close to Landauer limit. 1.0 = perfect efficiency."""
        if self.bits_erased <= 0:
            return 1.0
        actual_cost = self.energy_consumed / self.bits_erased
        return self.landauer_limit / actual_cost if actual_cost > 0 else 0.0

    @property
    def regime(self) -> ThermodynamicRegime:
        ratio = self.efficiency_ratio
        if ratio > 0.9:
            return ThermodynamicRegime.REVERSIBLE
        elif ratio > 0.1:
            return ThermodynamicRegime.NEAR_REVERSIBLE
        elif ratio > 0.01:
            return ThermodynamicRegime.EFFICIENT
        elif ratio > 1e-6:
            return ThermodynamicRegime.STANDARD
        return ThermodynamicRegime.WASTEFUL


class BekensteinBound:
    """
    Bekenstein bound: maximum information content of a finite region.
    S ≤ (2π R E) / (ℏ c ln2)
    
    This is the absolute ceiling on information density.
    No system can encode more bits than this bound allows.
    """

    @staticmethod
    def max_bits(radius_m: float, energy_j: float) -> float:
        """Maximum bits encodable in a sphere of given radius and energy."""
        hbar = PLANCK_H / (2 * math.pi)
        return (2 * math.pi * radius_m * energy_j) / (hbar * SPEED_OF_LIGHT * math.log(2))

    @staticmethod
    def information_density(radius_m: float, energy_j: float) -> float:
        """Bits per cubic meter at the Bekenstein limit."""
        volume = (4 / 3) * math.pi * radius_m ** 3
        if volume <= 0:
            return 0.0
        return BekensteinBound.max_bits(radius_m, energy_j) / volume

    @staticmethod
    def required_energy(radius_m: float, target_bits: float) -> float:
        """Minimum energy to encode target_bits in given radius."""
        hbar = PLANCK_H / (2 * math.pi)
        return (target_bits * hbar * SPEED_OF_LIGHT * math.log(2)) / (2 * math.pi * radius_m)


class LandauerEngine:
    """
    Manages thermodynamic accounting for information operations.
    Tracks energy cost of every bit erasure, enforcing physical constraints.
    """

    def __init__(self, temperature: float = 300.0, energy_budget: float = 1.0,
                 overhead_factor: float = 1e6):
        """
        Args:
            temperature: System temperature in Kelvin
            energy_budget: Total energy available in Joules
            overhead_factor: Multiplier above Landauer limit (realistic hardware)
        """
        self.state = ThermodynamicState(
            temperature=temperature,
            energy_budget=energy_budget
        )
        self.overhead_factor = overhead_factor
        self._operation_log: list[dict[str, Any]] = []

    @property
    def cost_per_bit(self) -> float:
        """Actual energy cost per bit erasure including overhead."""
        return self.state.landauer_limit * self.overhead_factor

    def can_afford(self, n_bits: int) -> bool:
        """Check if we have enough energy to erase n_bits."""
        return self.state.energy_remaining >= n_bits * self.cost_per_bit

    def erase_bits(self, n_bits: int, label: str = "operation") -> bool:
        """
        Perform irreversible bit erasure with thermodynamic accounting.
        Returns True if operation succeeded within energy budget.
        """
        cost = n_bits * self.cost_per_bit
        if cost > self.state.energy_remaining:
            self._operation_log.append({
                "label": label,
                "bits": n_bits,
                "cost": cost,
                "status": "REJECTED_ENERGY",
                "timestamp": time.time()
            })
            return False

        self.state.bits_erased += n_bits
        self.state.bits_processed += n_bits
        self.state.energy_consumed += cost
        self.state.entropy_produced += n_bits
        self.state.timestamp = time.time()

        self._operation_log.append({
            "label": label,
            "bits": n_bits,
            "cost": cost,
            "status": "COMPLETED",
            "timestamp": time.time()
        })
        return True

    def process_bits(self, n_bits: int, erasure_fraction: float = 0.1,
                     label: str = "processing") -> bool:
        """
        Process bits with partial erasure (most processing is reversible).
        erasure_fraction: fraction of bits that are irreversibly erased.
        """
        bits_to_erase = max(1, int(n_bits * erasure_fraction))
        success = self.erase_bits(bits_to_erase, label=label)
        if success:
            self.state.bits_processed += (n_bits - bits_to_erase)
        return success

    def get_report(self) -> dict[str, Any]:
        return {
            "temperature_K": self.state.temperature,
            "energy_budget_J": self.state.energy_budget,
            "energy_consumed_J": self.state.energy_consumed,
            "energy_remaining_J": self.state.energy_remaining,
            "bits_processed": self.state.bits_processed,
            "bits_erased": self.state.bits_erased,
            "entropy_produced_bits": self.state.entropy_produced,
            "efficiency_ratio": self.state.efficiency_ratio,
            "regime": self.state.regime.value,
            "operations_count": len(self._operation_log),
            "theoretical_bits_remaining": self.state.theoretical_bits_remaining,
        }


class EntropyBudget:
    """
    Manages the entropy budget for a learning system.
    
    Key insight from the paper: static datasets have finite entropy budget.
    Once exhausted, further scaling requires transition to a new regime.
    """

    def __init__(self, total_entropy_bits: float, regime: str = "static"):
        self.total_entropy = total_entropy_bits
        self.consumed_entropy = 0.0
        self.regime = regime
        self._regime_transitions: list[dict] = []

    @property
    def remaining_entropy(self) -> float:
        return max(0.0, self.total_entropy - self.consumed_entropy)

    @property
    def saturation_ratio(self) -> float:
        """How much of the entropy budget has been consumed. 1.0 = saturated."""
        if self.total_entropy <= 0:
            return 1.0
        return self.consumed_entropy / self.total_entropy

    @property
    def is_saturated(self) -> bool:
        return self.saturation_ratio >= 0.99

    def consume(self, bits: float) -> float:
        """Consume entropy from budget. Returns actually consumed amount."""
        actual = min(bits, self.remaining_entropy)
        self.consumed_entropy += actual

        if self.is_saturated and self.regime == "static":
            self._trigger_regime_transition()

        return actual

    def _trigger_regime_transition(self):
        """Signal that static regime is exhausted → transition to physical."""
        self.regime = "physical_interaction"
        self._regime_transitions.append({
            "from": "static",
            "to": "physical_interaction",
            "timestamp": time.time(),
            "reason": "entropy_budget_exhausted"
        })

    def inject_entropy(self, bits: float, source: str = "physical_interaction"):
        """Add new entropy from physical interaction with environment."""
        self.total_entropy += bits
        if self.regime == "static" and source == "physical_interaction":
            self.regime = "hybrid"

    def get_state(self) -> dict:
        return {
            "regime": self.regime,
            "total_entropy_bits": self.total_entropy,
            "consumed_entropy_bits": self.consumed_entropy,
            "remaining_entropy_bits": self.remaining_entropy,
            "saturation_ratio": self.saturation_ratio,
            "is_saturated": self.is_saturated,
            "transitions": self._regime_transitions,
        }
