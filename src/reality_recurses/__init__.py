"""
RealityRecurses — Physical Reality as a Scaling Substrate for AGI.

Grounding intelligence in information physics: Landauer principle,
Bekenstein bound, entropy budgets, and thermodynamic memory.
"""

__version__ = "1.0.0"

from reality_recurses.information_physics import (
    LandauerEngine,
    BekensteinBound,
    EntropyBudget,
    ThermodynamicRegime,
)
from reality_recurses.architecture import (
    FractalScale,
    FractalNode,
    NucleusNode,
    SensorNode,
    CompressorNode,
    CausalModelerNode,
    RealityScalerNode,
    build_fractal_agent,
)
from reality_recurses.thermodynamic_memory import (
    MemoryTrace,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ThermodynamicMemorySystem,
)
from reality_recurses.causal_engine import (
    CausalEdge,
    CausalVariable,
    CausalGraph,
    CausalInferenceEngine,
)
from reality_recurses.divergent_engine import (
    Hypothesis,
    HypothesisPopulation,
    DivergentExplorer,
)
from reality_recurses.agent import (
    AgentConfig,
    RealityScalerAgent,
    PhysicsSimEnvironment,
    run_simulation,
)
from reality_recurses.toy_env import LinearTanhEnv
from reality_recurses.baselines import RandomBaseline, ZeroActionBaseline
from reality_recurses.defaults import DEFAULT_CONFIG, PRESETS

__all__ = [
    # Physics
    "LandauerEngine", "BekensteinBound", "EntropyBudget", "ThermodynamicRegime",
    # Architecture
    "FractalScale", "FractalNode", "NucleusNode", "SensorNode",
    "CompressorNode", "CausalModelerNode", "RealityScalerNode",
    "build_fractal_agent",
    # Memory
    "MemoryTrace", "WorkingMemory", "EpisodicMemory", "SemanticMemory",
    "ThermodynamicMemorySystem",
    # Causal
    "CausalEdge", "CausalVariable", "CausalGraph", "CausalInferenceEngine",
    # Divergent
    "Hypothesis", "HypothesisPopulation", "DivergentExplorer",
    # Agent
    "AgentConfig", "RealityScalerAgent", "PhysicsSimEnvironment", "run_simulation",
    # Utilities
    "LinearTanhEnv", "RandomBaseline", "ZeroActionBaseline",
    "DEFAULT_CONFIG", "PRESETS",
]
