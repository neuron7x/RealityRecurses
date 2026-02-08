"""
Default configuration for Reality Scaler Agent.
"""

DEFAULT_CONFIG = {
    # Dimensionality
    "state_dim": 64,
    "action_dim": 16,
    
    # Physics
    "temperature": 300.0,
    "energy_budget": 1.0,
    "overhead_factor": 1e6,
    
    # Fractal Architecture
    "fractal_depth": 4,
    
    # Memory Hierarchy
    "working_memory_capacity": 7,
    "episodic_memory_capacity": 1000,
    "semantic_memory_capacity": 500,
    
    # Divergent Explorer
    "n_hypothesis_populations": 3,
    "hypothesis_population_size": 15,
    "evolution_interval": 10,
    
    # Causal Engine
    "causal_decay_rate": 0.001,
    "causal_pruning_threshold": 0.01,
    
    # Scaling
    "initial_entropy_budget": 1e6,
    "regime_transition_threshold": 0.99,
    
    # Simulation
    "default_simulation_steps": 200,
    "verbose": True,
}

# Presets for different use cases
PRESETS = {
    "minimal": {
        "state_dim": 16,
        "action_dim": 4,
        "fractal_depth": 2,
        "working_memory_capacity": 4,
        "episodic_memory_capacity": 100,
        "hypothesis_population_size": 5,
        "default_simulation_steps": 50,
    },
    "standard": DEFAULT_CONFIG,
    "research": {
        "state_dim": 128,
        "action_dim": 32,
        "fractal_depth": 4,
        "energy_budget": 100.0,
        "working_memory_capacity": 12,
        "episodic_memory_capacity": 10000,
        "semantic_memory_capacity": 5000,
        "n_hypothesis_populations": 5,
        "hypothesis_population_size": 50,
        "default_simulation_steps": 1000,
    },
}
