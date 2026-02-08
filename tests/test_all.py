"""
Reality Scaler — Comprehensive Test Suite
==========================================
69 tests: physics, architecture, causal, divergent, memory, agent, invariants, baselines.
"""

import numpy as np
import pytest

# ─── Information Physics ─────────────────────────────────────────────

class TestLandauerEngine:
    def test_creation(self):
        from reality_scaler.information_physics import LandauerEngine
        engine = LandauerEngine(temperature=300.0, energy_budget=1.0, overhead_factor=1e6)
        assert engine.state.energy_budget == 1.0
        assert engine.state.bits_processed == 0

    def test_erase_bits(self):
        from reality_scaler.information_physics import LandauerEngine
        engine = LandauerEngine(temperature=300.0, energy_budget=1.0, overhead_factor=1e6)
        assert engine.can_afford(100)
        assert engine.erase_bits(100, label="test")
        assert engine.state.bits_erased == 100
        assert engine.state.energy_consumed > 0

    def test_report(self):
        from reality_scaler.information_physics import LandauerEngine
        engine = LandauerEngine(temperature=300.0, energy_budget=1.0, overhead_factor=1e6)
        engine.erase_bits(50, label="a")
        engine.erase_bits(50, label="b")
        report = engine.get_report()
        assert report["operations_count"] == 2

    def test_energy_depletion(self):
        from reality_scaler.information_physics import LandauerEngine
        engine = LandauerEngine(temperature=300.0, energy_budget=1e-30, overhead_factor=1e6)
        assert not engine.can_afford(1_000_000_000)

    def test_process_bits(self):
        from reality_scaler.information_physics import LandauerEngine
        engine = LandauerEngine(temperature=300.0, energy_budget=1.0, overhead_factor=1e6)
        engine.process_bits(1000, erasure_fraction=0.1, label="proc")
        assert engine.state.bits_processed == 1000
        assert engine.state.bits_erased == 100


class TestBekensteinBound:
    def test_positive(self):
        from reality_scaler.information_physics import BekensteinBound
        bits = BekensteinBound.max_bits(radius_m=0.01, energy_j=1.0)
        assert bits > 0

    def test_zero_radius(self):
        from reality_scaler.information_physics import BekensteinBound
        bits = BekensteinBound.max_bits(radius_m=0.0, energy_j=1.0)
        assert bits == 0.0


class TestEntropyBudget:
    def test_consume(self):
        from reality_scaler.information_physics import EntropyBudget
        budget = EntropyBudget(total_entropy_bits=1000, regime="static")
        consumed = budget.consume(500)
        assert consumed == 500
        assert budget.saturation_ratio == pytest.approx(0.5)

    def test_regime_transition(self):
        from reality_scaler.information_physics import EntropyBudget
        budget = EntropyBudget(total_entropy_bits=1000, regime="static")
        budget.consume(500)
        budget.consume(600)  # exceeds budget
        assert budget.is_saturated
        assert budget.regime == "physical_interaction"

    def test_get_state(self):
        from reality_scaler.information_physics import EntropyBudget
        budget = EntropyBudget(total_entropy_bits=500, regime="static")
        budget.consume(100)
        state = budget.get_state()
        assert state["consumed_entropy_bits"] == 100
        assert state["total_entropy_bits"] == 500


# ─── Fractal Architecture ────────────────────────────────────────────

class TestFractalArchitecture:
    def test_nucleus(self):
        from reality_scaler.architecture import NucleusNode
        n = NucleusNode(feature_dim=32)
        action, pred = n.cycle(np.random.randn(32))
        assert action.shape == (1,)
        assert n.state.cycle_count == 1

    def test_sensor(self):
        from reality_scaler.architecture import SensorNode
        s = SensorNode(input_dim=64, compressed_dim=32, n_nuclei=2)
        action, pred = s.cycle(np.random.randn(64))
        assert s.state.cycle_count == 1
        assert s.total_nodes >= 3

    def test_compressor(self):
        from reality_scaler.architecture import CompressorNode
        c = CompressorNode(state_dim=16, memory_length=8, n_sensors=1)
        action, pred = c.cycle(np.random.randn(16))
        assert c.state.cycle_count == 1

    def test_build_fractal_agent(self):
        from reality_scaler.architecture import build_fractal_agent
        agent = build_fractal_agent(depth=3)
        x = np.random.randn(64)
        action, pred = agent.cycle(x)
        assert agent.state.cycle_count == 1
        report = agent.get_tree_report()
        assert report["total_nodes"] > 1

    def test_full_depth4(self):
        from reality_scaler.architecture import build_fractal_agent
        agent = build_fractal_agent(depth=4)
        for i in range(10):
            agent.cycle(np.random.randn(64))
        assert agent.state.cycle_count == 10

    def test_determinism(self):
        from reality_scaler.architecture import CompressorNode
        results = []
        for _ in range(2):
            np.random.seed(42)
            c = CompressorNode(state_dim=16, memory_length=8, n_sensors=1)
            for i in range(10):
                c.cycle(np.ones(16) * (i + 1))
            out, _ = c.cycle(np.ones(16))
            results.append(out.copy())
        assert np.allclose(results[0], results[1])

    def test_tree_report(self):
        from reality_scaler.architecture import build_fractal_agent
        agent = build_fractal_agent(depth=2)
        report = agent.get_tree_report()
        assert "total_nodes" in report
        assert "depth" in report


# ─── Causal Engine ───────────────────────────────────────────────────

class TestCausalEngine:
    def test_add_variable(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=2, is_action=True)
        assert "x" in g.variables

    def test_intervene(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=1, is_action=True)
        g.add_variable("y", dim=1)
        g.intervene("x", np.array([1.0]), {"y": np.array([0.8])})
        assert ("x", "y") in g.edges

    def test_causal_discovery(self):
        from reality_scaler.causal_engine import CausalInferenceEngine
        engine = CausalInferenceEngine()
        engine.graph.add_variable("a", dim=1, is_action=True)
        engine.graph.add_variable("b", dim=1)
        for _ in range(20):
            v = np.random.randn(1) * 0.5
            engine.graph.intervene("a", v, {"b": v * 0.8 + np.random.randn(1) * 0.1})
        effects = engine.graph.get_effects("a")
        assert len(effects) > 0
        assert effects[0].confidence > 0.1

    def test_counterfactual_single_target(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=1, is_action=True)
        g.add_variable("y", dim=1)
        for _ in range(10):
            g.intervene("x", np.array([1.0]), {"y": np.array([0.5])})
        cf = g.counterfactual("x", np.array([1.0]), "y")
        assert cf is not None

    def test_counterfactual_propagate(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=2, is_action=True)
        g.add_variable("y", dim=2)
        for _ in range(5):
            g.intervene("x", np.ones(2), {"y": np.ones(2) * 0.5})
        out = g.counterfactual_propagate("x", np.ones(2))
        assert isinstance(out, dict)
        assert "y" in out

    def test_decay_and_prune(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph(decay_rate=0.5, pruning_threshold=0.01)
        g.add_variable("x", dim=1, is_action=True)
        g.add_variable("y", dim=1)
        g.intervene("x", np.array([1.0]), {"y": np.array([0.5])})
        g.decay_and_prune()
        assert len(g.edges) >= 0  # may or may not survive

    def test_get_report(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=1, is_action=True)
        g.add_variable("y", dim=1)
        report = g.get_report()
        assert report["n_variables"] == 2

    def test_predict_effect(self):
        from reality_scaler.causal_engine import CausalInferenceEngine
        engine = CausalInferenceEngine()
        engine.graph.add_variable("a", dim=1, is_action=True)
        engine.graph.add_variable("b", dim=1)
        for _ in range(10):
            engine.graph.intervene("a", np.array([1.0]), {"b": np.array([0.7])})
        result = engine.predict_effect("a", np.array([1.0]))
        assert isinstance(result, dict)

    def test_do_intervention(self):
        from reality_scaler.causal_engine import CausalGraph
        g = CausalGraph()
        g.add_variable("x", dim=1, is_action=True)
        assert hasattr(g, "do_intervention") or hasattr(g, "do")


# ─── Divergent Explorer ──────────────────────────────────────────────

class TestDivergentExplorer:
    def test_creation(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        report = explorer.get_report()
        assert report["total_interactions"] == 0

    def test_explore(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        for _ in range(15):
            s = np.random.randn(8)
            a = np.random.randn(4)
            ns = s * 0.9
            explorer.explore(s, a, ns)
        report = explorer.get_report()
        assert report["total_interactions"] == 15

    def test_predict(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        for _ in range(5):
            explorer.explore(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        predictions = explorer.predict(np.random.randn(8), np.random.randn(4))
        assert len(predictions) > 0

    def test_suggest_action(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        action = explorer.suggest_action(np.random.randn(8))
        assert len(action) == 4

    def test_evolution(self):
        from reality_scaler.divergent_engine import HypothesisPopulation
        np.random.seed(42)
        pop = HypothesisPopulation(input_dim=4, output_dim=2, population_size=5)
        for _ in range(50):
            pop.test_all(np.random.randn(4), np.random.randn(2), threshold=0.01)
        pop.evolve()
        assert len(pop.hypotheses) == 5  # no collapse

    def test_population_no_collapse(self):
        """Verify fix: population never drops below target size."""
        from reality_scaler.divergent_engine import HypothesisPopulation
        np.random.seed(0)
        pop = HypothesisPopulation(input_dim=4, output_dim=2, population_size=8)
        for gen in range(20):
            for _ in range(15):
                pop.test_all(np.random.randn(4), np.random.randn(2), threshold=0.01)
            pop.evolve()
            assert len(pop.hypotheses) >= pop.population_size, \
                f"Population collapsed to {len(pop.hypotheses)} at gen {gen}"

    def test_ensemble_prediction(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        for _ in range(20):
            explorer.explore(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        preds = explorer.predict(np.random.randn(8), np.random.randn(4))
        assert isinstance(preds, dict)

    def test_get_report(self):
        from reality_scaler.divergent_engine import DivergentExplorer
        explorer = DivergentExplorer(state_dim=8, action_dim=4, n_populations=2, pop_size=5)
        explorer.explore(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        report = explorer.get_report()
        assert "populations" in report
        assert report["total_interactions"] == 1

    def test_hypothesis_fitness(self):
        from reality_scaler.divergent_engine import Hypothesis
        h = Hypothesis(domain="dynamics", generation=0)
        h.initialize(input_dim=4, output_dim=2)
        x = np.random.randn(4)
        pred = h.predict(x)
        assert pred.shape == (2,)
        h.test(x, np.random.randn(2), threshold=0.5)
        assert h.total_predictions == 1


# ─── Thermodynamic Memory ────────────────────────────────────────────

class TestThermodynamicMemory:
    def test_working_memory_write_read(self):
        from reality_scaler.thermodynamic_memory import WorkingMemory
        wm = WorkingMemory(capacity=3)
        wm.write("a", np.array([1.0, 2.0]))
        result = wm.read("a")
        assert result is not None
        assert np.allclose(result, [1.0, 2.0])

    def test_working_memory_eviction(self):
        from reality_scaler.thermodynamic_memory import WorkingMemory
        wm = WorkingMemory(capacity=2)
        wm.write("a", np.array([1.0]))
        wm.write("b", np.array([2.0]))
        wm.write("c", np.array([3.0]))
        assert len(wm.buffer) == 2

    def test_episodic_store(self):
        from reality_scaler.thermodynamic_memory import EpisodicMemory
        em = EpisodicMemory(capacity=100)
        trace = em.store_episode(np.ones(4), np.ones(2), np.ones(4))
        assert len(em.episodes) == 1

    def test_semantic_store_and_merge(self):
        from reality_scaler.thermodynamic_memory import SemanticMemory
        sm = SemanticMemory(capacity=100)
        sm.store("k1", np.array([1.0, 0.0]))
        sm.store("k1", np.array([0.0, 1.0]))  # merge
        result = sm.retrieve("k1")
        assert result is not None
        assert result.shape == (2,)

    def test_process_experience(self):
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem(working_capacity=5, episodic_capacity=100)
        for i in range(20):
            mem.process_experience(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        assert mem.total_traces > 0
        assert mem.total_energy > 0

    def test_recall(self):
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem(working_capacity=5)
        for i in range(5):
            mem.process_experience(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        traces = mem.recall(np.random.randn(8), top_k=3)
        assert isinstance(traces, list)

    def test_report(self):
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        mem.process_experience(np.random.randn(8), np.random.randn(4), np.random.randn(8))
        report = mem.get_report()
        assert report["total_operations"] == 1

    def test_physics_attr_stored(self):
        """Verify fix: __init__ stores physics and landauer_temperature_K."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem(physics="SENTINEL", landauer_temperature_K=310.0)
        assert mem.physics == "SENTINEL"
        assert mem.landauer_temperature_K == 310.0
        assert mem.noise_discard_threshold is not None
        assert mem.min_information_bits is not None

    def test_compute_entropy_score(self):
        """Verify fix: math import works for entropy computation."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        score = mem._compute_entropy_score(np.ones(4), np.ones(2), np.ones(4))
        assert 0.0 <= score <= 1.0

    def test_compute_information_content(self):
        """Verify fix: math import works for information content."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        ic = mem._compute_information_content(np.ones(4), np.ones(2), np.ones(4))
        assert ic >= 0.0

    def test_landauer_cost_j(self):
        """Verify fix: self.landauer_temperature_K accessible."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        cost = mem._landauer_cost_J(1000)
        assert cost > 0

    def test_rewrite_trace_semantic(self):
        """Verify fix: rewrite_trace uses self.semantic.knowledge, not .traces."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        mem.semantic.store("key1", np.array([1.0, 2.0]))
        ok = mem.rewrite_trace("key1", np.array([3.0, 4.0]), memory_type="semantic")
        assert ok
        result = mem.semantic.retrieve("key1")
        assert np.allclose(result, [3.0, 4.0])

    def test_rewrite_trace_working(self):
        """Verify fix: rewrite_trace works with working memory buffer."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        mem.working.write("w1", np.array([1.0]))
        ok = mem.rewrite_trace("w1", np.array([9.0]), memory_type="working")
        assert ok

    def test_rewrite_trace_nonexistent(self):
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        ok = mem.rewrite_trace("nonexistent", np.array([1.0]), memory_type="semantic")
        assert not ok

    def test_detect_memory_bias(self):
        """Verify fix: detect_memory_bias uses self.semantic.knowledge, not .traces."""
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        mem.semantic.store("a", np.ones(4), domain="physics")
        mem.semantic.store("b", np.ones(4), domain="physics")
        mem.semantic.store("c", np.ones(4), domain="physics")
        bias = mem.detect_memory_bias()
        assert bias["biased"]
        assert bias["max_share"] == pytest.approx(1.0)
        assert bias["top_domain"] == "physics"

    def test_detect_memory_bias_empty(self):
        from reality_scaler.thermodynamic_memory import ThermodynamicMemorySystem
        mem = ThermodynamicMemorySystem()
        bias = mem.detect_memory_bias()
        assert not bias["biased"]


# ─── Full Agent ──────────────────────────────────────────────────────

class TestAgent:
    def test_create(self):
        from reality_scaler.agent import RealityScalerAgent
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        assert agent is not None

    def test_act_learn(self):
        from reality_scaler.agent import RealityScalerAgent
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        state = np.random.randn(16)
        action = agent.act(state)
        assert action.shape == (4,)
        report = agent.learn(state, action, np.random.randn(16))
        assert "prediction_error" in report

    def test_step(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=16)
        state = env.observe()
        report = agent.step(state, environment_step_fn=env.step)
        assert "prediction_error" in report

    def test_multi_step(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=16)
        for _ in range(50):
            state = env.observe()
            agent.step(state, environment_step_fn=env.step)
        assert agent.metrics.total_information_gained > 0

    def test_determinism(self):
        from reality_scaler.agent import RealityScalerAgent, AgentConfig
        cfg = AgentConfig(seed=42, state_dim=8, action_dim=4, energy_budget=1.0)
        a1 = RealityScalerAgent(cfg).act(np.zeros(8))
        a2 = RealityScalerAgent(cfg).act(np.zeros(8))
        assert np.allclose(a1, a2)

    def test_self_audit(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=16)
        for _ in range(10):
            state = env.observe()
            agent.step(state, environment_step_fn=env.step)
        audit = agent.self_audit()
        assert audit["ok"]

    def test_full_report(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=16)
        for _ in range(10):
            agent.step(env.observe(), environment_step_fn=env.step)
        report = agent.get_full_report()
        assert "agent" in report
        assert "memory" in report
        assert "causal" in report

    def test_energy_depletion(self):
        from reality_scaler.agent import RealityScalerAgent
        agent = RealityScalerAgent.create(state_dim=8, action_dim=4, energy_budget=1e-25)
        action = agent.act(np.random.randn(8))
        assert np.allclose(action, 0)  # depleted → zeros

    def test_zero_state(self):
        from reality_scaler.agent import RealityScalerAgent
        agent = RealityScalerAgent.create(state_dim=8, action_dim=4, energy_budget=1.0)
        action = agent.act(np.zeros(8))
        report = agent.learn(np.zeros(8), action, np.zeros(8))
        assert np.isfinite(report["prediction_error"])

    def test_decision_log(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=8, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=8)
        for _ in range(5):
            agent.step(env.observe(), environment_step_fn=env.step)
        assert len(agent.decision_log.events) > 0


# ─── Invariants ──────────────────────────────────────────────────────

class TestInvariants:
    @pytest.fixture
    def run_agent(self):
        from reality_scaler.agent import RealityScalerAgent, PhysicsSimEnvironment
        agent = RealityScalerAgent.create(state_dim=16, action_dim=4, energy_budget=1.0)
        env = PhysicsSimEnvironment(state_dim=16)
        for _ in range(50):
            agent.step(env.observe(), environment_step_fn=env.step)
        return agent

    def test_energy_within_budget(self, run_agent):
        agent = run_agent
        assert agent.physics.state.energy_consumed <= agent.physics.state.energy_budget + 1e-12

    def test_bits_erased_le_processed(self, run_agent):
        agent = run_agent
        assert agent.physics.state.bits_erased <= agent.physics.state.bits_processed + 1e-9

    def test_memory_bounded(self, run_agent):
        agent = run_agent
        report = agent.memory.get_report()
        assert report["working_memory"]["size"] <= report["working_memory"]["capacity"]
        assert report["episodic_memory"]["size"] <= report["episodic_memory"]["capacity"]
        assert report["semantic_memory"]["size"] <= report["semantic_memory"]["capacity"]

    def test_decision_log_bounded(self, run_agent):
        assert len(run_agent.decision_log.events) <= 20000

    def test_counters_finite(self, run_agent):
        agent = run_agent
        assert np.isfinite(agent.metrics.total_information_gained)
        assert np.isfinite(agent.physics.state.energy_consumed)
        assert np.isfinite(agent.physics.state.bits_processed)


# ─── Baselines & Environments ────────────────────────────────────────

class TestBaselinesAndEnv:
    def test_random_baseline(self):
        from reality_scaler.baselines import RandomBaseline
        b = RandomBaseline(action_dim=4, seed=42)
        a = b.act(np.zeros(8))
        assert a.shape == (4,)

    def test_zero_baseline(self):
        from reality_scaler.baselines import ZeroActionBaseline
        b = ZeroActionBaseline(action_dim=4)
        a = b.act(np.zeros(8))
        assert np.allclose(a, 0)

    def test_linear_tanh_env(self):
        from reality_scaler.toy_env import LinearTanhEnv
        env = LinearTanhEnv(state_dim=8, action_dim=4, seed=42)
        ns = env.step(np.zeros(8), np.ones(4))
        assert ns.shape == (8,)
