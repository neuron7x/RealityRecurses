"""
Thermodynamic Memory System
=============================
Memory as a physical process. Every write costs energy. Every read
generates entropy. Forgetting is not a bug — it's thermodynamic necessity.

Principles:
1. Memory has physical cost (Landauer)
2. Compression is the foundation of generalization
3. Forgetting is entropy management
4. Retrieval priority follows information density, not recency

    ┌─────────────────────────────────────────────┐
    │              MEMORY HIERARCHY                │
    │                                              │
    │  ┌──────────┐  high energy    ┌──────────┐  │
    │  │ WORKING  │  per access     │ EPISODIC │  │
    │  │ MEMORY   │────────────────→│ MEMORY   │  │
    │  │ (fast)   │  compression    │ (medium) │  │
    │  └──────────┘                 └──────────┘  │
    │       │                            │        │
    │       │     consolidation          │        │
    │       ▼                            ▼        │
    │  ┌──────────┐  low energy     ┌──────────┐  │
    │  │ SEMANTIC │  per access     │ CAUSAL   │  │
    │  │ MEMORY   │────────────────→│ MEMORY   │  │
    │  │ (slow)   │  abstraction    │ (deep)   │  │
    │  └──────────┘                 └──────────┘  │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np




@dataclass
class MemoryTrace:
    """A single memory trace with thermodynamic + provenance metadata."""
    trace_id: str
    content: np.ndarray
    compressed: Optional[np.ndarray] = None

    # Thermodynamic properties
    energy_cost_write: float = 0.0
    energy_cost_reads: float = 0.0
    n_reads: int = 0
    n_consolidations: int = 0

    # Information properties
    information_content: float = 0.0
    compression_ratio: float = 1.0
    relevance_score: float = 1.0

    # Entropy / noise
    entropy_score: float = 0.0  # 0=low entropy, 1=high entropy
    noise_flag: bool = False

    # Temporal properties
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Metadata
    domain: str = "general"
    memory_type: str = "working"  # working | episodic | semantic | causal
    provenance: dict[str, Any] = field(default_factory=dict)
    rewrite_count: int = 0
    bits_erased_on_forget: float = 0.0

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def information_density(self) -> float:
        """Information per unit energy."""
        total_energy = self.energy_cost_write + self.energy_cost_reads
        if total_energy <= 0:
            return 0.0
        return float(self.information_content) / float(total_energy)

    @property
    def access_value(self) -> float:
        """Priority for retention/access (higher = keep)."""
        # Prefer high info density + relevance, penalize entropy and age.
        age_penalty = 1.0 / (1.0 + max(0.0, self.age))
        entropy_penalty = 1.0 - max(0.0, min(1.0, self.entropy_score))
        return float(self.information_density + self.relevance_score * entropy_penalty * age_penalty)

    def mark_access(self, read_energy_cost: float = 0.0) -> None:
        self.n_reads += 1
        self.energy_cost_reads += float(read_energy_cost)
        self.last_accessed = time.time()

    def access(self, energy_cost: float = 0.0) -> None:
        """Backward-compatible alias for mark_access."""
        self.mark_access(read_energy_cost=energy_cost)



    def rewrite(self, new_content: np.ndarray, write_energy_cost: float = 0.0, *, note: str = "rewrite") -> None:
        """Rewrite memory trace content (physically costly)."""
        self.content = np.array(new_content, copy=True)
        self.updated_at = time.time()
        self.rewrite_count += 1
        self.energy_cost_write += float(write_energy_cost)
        self.provenance.setdefault("rewrites", []).append({
            "ts": self.updated_at,
            "note": note,
        })

class WorkingMemory:
    """
    Fast, limited-capacity buffer for current processing.
    High energy cost per access, but immediate availability.
    """

    def __init__(self, capacity: int = 7, energy_per_access: float = 1e-6):
        self.capacity = capacity
        self.energy_per_access = energy_per_access
        self.buffer: list[MemoryTrace] = []
        self.total_energy = 0.0

    @property
    def utilization(self) -> float:
        return len(self.buffer) / self.capacity

    def write(self, trace_id: str, content: np.ndarray, domain: str = "general") -> MemoryTrace:
        """Write to working memory. Evicts least valuable if full."""
        info_bits = float(content.size * np.log2(max(2, np.max(np.abs(content)) * 100)))
        energy = self.energy_per_access * content.size
        self.total_energy += energy

        trace = MemoryTrace(
            trace_id=trace_id,
            content=content.copy(),
            energy_cost_write=energy,
            information_content=info_bits,
            domain=domain,
            memory_type="working"
        )

        if len(self.buffer) >= self.capacity:
            # Evict least valuable
            self.buffer.sort(key=lambda t: t.access_value)
            evicted = self.buffer.pop(0)
            # Evicted trace should be consolidated to episodic memory
            return_trace = evicted  # Caller can consolidate

        self.buffer.append(trace)
        return trace

    def read(self, trace_id: str) -> Optional[np.ndarray]:
        for trace in self.buffer:
            if trace.trace_id == trace_id:
                trace.access(self.energy_per_access)
                self.total_energy += self.energy_per_access
                return trace.content.copy()
        return None

    def read_most_relevant(self, query: np.ndarray, top_k: int = 1) -> list[MemoryTrace]:
        """Retrieve most relevant traces by cosine similarity."""
        if not self.buffer:
            return []
        scores = []
        for trace in self.buffer:
            q = query.flatten()
            c = trace.content.flatten()
            min_len = min(len(q), len(c))
            q, c = q[:min_len], c[:min_len]
            norm = (np.linalg.norm(q) * np.linalg.norm(c))
            sim = float(np.dot(q, c) / norm) if norm > 0 else 0.0
            scores.append((sim, trace))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [trace for _, trace in scores[:top_k]]
        for trace in results:
            trace.access(self.energy_per_access)
        return results


class EpisodicMemory:
    """
    Medium-term memory of specific experiences (episodes).
    Stores state-action-outcome sequences with temporal context.
    """

    def __init__(self, capacity: int = 1000, energy_per_access: float = 1e-8):
        self.capacity = capacity
        self.energy_per_access = energy_per_access
        self.episodes: list[MemoryTrace] = []
        self.total_energy = 0.0
        self._index = 0

    def store_episode(self, state: np.ndarray, action: np.ndarray,
                      outcome: np.ndarray, domain: str = "interaction") -> MemoryTrace:
        """Store a complete state-action-outcome episode."""
        content = np.concatenate([state.flatten(), action.flatten(), outcome.flatten()])
        info_bits = float(content.size * 4)  # Rough estimate
        energy = self.energy_per_access * content.size
        self.total_energy += energy

        trace = MemoryTrace(
            trace_id=f"ep_{self._index}",
            content=content,
            energy_cost_write=energy,
            information_content=info_bits,
            domain=domain,
            memory_type="episodic"
        )
        self._index += 1

        if len(self.episodes) >= self.capacity:
            # Remove least informative episode
            self.episodes.sort(key=lambda t: t.information_density)
            self.episodes.pop(0)

        self.episodes.append(trace)
        return trace

    def recall_similar(self, query_state: np.ndarray, top_k: int = 5) -> list[MemoryTrace]:
        """Recall episodes similar to query state."""
        if not self.episodes:
            return []
        q = query_state.flatten()
        scores = []
        for ep in self.episodes:
            c = ep.content[:len(q)]
            if len(c) < len(q):
                c = np.pad(c, (0, len(q) - len(c)))
            norm = np.linalg.norm(q) * np.linalg.norm(c)
            sim = float(np.dot(q, c) / norm) if norm > 0 else 0.0
            scores.append((sim, ep))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scores[:top_k]]

    def consolidate_to_semantic(self) -> Optional[np.ndarray]:
        """
        Extract statistical regularities from episodes.
        This is the compression step: many episodes → abstract knowledge.
        """
        if len(self.episodes) < 10:
            return None
        contents = np.array([ep.content for ep in self.episodes[-100:]])
        # Extract mean and principal components
        mean = np.mean(contents, axis=0)
        centered = contents - mean
        if centered.shape[0] > 1:
            cov = np.cov(centered.T)
            # Keep top eigenvalues (compression)
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                n_keep = max(1, np.sum(eigenvalues > np.max(eigenvalues) * 0.01))
                compression_ratio = n_keep / len(eigenvalues)
            except np.linalg.LinAlgError:
                compression_ratio = 1.0
        else:
            compression_ratio = 1.0
        return mean


class SemanticMemory:
    """
    Long-term abstract knowledge compressed from episodic memories.
    Low energy per access, high compression, generalized knowledge.
    """

    def __init__(self, capacity: int = 500, energy_per_access: float = 1e-10):
        self.capacity = capacity
        self.energy_per_access = energy_per_access
        self.knowledge: dict[str, MemoryTrace] = {}
        self.total_energy = 0.0

    def store(self, key: str, content: np.ndarray, domain: str = "abstract"):
        """Store or update a semantic memory."""
        energy = self.energy_per_access * content.size
        self.total_energy += energy

        if key in self.knowledge:
            # Merge with existing knowledge (Bayesian update)
            existing = self.knowledge[key].content
            min_len = min(len(existing), len(content))
            merged = 0.7 * existing[:min_len] + 0.3 * content[:min_len]
            self.knowledge[key].content = merged
            self.knowledge[key].n_consolidations += 1
            self.knowledge[key].information_content *= 1.1  # Consolidation increases value
        else:
            self.knowledge[key] = MemoryTrace(
                trace_id=f"sem_{key}",
                content=content.copy(),
                energy_cost_write=energy,
                information_content=float(content.size * 2),
                domain=domain,
                memory_type="semantic"
            )

    def retrieve(self, key: str) -> Optional[np.ndarray]:
        if key in self.knowledge:
            self.knowledge[key].access(self.energy_per_access)
            return self.knowledge[key].content.copy()
        return None

    def get_all_keys(self) -> list[str]:
        return list(self.knowledge.keys())


class ThermodynamicMemorySystem:
    """
    Unified memory system with thermodynamic constraints.
    Orchestrates working, episodic, semantic, and causal memories.
    """

    def __init__(self, working_capacity: int = 7, episodic_capacity: int = 1000,
                 semantic_capacity: int = 500,
                 *,
                 physics: Optional[Any] = None,
                 landauer_temperature_K: float = 300.0,
                 noise_discard_threshold: float = 0.92,
                 min_information_bits: float = 0.01):
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory(capacity=episodic_capacity)
        self.semantic = SemanticMemory(capacity=semantic_capacity)

        self.physics = physics
        self.landauer_temperature_K = landauer_temperature_K
        self.noise_discard_threshold = noise_discard_threshold
        self.min_information_bits = min_information_bits
        
        self._consolidation_count = 0
        self._total_operations = 0

    @property
    def total_energy(self) -> float:
        return self.working.total_energy + self.episodic.total_energy + self.semantic.total_energy

    @property
    def total_traces(self) -> int:
        return len(self.working.buffer) + len(self.episodic.episodes) + len(self.semantic.knowledge)

    def _compute_entropy_score(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """Heuristic entropy/noise estimator in [0,1]."""
        x = np.concatenate([state.flatten(), action.flatten(), next_state.flatten()]).astype(np.float64)
        if x.size == 0:
            return 1.0
        # Higher variance and high-frequency sign flips => higher entropy proxy
        var = float(np.var(x))
        diffs = np.diff(x)
        flips = float(np.mean((diffs[:-1] * diffs[1:] < 0))) if diffs.size >= 2 else 0.0
        # Normalize to [0,1] with soft saturation
        score = 1.0 - math.exp(-0.5 * var)  # 0..1
        score = 0.7 * score + 0.3 * min(1.0, flips * 2.0)
        return float(max(0.0, min(1.0, score)))

    def _compute_information_content(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """Heuristic information gain proxy in bits."""
        # Use prediction-error-like compressibility proxy: smaller residual => more predictable => lower info
        x = np.concatenate([state.flatten(), action.flatten()]).astype(np.float64)
        y = next_state.flatten().astype(np.float64)
        if y.size == 0:
            return 0.0
        # cheap linear predictor on last dim slice (bounded compute)
        k = min(32, x.size, y.size)
        if k == 0:
            return 0.0
        pred = x[-k:]
        tgt = y[:k]
        err = float(np.linalg.norm(tgt - pred)) / (float(np.linalg.norm(tgt)) + 1e-9)
        err = max(1e-9, min(1.0, err))
        return float(max(0.0, -math.log(err, 2)))

    def _landauer_cost_J(self, bits_erased: float) -> float:
        kB = 1.380649e-23
        return float(bits_erased) * kB * self.landauer_temperature_K * math.log(2.0)

    def _account_physics(self, bits_processed: int, bits_erased: float, label: str) -> None:
        if self.physics is None:
            return
        # Prefer explicit LandauerEngine API if present
        try:
            self.physics.process_bits(int(bits_processed), erasure_fraction=float(bits_erased) / max(1.0, float(bits_processed)), label=label)
        except Exception:
            # best-effort: ignore if not compatible
            return

    def rewrite_trace(self, trace_id: str, new_content: np.ndarray, *, memory_type: str = "semantic", note: str = "rewrite") -> bool:
        """Rewrite an existing trace. Returns True if rewritten."""
        trace: Optional[MemoryTrace] = None
        if memory_type == "working":
            for t in self.working.buffer:
                if t.trace_id == trace_id:
                    trace = t
                    break
        elif memory_type == "episodic":
            for t in self.episodic.episodes:
                if t.trace_id == trace_id:
                    trace = t
                    break
        elif memory_type == "semantic":
            trace = self.semantic.knowledge.get(trace_id)
        else:
            return False
        if trace is None:
            return False
        write_bits = int(new_content.size * 32)
        self._account_physics(write_bits, bits_erased=0.05 * write_bits, label="memory_rewrite")
        trace.rewrite(new_content, write_energy_cost=self._landauer_cost_J(0.05 * write_bits), note=note)
        return True

    def detect_memory_bias(self) -> dict[str, Any]:
        """Detect simple distributional bias induced by semantic memory."""
        domains = [t.domain for t in self.semantic.knowledge.values()]
        if not domains:
            return {"biased": False, "max_share": 0.0, "top_domain": None}
        from collections import Counter
        c = Counter(domains)
        top_domain, top_n = c.most_common(1)[0]
        share = float(top_n) / float(len(domains))
        return {"biased": bool(share > 0.8), "max_share": share, "top_domain": top_domain}

    def process_experience(self, state: np.ndarray, action: np.ndarray,
                          outcome: np.ndarray, domain: str = "interaction") -> dict[str, Any]:
        """
        Full memory processing pipeline:
        1. Write to working memory
        2. Store episode
        3. Periodically consolidate to semantic
        """
        self._total_operations += 1

        # Working memory
        wm_trace = self.working.write(f"wm_{self._total_operations}", state, domain)

        # Episodic memory
        ep_trace = self.episodic.store_episode(state, action, outcome, domain)

        # Periodic consolidation
        consolidated = False
        if self._total_operations % 50 == 0:
            semantic_summary = self.episodic.consolidate_to_semantic()
            if semantic_summary is not None:
                self.semantic.store(f"consolidated_{self._consolidation_count}", semantic_summary, domain)
                self._consolidation_count += 1
                consolidated = True

        return {
            "working_utilization": self.working.utilization,
            "episodic_size": len(self.episodic.episodes),
            "semantic_size": len(self.semantic.knowledge),
            "consolidated": consolidated,
            "total_energy": self.total_energy,
        }

    def recall(self, query: np.ndarray, memory_type: str = "all",
               top_k: int = 3) -> list[MemoryTrace]:
        """Recall from specified memory type(s)."""
        results = []
        if memory_type in ("all", "working"):
            results.extend(self.working.read_most_relevant(query, top_k))
        if memory_type in ("all", "episodic"):
            results.extend(self.episodic.recall_similar(query, top_k))
        # Sort by access value
        results.sort(key=lambda t: t.access_value, reverse=True)
        return results[:top_k]

    def get_report(self) -> dict[str, Any]:
        return {
            "total_operations": self._total_operations,
            "total_energy_J": self.total_energy,
            "total_traces": self.total_traces,
            "working_memory": {
                "size": len(self.working.buffer),
                "capacity": self.working.capacity,
                "utilization": self.working.utilization,
            },
            "episodic_memory": {
                "size": len(self.episodic.episodes),
                "capacity": self.episodic.capacity,
            },
            "semantic_memory": {
                "size": len(self.semantic.knowledge),
                "capacity": self.semantic.capacity,
                "keys": self.semantic.get_all_keys()[:10],
            },
            "consolidations": self._consolidation_count,
        }