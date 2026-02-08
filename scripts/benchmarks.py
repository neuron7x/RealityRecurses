#!/usr/bin/env python3
"""Benchmarks & Metrics for RealityRecurses.

Outputs a JSON metrics bundle intended for PR-gating and trend tracking.

Design goals:
- Runs fast in CI (--quick)
- Produces stable schema (tests validate schema)
- Avoids external deps beyond the project itself
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

# Ensure src/ is importable when running from repo root without install
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from reality_recurses.agent import run_simulation  # type: ignore


def benchmark(steps: int, state_dim: int, action_dim: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    report = run_simulation(n_steps=steps, state_dim=state_dim, action_dim=action_dim, verbose=False)
    t1 = time.perf_counter()

    wall = t1 - t0
    steps_per_sec = float(steps) / wall if wall > 0 else 0.0

    # Pull best-effort metrics from report (fail-closed schema is enforced by tests)
    metrics = report.get("metrics") or {}
    physics = report.get("physics") or {}
    return {
        "schema_version": "1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": {
            "steps": int(steps),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
        },
        "timing": {
            "total_wall_sec": float(wall),
            "steps_per_sec": float(steps_per_sec),
            "mean_step_wall_ms": float((wall / steps) * 1000.0) if steps > 0 else 0.0,
        },
        "agent_metrics": metrics,
        "physics": physics,
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fast CI run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--action-dim", type=int, default=16)
    ap.add_argument("--out", type=str, default="benchmarks.json")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else (40 if args.quick else 200)

    out = benchmark(steps=steps, state_dim=args.state_dim, action_dim=args.action_dim)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
