#!/usr/bin/env python3
"""
Run a Reality Scaler simulation from the command line.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --steps 500 --state-dim 32 --action-dim 8
"""
import argparse
import json
import sys

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reality_recurses.agent import RealityScalerAgent, PhysicsSimEnvironment


def main():
    parser = argparse.ArgumentParser(description="Reality Scaler Simulation")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps (default 200)")
    parser.add_argument("--state-dim", type=int, default=64, help="State dimensionality (default 64)")
    parser.add_argument("--action-dim", type=int, default=16, help="Action dimensionality (default 16)")
    parser.add_argument("--energy-budget", type=float, default=1.0, help="Energy budget (default 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args()

    agent = RealityScalerAgent.create(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        energy_budget=args.energy_budget,
        seed=args.seed,
    )
    env = PhysicsSimEnvironment(state_dim=args.state_dim)

    for step in range(args.steps):
        state = env.observe()
        report = agent.step(state, environment_step_fn=env.step)
        if not args.json and (step + 1) % 50 == 0:
            print(f"  step {step+1:5d} | error={report['prediction_error']:.4f} | info={report['information_gained']:.2f}")

    full = agent.get_full_report()
    if args.json:
        print(json.dumps(full, indent=2, default=str))
    else:
        print(f"\nSimulation complete: {args.steps} steps")
        print(f"  Total information: {full['agent']['total_information_gained']:.2f} bits")
        print(f"  Energy consumed:   {full['physics']['energy_consumed_J']:.2e} J")
        print(f"  Regime:            {full['entropy_budget']['regime']}")
        print(f"  Causal edges:      {full['causal']['n_edges']}")
        print(f"  Memory traces:     {full['memory']['total_traces']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
