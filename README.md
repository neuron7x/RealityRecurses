# RealityRecurses

RealityRecurses is a **physically bounded, causal agent architecture** for building **reproducible and verifiable** intelligent behavior under explicit constraints: energy/entropy accounting, bounded memory, causal interventions, and self-audit invariants.

## Quickstart

```bash
python -m pip install -e .
pytest -q
python scripts/audit.py
python scripts/run_simulation.py --steps 5 --json
```


## PR quality gates

```bash
python scripts/pr_gate.py --quick
python scripts/pr_gate.py --full
```

## What this repository provides

- Agent loop with explicit budgets (energy, entropy, information throughput)
- Causal core with intervention (`do`) and counterfactual hooks
- Thermodynamic memory (working/episodic/semantic) with bounded growth and controlled forgetting
- Self-audit that produces a pass/fail report suitable for CI gating
- GitHub Actions CI for pull requests (tests + audit)

## License (non-commercial, no resale)

This repository is the author's intellectual property and is licensed under **CC BY-NC-ND 4.0**.

You may **read, run, and share unmodified copies** for research, study, and personal use.

You may **not**:
- use it for commercial purposes (including selling, paid services, SaaS, consulting, courses, or embedding in commercial products),
- redistribute modified versions,
- relicense or remove attribution.

See `LICENSE` for details.
