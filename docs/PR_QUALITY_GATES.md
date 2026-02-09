# PR Quality Gates (merge-blocking)

This repository enforces **merge-blocking** quality gates. The authoritative runner is:

```bash
python scripts/pr_gate.py --quick
python scripts/pr_gate.py --full
```

## Gate registry

- **GATE-001: Lint**
  - Tool: `ruff check .`
  - Fails on: static errors, unused imports, common bug patterns.

- **GATE-002: Format**
  - Tool: `black --check .`
  - Fails on: formatting divergence.

- **GATE-003: Types**
  - Tool: `mypy src`
  - Fails on: type errors across `src/`.

- **GATE-004: Unit/Integration/Regression tests**
  - Tool: `pytest -m "not validation"`
  - Fails on: any test failure.

- **GATE-005: Coverage**
  - Tool: `pytest-cov` with `--cov-fail-under=90`
  - Threshold: **>= 90% line coverage**.

- **GATE-006: Benchmarks (contract)**
  - Tool: `python scripts/benchmarks.py --quick`
  - Fails on: non-zero exit, invalid schema (validated by tests).

- **GATE-007: Audit**
  - Tool: `python scripts/audit.py`
  - Fails on: any of the 100 checks failing.

- **GATE-008: Validation suite (optional / full)**
  - Tool: `pytest -m validation --no-cov`
  - Intended: broader checks that can be slower or more integration-heavy.

## Marker policy

Use markers to keep PR feedback loops tight:

- `unit`: fast, isolated
- `integration`: crosses module boundaries
- `regression`: guards a previously-fixed bug
- `validation`: slower / broader; excluded from default CI unit suite

## Local developer setup

```bash
python -m pip install -e .[dev]
pre-commit install
python scripts/pr_gate.py --quick
```
