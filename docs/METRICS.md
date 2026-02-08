# Metrics

This repository ships **machine-verifiable metrics** to keep changes measurable and reviewable.

## CI artifacts
Every Pull Request runs:
- `pytest`
- `python scripts/audit.py`
- `python scripts/benchmarks.py --quick --out benchmarks.json`

Artifacts uploaded by CI:
- `benchmarks.json` (stable schema)
- `audit_100_report.json` (self-audit report)

## benchmarks.json schema (v1)
Top-level keys:
- `schema_version` (string)
- `timestamp_utc` (ISO-8601 UTC)
- `params`:
  - `steps` (int)
  - `state_dim` (int)
  - `action_dim` (int)
- `timing`:
  - `total_wall_sec` (float)
  - `steps_per_sec` (float)
  - `mean_step_wall_ms` (float)
- `agent_metrics` (dict, best-effort passthrough from run report)
- `physics` (dict, best-effort passthrough from run report)
- `status` ("ok")

## PR expectations
A PR is considered **mergeable** only if CI is green:
- tests pass
- audit passes
- benchmarks generate a valid metrics bundle
