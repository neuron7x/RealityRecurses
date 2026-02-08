# Contributing

Non-commercial collaboration is welcome under the terms in `LICENSE`.

## Requirements for every PR
- Tests: `pytest -q`
- Audit gate: `python scripts/audit.py`
- Evidence: paste command output or link the CI run.

## Local
```bash
python -m pip install -e .
pytest -q
python scripts/audit.py
```
