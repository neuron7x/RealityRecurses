#!/usr/bin/env python3
"""PR Quality Gate runner (merge-blocking).

This is the single, authoritative entrypoint for local + CI checks.

Quick:
  python scripts/pr_gate.py --quick

Full:
  python scripts/pr_gate.py --full

Design:
- fail-fast
- deterministic command set
- explicit separation of unit suite vs validation suite
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Cmd:
    name: str
    argv: Sequence[str]


def _run(cmd: Cmd) -> int:
    print(f"\n==> {cmd.name}\n$ {' '.join(cmd.argv)}")
    return subprocess.call(list(cmd.argv))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fast checks suitable for tight local loops")
    ap.add_argument("--full", action="store_true", help="Full checks (includes validation suite)")
    args = ap.parse_args()

    # Default: quick, unless explicitly --full
    quick = args.quick or not args.full
    full = args.full

    python = sys.executable

    cmds: list[Cmd] = [
        Cmd("ruff (lint)", [python, "-m", "ruff", "check", "."]),
        Cmd("black (format check)", [python, "-m", "black", "--check", "."]),
        Cmd("mypy (types)", [python, "-m", "mypy", "src"]),
        # Unit+integration+regression: exclude validation by default
        Cmd(
            "pytest (unit/integration/regression) + coverage",
            [
                python,
                "-m",
                "pytest",
                "-m",
                "not validation",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-fail-under=90",
            ],
        ),
        Cmd("benchmarks (quick contract)", [python, "scripts/benchmarks.py", "--quick", "--out", "benchmarks.json"]),
        Cmd("audit (100-point)", [python, "scripts/audit.py"]),
    ]

    if full and not quick:
        # Validation suite: run without coverage to avoid subprocess coverage pitfalls.
        cmds.append(Cmd("pytest (validation suite)", [python, "-m", "pytest", "-m", "validation", "--no-cov"]))

    for c in cmds:
        rc = _run(c)
        if rc != 0:
            print(f"\nFAIL: {c.name} (exit {rc})")
            return rc

    print("\nOK: PR gate passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
