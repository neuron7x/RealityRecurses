import json
import os
import subprocess
import sys


def test_benchmarks_quick_generates_valid_json(tmp_path):
    out = tmp_path / "bench.json"
    cmd = [sys.executable, "scripts/benchmarks.py", "--quick", "--out", str(out)]
    subprocess.check_call(cmd)

    data = json.loads(out.read_text(encoding="utf-8"))

    # Schema invariants
    assert data["schema_version"] == "1"
    assert data["status"] == "ok"
    assert "timestamp_utc" in data

    params = data["params"]
    assert params["steps"] > 0
    assert params["state_dim"] > 0
    assert params["action_dim"] > 0

    timing = data["timing"]
    assert timing["total_wall_sec"] >= 0.0
    assert timing["steps_per_sec"] >= 0.0
    assert timing["mean_step_wall_ms"] >= 0.0

    # Best-effort: keys exist even if empty dict
    assert "agent_metrics" in data
    assert "physics" in data
