import subprocess
import sys


def test_audit_runs_clean():
    subprocess.check_call([sys.executable, "scripts/audit.py"])
