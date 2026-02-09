import json
import subprocess
import sys

import pytest


@pytest.mark.validation
def test_validation_pro006_self_audit_emits_100_checks_and_ok():
    # Use audit.py as authoritative contract
    out = subprocess.check_output([sys.executable, "scripts/audit.py", "--json"], text=True)
    report = json.loads(out)
    assert report["summary"]["total"] == 100
    assert report["ok"] is True
