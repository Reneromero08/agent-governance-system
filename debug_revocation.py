
import os
import sys
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path.cwd()
CAP = "46d06ff771e5857d84895ad4af4ac94196dfa5bf3f60a47140af039985f79e34"

def test_manual():
    tmp_path = REPO_ROOT / "temp_reproduce"
    tmp_path.mkdir(exist_ok=True)
    
    revokes_path = tmp_path / "REVOKES.json"
    revokes_path.write_bytes(json.dumps({"revokes_version": "1.0.0", "revoked_capabilities": [CAP]}, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": CAP}]}))

    env = dict(os.environ)
    # Only override Revokes, use real Caps/Pins from repo
    env["CATALYTIC_REVOKES_PATH"] = str(revokes_path)
    
    cmd = [sys.executable, "-m", "TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", "debug-fail"]
    
    print(f"Running: {cmd}")
    print(f"REVOKES_PATH: {env['CATALYTIC_REVOKES_PATH']}")
    
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))
    print(f"RC: {r.returncode}")
    print(f"STDOUT: {r.stdout}")
    print(f"STDERR: {r.stderr}")
    
    if "REVOKED_CAPABILITY" in r.stderr:
        print("SUCCESS: Revocation detected.")
    else:
        print("FAILURE: Revocation NOT detected.")

if __name__ == "__main__":
    test_manual()
