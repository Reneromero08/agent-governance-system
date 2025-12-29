"""
Phase 8: Model Binding Tests

Tests verify router receipt artifacts and fail-closed behavior:
1. Router receipts are created (ROUTER.json, ROUTER_OUTPUT.json, ROUTER_TRANSCRIPT_HASH)
2. Router over-output fails closed
3. Router stderr fails closed
4. Malformed plan fails schema validation
5. Capability escalation attempts fail closed
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd, env=None):
    return subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env or os.environ
    )


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_router_receipts_created(tmp_path):
    """Router execution must create receipt artifacts."""
    # Create a simple router that outputs a valid plan
    router_script = tmp_path / "router.py"
    router_script.write_text(
        """
import json
import sys
plan = {
    "plan_version": "1.0",
    "steps": [
        {
            "step_id": "test-step",
            "command": ["echo", "hello"],
            "jobspec": {
                "job_id": "test-job",
                "phase": 8,
                "task_type": "test_execution",
                "intent": "Test router receipts",
                "inputs": {},
                "outputs": {
                    "durable_paths": [],
                    "validation_criteria": {}
                },
                "catalytic_domains": [],
                "determinism": "deterministic"
            }
        }
    ]
}
sys.stdout.write(json.dumps(plan))
""",
        encoding="utf-8",
    )

    plan_output = tmp_path / "plan.json"
    receipt_dir = tmp_path / ".router_receipts"

    try:
        _rm(receipt_dir)

        cmd = [
            sys.executable,
            "-m",
            "TOOLS.ags",
            "plan",
            "--router",
            sys.executable,
            "--router-arg",
            str(router_script),
            "--out",
            str(plan_output),
        ]
        r = _run(cmd)

        assert r.returncode == 0, f"Router should succeed: {r.stderr}"

        # Verify receipt artifacts exist
        assert receipt_dir.exists(), "Receipt directory should be created"

        router_json = receipt_dir / "plan_ROUTER.json"
        router_output = receipt_dir / "plan_ROUTER_OUTPUT.json"
        transcript_hash = receipt_dir / "plan_ROUTER_TRANSCRIPT_HASH"

        assert router_json.exists(), "ROUTER.json should exist"
        assert router_output.exists(), "ROUTER_OUTPUT.json should exist"
        assert transcript_hash.exists(), "ROUTER_TRANSCRIPT_HASH should exist"

        # Verify ROUTER.json content
        router_data = json.loads(router_json.read_text())
        assert "router_executable" in router_data
        assert "router_hash_sha256" in router_data
        assert router_data["router_exit_code"] == 0
        assert router_data["router_stderr_bytes"] == 0

        # Verify ROUTER_OUTPUT.json is canonical JSON
        output_data = json.loads(router_output.read_text())
        assert output_data["plan_version"] == "1.0"
        assert len(output_data["steps"]) == 1

        # Verify transcript hash is a valid SHA-256
        hash_value = transcript_hash.read_text()
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    finally:
        _rm(receipt_dir)
        _rm(plan_output)


def test_router_over_output_fails_closed(tmp_path):
    """Router producing too much output must fail closed."""
    # Create a router that outputs excessive data
    router_script = tmp_path / "router_spam.py"
    router_script.write_text(
        """
import sys
# Output way more than the default max (10MB)
for i in range(100000):
    sys.stdout.write("x" * 1000)
""",
        encoding="utf-8",
    )

    plan_output = tmp_path / "plan.json"

    cmd = [
        sys.executable,
        "-m",
        "TOOLS.ags",
        "plan",
        "--router",
        sys.executable,
        "--router-arg",
        str(router_script),
        "--out",
        str(plan_output),
    ]
    r = _run(cmd)

    assert r.returncode != 0, "Router with excessive output should fail"
    assert "ROUTER_OUTPUT_TOO_LARGE" in (r.stderr + r.stdout)


def test_router_stderr_fails_closed(tmp_path):
    """Router producing stderr must fail closed."""
    router_script = tmp_path / "router_stderr.py"
    router_script.write_text(
        """
import sys
import json
sys.stderr.write("ERROR: something went wrong\\n")
plan = {"plan_version": "1.0", "steps": []}
sys.stdout.write(json.dumps(plan))
""",
        encoding="utf-8",
    )

    plan_output = tmp_path / "plan.json"

    cmd = [
        sys.executable,
        "-m",
        "TOOLS.ags",
        "plan",
        "--router",
        sys.executable,
        "--router-arg",
        str(router_script),
        "--out",
        str(plan_output),
    ]
    r = _run(cmd)

    assert r.returncode != 0, "Router with stderr should fail"
    assert "ROUTER_STDERR_NOT_EMPTY" in (r.stderr + r.stdout)


def test_malformed_plan_fails_schema_validation(tmp_path):
    """Router producing malformed JSON must fail schema validation."""
    router_script = tmp_path / "router_malformed.py"
    router_script.write_text(
        """
import sys
# Missing required fields
sys.stdout.write('{"steps": "not-a-list"}')
""",
        encoding="utf-8",
    )

    plan_output = tmp_path / "plan.json"

    cmd = [
        sys.executable,
        "-m",
        "TOOLS.ags",
        "plan",
        "--router",
        sys.executable,
        "--router-arg",
        str(router_script),
        "--out",
        str(plan_output),
    ]
    r = _run(cmd)

    assert r.returncode != 0, "Router with malformed plan should fail"


def test_capability_escalation_fails_closed(tmp_path):
    """Router attempting capability escalation must fail closed."""
    # Create a router that tries to use a revoked capability
    router_script = tmp_path / "router_escalate.py"
    router_script.write_text(
        """
import json
import sys
# Try to use a capability that's revoked
plan = {
    "plan_version": "1.0",
    "steps": [{
        "step_id": "escalate",
        "capability_hash": "e8e7e5234b43278a1a257b9257186b8bca5fdae9ab9096572942da1c5fb90f36"
    }]
}
sys.stdout.write(json.dumps(plan))
""",
        encoding="utf-8",
    )

    # Create a revokes file
    revokes_file = tmp_path / "REVOKES.json"
    revokes_file.write_bytes(
        json.dumps(
            {
                "revokes_version": "1.0.0",
                "revoked_capabilities": [
                    "e8e7e5234b43278a1a257b9257186b8bca5fdae9ab9096572942da1c5fb90f36"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    plan_output = tmp_path / "plan.json"

    env = dict(os.environ)
    env["CATALYTIC_CAPABILITIES_PATH"] = str(
        REPO_ROOT / "CATALYTIC-DPT" / "CAPABILITIES.json"
    )
    env["CATALYTIC_PINS_PATH"] = str(
        REPO_ROOT / "CATALYTIC-DPT" / "CAPABILITY_PINS.json"
    )
    env["CATALYTIC_REVOKES_PATH"] = str(revokes_file)

    cmd = [
        sys.executable,
        "-m",
        "TOOLS.ags",
        "plan",
        "--router",
        sys.executable,
        "--router-arg",
        str(router_script),
        "--out",
        str(plan_output),
    ]
    r = _run(cmd, env)

    assert r.returncode != 0, "Router with revoked capability should fail"
    assert "REVOKED_CAPABILITY" in (r.stderr + r.stdout)
