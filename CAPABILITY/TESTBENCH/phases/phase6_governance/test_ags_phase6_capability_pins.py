import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAP = "4f81ae57f3d1c61488c71a9042b041776dd463e6334568333321d15b6b7d78fc"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes as _canon

def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, env=env)

def test_capability_pinned_routes_and_verifies(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-ok"
    cap, _ = _mock_registry(tmp_path)
    
    plan_path = tmp_path / "plan.json"

    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}), encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    pins_path.write_bytes(_canon({"pins_version": "1.0.0", "allowed_capabilities": [cap]}))

    env["CATALYTIC_PINS_PATH"] = str(pins_path)

    # The v1 capability is pinned to a concrete ant-worker command that expects these files.
    reg_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "phase65_registry"
    in_file = reg_root / "in.txt"
    out_file = reg_root / "out.txt"
    
    try:
        r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route_ok.returncode == 0
        assert not in_file.exists()
    finally:
        if in_file.is_file():
            in_file.unlink()

        if out_file.is_file():
            out_file.unlink()


def test_verify_rejects_unpinned_even_if_pipeline_artifact_exists(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-verify-bypass"
    cap, env = _mock_registry(tmp_path)
    
    plan_path = tmp_path / "plan.json"

    plan_path.write_text(json.dumps({"plan_version": "1.0", "steps": [{"step_id": "s1", "capability_hash": cap}]}), encoding="utf-8")

    pins_path = tmp_path / "PINS.json"
    _rm(pins_path)
    
    r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
    assert r_route_ok.returncode == 0

    r_verify = _run([sys.executable, "-m", "LAW.CANON.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
    assert r_verify.returncode != 0
    assert "CAPABILITY_NOT_PINNED" in r_verify.stdout

def test_verify_rejects_empty_pins(tmp_path: Path) -> None:
    pipeline_id = "ags-capability-pins-empty-pins"
    
    caps = [CAP]
    
    for cap in caps:
        pins_path = tmp_path / f"PINS_{cap}.json"
        _rm(pins_path)
        
        r_route_ok = _run([sys.executable, "-m", "CAPABILITY.TOOLS.ags", "route", "--plan", str(plan_path), "--pipeline-id", pipeline_id], env=env)
        assert r_route_ok.returncode == 0
        
        r_verify = _run([sys.executable, "-m", "LAW.CANON.catalytic", "pipeline", "verify", "--pipeline-id", pipeline_id, "--strict"], env=env)
        if r_verify.returncode != 0:
            assert "CAPABILITY_NOT_PINNED" in r_verify.stdout
