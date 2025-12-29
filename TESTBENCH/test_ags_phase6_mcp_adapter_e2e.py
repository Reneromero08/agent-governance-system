
import json
import subprocess
import sys
import pytest
from pathlib import Path

# Fix REPO_ROOT calculation:
# File: CATALYTIC-DPT/TESTBENCH/test_ags_phase6_mcp_adapter_e2e.py
# parents[0]=TESTBENCH, parents[1]=CATALYTIC-DPT, parents[2]=root
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "CATALYTIC-DPT" / "SKILLS"
FIXTURES_DIR = REPO_ROOT / "CATALYTIC-DPT" / "TESTBENCH" / "fixtures"

WRAPPER_PATH = SKILLS_DIR / "mcp-adapter" / "scripts" / "wrapper.py"
SERVER_PATH = FIXTURES_DIR / "dummy_mcp.py"

def run_wrapper(config: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    config_path = tmp_path / "config.json"
    out_path = tmp_path / "out.json"
    config_path.write_text(json.dumps(config))
    
    return subprocess.run(
        [sys.executable, str(WRAPPER_PATH), str(config_path), str(out_path)],
        capture_output=True,
        text=True
    )

def test_happy_echo(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "echo",
            "params": {"hello": "world"},
            "id": 1
        },
        "caps": {
            "max_stdout_bytes": 1024,
            "max_stderr_bytes": 0,
            "timeout_ms": 1000,
            "allowed_exit_codes": [0]
        }
    }
    
    res = run_wrapper(config, tmp_path)
    if res.returncode != 0:
        pytest.fail(f"Wrapper failed. Stderr: {res.stderr}")
    
    assert res.returncode == 0
    
    out_json = json.loads((tmp_path / "out.json").read_text())
    assert out_json["status"] == "success"
    assert out_json["response"]["result"]["hello"] == "world"

def test_fail_stderr(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "stderr",
            "params": {},
            "id": 1
        },
        "caps": {
            "max_stdout_bytes": 1024,
            "max_stderr_bytes": 0, # Strict
            "timeout_ms": 1000
        }
    }
    res = run_wrapper(config, tmp_path)
    assert res.returncode != 0
    assert "STDERR_EMITTED" in res.stderr

def test_fail_timeout(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "sleep",
            "params": {"ms": 2000},
            "id": 1
        },
        "caps": {
            "timeout_ms": 500 # Short timeout
        }
    }
    res = run_wrapper(config, tmp_path)
    assert res.returncode != 0
    assert "TIMEOUT" in res.stderr

def test_fail_bloat(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "bloat",
            "params": {"size": 2000},
            "id": 1
        },
        "caps": {
            "max_stdout_bytes": 1000 # Too small
        }
    }
    res = run_wrapper(config, tmp_path)
    assert res.returncode != 0
    assert "STDOUT_OVERFLOW" in res.stderr

def test_fail_malformed_output(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "malformed",
            "params": {},
            "id": 1
        },
    }
    res = run_wrapper(config, tmp_path)
    assert res.returncode != 0
    assert "INVALID_JSON_OUTPUT" in res.stderr

def test_fail_crash(tmp_path):
    config = {
        "server_command": [sys.executable, str(SERVER_PATH)],
        "request_envelope": {
            "jsonrpc": "2.0",
            "method": "crash",
            "params": {},
            "id": 1
        },
        "caps": {
            "allowed_exit_codes": [0]
        }
    }
    res = run_wrapper(config, tmp_path)
    assert res.returncode != 0
    assert "BAD_EXIT_CODE" in res.stderr
