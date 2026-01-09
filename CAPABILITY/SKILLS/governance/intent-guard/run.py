#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CATALYTIC-DPT"))

from CAPABILITY.PIPELINES.pipeline_runtime import _slug  # type: ignore
from CAPABILITY.PRIMITIVES.restore_proof import canonical_json_bytes  # type: ignore

from CAPABILITY.TOOLS.utilities.intent import generate_intent

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")  # guarded


def _setup_pipeline(pipeline_id: str, runs_root: str, writer: Any) -> Path:
    slug = _slug(pipeline_id)
    pipeline_dir = PROJECT_ROOT / runs_root / "_pipelines" / slug
    if pipeline_dir.exists():
        shutil.rmtree(pipeline_dir) # scanner: rmtree (policy allows for pipeline reset)
    
    step_dir = pipeline_dir / "steps" / "step1"
    writer.mkdir_auto(str(step_dir))
    
    jobspec_path = step_dir / "JOBSPEC.json"
    jobspec_bytes = canonical_json_bytes({"step": "step1"})
    # Convert bytes to latin1 string for transparent write if writer only supports str?
    # Or just use write_bytes if writer supported it. GuardedWriter takes str.
    # use decode('utf-8') if json.
    writer.write_auto(str(jobspec_path), jobspec_bytes.decode('utf-8'))
    
    pipeline_spec = {
        "pipeline_id": pipeline_id,
        "steps": [
            {
                "step_id": "step1",
                "jobspec_path": _repo_rel(jobspec_path),
                "cmd": ["echo", "test"],
                "strict": True,
                "memoize": True,
            }
        ],
    }
    writer.mkdir_auto(str(pipeline_dir))
    # PIPELINE.json write
    writer.write_auto(str(pipeline_dir / "PIPELINE.json"), canonical_json_bytes(pipeline_spec).decode('utf-8'))
    return pipeline_dir


def main(input_path: Path, actual_path: Path) -> int:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    pipeline_id = data["pipeline_id"]
    runs_root = data.get("runs_root", "LAW/CONTRACTS/_runs")
    mode = data.get("mode", "artifact-only")
    allow_repo_write = bool(data.get("allow_repo_write", False))
    run_id = data.get("run_id")

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(PROJECT_ROOT, tmp_roots=["LAW/CONTRACTS/_runs/_tmp"], durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
    writer.open_commit_gate()

    pipeline_dir = _setup_pipeline(pipeline_id, runs_root, writer)
    intent_root = actual_path.parent / "intent"
    intent_path, intent_data = generate_intent(
        pipeline_id,
        runs_root,
        mode=mode,
        allow_repo_write=allow_repo_write,
        run_id=run_id,
        intent_root=intent_root,
    )
    intent_path_2, intent_data_2 = generate_intent(
        pipeline_id,
        runs_root,
        mode=mode,
        allow_repo_write=allow_repo_write,
        run_id=run_id,
        intent_root=intent_root,
    )

    admit_res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "governance" / "admission.py"), "--intent", str(intent_path)],
        cwd=str(PROJECT_ROOT),
    )

    actual = {
        "intent": intent_data,
        "repeat_same": intent_data == intent_data_2,
        "admission_rc": admit_res.returncode,
    }
    # Convert absolute path to relative path from repo root
    rel_actual_path = actual_path.resolve().relative_to(PROJECT_ROOT)
    writer.write_auto(
        str(rel_actual_path),
        json.dumps(actual, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <actual.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
