#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PIPELINES.pipeline_runtime import _slug  # type: ignore
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
DEFAULT_RUNS_ROOT = "LAW/CONTRACTS/_runs"

writer = GuardedWriter(
    project_root=REPO_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs"]
)


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _sorted_paths(paths: Set[str]) -> List[str]:
    return sorted(paths)


def _write_json(path: Path, data: Dict[str, object], writer_obj: Optional[GuardedWriter] = None) -> None:
    writer_instance = writer_obj or writer
    writer_instance.mkdir_durable(str(path.parent.relative_to(REPO_ROOT)), parents=True, exist_ok=True)
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if writer_obj is None:
        writer_instance.open_commit_gate()
    writer_instance.write_durable(str(path.relative_to(REPO_ROOT)), serialized)


def _load_pipeline_spec(pipeline_dir: Path) -> Tuple[List[str], List[str]]:
    read_paths: Set[str] = set()
    spec_path = pipeline_dir / "PIPELINE.json"
    if not spec_path.exists():
        return [], []
    read_paths.add(_repo_rel(spec_path))
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        steps = spec.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                jobspec = step.get("jobspec_path")
                if isinstance(jobspec, str) and jobspec:
                    read_paths.add(jobspec)
    except Exception:
        pass
    write_targets: Set[str] = set()
    write_targets.add(_repo_rel(pipeline_dir))
    return list(read_paths), list(write_targets)


def generate_intent(
    pipeline_id: str,
    runs_root: str = DEFAULT_RUNS_ROOT,
    *,
    mode: str = "artifact-only",
    allow_repo_write: bool = False,
    run_id: Optional[str] = None,
    intent_root: Optional[Path] = None,
) -> Tuple[Path, Dict[str, object]]:
    if mode not in {"artifact-only", "repo-write"}:
        raise ValueError("mode must be 'artifact-only' or 'repo-write'")

    slug = _slug(pipeline_id)
    effective_run_id = run_id or slug
    runs_root_path = REPO_ROOT / runs_root
    pipeline_dir = runs_root_path / "_pipelines" / slug

    read_paths, write_targets = _load_pipeline_spec(pipeline_dir)

    artifact_root_rel = str(Path(runs_root).as_posix())
    write_paths: Set[str] = {artifact_root_rel}
    write_paths.update(write_targets)

    sorted_read = _sorted_paths(set(read_paths))
    sorted_write = _sorted_paths(write_paths)

    base_intent_root = Path(intent_root if intent_root is not None else runs_root_path / "intent")
    if not base_intent_root.is_absolute():
        base_intent_root = REPO_ROOT / base_intent_root
    intent_path = base_intent_root / effective_run_id / "intent.json"

    intent_data = {
        "mode": mode,
        "paths": {"read": sorted_read, "write": sorted_write},
        "allow_repo_write": bool(allow_repo_write),
    }

    _write_json(intent_path, intent_data)
    return intent_path, intent_data
