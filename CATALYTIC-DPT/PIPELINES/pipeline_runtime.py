from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PRIMITIVES.restore_proof import canonical_json_bytes
from PRIMITIVES.preflight import PreflightValidator
from PIPELINES.pipeline_chain import build_chain_obj, compute_step_entry, verify_chain


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_canonical_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(obj))


_RUN_ID_SAFE = re.compile(r"[^a-z0-9-]+")


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", "-")
    s = _RUN_ID_SAFE.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        raise ValueError("cannot slugify empty string")
    return s


@dataclass(frozen=True)
class StepSpec:
    step_id: str
    jobspec_path: str
    cmd: List[str]
    strict: bool = True
    memoize: bool = True


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    steps: List[StepSpec]
    validator_semver: str = "0.1.0"
    validator_build_id: str = "phase0-canonical"
    timestamp: str = "CATALYTIC-DPT-02_CONFIG"


class PipelineRuntime:
    """
    Minimal deterministic pipeline runner.

    Output location (canon-compliant):
      CONTRACTS/_runs/_pipelines/<pipeline_id>/
        - PIPELINE.json (canonical JSON)
        - STATE.json (canonical JSON, atomically updated)
        - CHAIN.json (canonical JSON proof chain)
        - steps/<step_id>/RUN_REF.json

    Note: The user-facing spec sometimes references CONTRACTS/_pipelines/.
    This implementation uses CONTRACTS/_runs/_pipelines/ to comply with repo invariants
    restricting system artifacts to CONTRACTS/_runs/.
    """

    def __init__(self, *, project_root: Path):
        self.project_root = project_root
        self.schemas_dir = self.project_root / "CATALYTIC-DPT" / "SCHEMAS"
        self.jobspec_schema_path = self.schemas_dir / "jobspec.schema.json"

    def pipeline_dir(self, pipeline_id: str) -> Path:
        return self.project_root / "CONTRACTS" / "_runs" / "_pipelines" / _slug(pipeline_id)

    def _init_state_if_missing(self, *, pipeline_id: str, pdir: Path) -> None:
        spec_path = pdir / "PIPELINE.json"
        state_path = pdir / "STATE.json"
        if not spec_path.exists() or state_path.exists():
            return
        spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = self._parse_spec(pipeline_id=pipeline_id, obj=spec_obj)
        _atomic_write_canonical_json(state_path, self._initial_state(spec))

    def init_from_spec_path(self, *, pipeline_id: str, spec_path: Path) -> PipelineSpec:
        spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = self._parse_spec(pipeline_id=pipeline_id, obj=spec_obj)
        pdir = self.pipeline_dir(pipeline_id)
        pdir.mkdir(parents=True, exist_ok=True)
        _atomic_write_canonical_json(pdir / "PIPELINE.json", self._spec_to_json(spec))
        state = self._initial_state(spec)
        _atomic_write_canonical_json(pdir / "STATE.json", state)
        return spec

    def load(self, *, pipeline_id: str) -> Tuple[PipelineSpec, Dict[str, Any]]:
        pdir = self.pipeline_dir(pipeline_id)
        spec_path = pdir / "PIPELINE.json"
        state_path = pdir / "STATE.json"
        if spec_path.exists() and not state_path.exists():
            self._init_state_if_missing(pipeline_id=pipeline_id, pdir=pdir)
        if not spec_path.exists() or not state_path.exists():
            raise FileNotFoundError(f"pipeline not initialized: {pdir}")
        spec_obj = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = self._parse_spec(pipeline_id=pipeline_id, obj=spec_obj)
        state_obj = json.loads(state_path.read_text(encoding="utf-8"))
        state = self._coerce_state(spec, state_obj)
        self._assert_state_consistent(spec, state)
        return spec, state

    def status_text(self, *, pipeline_id: str) -> str:
        spec, state = self.load(pipeline_id=pipeline_id)
        total = len(spec.steps)
        completed = state["completed_steps"]
        lines = [
            f"pipeline_id: {spec.pipeline_id}",
            f"steps_total: {total}",
            f"steps_completed: {len(completed)}",
        ]
        next_step = None
        for step in spec.steps:
            if step.step_id not in completed:
                next_step = step.step_id
                break
        lines.append(f"next_step: {next_step if next_step is not None else 'NONE'}")
        for step in spec.steps:
            status = "DONE" if step.step_id in completed else "PENDING"
            run_id = state["step_run_ids"].get(step.step_id)
            if run_id is None:
                lines.append(f"{step.step_id}: {status}")
            else:
                lines.append(f"{step.step_id}: {status} run_id={run_id}")
        return "\n".join(lines) + "\n"

    def run(self, *, pipeline_id: str, spec_path: Optional[Path] = None, max_steps: Optional[int] = None) -> None:
        pdir = self.pipeline_dir(pipeline_id)
        if not pdir.exists():
            if spec_path is None:
                raise FileNotFoundError(f"pipeline missing and no spec provided: {pdir}")
            self.init_from_spec_path(pipeline_id=pipeline_id, spec_path=spec_path)
        else:
            self._init_state_if_missing(pipeline_id=pipeline_id, pdir=pdir)

        spec, state = self.load(pipeline_id=pipeline_id)
        completed_steps = set(state["completed_steps"])
        attempts: Dict[str, int] = dict(state.get("attempts", {}))

        steps_executed = 0
        for step_index, step in enumerate(spec.steps):
            if step.step_id in completed_steps:
                continue

            if max_steps is not None and steps_executed >= max_steps:
                break

            # Fail closed: if resuming from prior completed steps, require a valid chain.
            if completed_steps:
                v = verify_chain(project_root=self.project_root, pipeline_dir=pdir)
                if not v.get("ok", False):
                    raise RuntimeError(f"pipeline chain invalid before step {step.step_id}: {v.get('code')}")

            attempt = attempts.get(step.step_id, 0) + 1
            attempts[step.step_id] = attempt
            run_id = self._make_run_id(spec.pipeline_id, step.step_id, attempt)

            run_dir = self.project_root / "CONTRACTS" / "_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            self._execute_step(spec=spec, step=step, run_id=run_id)
            self._assert_step_outputs(run_dir)

            # Update CHAIN.json (deterministic, fail-closed).
            chain_entries: List[Dict[str, Any]] = []
            prev_hash: Optional[str] = None
            for s in spec.steps:
                sid = s.step_id
                if sid not in completed_steps and sid != step.step_id:
                    continue
                rid = state["step_run_ids"].get(sid)
                if sid == step.step_id:
                    rid = run_id
                if not isinstance(rid, str) or not rid:
                    raise RuntimeError(f"missing run_id for chained step {sid}")
                entry = compute_step_entry(
                    project_root=self.project_root,
                    pipeline_id=spec.pipeline_id,
                    step_id=sid,
                    run_id=rid,
                    prev_step_proof_hash=prev_hash,
                )
                prev_hash = entry["step_proof_hash"]
                chain_entries.append(entry)

            chain_obj = build_chain_obj(project_root=self.project_root, pipeline_id=spec.pipeline_id, ordered_steps=chain_entries)
            _atomic_write_canonical_json(pdir / "CHAIN.json", chain_obj)
            v = verify_chain(project_root=self.project_root, pipeline_dir=pdir)
            if not v.get("ok", False):
                raise RuntimeError(f"pipeline chain invalid after step {step.step_id}: {v.get('code')}")

            # Persist step run ref
            step_dir = pdir / "steps" / step.step_id
            step_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_canonical_json(step_dir / "RUN_REF.json", {"run_id": run_id})

            # Update state (atomic, deterministic)
            completed_steps.add(step.step_id)
            state["completed_steps"] = [s.step_id for s in spec.steps if s.step_id in completed_steps]
            state["current_step_index"] = max(step_index + 1, state.get("current_step_index", 0))
            state["step_run_ids"][step.step_id] = run_id
            state["attempts"] = dict(sorted(attempts.items(), key=lambda kv: kv[0]))
            _atomic_write_canonical_json(pdir / "STATE.json", state)

            steps_executed += 1

    # ----------------------------
    # Internals
    # ----------------------------

    def _parse_spec(self, *, pipeline_id: str, obj: Dict[str, Any]) -> PipelineSpec:
        if not isinstance(obj, dict):
            raise ValueError("pipeline spec must be a JSON object")
        pid = obj.get("pipeline_id", pipeline_id)
        if pid != pipeline_id:
            raise ValueError("pipeline_id mismatch between CLI and spec")
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError("pipeline_id must be a non-empty string")

        steps_obj = obj.get("steps")
        if not isinstance(steps_obj, list) or not steps_obj:
            raise ValueError("steps must be a non-empty list")

        steps: List[StepSpec] = []
        seen_ids: set[str] = set()
        for idx, raw in enumerate(steps_obj):
            if not isinstance(raw, dict):
                raise ValueError(f"step[{idx}] must be an object")
            step_id = raw.get("step_id")
            jobspec_path = raw.get("jobspec_path")
            cmd = raw.get("cmd")
            strict = raw.get("strict", True)
            memoize = raw.get("memoize", True)
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError(f"step[{idx}].step_id must be a non-empty string")
            if step_id in seen_ids:
                raise ValueError(f"duplicate step_id: {step_id}")
            seen_ids.add(step_id)
            if not isinstance(jobspec_path, str) or not jobspec_path.strip():
                raise ValueError(f"step[{idx}].jobspec_path must be a non-empty string")
            if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
                raise ValueError(f"step[{idx}].cmd must be a non-empty list of strings")
            if not isinstance(strict, bool):
                raise ValueError(f"step[{idx}].strict must be boolean")
            if not isinstance(memoize, bool):
                raise ValueError(f"step[{idx}].memoize must be boolean")

            steps.append(StepSpec(step_id=step_id, jobspec_path=jobspec_path, cmd=cmd, strict=strict, memoize=memoize))

        validator_semver = obj.get("validator_semver", "0.1.0")
        validator_build_id = obj.get("validator_build_id", "phase0-canonical")
        timestamp = obj.get("timestamp", "CATALYTIC-DPT-02_CONFIG")
        if not isinstance(validator_semver, str) or not validator_semver:
            raise ValueError("validator_semver must be a non-empty string")
        if not isinstance(validator_build_id, str) or not validator_build_id:
            raise ValueError("validator_build_id must be a non-empty string")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty string")

        return PipelineSpec(
            pipeline_id=pid,
            steps=steps,
            validator_semver=validator_semver,
            validator_build_id=validator_build_id,
            timestamp=timestamp,
        )

    def _spec_to_json(self, spec: PipelineSpec) -> Dict[str, Any]:
        return {
            "pipeline_id": spec.pipeline_id,
            "validator_semver": spec.validator_semver,
            "validator_build_id": spec.validator_build_id,
            "timestamp": spec.timestamp,
            "steps": [
                {
                    "step_id": s.step_id,
                    "jobspec_path": s.jobspec_path,
                    "cmd": list(s.cmd),
                    "strict": s.strict,
                    "memoize": s.memoize,
                }
                for s in spec.steps
            ],
        }

    def _initial_state(self, spec: PipelineSpec) -> Dict[str, Any]:
        return {
            "pipeline_id": spec.pipeline_id,
            "current_step_index": 0,
            "completed_steps": [],
            "step_run_ids": {},
            "attempts": {},
        }

    def _coerce_state(self, spec: PipelineSpec, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return self._initial_state(spec)
        if state.get("pipeline_id") != spec.pipeline_id:
            return self._initial_state(spec)
        completed = state.get("completed_steps", [])
        if not isinstance(completed, list):
            completed = []
        completed_set = {x for x in completed if isinstance(x, str)}
        completed_ordered = [s.step_id for s in spec.steps if s.step_id in completed_set]
        step_run_ids = state.get("step_run_ids", {})
        if not isinstance(step_run_ids, dict):
            step_run_ids = {}
        step_run_ids = {k: v for k, v in step_run_ids.items() if isinstance(k, str) and isinstance(v, str)}
        attempts = state.get("attempts", {})
        if not isinstance(attempts, dict):
            attempts = {}
        attempts = {k: int(v) for k, v in attempts.items() if isinstance(k, str) and isinstance(v, int)}
        return {
            "pipeline_id": spec.pipeline_id,
            "current_step_index": int(state.get("current_step_index", len(completed_ordered))),
            "completed_steps": completed_ordered,
            "step_run_ids": step_run_ids,
            "attempts": dict(sorted(attempts.items(), key=lambda kv: kv[0])),
        }

    def _assert_state_consistent(self, spec: PipelineSpec, state: Dict[str, Any]) -> None:
        completed = state.get("completed_steps", [])
        step_run_ids = state.get("step_run_ids", {})
        if not isinstance(completed, list) or not isinstance(step_run_ids, dict):
            raise ValueError("pipeline state is invalid")

        # Fail closed: a completed step must have a recorded run_id, and that run must be verifiable.
        missing_run_ids = [sid for sid in completed if sid not in step_run_ids]
        if missing_run_ids:
            raise ValueError(f"inconsistent pipeline state: completed steps missing run_ids: {missing_run_ids}")

        for step_id in completed:
            run_id = step_run_ids.get(step_id)
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"inconsistent pipeline state: invalid run_id for step {step_id!r}")
            run_dir = self.project_root / "CONTRACTS" / "_runs" / run_id
            required = ["PROOF.json", "DOMAIN_ROOTS.json", "LEDGER.jsonl"]
            missing = [name for name in required if not (run_dir / name).exists()]
            if missing:
                raise ValueError(f"inconsistent pipeline state: run {run_id} missing artifacts: {missing}")

    def _make_run_id(self, pipeline_id: str, step_id: str, attempt: int) -> str:
        return _slug(f"pipeline-{pipeline_id}-{step_id}-a{attempt}")

    def _load_jobspec(self, jobspec_path: str) -> Dict[str, Any]:
        rel = jobspec_path.replace("\\", "/")
        if Path(rel).is_absolute():
            raise ValueError("jobspec_path must be repo-relative")
        abs_path = self.project_root / rel
        obj = json.loads(abs_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("jobspec must be an object")
        preflight = PreflightValidator(self.jobspec_schema_path)
        valid, errors = preflight.validate(obj, self.project_root)
        if not valid:
            first = errors[0] if errors else {"code": "JOBSPEC_INVALID", "message": "invalid"}
            raise ValueError(f"jobspec invalid: {first.get('code')}: {first.get('message')}")
        return obj

    def _execute_step(self, *, spec: PipelineSpec, step: StepSpec, run_id: str) -> None:
        jobspec = self._load_jobspec(step.jobspec_path)
        catalytic_domains = jobspec.get("catalytic_domains", [])
        durable_paths = jobspec.get("outputs", {}).get("durable_paths", [])
        intent = jobspec.get("intent", "")
        job_id = jobspec.get("job_id", run_id)
        if not isinstance(catalytic_domains, list) or not all(isinstance(x, str) and x for x in catalytic_domains):
            raise ValueError("jobspec catalytic_domains must be list[str]")
        if not isinstance(durable_paths, list) or not all(isinstance(x, str) and x for x in durable_paths):
            raise ValueError("jobspec outputs.durable_paths must be list[str]")
        if not isinstance(intent, str) or not intent:
            raise ValueError("jobspec intent must be non-empty string")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("jobspec job_id must be non-empty string")

        cmd = [
            "python3",
            str(self.project_root / "TOOLS" / "catalytic_runtime.py"),
            "--run-id",
            run_id,
            "--job-id",
            job_id,
            "--catalytic-domains",
            *catalytic_domains,
            "--durable-outputs",
            *durable_paths,
            "--intent",
            intent,
            "--timestamp",
            spec.timestamp,
            "--validator-semver",
            spec.validator_semver,
            "--validator-build-id",
            spec.validator_build_id,
        ]
        if not step.memoize:
            cmd.append("--no-memoize")
        if not step.strict:
            cmd.append("--non-strict")
        cmd.append("--")
        cmd.extend(step.cmd)

        res = subprocess.run(cmd, cwd=str(self.project_root))
        if res.returncode != 0:
            raise RuntimeError(f"step {step.step_id} failed: rc={res.returncode}")

    def _assert_step_outputs(self, run_dir: Path) -> None:
        required = ["PROOF.json", "DOMAIN_ROOTS.json", "LEDGER.jsonl"]
        missing = [f for f in required if not (run_dir / f).exists()]
        if missing:
            raise RuntimeError(f"step run is missing required artifacts: {missing}")
