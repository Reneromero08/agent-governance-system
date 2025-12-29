#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Suppress jsonschema/RefResolver deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PIPELINES.pipeline_runtime import _slug  # type: ignore
from PRIMITIVES.cas_store import normalize_relpath  # type: ignore
from PRIMITIVES.restore_proof import canonical_json_bytes  # type: ignore
from PRIMITIVES.preflight import PreflightValidator  # type: ignore
from PRIMITIVES.registry_validators import (
    validate_capabilities_registry,
    validate_capability_pins,
    validate_capability_revokes,
)  # type: ignore
from PRIMITIVES.skills import SkillRegistry  # type: ignore

from jsonschema import Draft7Validator, RefResolver

from TOOLS.utilities.intent import generate_intent


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def _atomic_write_canon_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(obj))

def _load_json_output(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _pipeline_dir(pipeline_id: str) -> Path:
    slugged = _slug(pipeline_id)
    path = Path("CONTRACTS") / "_runs" / "_pipelines" / slugged
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _write_policy(pipeline_id: str, policy: Dict[str, Any]) -> None:
    pdir = _pipeline_dir(pipeline_id)
    pdir.mkdir(parents=True, exist_ok=True)
    policy_path = pdir / "POLICY.json"
    _atomic_write_bytes(policy_path, canonical_json_bytes(policy))


def _write_idempotent(path: Path, data: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"REFUSE_OVERWRITE: {path} differs")
        return
    _atomic_write_bytes(path, data)


def _as_repo_relpath(path: Path) -> str:
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    if Path(rel).is_absolute():
        raise ValueError("path must be repo-relative")
    return rel


MAX_PLAN_BYTES_DEFAULT = 262_144
MAX_STEPS = 64
MAX_JOBSPEC_BYTES = 65_536

# Global strict caps for bounded dereference (fail-closed).
GLOBAL_DEREF_MAX_BYTES = 65_536
GLOBAL_DEREF_MAX_MATCHES = 20
GLOBAL_DEREF_MAX_NODES = 2_000
GLOBAL_DEREF_MAX_DEPTH = 32

# Default capabilities registry path (repo-relative).
DEFAULT_CAPABILITIES_PATH = "CATALYTIC-DPT/CAPABILITIES.json"
DEFAULT_PINS_PATH = "CATALYTIC-DPT/CAPABILITY_PINS.json"
DEFAULT_REVOKES_PATH = "CATALYTIC-DPT/CAPABILITY_REVOKES.json"
DEFAULT_SKILLS_PATH = "CATALYTIC-DPT/SKILLS/registry.json"


def _read_bytes_bounded(path: Path, max_bytes: int) -> bytes:
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"plan exceeds max bytes: {len(data)} > {max_bytes}")
    return data


def _load_plan(path: Path, *, max_bytes: int = MAX_PLAN_BYTES_DEFAULT) -> Dict[str, Any]:
    raw = _read_bytes_bounded(path, max_bytes)
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("plan must not start with BOM")
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("plan must be a JSON object")
    return obj


def _reject_control_chars(s: str) -> None:
    for ch in s:
        o = ord(ch)
        if o < 32 or o == 127:
            raise ValueError("step_id contains control characters")


def _validate_jobspec_obj(jobspec: Dict[str, Any]) -> None:
    jobspec_bytes = canonical_json_bytes(jobspec)
    if len(jobspec_bytes) > MAX_JOBSPEC_BYTES:
        raise ValueError(f"jobspec exceeds max bytes: {len(jobspec_bytes)} > {MAX_JOBSPEC_BYTES}")
    schema_path = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "jobspec.schema.json"
    preflight = PreflightValidator(schema_path)
    valid, errors = preflight.validate(jobspec, REPO_ROOT)
    if not valid:
        first = errors[0] if errors else {"code": "JOBSPEC_INVALID", "message": "invalid"}
        raise ValueError(f"jobspec invalid: {first.get('code')}: {first.get('message')}")


def _load_ags_plan_schema() -> Draft7Validator:
    schemas_dir = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS"
    plan_path = (schemas_dir / "ags_plan.schema.json").resolve()
    adapter_path = (schemas_dir / "adapter.schema.json").resolve()
    jobspec_path = (schemas_dir / "jobspec.schema.json").resolve()

    plan_schema = json.loads(plan_path.read_text(encoding="utf-8"))
    adapter_schema = json.loads(adapter_path.read_text(encoding="utf-8"))
    jobspec_schema = json.loads(jobspec_path.read_text(encoding="utf-8"))

    store: dict[str, Any] = {
        plan_schema.get("$id", "ags_plan.schema.json"): plan_schema,
        adapter_schema.get("$id", "adapter.schema.json"): adapter_schema,
        jobspec_schema.get("$id", "jobspec.schema.json"): jobspec_schema,
        "ags_plan.schema.json": plan_schema,
        "ags_plan.schema.json#": plan_schema,
        "adapter.schema.json": adapter_schema,
        "adapter.schema.json#": adapter_schema,
        "jobspec.schema.json": jobspec_schema,
        "jobspec.schema.json#": jobspec_schema,
        plan_path.as_uri(): plan_schema,
        plan_path.as_uri() + "#": plan_schema,
        adapter_path.as_uri(): adapter_schema,
        adapter_path.as_uri() + "#": adapter_schema,
        jobspec_path.as_uri(): jobspec_schema,
        jobspec_path.as_uri() + "#": jobspec_schema,
    }

    resolver = RefResolver.from_schema(plan_schema, store=store)
    return Draft7Validator(plan_schema, resolver=resolver)


_AGS_PLAN_VALIDATOR = _load_ags_plan_schema()

def _load_adapter_schema() -> Draft7Validator:
    schemas_dir = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS"
    adapter_path = (schemas_dir / "adapter.schema.json").resolve()
    jobspec_path = (schemas_dir / "jobspec.schema.json").resolve()

    adapter_schema = json.loads(adapter_path.read_text(encoding="utf-8"))
    jobspec_schema = json.loads(jobspec_path.read_text(encoding="utf-8"))

    store: dict[str, Any] = {
        adapter_schema.get("$id", "adapter.schema.json"): adapter_schema,
        jobspec_schema.get("$id", "jobspec.schema.json"): jobspec_schema,
        "adapter.schema.json": adapter_schema,
        "adapter.schema.json#": adapter_schema,
        "jobspec.schema.json": jobspec_schema,
        "jobspec.schema.json#": jobspec_schema,
        adapter_path.as_uri(): adapter_schema,
        adapter_path.as_uri() + "#": adapter_schema,
        jobspec_path.as_uri(): jobspec_schema,
        jobspec_path.as_uri() + "#": jobspec_schema,
    }

    resolver = RefResolver.from_schema(adapter_schema, store=store)
    return Draft7Validator(adapter_schema, resolver=resolver)


_ADAPTER_VALIDATOR = _load_adapter_schema()


def _validate_plan_schema(obj: Dict[str, Any]) -> None:
    errors = sorted(_AGS_PLAN_VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise ValueError(f"plan schema invalid at {list(first.path)}: {first.message}")


def _parse_step_for_route(step: Any, idx: int) -> Tuple[str, Dict[str, Any], List[str], bool, bool]:
    if not isinstance(step, dict):
        raise ValueError(f"steps[{idx}] must be an object")
    step_id = step.get("step_id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise ValueError(f"steps[{idx}].step_id must be non-empty string")
    _reject_control_chars(step_id)
    if "adapter" in step:
        adapter = step.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError(f"steps[{idx}].adapter must be an object")
        cmd, jobspec = _validate_adapter(adapter, strict=True)
        return step_id, jobspec, cmd, True, True

    jobspec = step.get("jobspec")
    if not isinstance(jobspec, dict):
        raise ValueError(f"steps[{idx}].jobspec must be an object")
    _validate_jobspec_obj(jobspec)
    cmd = step.get("command", step.get("cmd"))
    if cmd is None:
        raise ValueError("MISSING_STEP_COMMAND")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        raise ValueError(f"steps[{idx}].command must be a non-empty list[str]")
    strict = step.get("strict", True)
    memoize = step.get("memoize", True)
    if not isinstance(strict, bool):
        raise ValueError(f"steps[{idx}].strict must be boolean")
    if not isinstance(memoize, bool):
        raise ValueError(f"steps[{idx}].memoize must be boolean")
    return step_id, jobspec, cmd, strict, memoize


def _validate_adapter(adapter: Dict[str, Any], *, strict: bool) -> Tuple[List[str], Dict[str, Any]]:
    errors = sorted(_ADAPTER_VALIDATOR.iter_errors(adapter), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise ValueError(f"adapter schema invalid at {list(first.path)}: {first.message}")

    side_effects = adapter.get("side_effects", {})
    if strict:
        if any(side_effects.get(k) is True for k in ["network", "clock", "filesystem_unbounded", "nondeterministic"]):
            raise ValueError("ADAPTER_SIDE_EFFECTS_FORBIDDEN")

    # Path normalization: keys must already be normalized repo-relative paths.
    inputs = adapter.get("inputs", {})
    outputs = adapter.get("outputs", {})
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("adapter inputs/outputs must be objects")

    def _check_paths(mapping: Dict[str, Any]) -> None:
        for raw in mapping.keys():
            if not isinstance(raw, str) or not raw:
                raise ValueError("adapter path keys must be non-empty strings")
            try:
                norm = normalize_relpath(raw)
            except Exception:
                raise ValueError("NON_NORMALIZED_PATH")
            if norm != raw or norm == ".":
                raise ValueError("NON_NORMALIZED_PATH")

    _check_paths(inputs)
    _check_paths(outputs)

    overlap = set(inputs.keys()) & set(outputs.keys())
    if overlap:
        raise ValueError("INPUT_OUTPUT_OVERLAP")

    caps = adapter.get("deref_caps", {})
    if not isinstance(caps, dict):
        raise ValueError("deref_caps must be object")
    if int(caps.get("max_bytes", -1)) > GLOBAL_DEREF_MAX_BYTES:
        raise ValueError("DEREF_CAPS_TOO_LARGE")
    if int(caps.get("max_matches", -1)) > GLOBAL_DEREF_MAX_MATCHES:
        raise ValueError("DEREF_CAPS_TOO_LARGE")
    if int(caps.get("max_nodes", -1)) > GLOBAL_DEREF_MAX_NODES:
        raise ValueError("DEREF_CAPS_TOO_LARGE")
    if int(caps.get("max_depth", -1)) > GLOBAL_DEREF_MAX_DEPTH:
        raise ValueError("DEREF_CAPS_TOO_LARGE")

    cmd = adapter.get("command")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        raise ValueError("MISSING_COMMAND")

    jobspec = adapter.get("jobspec")
    if not isinstance(jobspec, dict):
        raise ValueError("adapter jobspec must be object")
    _validate_jobspec_obj(jobspec)

    artifacts = adapter.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("MISSING_ARTIFACTS")

    return cmd, jobspec


def _capabilities_path() -> Path:
    env = os.environ.get("CATALYTIC_CAPABILITIES_PATH")
    rel = env if isinstance(env, str) and env else DEFAULT_CAPABILITIES_PATH
    p = Path(rel)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _pins_path() -> Path:
    env = os.environ.get("CATALYTIC_PINS_PATH")
    rel = env if isinstance(env, str) and env else DEFAULT_PINS_PATH
    p = Path(rel)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _revokes_path() -> Path:
    env = os.environ.get("CATALYTIC_REVOKES_PATH")
    rel = env if isinstance(env, str) and env else DEFAULT_REVOKES_PATH
    p = Path(rel)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _skills_path() -> Path:
    env = os.environ.get("CATALYTIC_SKILLS_PATH")
    rel = env if isinstance(env, str) and env else DEFAULT_SKILLS_PATH
    p = Path(rel)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _load_capabilities_registry() -> Dict[str, Any]:
    path = _capabilities_path()
    v = validate_capabilities_registry(path)
    if not v.ok:
        if v.code == "REGISTRY_TAMPERED":
            raise ValueError("CAPABILITY_HASH_MISMATCH")
        raise ValueError(v.code)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("CAPABILITIES_REGISTRY_INVALID")
    if obj.get("registry_version") != "1.0.0":
        raise ValueError("CAPABILITIES_REGISTRY_INVALID_VERSION")
    caps = obj.get("capabilities")
    if not isinstance(caps, dict):
        raise ValueError("CAPABILITIES_REGISTRY_INVALID")
    return obj


def _load_pins() -> Dict[str, Any]:
    path = _pins_path()
    v = validate_capability_pins(path)
    if not v.ok:
        raise ValueError(v.code)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("PINS_INVALID")
    if obj.get("pins_version") != "1.0.0":
        raise ValueError("PINS_INVALID_VERSION")
    allowed = obj.get("allowed_capabilities")
    if not isinstance(allowed, list) or not all(isinstance(x, str) and x for x in allowed):
        raise ValueError("PINS_INVALID")
    return obj


def _load_revokes() -> Dict[str, Any]:
    path = _revokes_path()
    if not path.exists():
        return {"revokes_version": "1.0.0", "revoked_capabilities": []}
    v = validate_capability_revokes(path)
    if not v.ok:
        raise ValueError(v.code)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("REVOKES_INVALID")
    if obj.get("revokes_version") != "1.0.0":
        raise ValueError("REVOKES_INVALID_VERSION")
    revoked = obj.get("revoked_capabilities")
    if not isinstance(revoked, list) or not all(isinstance(x, str) and x for x in revoked):
        raise ValueError("REVOKES_INVALID")
    return obj


def _is_capability_revoked(capability_hash: str) -> bool:
    revokes = _load_revokes()
    revoked = revokes.get("revoked_capabilities", [])
    return capability_hash in revoked


def _is_capability_pinned(capability_hash: str) -> bool:
    pins = _load_pins()
    allowed = pins.get("allowed_capabilities", [])
    return capability_hash in allowed


def _canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def _resolve_capability_hash(capability_hash: str) -> Dict[str, Any]:
    registry = _load_capabilities_registry()
    caps = registry.get("capabilities", {})
    entry = caps.get(capability_hash)
    if not isinstance(entry, dict):
        raise ValueError("UNKNOWN_CAPABILITY")
    if _is_capability_revoked(capability_hash):
        raise ValueError("REVOKED_CAPABILITY")
    if not _is_capability_pinned(capability_hash):
        raise ValueError("CAPABILITY_NOT_PINNED")
    adapter = entry.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("CAPABILITIES_REGISTRY_INVALID")
    computed = _canonical_hash(adapter)
    if computed != capability_hash:
        raise ValueError("CAPABILITY_HASH_MISMATCH")
    spec_hash = entry.get("adapter_spec_hash")
    if not isinstance(spec_hash, str) or spec_hash != computed:
        raise ValueError("CAPABILITY_HASH_MISMATCH")
    return adapter


def _resolve_skill_to_hash(skill_id: str) -> str:
    path = _skills_path()
    registry = SkillRegistry.load(path)
    skill = registry.resolve(skill_id)
    return skill.capability_hash


def _validate_and_extract_steps_for_route(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "plan_version" in plan:
        _validate_plan_schema(plan)
        steps = plan.get("steps")
        assert isinstance(steps, list)
        if len(steps) > MAX_STEPS:
            raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")
        return steps

    # Legacy Phase 6.1 plan format (no plan_version). Keep for backward compatibility.
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan.steps must be a non-empty list")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")
    return steps


def ags_route(*, plan_path: Path, pipeline_id: str, runs_root: str) -> int:
    if runs_root != "CONTRACTS/_runs":
        raise RuntimeError("UNSUPPORTED_RUNS_ROOT (only CONTRACTS/_runs is supported)")

    plan = _load_plan(plan_path, max_bytes=MAX_PLAN_BYTES_DEFAULT)
    steps = _validate_and_extract_steps_for_route(plan)

    seen: set[str] = set()
    pipeline_dir = REPO_ROOT / runs_root / "_pipelines" / _slug(pipeline_id)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    pipeline_steps: List[Dict[str, Any]] = []
    for idx, raw in enumerate(steps):
        if isinstance(raw, dict) and "skill_id" in raw and "capability_hash" not in raw and "adapter" not in raw:
            sid = raw.get("skill_id")
            if not isinstance(sid, str) or not sid:
                raise ValueError("MISSING_SKILL_ID")
            # Resolve skill_id to capability_hash immediately
            # Note: We do not modify 'raw' in place to avoid side effects if reused, 
            # effectively treating skill_id as a pointer to a capability_hash.
            cap_hash = _resolve_skill_to_hash(sid)
            # Proceed to capability_hash branch logic by synthesizing a check
            # We copy raw logic below but specialized for this case to avoid messing with types
            
            adapter = _resolve_capability_hash(cap_hash)
            cmd, jobspec = _validate_adapter(adapter, strict=True)
            step_id = raw.get("step_id")
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError(f"steps[{idx}].step_id must be non-empty string")
            _reject_control_chars(step_id)
            strict = True
            memoize = True
            capability_hash = cap_hash

        elif isinstance(raw, dict) and "capability_hash" in raw:
            cap = raw.get("capability_hash")
            if not isinstance(cap, str) or not cap:
                raise ValueError("MISSING_CAPABILITY_HASH")
            adapter = _resolve_capability_hash(cap)
            cmd, jobspec = _validate_adapter(adapter, strict=True)
            step_id = raw.get("step_id")
            if not isinstance(step_id, str) or not step_id.strip():
                raise ValueError(f"steps[{idx}].step_id must be non-empty string")
            _reject_control_chars(step_id)
            strict = True
            memoize = True
            capability_hash = cap
        else:
            step_id, jobspec, cmd, strict, memoize = _parse_step_for_route(raw, idx)
            capability_hash = None

        if step_id in seen:
            raise ValueError(f"duplicate step_id: {step_id}")
        seen.add(step_id)

        step_dir = pipeline_dir / "steps" / step_id
        jobspec_path = step_dir / "JOBSPEC.json"
        jobspec_bytes = canonical_json_bytes(jobspec)
        _write_idempotent(jobspec_path, jobspec_bytes)

        pipeline_steps.append(
            {
                "step_id": step_id,
                "jobspec_path": _as_repo_relpath(jobspec_path),
                "cmd": cmd,
                "strict": strict,
                "memoize": memoize,
                **({"capability_hash": capability_hash} if isinstance(capability_hash, str) else {}),
            }
        )

    pipeline_spec = {"pipeline_id": pipeline_id, "steps": pipeline_steps}
    spec_bytes = canonical_json_bytes(pipeline_spec)
    _write_idempotent(pipeline_dir / "PIPELINE.json", spec_bytes)

    sys.stdout.write(f"OK wrote {runs_root}/_pipelines/{_slug(pipeline_id)}/PIPELINE.json\n")
    return 0


def ags_plan(
    *,
    router: str,
    router_args: List[str],
    out_path: Path,
    pipeline_id: Optional[str],
    max_bytes: int,
) -> int:
    env = dict(os.environ)
    if pipeline_id is not None:
        env["AGS_PIPELINE_ID"] = pipeline_id

    # Resolve router path for hashing
    router_path = Path(router)
    if not router_path.is_absolute():
        router_path = REPO_ROOT / router_path
    
    # Hash the router executable
    router_hash = None
    if router_path.exists() and router_path.is_file():
        router_hash = hashlib.sha256(router_path.read_bytes()).hexdigest()

    proc = subprocess.run(
        [router, *router_args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
    )
    if proc.stderr not in (b"", None):
        raise RuntimeError("ROUTER_STDERR_NOT_EMPTY")
    if proc.returncode != 0:
        raise RuntimeError(f"ROUTER_EXIT_{proc.returncode}")

    out = proc.stdout or b""
    if len(out) > max_bytes:
        raise RuntimeError(f"ROUTER_OUTPUT_TOO_LARGE: {len(out)} > {max_bytes}")

    # Phase 8: Hash the raw router output (transcript hash)
    router_transcript_hash = hashlib.sha256(out).hexdigest()

    text = out.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("router output must not start with BOM")
    plan_obj = json.loads(text)
    if not isinstance(plan_obj, dict):
        raise ValueError("router output must be a JSON object")

    # Build router receipt
    router_receipt = {
        "router_executable": str(router),
        "router_args": list(router_args),
        "router_hash": router_hash,
        "max_bytes": max_bytes,
        "transcript_hash": router_transcript_hash,
    }
    # Embed in plan for downstream verification/logging
    plan_obj["router"] = router_receipt

    _validate_plan_schema(plan_obj)
    steps = plan_obj.get("steps")
    assert isinstance(steps, list)
    if len(steps) > MAX_STEPS:
        raise ValueError(f"steps exceeds max: {len(steps)} > {MAX_STEPS}")

    seen: set[str] = set()
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{idx}] must be an object")
        sid = step.get("step_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"steps[{idx}].step_id must be non-empty string")
        _reject_control_chars(sid)
        if sid in seen:
            raise ValueError(f"duplicate step_id: {sid}")
        seen.add(sid)
        if "capability_hash" in step:
            cap = step.get("capability_hash")
            if not isinstance(cap, str) or not cap:
                raise ValueError("MISSING_CAPABILITY_HASH")
            _resolve_capability_hash(cap)
            continue
        if "adapter" in step:
            adapter = step.get("adapter")
            if not isinstance(adapter, dict):
                raise ValueError(f"steps[{idx}].adapter must be an object")
            _validate_adapter(adapter, strict=True)
            continue

        cmd = step.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
            raise ValueError("MISSING_STEP_COMMAND")
        jobspec = step.get("jobspec")
        if not isinstance(jobspec, dict):
            raise ValueError(f"steps[{idx}].jobspec must be an object")
        _validate_jobspec_obj(jobspec)

    if pipeline_id is not None:
        plan_obj["pipeline_id"] = pipeline_id
        
        # Persist router artifacts to pipeline directory
        pdir = _pipeline_dir(pipeline_id)
        pdir.mkdir(parents=True, exist_ok=True)
        _atomic_write_canon_json(pdir / "ROUTER.json", router_receipt)
        _atomic_write_canon_json(pdir / "ROUTER_OUTPUT.json", plan_obj)
        # Store raw bytes for the transcript hash
        (pdir / "ROUTER_TRANSCRIPT_HASH").write_text(router_transcript_hash, encoding="utf-8")

    out_bytes = canonical_json_bytes(plan_obj)
    if len(out_bytes) > MAX_PLAN_BYTES_DEFAULT:
        raise RuntimeError(f"PLAN_BYTES_TOO_LARGE_AFTER_CANON: {len(out_bytes)} > {MAX_PLAN_BYTES_DEFAULT}")

    # Phase 8: Write router receipt artifacts
    receipt_dir = out_path.parent / f".router_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    
    # ROUTER.json - what router ran
    router_receipt = {
        "router_executable": str(router_path.relative_to(REPO_ROOT)) if router_path.is_relative_to(REPO_ROOT) else str(router_path),
        "router_args": router_args,
        "router_hash_sha256": router_hash,
        "router_exit_code": proc.returncode,
        "router_stderr_bytes": len(proc.stderr) if proc.stderr else 0,
    }
    router_receipt_path = receipt_dir / f"{out_path.stem}_ROUTER.json"
    _atomic_write_bytes(router_receipt_path, canonical_json_bytes(router_receipt))
    
    # ROUTER_OUTPUT.json - canonical plan output
    router_output_path = receipt_dir / f"{out_path.stem}_ROUTER_OUTPUT.json"
    _atomic_write_bytes(router_output_path, out_bytes)
    
    # ROUTER_TRANSCRIPT_HASH - hash of raw bytes
    transcript_hash_path = receipt_dir / f"{out_path.stem}_ROUTER_TRANSCRIPT_HASH"
    _atomic_write_bytes(transcript_hash_path, router_transcript_hash.encode("utf-8"))

    _write_idempotent(out_path, out_bytes)
    sys.stdout.write(f"OK wrote {out_path}\n")
    sys.stdout.write(f"OK wrote router receipts to {receipt_dir}\n")
    return 0


def ags_run(*, pipeline_id: str, runs_root: str, strict: bool, repo_write: bool, allow_repo_write: bool, allow_dirty: bool, skip_preflight: bool = False) -> int:
    if runs_root != "CONTRACTS/_runs":
        raise RuntimeError("UNSUPPORTED_RUNS_ROOT (only CONTRACTS/_runs is supported)")

    strict = True if strict else True

    catalytic = [sys.executable, str(REPO_ROOT / "TOOLS" / "catalytic.py")]

    run_cmd = catalytic + [
        "pipeline",
        "run",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        runs_root,
    ]
    verify_cmd = catalytic + [
        "pipeline",
        "verify",
        "--pipeline-id",
        pipeline_id,
        "--runs-root",
        runs_root,
        "--strict",
    ]

    preflight_data = {}
    if not skip_preflight:
        preflight_cmd = [sys.executable, str(REPO_ROOT / "TOOLS" / "preflight.py"), "--json"]
        if allow_dirty:
            preflight_cmd.append("--allow-dirty-tracked")
        preflight_res = subprocess.run(preflight_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if preflight_res.returncode != 0:
            sys.stdout.write(f"FAIL preflight rc={preflight_res.returncode}\n")
            sys.stdout.write(preflight_res.stdout + preflight_res.stderr)
            return preflight_res.returncode
        preflight_data = _load_json_output(preflight_res.stdout)
    else:
        # Dummy preflight data when skipped
        preflight_data = {
            "verdict": "SKIPPED",
            "canon_sha256": "0" * 64,
            "cortex_sha256": "0" * 64,
            "git_head_sha": "0" * 64,
            "cortex_generated_at": "2025-01-01T00:00:00Z"
        }

    intent_mode = "repo-write" if repo_write else "artifact-only"
    intent_path, intent_data = generate_intent(
        pipeline_id,
        runs_root,
        mode=intent_mode,
        allow_repo_write=allow_repo_write,
    )
    admit_cmd = [sys.executable, str(REPO_ROOT / "TOOLS" / "admission.py"), "--intent", str(intent_path)]
    admit_res = subprocess.run(admit_cmd, cwd=str(REPO_ROOT))
    if admit_res.returncode != 0:
        sys.stdout.write(f"FAIL admission rc={admit_res.returncode}\n")
        return admit_res.returncode

    # preflight_data already loaded or dummy-fied above
    admission_data = _load_json_output(admit_res.stdout)
    intent_sha256 = hashlib.sha256(Path(intent_path).read_bytes()).hexdigest()
    
    revokes = _load_revokes()
    revoked_list = sorted(set(revokes.get("revoked_capabilities", [])))

    policy = {
        "policy_version": "1.0.0",
        "revoked_capabilities": revoked_list,
        "preflight": {
            "verdict": preflight_data.get("verdict"),
            "canon_sha256": preflight_data.get("canon_sha256"),
            "cortex_sha256": preflight_data.get("cortex_sha256"),
            "git_head_sha": preflight_data.get("git_head_sha"),
            "generated_at": preflight_data.get("cortex_generated_at"),
        },
        "admission": {
            "verdict": admission_data.get("verdict"),
            "intent_sha256": intent_sha256,
            "mode": intent_data.get("mode"),
            "reasons": admission_data.get("reasons", []),
        },
    }
    _write_policy(pipeline_id, policy)

    sys.stdout.write("AGS: running pipeline\n")
    res = subprocess.run(run_cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        sys.stdout.write(f"FAIL pipeline run rc={res.returncode}\n")
        return res.returncode

    sys.stdout.write("AGS: verifying pipeline (fail-closed)\n")
    res2 = subprocess.run(verify_cmd, cwd=str(REPO_ROOT))
    if res2.returncode != 0:
        sys.stdout.write(f"FAIL pipeline verify rc={res2.returncode}\n")
        return res2.returncode

    sys.stdout.write("OK\n")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ags", description="AGS CLI (model-free bridge)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="Run router to produce a validated plan JSON (fail-closed)")
    plan_p.add_argument("--router", required=True, help="Router executable path")
    plan_p.add_argument("--router-arg", action="append", default=[], help="Router argument (repeatable)")
    plan_p.add_argument("--out", required=True, help="Output plan.json path (repo-relative recommended)")
    plan_p.add_argument("--pipeline-id", default=None, help="Override pipeline_id")
    plan_p.add_argument("--max-bytes", type=int, default=MAX_PLAN_BYTES_DEFAULT, help="Max router stdout bytes")

    route_p = sub.add_parser("route", help="Emit deterministic PIPELINE.json from an explicit plan")
    route_p.add_argument("--plan", required=True, help="Path to JSON plan file")
    route_p.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    route_p.add_argument("--runs-root", default="CONTRACTS/_runs", help="Runs root (default: CONTRACTS/_runs)")

    run_p = sub.add_parser("run", help="Run and immediately verify a pipeline (fail-closed)")
    run_p.add_argument("--pipeline-id", required=True, help="Pipeline ID")
    run_p.add_argument("--runs-root", default="CONTRACTS/_runs", help="Runs root (default: CONTRACTS/_runs)")
    run_p.add_argument("--strict", action="store_true", help="Strict verification (always on)")
    run_p.add_argument("--repo-write", action="store_true", help="Derived mode indicates repo-write behavior")
    run_p.add_argument("--allow-repo-write", action="store_true", help="Allow repo writes when mode is repo-write")
    run_p.add_argument("--allow-dirty-tracked", action="store_true", help="Allow dirty tracked files in preflight")
    run_p.add_argument("--skip-preflight", action="store_true", help="Skip preflight check (dangerous)")

    preflight_p = sub.add_parser("preflight", help="Emit JSON preflight verdict (fail-closed)")
    preflight_p.add_argument("--strict", action="store_true", help="Treat untracked files as blocking")
    preflight_p.add_argument("--allow-dirty-tracked", action="store_true", help="Allow dirty tracked files (still reported)")
    preflight_p.add_argument("--json", action="store_true", help="Emit JSON (default)")

    admit_p = sub.add_parser("admit", help="Admission control (intent gate, JSON-only)")
    admit_p.add_argument("--intent", required=True, help="Path to intent.json")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "plan":
            return ags_plan(
                router=args.router,
                router_args=list(args.router_arg),
                out_path=Path(args.out),
                pipeline_id=args.pipeline_id,
                max_bytes=int(args.max_bytes),
            )
        if args.cmd == "route":
            return ags_route(
                plan_path=Path(args.plan),
                pipeline_id=args.pipeline_id,
                runs_root=args.runs_root,
            )
        if args.cmd == "run":
            return ags_run(
                pipeline_id=args.pipeline_id,
                runs_root=args.runs_root,
                strict=args.strict,
                repo_write=args.repo_write,
                allow_repo_write=args.allow_repo_write,
                allow_dirty=args.allow_dirty_tracked,
                skip_preflight=args.skip_preflight,
            )
        if args.cmd == "preflight":
            preflight = [sys.executable, str(REPO_ROOT / "TOOLS" / "preflight.py")]
            forwarded: List[str] = []
            if bool(args.strict):
                forwarded.append("--strict")
            if bool(args.allow_dirty_tracked):
                forwarded.append("--allow-dirty-tracked")
            if bool(args.json):
                forwarded.append("--json")
            res = subprocess.run(preflight + forwarded, cwd=str(REPO_ROOT))
            return int(res.returncode)
        if args.cmd == "admit":
            admit = [sys.executable, str(REPO_ROOT / "TOOLS" / "admission.py"), "--intent", str(args.intent)]
            res = subprocess.run(admit, cwd=str(REPO_ROOT))
            return int(res.returncode)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    sys.stderr.write("ERROR: unsupported command\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
