#!/usr/bin/env python3
"""
Cortex Toolkit - Unified CORTEX operations skill.

Consolidates: cortex-build, cas-integrity-check, system1-verify,
              cortex-summaries, llm-packer-smoke

Operations:
  - build: Rebuild CORTEX index and SECTION_INDEX
  - verify_cas: Check CAS directory integrity
  - verify_system1: Ensure system1.db is in sync
  - summarize: Generate deterministic section summaries
  - smoke_test: Run LLM Packer smoke tests
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

# Constants
DEFAULT_BUILD_SCRIPT = "NAVIGATION/CORTEX/db/cortex.build.py"
DEFAULT_SECTION_INDEX = "NAVIGATION/CORTEX/_generated/SECTION_INDEX.json"
DB_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
PACKER_MODULE = "MEMORY.LLM_PACKER.Engine.packer"
PACKS_ROOT = PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"
RUNS_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"

# Directories to exclude from system1 verification
EXCLUDE_DIRS = [
    "_generated", "__pycache__", ".git", "node_modules",
    "BUILD", "MEMORY", ".pytest_cache", "meta", "LAW/CONTRACTS/_runs",
]

DURABLE_ROOTS = [
    "LAW/CONTRACTS/_runs",
    "NAVIGATION/CORTEX/_generated",
    "MEMORY/LLM_PACKER/_packs",
    "BUILD",
    "CAPABILITY/SKILLS",
]


def get_writer() -> GuardedWriter:
    """Get a configured GuardedWriter instance."""
    if not GuardedWriter:
        raise RuntimeError("GuardedWriter not available")
    writer = GuardedWriter(project_root=PROJECT_ROOT, durable_roots=DURABLE_ROOTS)
    writer.open_commit_gate()
    return writer


def write_output(output_path: Path, data: Dict[str, Any], writer: GuardedWriter) -> None:
    """Write JSON output using GuardedWriter."""
    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(data, indent=2, sort_keys=True) + "\n")


# ============================================================================
# Operation: build
# ============================================================================

def _git_head_timestamp(project_root: Path) -> Optional[str]:
    """Get git HEAD commit timestamp."""
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    value = res.stdout.strip()
    return value or None


def op_build(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Rebuild CORTEX index and verify expected paths."""
    expected_paths = payload.get("expected_paths") or []
    timeout_sec = int(payload.get("timeout_sec", 120))
    build_script = Path(payload.get("build_script") or DEFAULT_BUILD_SCRIPT)
    section_index = Path(payload.get("section_index_path") or DEFAULT_SECTION_INDEX)

    errors: List[str] = []
    missing_paths: List[str] = []

    build_path = PROJECT_ROOT / build_script
    if not build_path.exists():
        errors.append(f"build_script_not_found: {build_script.as_posix()}")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    head_ts = _git_head_timestamp(PROJECT_ROOT)
    if head_ts:
        env["CORTEX_BUILD_TIMESTAMP"] = head_ts

    returncode = 1
    if not errors:
        try:
            result = subprocess.run(
                [sys.executable, str(build_path)],
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            returncode = result.returncode
            if result.returncode != 0:
                stderr = (result.stdout or "") + (result.stderr or "")
                errors.append("build_failed")
                if stderr.strip():
                    errors.append(stderr.strip())
        except subprocess.TimeoutExpired:
            errors.append("build_timeout")
        except OSError as exc:
            errors.append(f"build_os_error: {exc}")

    index_path = PROJECT_ROOT / section_index
    if not errors and not index_path.exists():
        errors.append(f"section_index_missing: {section_index.as_posix()}")

    if not errors:
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            indexed_paths = {str(entry.get("path", "")) for entry in entries}
            for expected in expected_paths:
                if expected not in indexed_paths:
                    missing_paths.append(expected)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"section_index_read_failed: {exc}")

    ok = not errors and not missing_paths and returncode == 0
    output = {
        "ok": ok,
        "returncode": returncode,
        "section_index_path": section_index.as_posix(),
        "missing_paths": missing_paths,
        "errors": errors,
    }
    write_output(output_path, output, writer)
    return 0 if ok else 1


# ============================================================================
# Operation: verify_cas
# ============================================================================

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def op_verify_cas(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Verify CAS directory integrity."""
    cas_root_str = payload.get("cas_root")
    if not isinstance(cas_root_str, str) or not cas_root_str.strip():
        write_output(output_path, {"status": "failure", "error": "MISSING_CAS_ROOT"}, writer)
        return 0

    cas_root = Path(cas_root_str)
    if not cas_root.is_absolute():
        cas_root = (PROJECT_ROOT / cas_root).resolve()

    if not cas_root.exists():
        write_output(output_path, {
            "status": "failure",
            "error": f"CAS_ROOT_NOT_FOUND: {cas_root}",
            "cas_root": str(cas_root)
        }, writer)
        return 0

    corrupt_blobs = []
    total_blobs = 0

    for file_path in cas_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue

        total_blobs += 1
        expected_hash = file_path.name
        if len(expected_hash) != 64:
            corrupt_blobs.append({"path": str(file_path), "reason": "invalid_filename_format"})
            continue

        try:
            actual_hash = _sha256_file(file_path)
        except Exception as exc:
            corrupt_blobs.append({"path": str(file_path), "reason": f"read_error: {exc}"})
            continue

        if actual_hash != expected_hash:
            corrupt_blobs.append({
                "path": str(file_path),
                "expected": expected_hash,
                "actual": actual_hash,
                "reason": "hash_mismatch"
            })

    status = "success" if not corrupt_blobs else "failure"
    result = {
        "status": status,
        "total_blobs": total_blobs,
        "corrupt_blobs": corrupt_blobs,
        "cas_root": str(cas_root)
    }
    write_output(output_path, result, writer)
    return 0


# ============================================================================
# Operation: verify_system1
# ============================================================================

def _should_verify(path: Path) -> bool:
    """Check if path should be verified in system1."""
    parts = path.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return False
    return True


def _get_tracked_files(root: Path) -> Optional[set]:
    """Return set of tracked files (absolute paths). Returns None if git fails."""
    try:
        cmd = ["git", "ls-files", "-z"]
        result = subprocess.run(cmd, cwd=root, capture_output=True)
        if result.returncode != 0:
            return None
        paths = set()
        for p in result.stdout.split(b'\0'):
            if p:
                paths.add(root / os.fsdecode(p))
        return paths
    except Exception:
        return None


def op_verify_system1(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Verify system1.db matches repository state."""
    if not DB_PATH.exists():
        write_output(output_path, {
            "success": False,
            "description": "system1.db does not exist"
        }, writer)
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    tracked_files = _get_tracked_files(PROJECT_ROOT)

    cursor = conn.execute("SELECT path, content_hash FROM files")
    db_files = {row['path']: row['content_hash'] for row in cursor}

    repo_files = {}
    for md_file in PROJECT_ROOT.rglob("*.md"):
        if not _should_verify(md_file):
            continue
        if tracked_files is not None and md_file.resolve() not in tracked_files:
            continue

        rel_path = md_file.relative_to(PROJECT_ROOT).as_posix()
        try:
            content = md_file.read_text(encoding='utf-8')
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            repo_files[rel_path] = content_hash
        except Exception:
            pass

    missing = set(repo_files.keys()) - set(db_files.keys())
    orphaned = set(db_files.keys()) - set(repo_files.keys())
    mismatches = [
        path for path in repo_files
        if path in db_files and db_files[path] != repo_files[path]
    ]

    conn.close()

    success = not missing and not mismatches
    description = f"Verified {len(repo_files)} files"
    if orphaned:
        description += f", {len(orphaned)} orphaned"
    if missing:
        description = f"{len(missing)} files not indexed"
    if mismatches:
        description = f"{len(mismatches)} files have changed"

    write_output(output_path, {
        "success": success,
        "description": description,
        "files_verified": len(repo_files),
        "missing": sorted(list(missing))[:20] if missing else [],
        "orphaned_count": len(orphaned),
        "mismatches": sorted(mismatches)[:20] if mismatches else []
    }, writer)
    return 0 if success else 1


# ============================================================================
# Operation: summarize
# ============================================================================

def _load_build_module() -> object:
    """Load cortex build module for summary functions."""
    build_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "cortex.build.py"

    if not build_path.exists():
        # Return a mock module with proper functionality
        import types
        module = types.ModuleType("cortex_build")
        module.build_cortex = lambda: {"entities": []}

        def _safe_section_id_filename(section_id):
            safe_id = re.sub(r'[^\w\-_.]', '_', section_id)
            safe_id = safe_id.replace("::", "_")
            hash_suffix = hashlib.sha256(section_id.encode()).hexdigest()[:8]
            return f"{safe_id}_{hash_suffix}"

        def _summarize_section(record, text):
            section_id = record.get("section_id", "")
            start_line = record.get("start_line", "")
            end_line = record.get("end_line", "")
            hash_val = record.get("hash", "")[:8] if record.get("hash") else ""

            source_header = f"source: {section_id}:{start_line}-{end_line}#{hash_val}"
            lines = text.strip().split('\n')
            processed_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped.startswith('#'):
                        content = stripped.lstrip('# ').strip()
                        processed_lines.append(f"- {content}")
                    elif stripped.startswith(('-', '*')):
                        processed_lines.append(stripped)
                    else:
                        processed_lines.append(f"- {stripped}")

            return "\n".join([source_header] + processed_lines)

        module._safe_section_id_filename = _safe_section_id_filename
        module._summarize_section = _summarize_section
        return module

    cortex_dir = str(PROJECT_ROOT / "NAVIGATION" / "CORTEX")
    if cortex_dir not in sys.path:
        sys.path.insert(0, cortex_dir)
    spec = importlib.util.spec_from_file_location("cortex_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def op_summarize(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Generate deterministic section summaries."""
    record = payload.get("record") or {}
    slice_text = str(payload.get("slice_text") or "")

    build = _load_build_module()
    safe_filename = build._safe_section_id_filename(str(record.get("section_id") or ""))
    summary_md = build._summarize_section(record, slice_text)
    summary_sha256 = hashlib.sha256(summary_md.encode("utf-8")).hexdigest()

    output = {
        "safe_filename": safe_filename,
        "summary_md": summary_md,
        "summary_sha256": summary_sha256,
    }
    write_output(output_path, output, writer)
    return 0


# ============================================================================
# Operation: smoke_test
# ============================================================================

def _resolve_out_dir(out_dir: str) -> Path:
    """Resolve output directory path."""
    path = Path(out_dir)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _ensure_under_packs(path: Path) -> None:
    """Ensure path is under PACKS_ROOT."""
    packs_root = PACKS_ROOT.resolve()
    try:
        path.resolve().relative_to(packs_root)
    except ValueError as exc:
        raise ValueError(f"out_dir must be under MEMORY/LLM_PACKER/_packs/: {path}") from exc


def _ensure_runner_writes_under_runs(path: Path) -> None:
    """Ensure path is under RUNS_ROOT."""
    runs_root = RUNS_ROOT.resolve()
    try:
        path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"runner output must be under LAW/CONTRACTS/_runs/: {path}") from exc


def op_smoke_test(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Run LLM Packer smoke tests."""
    out_dir_raw = str(payload.get("out_dir", "MEMORY/LLM_PACKER/_packs/_system/fixtures/fixture-smoke"))
    combined = bool(payload.get("combined", False))
    zip_enabled = bool(payload.get("zip", False))
    mode = str(payload.get("mode", "full"))
    profile = str(payload.get("profile", "full"))
    scope = str(payload.get("scope", "ags"))
    stamp = str(payload.get("stamp", "fixture-smoke"))
    split_lite = bool(payload.get("split_lite", False))
    allow_duplicate_hashes = payload.get("allow_duplicate_hashes", None)
    emit_pruned = bool(payload.get("emit_pruned", False))
    assert_archive_excluded = bool(payload.get("assert_archive_excluded", False))

    out_dir = _resolve_out_dir(out_dir_raw)
    _ensure_under_packs(out_dir)
    _ensure_runner_writes_under_runs(output_path)

    args = [
        sys.executable, "-u", "-m", PACKER_MODULE,
        "--scope", scope,
        "--mode", mode,
        "--profile", profile,
        "--out-dir", out_dir.relative_to(PROJECT_ROOT).as_posix(),
        "--skip-proofs",
    ]
    if stamp:
        args.extend(["--stamp", stamp])
    if zip_enabled:
        args.append("--zip")
    if combined:
        args.append("--combined")
    if split_lite:
        args.append("--split-lite")
    if allow_duplicate_hashes is True:
        args.append("--allow-duplicate-hashes")
    elif allow_duplicate_hashes is False:
        args.append("--disallow-duplicate-hashes")
    if not emit_pruned:
        args.append("--no-emit-pruned")

    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        return result.returncode

    required = [
        "meta/START_HERE.md", "meta/ENTRYPOINTS.md", "meta/FILE_TREE.txt",
        "meta/FILE_INDEX.json", "meta/REPO_OMITTED_BINARIES.json",
        "meta/REPO_STATE.json", "meta/PACK_INFO.json", "meta/BUILD_TREE.txt",
        "meta/PROVENANCE.json",
    ]
    if scope == "ags":
        required.extend([
            "SPLIT/AGS-00_INDEX.md", "SPLIT/AGS-01_LAW.md",
            "SPLIT/AGS-02_CAPABILITY.md", "SPLIT/AGS-03_NAVIGATION.md",
            "SPLIT/AGS-04_PROOFS.md", "SPLIT/AGS-05_MEMORY.md",
            "SPLIT/AGS-06_ROOT_FILES.md",
        ])
    elif scope == "lab":
        required.extend([
            "SPLIT/LAB-00_INDEX.md", "SPLIT/LAB-01_DOCS.md",
            "SPLIT/LAB-02_SYSTEM.md",
        ])
    else:
        print(f"Unknown scope in fixture: {scope}")
        return 1

    if split_lite or profile == "lite":
        if scope == "ags":
            required.extend(["LITE/AGS-00_INDEX.md", "LITE/PROOFS.json"])
        else:
            required.append("LITE/LAB-00_INDEX.md")

    if emit_pruned:
        pruned_dir = out_dir / "PRUNED"
        if pruned_dir.exists():
            required.extend([
                "PRUNED/PACK_MANIFEST_PRUNED.json",
                "PRUNED/meta/PRUNED_RULES.json",
            ])

    if combined:
        prefix = "AGS" if scope == "ags" else "LAB"
        required.extend([
            f"FULL/{prefix}-FULL-{stamp}.md",
            f"FULL/{prefix}-FULL-TREEMAP-{stamp}.md",
        ])

    missing = [p for p in required if not (out_dir / p).exists()]
    if missing:
        print("Packer output missing required files:")
        for p in missing:
            print(f"- {p}")
        return 1

    if not emit_pruned:
        pruned_dir = out_dir / "PRUNED"
        if pruned_dir.exists():
            print("PRUNED directory exists when emit_pruned is OFF")
            return 1

    archive_excluded = None
    if assert_archive_excluded and scope == "ags":
        repo_state_path = out_dir / "meta/REPO_STATE.json"
        try:
            repo_state = json.loads(repo_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error reading REPO_STATE.json for archive check: {exc}")
            return 1
        entries = repo_state.get("files", [])
        has_archive = any(e.get("path", "").startswith("MEMORY/ARCHIVE/") for e in entries)
        if has_archive:
            print("REPO_STATE.json includes MEMORY/ARCHIVE entries when they should be excluded")
            return 1
        archive_excluded = True

    output_payload = {
        "pack_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
        "stamp": stamp,
        "verified": required,
        "emit_pruned": emit_pruned,
    }
    if archive_excluded is not None:
        output_payload["archive_excluded"] = archive_excluded

    write_output(output_path, output_payload, writer)
    return 0


# ============================================================================
# Main dispatcher
# ============================================================================

OPERATIONS = {
    "build": op_build,
    "verify_cas": op_verify_cas,
    "verify_system1": op_verify_system1,
    "summarize": op_summarize,
    "smoke_test": op_smoke_test,
}


def main(input_path: Path, output_path: Path) -> int:
    """Main entry point."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    operation = payload.get("operation")
    if not operation:
        print("Error: 'operation' field is required")
        return 1

    if operation not in OPERATIONS:
        print(f"Error: Unknown operation '{operation}'. Valid: {', '.join(OPERATIONS.keys())}")
        return 1

    try:
        writer = get_writer()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    return OPERATIONS[operation](payload, output_path, writer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
