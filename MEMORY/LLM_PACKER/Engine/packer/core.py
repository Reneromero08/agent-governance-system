#!/usr/bin/env python3
"""
Core utilities for LLM Packer (Phase 1 modular implementation).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .proofs import refresh_proofs
from .firewall_writer import PackerWriter
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MEMORY_DIR = PROJECT_ROOT / "MEMORY"
LLM_PACKER_DIR = MEMORY_DIR / "LLM_PACKER"
PACKS_ROOT = LLM_PACKER_DIR / "_packs"
SYSTEM_DIR = PACKS_ROOT / "_system"
EXTERNAL_ARCHIVE_DIR = PACKS_ROOT / "_archive"
FIXTURE_PACKS_DIR = SYSTEM_DIR / "fixtures"
STATE_DIR = SYSTEM_DIR / "_state"
BASELINE_PATH = STATE_DIR / "baseline.json"

CANON_VERSION_FILE = PROJECT_ROOT / "LAW" / "CANON" / "VERSIONING.md"
GRAMMAR_VERSION = "1.0"
P2_MANIFEST_VERSION = "P2.0"

# Token estimation constants
CHARS_PER_TOKEN = 4
TOKEN_LIMIT_WARNING = 100_000
TOKEN_LIMIT_CRITICAL = 200_000

# ANSI Colors
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

TEXT_EXTENSIONS = {
    ".md", ".txt", ".json", ".py", ".js", ".mjs", ".cjs",
    ".css", ".html", ".php", ".ps1", ".cmd", ".bat", ".yml", ".yaml",
}

TEXT_BASENAMES = {".gitignore", ".gitattributes", ".editorconfig", ".htaccess", ".gitkeep", "LICENSE"}

def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()

def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _bucket_ordered(include_dirs: Sequence[str]) -> List[str]:
    # P.2 ordering rule
    preferred = ["LAW", "CAPABILITY", "NAVIGATION", "DIRECTION", "THOUGHT", "MEMORY", ".github"]
    present: set[str] = set()
    for d in include_dirs:
        try:
            present.add(Path(d).parts[-1])
        except Exception:
            continue
    return [b for b in preferred if b in present]

@contextmanager
def _override_cas_root(cas_root: Optional[Path]) -> Iterator[None]:
    if cas_root is None:
        yield
        return
    # Only used for deterministic tests; production uses default CAS root.
    from CAPABILITY.CAS import cas as cas_mod
    old = cas_mod._CAS_ROOT
    cas_mod._CAS_ROOT = Path(cas_root)
    try:
        yield
    finally:
        cas_mod._CAS_ROOT = old

def _git_repo_state(project_root: Path) -> Dict[str, str]:
    def run(args: List[str]) -> str:
        try:
            res = subprocess.run(args, cwd=str(project_root), capture_output=True, text=True, check=False)
            if res.returncode != 0:
                return ""
            return (res.stdout or "").strip()
        except Exception:
            return ""

    head = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    out: Dict[str, str] = {}
    if head:
        out["commit"] = head
    if branch and branch != "HEAD":
        out["branch"] = branch
    return out

def _validate_artifact_ref(ref: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
        raise ValueError(f"PACK_P2_INVALID_REF:{ref}")
    return ref

def _strip_sha256_prefix(ref: str) -> str:
    _validate_artifact_ref(ref)
    return ref.split(":", 1)[1]

def _write_run_roots(runs_dir: Path, *, roots: Sequence[str], writer: Optional[PackerWriter] = None) -> None:
    if writer is None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        roots_sorted = sorted(set(roots))
        # Strict format: 64 lowercase hex
        for h in roots_sorted:
            if not re.fullmatch(r"[0-9a-f]{64}", h):
                raise ValueError(f"PACK_P2_INVALID_ROOT_HASH:{h}")
        (runs_dir / "RUN_ROOTS.json").write_text(json.dumps(roots_sorted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        writer.mkdir(runs_dir, kind="durable", parents=True, exist_ok=True)
        roots_sorted = sorted(set(roots))
        # Strict format: 64 lowercase hex
        for h in roots_sorted:
            if not re.fullmatch(r"[0-9a-f]{64}", h):
                raise ValueError(f"PACK_P2_INVALID_ROOT_HASH:{h}")
        writer.write_text(runs_dir / "RUN_ROOTS.json", json.dumps(roots_sorted, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _emit_p2_lite_artifacts(
    pack_dir: Path,
    *,
    project_root: Path,
    include_paths: Sequence[str],
    scope: PackScope,
    runs_dir: Path,
    writer: Optional[PackerWriter] = None,
) -> Dict[str, str]:
    """
    P.2: CAS-addressed LITE manifest + run records + root-audit gating.

    Returns refs:
      - manifest_ref (sha256:...)
      - task_spec_ref (64hex)
      - output_hashes_ref (64hex)
      - status_ref (64hex)
    """
    from CAPABILITY.ARTIFACTS.store import store_file, store_bytes
    from CAPABILITY.RUNS.records import put_task_spec, put_output_hashes, put_status
    from CAPABILITY.AUDIT.root_audit import root_audit
    from CAPABILITY.CAS import cas as cas_mod

    lite_dir = pack_dir / "LITE"
    if writer is None:
        lite_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(lite_dir, kind="durable", parents=True, exist_ok=True)

    # Store each included payload into CAS and build deterministic entries.
    entries: List[Dict[str, Any]] = []
    payload_hashes: List[str] = []

    source_root = project_root / scope.source_root_rel
    for rel in sorted(set(include_paths)):
        src = source_root / rel
        if not src.exists() or not src.is_file():
            raise ValueError(f"PACK_P2_MISSING_SOURCE:{Path(rel).as_posix()}")
        ref = _validate_artifact_ref(store_file(str(src)))
        payload_hashes.append(_strip_sha256_prefix(ref))
        entries.append(
            {
                "path": Path(rel).as_posix(),
                "ref": ref,
                "bytes": int(src.stat().st_size),
                "ext": src.suffix.lower(),
                "kind": "FILE",
            }
        )

    entries.sort(key=lambda e: e["path"])
    # Dedup by path (fail-closed)
    seen: set[str] = set()
    for e in entries:
        if e["path"] in seen:
            raise ValueError(f"PACK_P2_DUPLICATE_MANIFEST_PATH:{e['path']}")
        seen.add(e["path"])

    manifest = {
        "version": P2_MANIFEST_VERSION,
        "scope": scope.key,
        "repo_state": _git_repo_state(project_root),
        "buckets": _bucket_ordered(scope.include_dirs),
        "entries": entries,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    if writer is None:
        (lite_dir / "PACK_MANIFEST.json").write_bytes(manifest_bytes)
    else:
        writer.write_bytes(lite_dir / "PACK_MANIFEST.json", manifest_bytes)

    manifest_ref = _validate_artifact_ref(store_bytes(manifest_bytes))
    manifest_hash = _strip_sha256_prefix(manifest_ref)

    task_spec = {
        "scope": scope.key,
        "repo_state": _git_repo_state(project_root),
        "include_dirs": list(scope.include_dirs),
        "excluded_dir_parts": sorted(scope.excluded_dir_parts),
        "modes": ["FULL", "SPLIT", "LITE"],
        "manifest_version": P2_MANIFEST_VERSION,
        "grammar_version": GRAMMAR_VERSION,
    }
    task_spec_ref = put_task_spec(task_spec)

    required_hashes = [manifest_hash, *payload_hashes]
    output_hashes_ref = put_output_hashes(required_hashes)

    # First roots pass (no status_ref yet)
    roots_phase1 = sorted(set([task_spec_ref, output_hashes_ref, manifest_hash, *payload_hashes]))
    _write_run_roots(runs_dir, roots=roots_phase1, writer=writer)

    audit1 = root_audit(output_hashes_record=output_hashes_ref, dry_run=True, runs_dir=runs_dir, cas_root=cas_mod._CAS_ROOT)
    if audit1.get("verdict") != "PASS":
        status_payload = {
            "state": "FAILED",
            "verdict": "FAIL",
            "task_spec_ref": task_spec_ref,
            "output_hashes_ref": output_hashes_ref,
            "manifest_ref": manifest_ref,
            # Deterministic snapshot for this pack run (not global CAS state)
            "cas_snapshot_hash": _sha256_hex("\n".join(sorted(set(roots_phase1))).encode("utf-8")),
        }
        status_ref = put_status(status_payload)
        _write_run_roots(runs_dir, roots=sorted(set([*roots_phase1, status_ref])), writer=writer)
        audit2 = root_audit(output_hashes_record=output_hashes_ref, dry_run=True, runs_dir=runs_dir, cas_root=cas_mod._CAS_ROOT)
        raise ValueError(f"PACK_P2_ROOT_AUDIT_FAIL:{audit1.get('errors', [])}:{audit2.get('errors', [])}")

    status_payload = {
        "state": "COMPLETED",
        "verdict": "PASS",
        "task_spec_ref": task_spec_ref,
        "output_hashes_ref": output_hashes_ref,
        "manifest_ref": manifest_ref,
        "cas_snapshot_hash": _sha256_hex("\n".join(sorted(set(roots_phase1))).encode("utf-8")),
    }
    status_ref = put_status(status_payload)
    roots_final = sorted(set([*roots_phase1, status_ref]))
    _write_run_roots(runs_dir, roots=roots_final, writer=writer)
    audit_final = root_audit(output_hashes_record=output_hashes_ref, dry_run=True, runs_dir=runs_dir, cas_root=cas_mod._CAS_ROOT)
    if audit_final.get("verdict") != "PASS":
        raise ValueError(f"PACK_P2_ROOT_AUDIT_FAIL_FINAL:{audit_final.get('errors', [])}")

    run_refs = {
        "manifest_ref": manifest_ref,
        "task_spec_ref": task_spec_ref,
        "output_hashes_ref": output_hashes_ref,
        "status_ref": status_ref,
    }
    if writer is None:
        (lite_dir / "RUN_REFS.json").write_bytes(_canonical_json_bytes(run_refs))
    else:
        writer.write_bytes(lite_dir / "RUN_REFS.json", _canonical_json_bytes(run_refs))
    return run_refs


@dataclass(frozen=True)
class PackLimits:
    max_total_bytes: int
    max_entry_bytes: int
    max_entries: int
    allow_duplicate_hashes: Optional[bool]


@dataclass(frozen=True)
class PackScope:
    key: str
    title: str
    file_prefix: str
    include_dirs: Tuple[str, ...]
    root_files: Tuple[str, ...]
    anchors: Tuple[str, ...]
    excluded_dir_parts: frozenset[str]
    source_root_rel: str = "."


SCOPE_AGS = PackScope(
    key="ags",
    title="Agent Governance System (AGS)",
    file_prefix="AGS",
    include_dirs=("LAW", "CAPABILITY", "NAVIGATION", "DIRECTION", "THOUGHT", "MEMORY", ".github"),
    root_files=("README.md", "LICENSE", "AGENTS.md", ".gitignore", ".gitattributes", ".editorconfig"),
    anchors=(
        "AGENTS.md",
        "README.md",
        rel_posix("LAW", "CANON", "CONTRACT.md"),
        rel_posix("LAW", "CANON", "INVARIANTS.md"),
        rel_posix("LAW", "CANON", "VERSIONING.md"),
        rel_posix("NAVIGATION", "MAPS", "ENTRYPOINTS.md"),
        rel_posix("LAW", "CONTRACTS", "runner.py"),
        "MEMORY/LLM_PACKER/README.md",
    ),
    excluded_dir_parts=frozenset({
        ".git", "BUILD", "_runs", "_generated", "_packs",
        "LAB",
        "Original", "ORIGINAL", "research", "RESEARCH", "__pycache__", "node_modules",
    }),
    source_root_rel=".",
)

SCOPE_LAB = PackScope(
    key="lab",
    title="LAB (THOUGHT/LAB)",
    file_prefix="LAB",
    include_dirs=("THOUGHT/LAB",),
    root_files=(),
    anchors=(),
    excluded_dir_parts=frozenset({
        ".git", "BUILD", "_runs", "_generated", "_packs", "__pycache__", "node_modules",
    }),
    source_root_rel="THOUGHT/LAB",
)

SCOPES: Dict[str, PackScope] = {
    SCOPE_AGS.key: SCOPE_AGS,
    SCOPE_LAB.key: SCOPE_LAB,
}


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    return len(text) // CHARS_PER_TOKEN


def estimate_file_tokens(path: Path) -> int:
    try:
        return estimate_tokens(read_text(path))
    except Exception:
        return 0


def read_canon_version() -> str:
    if not CANON_VERSION_FILE.exists():
        return "unknown"
    text = read_text(CANON_VERSION_FILE)
    match = re.search(r"canon_version:\s*(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else "unknown"


def is_text_path(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext:
        return ext in TEXT_EXTENSIONS
    return path.name in TEXT_BASENAMES


def is_excluded_rel_path(rel_path: Path, *, excluded_dir_parts: frozenset[str]) -> bool:
    parts = set(rel_path.parts)
    if parts & excluded_dir_parts:
        return True
    # P.2: Roots are runtime artifacts and must never be packed (avoid self-reference cycles).
    if rel_path.name in {"RUN_ROOTS.json", "GC_PINS.json"}:
        return True
    if any(part.startswith(".") and part != ".github" for part in rel_path.parts):
        return True
    return False


def iter_repo_candidates(project_root: Path, *, scope: PackScope) -> Iterable[Path]:
    for directory in scope.include_dirs:
        base = project_root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            try:
                is_file = path.is_file()
            except OSError:
                is_file = False
            if not is_file:
                continue
            rel = path.relative_to(project_root)
            if is_excluded_rel_path(rel, excluded_dir_parts=scope.excluded_dir_parts):
                continue
            yield path

    for file_name in scope.root_files:
        path = project_root / file_name
        if path.exists() and path.is_file():
            yield path


def build_state_manifest(project_root: Path, *, scope: PackScope) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    canon_version = read_canon_version()
    files: List[Dict[str, Any]] = []
    omitted: List[Dict[str, Any]] = []

    source_root = project_root / scope.source_root_rel

    seen: set[str] = set()
    for abs_path in iter_repo_candidates(project_root, scope=scope):
        try:
            rel = abs_path.relative_to(source_root).as_posix()
        except ValueError:
            # Fallback for root files not inside source_root (unlikely with deep source roots unless explicitly included)
            # For CAT/LAB, root_files is empty, so this mostly applies to AGS where source_root is .
             rel = abs_path.relative_to(project_root).as_posix()

        if rel in seen:
            raise RuntimeError(f"PACK_DEDUP_DUPLICATE_PATH:{rel}")
        seen.add(rel)

        if not is_text_path(abs_path):
            omitted.append({
                "scope": "repo",
                "repoRelPath": rel,
                "bytes": abs_path.stat().st_size,
            })
            continue

        files.append({
            "path": rel,
            "hash": hash_file(abs_path),
            "size": abs_path.stat().st_size,
        })

    files.sort(key=lambda e: (e["path"], e["hash"]))
    manifest: Dict[str, Any] = {
        "canon_version": canon_version,
        "grammar_version": GRAMMAR_VERSION,
        "scope": scope.key,
        "files": files,
    }
    return manifest, omitted


def manifest_digest(manifest: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for entry in manifest.get("files", []):
        line = f"{entry['hash']} {entry['size']} {entry['path']}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def baseline_path_for_scope(scope: PackScope) -> Path:
    if scope.key == SCOPE_AGS.key:
        return BASELINE_PATH
    return STATE_DIR / f"baseline-{scope.key}.json"


def load_baseline(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(read_text(path))
    except Exception:
        return None


def write_json(path: Path, payload: Any, writer: Optional[PackerWriter] = None) -> None:
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        writer.mkdir(path.parent, kind="durable", parents=True, exist_ok=True)
        writer.write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_under_packs_root(out_dir: Path) -> Path:
    packs_root = PACKS_ROOT.resolve()
    out_dir_resolved = out_dir.resolve()
    try:
        out_dir_resolved.relative_to(packs_root)
    except ValueError as exc:
        raise ValueError(f"OutDir must be under MEMORY/LLM_PACKER/_packs/. Received: {out_dir}") from exc
    return out_dir_resolved


def _is_valid_archive_zip(zip_path: Path) -> bool:
    if not zip_path.exists() or not zip_path.is_file():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Ensure the zip isn't corrupted.
            if zf.testzip() is not None:
                return False
            names = zf.namelist()
    except Exception:
        return False

    has_meta = any(n.startswith("meta/") for n in names)
    has_repo = any(n.startswith("repo/") for n in names)
    return has_meta and has_repo


def _is_valid_external_archive_zip(zip_path: Path, *, expected_pack_name: str) -> bool:
    if not zip_path.exists() or not zip_path.is_file():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if zf.testzip() is not None:
                return False
            names = zf.namelist()
    except Exception:
        return False

    # Must contain internal archive zip and at least one split file for safety.
    internal_zip = f"{expected_pack_name}/archive/pack.zip"
    has_internal = internal_zip in set(names)
    has_split_index = any(n == f"{expected_pack_name}/SPLIT/AGS-00_INDEX.md" or n.endswith("/SPLIT/LAB-00_INDEX.md") for n in names)
    return has_internal and has_split_index


def _maybe_delete_previous_pack(*, current_pack_dir: Path, scope: "PackScope") -> None:
    """
    Delete the previous unzipped pack folder ONLY if it is safely zipped and archived.

    "Previous" is the most recently modified sibling pack dir matching the scope prefix, excluding the current pack.
    """
    if current_pack_dir.parent.resolve() != PACKS_ROOT.resolve():
        return

    prefixes: List[str] = []
    if scope.key == "ags":
        prefixes = ["ags-pack-", "llm-pack-ags-"]
    elif scope.key == "lab":
        prefixes = ["lab-pack-", "llm-pack-lab-"]
    else:
        prefixes = [f"{scope.key}-pack-", f"llm-pack-{scope.key}-"]

    candidates: List[Path] = []
    for p in PACKS_ROOT.iterdir():
        if not p.is_dir():
            continue
        if p.name in {"_system", "_archive", "_state", "archive"}:
            continue
        if p.resolve() == current_pack_dir.resolve():
            continue
        if not any(p.name.startswith(pref) for pref in prefixes):
            continue
        candidates.append(p)

    if not candidates:
        return

    previous = max(candidates, key=lambda d: d.stat().st_mtime)
    archived_zip = EXTERNAL_ARCHIVE_DIR / f"{previous.name}.zip"
    if not _is_valid_external_archive_zip(archived_zip, expected_pack_name=previous.name):
        return
    shutil.rmtree(previous)


def _migrate_system_archive() -> None:
    """
    Legacy: move `_packs/_system/archive/*` into `_packs/_archive/`.
    """
    legacy_dir = SYSTEM_DIR / "archive"
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return

    dest_dir = EXTERNAL_ARCHIVE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved_any = False
    for src in sorted(legacy_dir.iterdir()):
        if not src.is_file():
            continue
        dst = dest_dir / src.name

        # Be robust across filesystems: copy then unlink source (avoid "copy without delete" edge cases).
        if dst.exists():
            try:
                same_size = src.stat().st_size == dst.stat().st_size
                if same_size:
                    import hashlib

                    def _sha256(p: Path) -> str:
                        h = hashlib.sha256()
                        with p.open("rb") as f:
                            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                                h.update(chunk)
                        return h.hexdigest()

                    if _sha256(src) == _sha256(dst):
                        try:
                            src.unlink()
                            moved_any = True
                        except OSError:
                            pass
                        continue
            except OSError:
                pass

            # Preserve content without clobbering the existing external archive name.
            stem, suffix = src.stem, src.suffix
            i = 1
            while True:
                candidate = dest_dir / f"{stem}__legacy{i}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1

        shutil.copy2(src, dst)
        try:
            src.unlink()
        except OSError:
            # If we can't delete the legacy file, do not fail pack creation.
            pass
        moved_any = True

    if moved_any:
        # Best-effort cleanup.
        try:
            legacy_dir.rmdir()
        except OSError:
            pass


def enforce_included_repo_limits(entries: List[Dict[str, Any]], limits: PackLimits) -> Dict[str, Any]:
    total_bytes = sum(e["size"] for e in entries)
    if total_bytes > limits.max_total_bytes:
        raise ValueError(f"PACK_LIMIT_EXCEEDED:max_total_bytes ({total_bytes} > {limits.max_total_bytes})")
    
    if len(entries) > limits.max_entries:
        raise ValueError(f"PACK_LIMIT_EXCEEDED:max_entries ({len(entries)} > {limits.max_entries})")

    for e in entries:
        if e["size"] > limits.max_entry_bytes:
            raise ValueError(f"PACK_LIMIT_EXCEEDED:max_entry_bytes ({e['path']} {e['size']} > {limits.max_entry_bytes})")

    if limits.allow_duplicate_hashes is False:
        hashes = [e["hash"] for e in entries]
        if len(hashes) != len(set(hashes)):
            raise ValueError("PACK_LIMIT_EXCEEDED:duplicate_hashes")

    return {
        "repo_files": len(entries),
        "repo_bytes": total_bytes,
    }


def pack_dir_total_bytes(pack_dir: Path) -> int:
    return sum(f.stat().st_size for f in pack_dir.rglob("*") if f.is_file())


def copy_repo_files(pack_dir: Path, project_root: Path, included_paths: Sequence[str], scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    source_root = project_root / scope.source_root_rel
    for rel in included_paths:
        # rel is already relative to source_root as per build_state_manifest
        src = source_root / rel
        if not src.exists() or not src.is_file():
             # Fallback: maybe it was relative to project root? (Should not happen if manifest was built consistent)
             src = project_root / rel
             if not src.exists():
                continue

        dst = pack_dir / "repo" / rel
        if writer is None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        else:
            writer.mkdir(dst.parent, kind="durable", parents=True, exist_ok=True)
            writer.write_bytes(dst, src.read_bytes())


# --- Document Generators (Sanitized) ---

def write_start_here(pack_dir: Path, *, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    if scope.key == SCOPE_AGS.key:
        canon_contract = rel_posix("LAW", "CANON", "CONTRACT.md")
        maps_entrypoints = rel_posix("NAVIGATION", "MAPS", "ENTRYPOINTS.md")
        contracts_runner = rel_posix("LAW", "CONTRACTS", "runner.py")
        text = "\n".join(
            [
                "# START HERE",
                "",
                "This snapshot is meant to be shared with any LLM to continue work on the Agent Governance System (AGS) repository.",
                "",
                "## Read order",
                "1) `repo/AGENTS.md`",
                "2) `repo/README.md`",
                f"3) `repo/{canon_contract}`",
                f"4) `repo/{maps_entrypoints}`",
                f"5) `repo/{contracts_runner}`",
                "6) `meta/ENTRYPOINTS.md`",
                "",
                "## Notes",
                "- `BUILD` contents excluded.",
                "- Use `FULL/` for single-file output or `SPLIT/` for sectioned reading.",
                "",
            ]
        )
    elif scope.key == SCOPE_LAB.key:
        text = "\n".join(
            [
                "# START HERE (LAB)",
                "",
                "This snapshot contains ALL research and experimental code under `THOUGHT/LAB`. It is volatile.",
                "",
                "## Read order",
                "1) `repo/THOUGHT/LAB/`",
                "",
                "## Notes",
                "- Use `FULL/` for single-file output or `SPLIT/` for sectioned reading.",
                "",
            ]
        )
    else:
        raise ValueError(f"Unsupported scope: {scope.key}")

    if writer is None:
        (pack_dir / "meta" / "START_HERE.md").write_text(text, encoding="utf-8")
    else:
        writer.write_text(pack_dir / "meta" / "START_HERE.md", text, encoding="utf-8")


def write_entrypoints(pack_dir: Path, *, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    if scope.key == SCOPE_AGS.key:
        canon_contract = rel_posix("LAW", "CANON", "CONTRACT.md")
        maps_entrypoints = rel_posix("NAVIGATION", "MAPS", "ENTRYPOINTS.md")
        skills_dir = rel_posix("CAPABILITY", "SKILLS")
        contracts_runner = rel_posix("LAW", "CONTRACTS", "runner.py")
        text = "\n".join(
            [
                "# Snapshot Entrypoints",
                "",
                "Key entrypoints for `AGS`:",
                "",
                "- `repo/AGENTS.md`",
                "- `repo/README.md`",
                f"- `repo/{canon_contract}`",
                f"- `repo/{maps_entrypoints}`",
                f"- `repo/{skills_dir}/`",
                f"- `repo/{contracts_runner}`",
                "",
                "Notes:",
                "- `FULL/` contains single-file bundles.",
                "- `SPLIT/` contains chunked sections.",
                "",
            ]
        )
    elif scope.key == SCOPE_LAB.key:
        text = "\n".join(
            [
                "# Snapshot Entrypoints",
                "",
                f"Key entrypoints for `{scope.title}`:",
                "",
                "- `repo/THOUGHT/LAB/`",
                "",
                "Notes:",
                "- `FULL/` contains single-file bundles.",
                "- `SPLIT/` contains chunked sections.",
                "",
            ]
        )
    else:
        raise ValueError(f"Unsupported scope for ENTRYPOINTS: {scope.key}")

    if writer is None:
        (pack_dir / "meta" / "ENTRYPOINTS.md").write_text(text, encoding="utf-8")
    else:
        writer.write_text(pack_dir / "meta" / "ENTRYPOINTS.md", text, encoding="utf-8")


def write_build_tree(pack_dir: Path, project_root: Path, writer: Optional[PackerWriter] = None) -> None:
    tree_path = pack_dir / "meta" / "BUILD_TREE.txt"
    if writer is None:
        tree_path.write_text("BUILD is excluded from packs by contract.\n", encoding="utf-8")
    else:
        writer.write_text(tree_path, "BUILD is excluded from packs by contract.\n", encoding="utf-8")


def write_pack_info(pack_dir: Path, scope: PackScope, stamp: str, writer: Optional[PackerWriter] = None) -> None:
    info = {
        "scope": scope.key,
        "title": scope.title,
        "stamp": stamp,
        "version": read_canon_version(),
    }
    write_json(pack_dir / "meta" / "PACK_INFO.json", info, writer=writer)

def write_provenance(pack_dir: Path, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    import os

    prov = {
        "generator": "LLM_PACKER",
        # P.2: deterministic by default (no timestamps / machine identity)
        "scope": scope.key,
    }
    write_json(pack_dir / "meta" / "PROVENANCE.json", prov, writer=writer)

def write_omitted(pack_dir: Path, omitted: List[Dict[str, Any]], writer: Optional[PackerWriter] = None) -> None:
    # Always emit for determinism and fixture stability (empty list is meaningful).
    write_json(pack_dir / "meta" / "REPO_OMITTED_BINARIES.json", omitted, writer=writer)

def render_tree(paths: Sequence[str]) -> str:
    """Render a visual tree from a list of relative paths."""
    tree: Dict[str, Any] = {}
    for path in sorted(paths):
        # Normalize to forward slashes just in case
        parts = path.replace("\\", "/").split("/")
        node = tree
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]
            
    lines = ["."]
    
    def walk(node: Dict[str, Any], prefix: str = "") -> None:
        entries = sorted(node.keys())
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry}")
            
            children = node[entry]
            if children:
                extension = "    " if is_last else "│   "
                walk(children, prefix + extension)
                
    walk(tree)
    return "\n".join(lines)


def write_pack_file_tree_and_index(pack_dir: Path, *, scope: PackScope, stamp: str, combined: bool, writer: Optional[PackerWriter] = None) -> None:
    all_files = [p for p in pack_dir.rglob("*") if p.is_file()]
    rel_paths = sorted(p.relative_to(pack_dir).as_posix() for p in all_files)

    # Generate visual tree including predicted treemap files if combined
    final_paths = list(rel_paths)
    tm_md = f"FULL/{scope.file_prefix}-FULL-TREEMAP-{stamp}.md"
    tm_txt = f"FULL/{scope.file_prefix}-FULL-TREEMAP-{stamp}.txt"

    if combined:
        if tm_md not in final_paths: final_paths.append(tm_md)
        if tm_txt not in final_paths: final_paths.append(tm_txt)

    tree_text = render_tree(final_paths)

    if writer is None:
        (pack_dir / "meta" / "FILE_TREE.txt").write_text(tree_text + "\n", encoding="utf-8")
    else:
        writer.write_text(pack_dir / "meta" / "FILE_TREE.txt", tree_text + "\n", encoding="utf-8")

    if combined:
        full_dir = pack_dir / "FULL"
        if writer is None:
            full_dir.mkdir(parents=True, exist_ok=True)
        else:
            writer.mkdir(full_dir, kind="durable", parents=True, exist_ok=True)

        md_content = f"# {scope.file_prefix} TREEMAP\n\n```text\n{tree_text}\n```\n"
        txt_content = f"{scope.file_prefix} TREEMAP\n\n{tree_text}\n"

        if writer is None:
            (full_dir / tm_md.replace("FULL/", "")).write_text(md_content, encoding="utf-8")
        else:
            writer.write_text(full_dir / tm_md.replace("FULL/", ""), md_content, encoding="utf-8")
        # .txt treemap not allowed in FULL/ per req 5. Archive will generate sibling or use FILE_TREE.txt line 538 removal.

    file_index: List[Dict[str, Any]] = []
    # Use actual files for index (excluding treemaps if not written yet)
    for p in sorted(all_files, key=lambda x: x.relative_to(pack_dir).as_posix()):
        rel = p.relative_to(pack_dir).as_posix()
        size = p.stat().st_size
        file_index.append({"path": rel, "bytes": size, "sha256": hash_file(p)})
    file_index.sort(key=lambda e: (e["path"], e["sha256"]))
    write_json(pack_dir / "meta" / "FILE_INDEX.json", file_index, writer=writer)


# --- Full Output Generation ---

def build_combined_md_block(rel_path: str, text: str, byte_count: int) -> str:
    fence = "`````" if "````" in text else ("````" if "```" in text else "```")
    return "\n".join([f"## `{rel_path}` ({byte_count:,} bytes)", "", fence, text.rstrip(), fence, ""]) 

def build_combined_txt_block(rel_path: str, text: str, byte_count: int) -> str:
    limit = "-" * 80
    return "\n".join([limit, f"FILE: {rel_path} ({byte_count:,} bytes)", limit, text.rstrip(), ""])

def write_full_outputs(pack_dir: Path, *, stamp: str, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    full_dir = pack_dir / "FULL"
    if writer is None:
        full_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(full_dir, kind="durable", parents=True, exist_ok=True)

    combined_md_rel = f"{scope.file_prefix}-FULL-{stamp}.md"
    combined_txt_rel = f"{scope.file_prefix}-FULL-{stamp}.txt"

    # We will compute the tree later and append it if needed, or write it separate
    # Roadmap says: FULL/ ...md, ...txt.
    # Logic adapted from legacy but pointing to FULL_DIR

    combined_md_lines = [f"# {scope.file_prefix} FULL", ""]
    combined_txt_lines = [f"{scope.file_prefix} FULL", ""]

    base_paths = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())

    for rel in base_paths:
        # Exclude generated output dirs to avoid recursion/duplication
        if rel.startswith("FULL/") or rel.startswith("SPLIT/") or rel.startswith("LITE/"):
            continue

        abs_path = pack_dir / rel
        text = read_text(abs_path)
        size = abs_path.stat().st_size
        combined_md_lines.append(build_combined_md_block(rel, text, size))
        combined_txt_lines.append(build_combined_txt_block(rel, text, size))

    md_content = "\n".join(combined_md_lines).rstrip() + "\n"
    txt_content = "\n".join(combined_txt_lines).rstrip() + "\n"

    if writer is None:
        (full_dir / combined_md_rel).write_text(md_content, encoding="utf-8")
    else:
        writer.write_text(full_dir / combined_md_rel, md_content, encoding="utf-8")
    # NO TXT output in FULL/ as per requirements
    # (full_dir / combined_txt_rel).write_text(txt_content, encoding="utf-8")


def verify_manifest(pack_dir: Path) -> Tuple[bool, List[str]]:
    manifest_path = pack_dir / "meta" / "REPO_STATE.json"
    if not manifest_path.exists():
        return False, ["Manifest missing: meta/REPO_STATE.json"]
    
    try:
        manifest = json.loads(read_text(manifest_path))
    except Exception as e:
        return False, [f"Manifest invalid JSON: {e}"]
        
    errors = []
    repo_dir = pack_dir / "repo"
    
    files = manifest.get("files", [])
    for entry in files:
        rel_path = entry.get("path")
        expected_hash = entry.get("hash")
        
        # Determine file path (handle potential subdirectory nesting)
        file_path = repo_dir.joinpath(rel_path)
        
        if not file_path.exists():
            errors.append(f"File missing: {rel_path}")
            continue
            
        computed = hash_file(file_path)
        if computed != expected_hash:
            errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {computed}")
            
    return len(errors) == 0, errors


def make_pack(
    *,
    scope_key: str,
    mode: str,
    profile: str,
    split_lite: bool,
    out_dir: Optional[Path],
    combined: bool,
    stamp: Optional[str],
    zip_enabled: bool,
    max_total_bytes: int,
    max_entry_bytes: int,
    max_entries: int,
    allow_duplicate_hashes: Optional[bool],
    project_root: Optional[Path] = None,
    p2_runs_dir: Optional[Path] = None,
    p2_cas_root: Optional[Path] = None,
    skip_proofs: bool = False,
    writer: Optional[PackerWriter] = None,
) -> Path:
    from .split import write_split_pack
    from .lite import write_split_pack_lite
    from .archive import write_pack_internal_archives, write_pack_external_archive
    from CAPABILITY.CAS import cas as cas_mod

    scope = SCOPES.get(scope_key)
    if not scope:
        raise ValueError(f"Unknown scope: {scope_key}")

    source_project_root = (project_root or PROJECT_ROOT).resolve()
    use_shared_baseline = source_project_root == PROJECT_ROOT.resolve()


    with _override_cas_root(p2_cas_root):
        _migrate_system_archive()

        # 0. Proof Refresh (Fail-closed)
        # Must happen BEFORE manifest build so new proofs are picked up.
        if scope.key == SCOPE_AGS.key and not skip_proofs:
            # We use a derived stamp for the proof run if one isn't provided yet.
            # However, core.py generates digest-based stamps later.
            # For proofs, we'll use a temp timestamp if stamp is None,
            # but the Pack ID will stick to the digest or user stamp.
            proof_stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            print(f"Refreshing proofs (stamp={proof_stamp})...")
            success, err = refresh_proofs(source_project_root, stamp=proof_stamp)
            if not success:
                raise RuntimeError(err)

        manifest, omitted = build_state_manifest(source_project_root, scope=scope)
        digest = manifest_digest(manifest)

        if out_dir is None:
            out_dir = PACKS_ROOT / f"llm-pack-{scope.key}-{digest[:12]}"
        out_dir = ensure_under_packs_root(out_dir)

        SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
        EXTERNAL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        FIXTURE_PACKS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        baseline_path = baseline_path_for_scope(scope)
        baseline = load_baseline(baseline_path) if use_shared_baseline else None
        baseline_files_by_path = {f["path"]: f for f in (baseline or {}).get("files", [])}
        current_files_by_path = {f["path"]: f for f in manifest.get("files", [])}

        if allow_duplicate_hashes is None:
            allow_duplicate_hashes = True

        # validate_repo_state_manifest logic (simplified inline or stubbed)
        # We trust build_state_manifest for Phase 1 refactor

        anchors = scope.anchors if use_shared_baseline else ()

        if mode == "delta" and baseline is not None:
            changed = []
            for path, entry in current_files_by_path.items():
                prev = baseline_files_by_path.get(path)
                if prev is None or prev.get("hash") != entry.get("hash") or prev.get("size") != entry.get("size"):
                    changed.append(path)
            deleted = sorted(set(baseline_files_by_path.keys()) - set(current_files_by_path.keys()))
            include_paths = sorted(set(changed) | set(anchors))
        else:
            include_paths = sorted(set(current_files_by_path.keys()) | set(anchors))
            deleted = []

        limits = PackLimits(
            max_total_bytes=max_total_bytes,
            max_entry_bytes=max_entry_bytes,
            max_entries=max_entries,
            allow_duplicate_hashes=allow_duplicate_hashes,
        )
        included_entries = [current_files_by_path[p] for p in include_paths if p in current_files_by_path]
        included_stats = enforce_included_repo_limits(included_entries, limits=limits)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        (out_dir / "repo").mkdir(parents=True, exist_ok=True)

        # 1. Copy Repo
        copy_repo_files(out_dir, source_project_root, include_paths, scope=scope, writer=writer)
        write_json(out_dir / "meta" / "REPO_STATE.json", manifest, writer=writer)
        write_build_tree(out_dir, source_project_root, writer=writer)

        # 2. Meta Docs
        write_start_here(out_dir, scope=scope, writer=writer)
        write_entrypoints(out_dir, scope=scope, writer=writer)
        write_pack_info(out_dir, scope=scope, stamp=stamp or digest[:12], writer=writer)
        write_provenance(out_dir, scope, writer=writer)
        write_omitted(out_dir, omitted, writer=writer)

        # 3. SPLIT Output (Strictly SPLIT/)
        repo_pack_paths = [f"repo/{p}" for p in include_paths]
        write_split_pack(out_dir, repo_pack_paths, scope=scope)

        # 4. FULL Output (Strictly FULL/) - if combined requested (renamed from legacy flag)
        # We respect the legacy flag '--combined' but map it to producing FULL/ output
        if combined:
            effective_stamp = stamp or digest[:12]
            write_full_outputs(out_dir, stamp=effective_stamp, scope=scope, writer=writer)

        # 5. LITE Output (Strictly LITE/) + P.2 CAS artifacts
        if split_lite or profile == "lite":
            write_split_pack_lite(out_dir, scope=scope, project_root=source_project_root)
            effective_runs_dir = p2_runs_dir or Path("CAPABILITY/RUNS")
            _emit_p2_lite_artifacts(
                out_dir,
                project_root=source_project_root,
                include_paths=include_paths,
                scope=scope,
                runs_dir=effective_runs_dir,
                writer=writer,
            )

        # 6. File Inventory
        # effective_stamp used here if combined, otherwise stamp or digest
        eff_stamp_for_tree = stamp or digest[:12]
        write_pack_file_tree_and_index(out_dir, scope=scope, stamp=eff_stamp_for_tree, combined=combined, writer=writer)

        # 7. Archives
        if zip_enabled:
            # Internal Archive lives inside the pack folder (archive/).
            write_pack_internal_archives(out_dir, scope=scope)
            
            # Cleanup root meta/ and repo/ (they are safely inside the Internal Archive zip).
            try:
                shutil.rmtree(out_dir / "meta")
                shutil.rmtree(out_dir / "repo")
            except Exception as exc:
                print(f"PACKER_WARNING: Failed to cleanup root folders: {exc}")

            # External Archive is a zip of the final pack folder under `_packs/_archive/`.
            write_pack_external_archive(out_dir, scope=scope)

            _maybe_delete_previous_pack(current_pack_dir=out_dir, scope=scope)

        # Update baseline
        if use_shared_baseline:
            write_json(baseline_path, manifest, writer=writer)

        return out_dir
