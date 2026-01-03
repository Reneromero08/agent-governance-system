#!/usr/bin/env python3
"""
Core utilities for LLM Packer (Phase 1 modular implementation).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MEMORY_DIR = PROJECT_ROOT / "MEMORY"
LLM_PACKER_DIR = MEMORY_DIR / "LLM_PACKER"
PACKS_ROOT = LLM_PACKER_DIR / "_packs"
SYSTEM_DIR = PACKS_ROOT / "_system"
FIXTURE_PACKS_DIR = SYSTEM_DIR / "fixtures"
STATE_DIR = SYSTEM_DIR / "_state"
BASELINE_PATH = STATE_DIR / "baseline.json"

CANON_VERSION_FILE = PROJECT_ROOT / "LAW" / "CANON" / "VERSIONING.md"
GRAMMAR_VERSION = "1.0"

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
        "Original", "ORIGINAL", "research", "RESEARCH", "__pycache__", "node_modules",
    }),
    source_root_rel=".",
)

SCOPE_CATALYTIC_DPT = PackScope(
    key="catalytic-dpt",
    title="CATALYTIC-DPT (MAIN, no LAB)",
    file_prefix="CATALYTIC-DPT",
    include_dirs=("CAPABILITY", "LAW", "DIRECTION", "NAVIGATION"),
    root_files=(),
    anchors=(
        "AGENTS.md",
        "README.md",
        rel_posix("LAW", "CANON", "CONTRACT.md"),
        rel_posix("LAW", "CANON", "INVARIANTS.md"),
        "CAPABILITY/TESTBENCH/README.md",
    ),
    excluded_dir_parts=frozenset({
        ".git", "BUILD", "LAB", "_runs", "_generated", "_packs", "__pycache__", "node_modules",
    }),
    source_root_rel=".",
)

SCOPE_LAB = PackScope(
    key="lab",
    title="CATALYTIC-DPT (LAB)",
    file_prefix="CATALYTIC-DPT-LAB",
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
    SCOPE_CATALYTIC_DPT.key: SCOPE_CATALYTIC_DPT,
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_under_packs_root(out_dir: Path) -> Path:
    packs_root = PACKS_ROOT.resolve()
    out_dir_resolved = out_dir.resolve()
    try:
        out_dir_resolved.relative_to(packs_root)
    except ValueError as exc:
        raise ValueError(f"OutDir must be under MEMORY/LLM_PACKER/_packs/. Received: {out_dir}") from exc
    return out_dir_resolved


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


def copy_repo_files(pack_dir: Path, project_root: Path, included_paths: Sequence[str], scope: PackScope) -> None:
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


# --- Document Generators (Sanitized) ---

def write_start_here(pack_dir: Path, *, scope: PackScope) -> None:
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
    elif scope.key == SCOPE_CATALYTIC_DPT.key:
        text = "\n".join(
            [
                "# START HERE",
                "",
                f"This snapshot is meant to be shared with any LLM to continue work on `{scope.title}`.",
                "",
                "## Read order",
                "1) `repo/CATALYTIC-DPT/AGENTS.md`",
                "2) `repo/CATALYTIC-DPT/README.md`",
                "3) `repo/CATALYTIC-DPT/ROADMAP_V2.1.md`",
                "4) `repo/CATALYTIC-DPT/swarm_config.json`",
                "5) `repo/CATALYTIC-DPT/CHANGELOG.md`",
                "6) `meta/ENTRYPOINTS.md`",
                "",
                "## Notes",
                "- Use `FULL/` for single-file output or `SPLIT/` for sectioned reading.",
                "",
            ]
        )
    elif scope.key == SCOPE_LAB.key:
        text = "\n".join(
            [
                "# START HERE (LAB)",
                "",
                "This snapshot contains ALL research and experimental code under `CATALYTIC-DPT/LAB`. It is volatile.",
                "",
                "## Read order",
                "1) `repo/CATALYTIC-DPT/LAB/ROADMAP_PATCH_SEMIOTIC.md`",
                "2) `repo/CATALYTIC-DPT/LAB/COMMONSENSE/`",
                "3) `repo/CATALYTIC-DPT/LAB/MCP/`",
                "",
                "## Notes",
                "- Use `FULL/` for single-file output or `SPLIT/` for sectioned reading.",
                "",
            ]
        )
    else:
        raise ValueError(f"Unsupported scope: {scope.key}")

    (pack_dir / "meta" / "START_HERE.md").write_text(text, encoding="utf-8")


def write_entrypoints(pack_dir: Path, *, scope: PackScope) -> None:
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
    elif scope.key == SCOPE_CATALYTIC_DPT.key:
        text = "\n".join(
            [
                "# Snapshot Entrypoints",
                "",
                f"Key entrypoints for `{scope.title}`:",
                "",
                "- `repo/CATALYTIC-DPT/AGENTS.md`",
                "- `repo/CATALYTIC-DPT/README.md`",
                "- `repo/CATALYTIC-DPT/swarm_config.json`",
                "- `repo/CATALYTIC-DPT/TESTBENCH/`",
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
                "- `repo/CATALYTIC-DPT/LAB/`",
                "- `repo/CATALYTIC-DPT/LAB/ROADMAP_PATCH_SEMIOTIC.md`",
                "- `repo/CATALYTIC-DPT/LAB/COMMONSENSE/`",
                "- `repo/CATALYTIC-DPT/LAB/MCP/`",
                "",
                "Notes:",
                "- `FULL/` contains single-file bundles.",
                "- `SPLIT/` contains chunked sections.",
                "",
            ]
        )
    else:
        raise ValueError(f"Unsupported scope for ENTRYPOINTS: {scope.key}")

    (pack_dir / "meta" / "ENTRYPOINTS.md").write_text(text, encoding="utf-8")


def write_build_tree(pack_dir: Path, project_root: Path) -> None:
    tree_path = pack_dir / "meta" / "BUILD_TREE.txt"
    tree_path.write_text("BUILD is excluded from packs by contract.\n", encoding="utf-8")


def write_pack_info(pack_dir: Path, scope: PackScope, stamp: str) -> None:
    info = {
        "scope": scope.key,
        "title": scope.title,
        "stamp": stamp,
        "version": read_canon_version(),
    }
    write_json(pack_dir / "meta" / "PACK_INFO.json", info)

def write_provenance(pack_dir: Path, scope: PackScope) -> None:
    import os
    import datetime
    
    # Allow deterministic timestamp for tests
    timestamp = os.getenv("LLM_PACKER_DETERMINISTIC_TIMESTAMP")
    if not timestamp:
        timestamp = datetime.datetime.now().isoformat()
        
    prov = {
        "generator": "LLM_PACKER",
        "generated_at": timestamp,
        "host": os.getenv("COMPUTERNAME", "unknown"),
        "user": os.getenv("USERNAME", "unknown"),
        "scope": scope.key
    }
    write_json(pack_dir / "meta" / "PROVENANCE.json", prov)

def write_omitted(pack_dir: Path, omitted: List[Dict[str, Any]]) -> None:
    if omitted:
        write_json(pack_dir / "meta" / "REPO_OMITTED_BINARIES.json", omitted)

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


def write_pack_file_tree_and_index(pack_dir: Path, *, scope: PackScope, stamp: str, combined: bool) -> None:
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

    (pack_dir / "meta" / "FILE_TREE.txt").write_text(tree_text + "\n", encoding="utf-8")

    if combined:
        full_dir = pack_dir / "FULL"
        full_dir.mkdir(parents=True, exist_ok=True)
        
        md_content = f"# {scope.file_prefix} TREEMAP\n\n```text\n{tree_text}\n```\n"
        txt_content = f"{scope.file_prefix} TREEMAP\n\n{tree_text}\n"
        
        (full_dir / tm_md.replace("FULL/", "")).write_text(md_content, encoding="utf-8")
        # .txt treemap not allowed in FULL/ per req 5. Archive will generate sibling or use FILE_TREE.txt line 538 removal.

    file_index: List[Dict[str, Any]] = []
    # Use actual files for index (excluding treemaps if not written yet)
    for p in sorted(all_files, key=lambda x: x.relative_to(pack_dir).as_posix()):
        rel = p.relative_to(pack_dir).as_posix()
        size = p.stat().st_size
        file_index.append({"path": rel, "bytes": size, "sha256": hash_file(p)})
    file_index.sort(key=lambda e: (e["path"], e["sha256"]))
    write_json(pack_dir / "meta" / "FILE_INDEX.json", file_index)


# --- Full Output Generation ---

def build_combined_md_block(rel_path: str, text: str, byte_count: int) -> str:
    fence = "`````" if "````" in text else ("````" if "```" in text else "```")
    return "\n".join([f"## `{rel_path}` ({byte_count:,} bytes)", "", fence, text.rstrip(), fence, ""]) 

def build_combined_txt_block(rel_path: str, text: str, byte_count: int) -> str:
    limit = "-" * 80
    return "\n".join([limit, f"FILE: {rel_path} ({byte_count:,} bytes)", limit, text.rstrip(), ""])

def write_full_outputs(pack_dir: Path, *, stamp: str, scope: PackScope) -> None:
    full_dir = pack_dir / "FULL"
    full_dir.mkdir(parents=True, exist_ok=True)

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
        if rel.startswith("FULL/") or rel.startswith("SPLIT/") or rel.startswith("LITE/") or rel.startswith("archive/"):
            continue

        abs_path = pack_dir / rel
        text = read_text(abs_path)
        size = abs_path.stat().st_size
        combined_md_lines.append(build_combined_md_block(rel, text, size))
        combined_txt_lines.append(build_combined_txt_block(rel, text, size))

    md_content = "\n".join(combined_md_lines).rstrip() + "\n"
    txt_content = "\n".join(combined_txt_lines).rstrip() + "\n"

    (full_dir / combined_md_rel).write_text(md_content, encoding="utf-8")
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
) -> Path:
    from .split import write_split_pack
    from .lite import write_split_pack_lite
    from .archive import write_pack_internal_archives

    scope = SCOPES.get(scope_key)
    if not scope:
        raise ValueError(f"Unknown scope: {scope_key}")

    manifest, omitted = build_state_manifest(PROJECT_ROOT, scope=scope)
    digest = manifest_digest(manifest)

    if out_dir is None:
        out_dir = PACKS_ROOT / f"llm-pack-{scope.key}-{digest[:12]}"
    out_dir = ensure_under_packs_root(out_dir)

    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_PACKS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = baseline_path_for_scope(scope)
    baseline = load_baseline(baseline_path)
    baseline_files_by_path = {f["path"]: f for f in (baseline or {}).get("files", [])}
    current_files_by_path = {f["path"]: f for f in manifest.get("files", [])}

    if allow_duplicate_hashes is None:
        allow_duplicate_hashes = True

    # validate_repo_state_manifest logic (simplified inline or stubbed)
    # We trust build_state_manifest for Phase 1 refactor

    if mode == "delta" and baseline is not None:
        changed = []
        for path, entry in current_files_by_path.items():
            prev = baseline_files_by_path.get(path)
            if prev is None or prev.get("hash") != entry.get("hash") or prev.get("size") != entry.get("size"):
                changed.append(path)
        deleted = sorted(set(baseline_files_by_path.keys()) - set(current_files_by_path.keys()))
        include_paths = sorted(set(changed) | set(scope.anchors))
    else:
        include_paths = sorted(set(current_files_by_path.keys()) | set(scope.anchors))
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
    copy_repo_files(out_dir, PROJECT_ROOT, include_paths, scope=scope)
    write_json(out_dir / "meta" / "REPO_STATE.json", manifest)
    write_build_tree(out_dir, PROJECT_ROOT)

    # 2. Meta Docs
    write_start_here(out_dir, scope=scope)
    write_entrypoints(out_dir, scope=scope)
    write_pack_info(out_dir, scope=scope, stamp=stamp or digest[:12])
    write_provenance(out_dir, scope)
    write_omitted(out_dir, omitted)

    # 3. SPLIT Output (Strictly SPLIT/)
    repo_pack_paths = [f"repo/{p}" for p in include_paths]
    write_split_pack(out_dir, repo_pack_paths, scope=scope)

    # 4. FULL Output (Strictly FULL/) - if combined requested (renamed from legacy flag)
    # We respect the legacy flag '--combined' but map it to producing FULL/ output
    if combined:
        effective_stamp = stamp or digest[:12]
        write_full_outputs(out_dir, stamp=effective_stamp, scope=scope)

    # 5. LITE Output (Strictly LITE/)
    if split_lite or profile == "lite":
        write_split_pack_lite(out_dir, scope=scope)

    # 6. File Inventory
    # effective_stamp used here if combined, otherwise stamp or digest
    eff_stamp_for_tree = stamp or digest[:12]
    write_pack_file_tree_and_index(out_dir, scope=scope, stamp=eff_stamp_for_tree, combined=combined)

    # 7. Archives (Strictly archive/)
    if zip_enabled:
        write_pack_internal_archives(out_dir, scope=scope, system_archive_dir=SYSTEM_DIR / "archive")
        
        # Cleanup root meta/ and repo/ as per naming requirement 4
        # "Pack root must contain ONLY: FULL/, SPLIT/, LITE/, archive/"
        # Since they are safely inside pack.zip, we delete the folders.
        try:
            shutil.rmtree(out_dir / "meta")
            shutil.rmtree(out_dir / "repo")
        except Exception as exc:
            print(f"PACKER_WARNING: Failed to cleanup root folders: {exc}")


    # Update baseline
    write_json(baseline_path, manifest)

    return out_dir
