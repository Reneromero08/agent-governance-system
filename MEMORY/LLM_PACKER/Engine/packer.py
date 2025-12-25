#!/usr/bin/env python3

"""
Packer for AGS memory packs.

This script produces two related artifacts:

1) A repository state manifest (hashes + sizes) used to compare snapshots and drive delta
   packs.
2) A shareable "LLM pack" directory with curated entrypoints, indices and optional combined
   markdown suitable for handoff to another model.


All outputs are written under `MEMORY/LLM_PACKER/_packs/`.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Token Limits
TOKEN_LIMIT_WARNING = 120000   # Warn approaching standard 128k window
TOKEN_LIMIT_CRITICAL = 190000  # Critical danger zone (leaving buffer for output)

# ANSI Colors
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEMORY_DIR = PROJECT_ROOT / "MEMORY"
LLM_PACKER_DIR = MEMORY_DIR / "LLM_PACKER"
PACKS_ROOT = LLM_PACKER_DIR / "_packs"
STATE_DIR = PACKS_ROOT / "_state"
BASELINE_PATH = STATE_DIR / "baseline.json"

@dataclass(frozen=True)
class PackScope:
    key: str
    title: str
    file_prefix: str
    include_dirs: Tuple[str, ...]
    root_files: Tuple[str, ...]
    anchors: Tuple[str, ...]
    excluded_dir_parts: frozenset[str]


SCOPE_AGS = PackScope(
    key="ags",
    title="Agent Governance System (AGS)",
    file_prefix="AGS",
    include_dirs=(
        "CANON",
        "CONTEXT",
        "MAPS",
        "SKILLS",
        "CONTRACTS",
        "MEMORY",
        "CORTEX",
        "TOOLS",
        ".github",
    ),
    root_files=(
        "README.md",
        "LICENSE",
        "AGENTS.md",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
    ),
    anchors=(
        "AGENTS.md",
        "README.md",
        "CONTEXT/archive/planning/INDEX.md",
        "CANON/CONTRACT.md",
        "CANON/INVARIANTS.md",
        "CANON/VERSIONING.md",
        "MAPS/ENTRYPOINTS.md",
        "CONTRACTS/runner.py",
        "MEMORY/packer.py",
    ),
    excluded_dir_parts=frozenset(
        {
            ".git",
            "BUILD",
            "_runs",
            "_generated",
            "_packs",
            "Original",
            "ORIGINAL",
            "research",
            "RESEARCH",
            "__pycache__",
            "node_modules",
        }
    ),
)


SCOPE_CATALYTIC_DPT = PackScope(
    key="catalytic-dpt",
    title="CATALYTIC-DPT",
    file_prefix="CATALYTIC-DPT",
    include_dirs=("CATALYTIC-DPT",),
    root_files=(),
    anchors=(
        "CATALYTIC-DPT/AGENTS.md",
        "CATALYTIC-DPT/README.md",
        "CATALYTIC-DPT/ROADMAP_V2.1.md",
        "CATALYTIC-DPT/swarm_config.json",
        "CATALYTIC-DPT/CHANGELOG.md",
    ),
    excluded_dir_parts=frozenset(
        {
            ".git",
            "BUILD",
            "_runs",
            "_generated",
            "_packs",
            "__pycache__",
            "node_modules",
        }
    ),
)


SCOPES: Dict[str, PackScope] = {
    SCOPE_AGS.key: SCOPE_AGS,
    SCOPE_CATALYTIC_DPT.key: SCOPE_CATALYTIC_DPT,
}

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".css",
    ".html",
    ".php",
    ".ps1",
    ".cmd",
    ".bat",
    ".yml",
    ".yaml",
}

TEXT_BASENAMES = {".gitignore", ".gitattributes", ".editorconfig", ".htaccess", ".gitkeep", "LICENSE"}

CANON_VERSION_FILE = PROJECT_ROOT / "CANON" / "VERSIONING.md"
GRAMMAR_VERSION = "1.0"

def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# Token estimation constants
CHARS_PER_TOKEN = 4  # Rough estimate for English text
TOKEN_LIMIT_WARNING = 100_000
TOKEN_LIMIT_CRITICAL = 200_000


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for text."""
    if tiktoken:
        try:
            # Use o200k_base for gpt-4o/o1
            encoding = tiktoken.get_encoding("o200k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    # Rough approximation
    return len(text) // CHARS_PER_TOKEN


def estimate_file_tokens(path: Path) -> int:
    """Estimate token count for a file."""
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
    # Allow `.github/` but avoid other hidden folders by default.
    if any(part.startswith(".") and part != ".github" for part in rel_path.parts):
        return True
    return False


def iter_repo_candidates(project_root: Path, *, scope: PackScope) -> Iterable[Path]:
    for directory in scope.include_dirs:
        base = project_root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
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

    seen: set[str] = set()
    for abs_path in iter_repo_candidates(project_root, scope=scope):
        rel = abs_path.relative_to(project_root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)

        if not is_text_path(abs_path):
            omitted.append(
                {
                    "scope": "repo",
                    "repoRelPath": rel,
                    "bytes": abs_path.stat().st_size,
                }
            )
            continue

        files.append(
            {
                "path": rel,
                "hash": hash_file(abs_path),
                "size": abs_path.stat().st_size,
            }
        )

    files.sort(key=lambda e: e["path"])
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

def _extract_markdown_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == f"## {heading}".lower():
            start = idx + 1
            break
    if start is None:
        return ""
    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        out.append(line.rstrip())
        if len(out) >= 30:
            break
    return "\n".join([l for l in out if l.strip()]).strip()


def _ast_signature(node: ast.AST) -> Dict[str, Any]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = node.args
    posonly = [a.arg for a in getattr(args, "posonlyargs", [])]
    normal = [a.arg for a in args.args]
    kwonly = [a.arg for a in args.kwonlyargs]
    return {
        "posonlyargs": posonly,
        "args": normal,
        "vararg": args.vararg.arg if args.vararg else None,
        "kwonlyargs": kwonly,
        "kwarg": args.kwarg.arg if args.kwarg else None,
    }


def _extract_code_symbols(source_text: str, module_path: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return {"module": module_path, "error": str(exc), "symbols": []}
    symbols: List[Dict[str, Any]] = []

    module_doc = ast.get_docstring(tree) or ""

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "qualname": node.name,
                    "signature": _ast_signature(node),
                    "docstring": ast.get_docstring(node) or "",
                    "lineno": getattr(node, "lineno", None),
                }
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "qualname": node.name,
                    "bases": [getattr(b, "id", getattr(b, "attr", None)) for b in node.bases],
                    "docstring": ast.get_docstring(node) or "",
                    "lineno": getattr(node, "lineno", None),
                }
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        {
                            "kind": "method",
                            "name": child.name,
                            "qualname": f"{node.name}.{child.name}",
                            "signature": _ast_signature(child),
                            "docstring": ast.get_docstring(child) or "",
                            "lineno": getattr(child, "lineno", None),
                        }
                    )

    return {"module": module_path, "docstring": module_doc, "symbols": symbols}


def _collect_fixture_preview(project_root: Path, rel_path: str, size_bytes: int) -> Dict[str, Any]:
    preview: Dict[str, Any] = {"type": None}
    if not rel_path.endswith(".json"):
        return preview
    if size_bytes > 256 * 1024:
        preview["type"] = "json"
        preview["keys"] = None
        preview["note"] = "skipped_large_json"
        return preview
    try:
        payload = json.loads((project_root / rel_path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        preview["type"] = "json"
        preview["keys"] = None
        preview["note"] = "parse_error"
        return preview
    if isinstance(payload, dict):
        preview["type"] = "object"
        preview["keys"] = sorted(list(payload.keys()))[:30]
    elif isinstance(payload, list):
        preview["type"] = "array"
        preview["length"] = len(payload)
    else:
        preview["type"] = type(payload).__name__
    return preview


def write_lite_indexes(
    pack_dir: Path,
    *,
    project_root: Path,
    include_paths: Sequence[str],
    omitted_paths: Sequence[str],
    files_by_path: Dict[str, Dict[str, Any]],
) -> None:
    meta_dir = pack_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    allowlist = {
        "profile": "lite",
        "required_includes": [
            "AGENTS.md",
            "README.md",
            "CANON/**",
            "MAPS/**",
            "CONTRACTS/runner.py",
            "CORTEX/query.py",
            "TOOLS/critic.py",
            "SKILLS/**/SKILL.md",
            "SKILLS/**/version.json",
        ],
        "excludes": [
            "**/fixtures/**",
            "**/_runs/**",
            "**/_generated/**",
            "CONTEXT/research/**",
            "CONTEXT/archive/**",
            "MEMORY/**/_packs/**",
            "**/*.cmd",
            "**/*.ps1",
        ],
    }
    write_json(meta_dir / "LITE_ALLOWLIST.json", allowlist)

    omitted: List[Dict[str, Any]] = []
    for rel in sorted(omitted_paths):
        entry = files_by_path.get(rel)
        if not entry:
            continue
        omitted.append(
            {
                "path": rel,
                "bytes": entry.get("size"),
                "sha256": entry.get("hash"),
            }
        )
    write_json(meta_dir / "LITE_OMITTED.json", omitted)

    lite_start_here = "\n".join(
        [
            "# LITE Pack: START HERE",
            "",
            "This is a discussion-first pack profile. It includes high-signal governance + interfaces and omits bulk payload (fixtures, archives, and most code).",
            "",
            "## What is included",
            "- `COMBINED/SPLIT/*` (00..07) for read-order and contracts summaries",
            "- `meta/*` inventories (FILE_TREE, FILE_INDEX, PACK_INFO, etc.)",
            "- Core repo entrypoints (AGENTS, CANON, MAPS, runner/query/critic, skill manifests)",
            "",
            "## What is omitted",
            "- Fixture trees and large snapshots are not copied into `repo/**` in this profile.",
            "- See `meta/LITE_OMITTED.json` for the omitted path list (with sizes and hashes).",
            "",
            "## When you need FULL",
            "- Use the FULL profile to access the complete `repo/**` snapshot for deep dives and exact reconstruction.",
            "",
        ]
    ).rstrip() + "\n"
    (meta_dir / "LITE_START_HERE.md").write_text(lite_start_here, encoding="utf-8")

    # SKILL_INDEX.json (from SKILL.md files that are included in LITE)
    skill_index: List[Dict[str, Any]] = []
    skill_manifests = [p for p in include_paths if p.startswith("SKILLS/") and p.endswith("/SKILL.md")]
    for rel in sorted(skill_manifests):
        skill_name = rel.split("/")[1] if len(rel.split("/")) >= 2 else rel
        text = read_text(project_root / rel)
        skill_index.append(
            {
                "name": skill_name,
                "path": f"repo/{rel}",
                "required_canon_version": next(
                    (line.split(":", 1)[1].strip() for line in text.splitlines() if "required_canon_version" in line),
                    "",
                ),
                "inputs": _extract_markdown_section(text, "Inputs"),
                "outputs": _extract_markdown_section(text, "Outputs"),
                "constraints": _extract_markdown_section(text, "Constraints"),
            }
        )
    write_json(meta_dir / "SKILL_INDEX.json", skill_index)

    # FIXTURE_INDEX.json (inventory only; do not copy blobs)
    fixture_index: List[Dict[str, Any]] = []
    for rel, entry in sorted(files_by_path.items(), key=lambda kv: kv[0]):
        if "/fixtures/" not in rel:
            continue
        if not rel.endswith(".json"):
            continue
        size = int(entry.get("size") or 0)
        fixture_index.append(
            {
                "path": rel,
                "bytes": size,
                "sha256": entry.get("hash"),
                "preview": _collect_fixture_preview(project_root, rel, size),
            }
        )
        if len(fixture_index) >= 5000:
            break
    write_json(meta_dir / "FIXTURE_INDEX.json", fixture_index)

    # CODEBOOK.md (hot entrypoints table)
    hot_paths: List[str] = [
        "CONTRACTS/runner.py",
        "CORTEX/query.py",
        "TOOLS/critic.py",
        "MEMORY/LLM_PACKER/Engine/packer.py",
    ]
    for rel in sorted({p.replace("SKILLS/", "SKILLS/") for p in files_by_path.keys() if p.startswith("SKILLS/") and p.endswith("/run.py")}):
        hot_paths.append(rel)

    def module_purpose(path_rel: str) -> str:
        abs_path = project_root / path_rel
        if not abs_path.exists():
            return "not present in repo"
        text = read_text(abs_path)
        symbols = _extract_code_symbols(text, path_rel)
        doc = (symbols.get("docstring") or "").strip().splitlines()
        return doc[0].strip() if doc else "see file"

    codebook_lines = [
        "# CODEBOOK (LITE)",
        "",
        "Symbolic table of hot entrypoints. This file does not embed full source bodies.",
        "",
        "| Path | Included in LITE | Purpose |",
        "|---|---:|---|",
    ]
    include_set = set(include_paths)
    for rel in sorted(set(hot_paths)):
        included = "yes" if rel in include_set else "no"
        purpose = module_purpose(rel).replace("|", "\\|")
        codebook_lines.append(f"| `repo/{rel}` | {included} | {purpose} |")
    codebook_lines.append("")
    (meta_dir / "CODEBOOK.md").write_text("\n".join(codebook_lines), encoding="utf-8")

    # CODE_SYMBOLS.json (AST symbols for included code files only; no bodies)
    code_symbols: List[Dict[str, Any]] = []
    for rel in sorted(include_paths):
        if not rel.endswith(".py"):
            continue
        abs_path = project_root / rel
        if not abs_path.exists():
            continue
        code_symbols.append(_extract_code_symbols(read_text(abs_path), rel))
    write_json(meta_dir / "CODE_SYMBOLS.json", code_symbols)


def verify_manifest(pack_dir: Path) -> Tuple[bool, List[str]]:
    """
    Verify pack integrity by checking file hashes against manifest.
    
    Returns:
        (is_valid, errors): True if all hashes match, list of errors if any.
    """
    errors: List[str] = []
    
    # Load the pack manifest
    manifest_path = pack_dir / "meta" / "REPO_STATE.json"
    if not manifest_path.exists():
        errors.append(f"Manifest not found: {manifest_path}")
        return False, errors
    
    try:
        manifest = json.loads(read_text(manifest_path))
    except Exception as e:
        errors.append(f"Failed to load manifest: {e}")
        return False, errors
    
    # Verify each file in the manifest
    for entry in manifest.get("files", []):
        rel_path = entry.get("path", "")
        expected_hash = entry.get("hash", "")
        expected_size = entry.get("size", 0)
        
        file_path = pack_dir / "repo" / rel_path
        if not file_path.exists():
            errors.append(f"Missing file: {rel_path}")
            continue
        
        actual_size = file_path.stat().st_size
        if actual_size != expected_size:
            errors.append(f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}")
            continue
        
        actual_hash = hash_file(file_path)
        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash[:12]}..., got {actual_hash[:12]}...")
    
    return len(errors) == 0, errors


def load_and_verify_pack(pack_dir: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Load a pack and verify its integrity.
    
    Returns:
        (manifest, errors): The manifest dict if valid, None if invalid. List of errors.
    """
    is_valid, errors = verify_manifest(pack_dir)
    if not is_valid:
        return None, errors
    
    manifest_path = pack_dir / "meta" / "REPO_STATE.json"
    manifest = json.loads(read_text(manifest_path))
    return manifest, []


def choose_fence(text: str) -> str:
    matches = re.findall(r"`+", text)
    longest = max((len(m) for m in matches), default=0)
    return "`" * max(3, longest + 1)


def infer_lang(rel_path: str) -> str:
    suffix = Path(rel_path).suffix.lower()
    return {
        ".json": "json",
        ".md": "md",
        ".py": "python",
        ".js": "js",
        ".mjs": "js",
        ".cjs": "js",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".ps1": "powershell",
        ".cmd": "bat",
        ".bat": "bat",
        ".css": "css",
        ".html": "html",
        ".php": "php",
        ".txt": "text",
    }.get(suffix, "")


def build_combined_md_block(rel_path: str, text: str, byte_count: int) -> str:
    fence = choose_fence(text)
    lang = infer_lang(rel_path)
    fence_open = fence + (lang if lang else "")
    return "\n".join(
        [
            "",
            "-----",
            f"Source: `{rel_path}`",
            f"Bytes: {byte_count}",
            "-----",
            "",
            fence_open,
            text.rstrip("\n"),
            fence,
        ]
    )


def build_combined_txt_block(rel_path: str, text: str, byte_count: int) -> str:
    return "\n".join(
        [
            "",
            "-----",
            f"Source: {rel_path}",
            f"Bytes: {byte_count}",
            "-----",
            "",
            text.rstrip("\n"),
        ]
    )


class _TreeNode:
    def __init__(self) -> None:
        self.dirs: Dict[str, "_TreeNode"] = {}
        self.files: set[str] = set()


def _sort_key(name: str) -> Tuple[str, str]:
    folded = name.casefold()
    return (folded, name)


def _add_tree_path(root: _TreeNode, rel_path: str) -> None:
    parts = [p for p in rel_path.split("/") if p]
    if not parts:
        return
    node = root
    for idx, part in enumerate(parts):
        is_leaf = idx == len(parts) - 1
        if is_leaf:
            node.files.add(part)
            return
        node = node.dirs.setdefault(part, _TreeNode())


def _render_tree(node: _TreeNode, prefix: str, out_lines: List[str]) -> None:
    dir_names = sorted(node.dirs.keys(), key=_sort_key)
    file_names = sorted(node.files, key=_sort_key)
    entries: List[Tuple[str, str]] = [("dir", d) for d in dir_names] + [("file", f) for f in file_names]

    for idx, (kind, name) in enumerate(entries):
        is_last = idx == len(entries) - 1
        connector = "\\-- " if is_last else "|-- "
        child_prefix = prefix + ("    " if is_last else "|   ")
        if kind == "dir":
            out_lines.append(prefix + connector + name + "/")
            _render_tree(node.dirs[name], child_prefix, out_lines)
        else:
            out_lines.append(prefix + connector + name)


def build_pack_tree_text(paths: Sequence[str], extra_paths: Sequence[str]) -> str:
    root = _TreeNode()
    for rel in paths:
        _add_tree_path(root, rel)
    for rel in extra_paths:
        _add_tree_path(root, rel)

    lines: List[str] = ["PACK/"]
    _render_tree(root, "", lines)
    return "\n".join(lines).rstrip() + "\n"


def write_split_pack_ags(pack_dir: Path, included_repo_paths: Sequence[str]) -> None:
    split_dir = pack_dir / "COMBINED" / "SPLIT"
    split_dir.mkdir(parents=True, exist_ok=True)

    def section(paths: Sequence[str]) -> str:
        out_lines: List[str] = []
        for rel in paths:
            src = pack_dir / rel
            if not src.exists():
                continue
            text = read_text(src)
            fence = choose_fence(text)
            out_lines.append(f"## `{rel}`")
            out_lines.append("")
            out_lines.append(fence)
            out_lines.append(text.rstrip("\n"))
            out_lines.append(fence)
            out_lines.append("")
        return "\n".join(out_lines).rstrip() + "\n"

    canon_paths = [p for p in included_repo_paths if p.startswith("repo/CANON/")]
    root_paths = [p for p in included_repo_paths if p.startswith("repo/") and p.count("/") == 1]
    maps_paths = [p for p in included_repo_paths if p.startswith("repo/MAPS/")]
    context_paths = [p for p in included_repo_paths if p.startswith("repo/CONTEXT/")]
    skills_paths = [p for p in included_repo_paths if p.startswith("repo/SKILLS/")]
    contracts_paths = [p for p in included_repo_paths if p.startswith("repo/CONTRACTS/")]
    cortex_paths = [p for p in included_repo_paths if p.startswith("repo/CORTEX/")]
    memory_paths = [p for p in included_repo_paths if p.startswith("repo/MEMORY/")]
    tools_paths = [p for p in included_repo_paths if p.startswith("repo/TOOLS/")]
    github_paths = [p for p in included_repo_paths if p.startswith("repo/.github/")]

    # Also discover meta files for snapshots
    meta_dir = pack_dir / "meta"
    meta_paths = []
    if meta_dir.exists():
        meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()])


    (split_dir / "AGS-00_INDEX.md").write_text(
        "\n".join(
            [
                "# AGS Pack Index",
                "",
                "This directory contains a generated snapshot of the repository intended for LLM handoff.",
                "",
                "## Read order",
                "1) `repo/AGENTS.md`",
                "2) `repo/README.md` and `repo/CONTEXT/archive/planning/INDEX.md`",
                "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md`",
                "4) `repo/MAPS/ENTRYPOINTS.md`",
                "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/`",
                "6) `repo/CORTEX/` and `repo/TOOLS/`",
                "7) `meta/ENTRYPOINTS.md` and `meta/CONTEXT.txt` (Snapshot specific)",
                "",
                "## Notes",
        "- `BUILD` contents are not included. Only a file tree inventory is captured in `meta/BUILD_TREE.txt`.",
                "- Research under `repo/CONTEXT/research/` is non-binding and opt-in.",
                "- If `--combined` is enabled, `COMBINED/` contains `AGS-FULL-COMBINED-*` and `AGS-FULL-TREEMAP-*` outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / "AGS-01_CANON.md").write_text("# Canon\n\n" + section(canon_paths), encoding="utf-8")
    (split_dir / "AGS-02_ROOT.md").write_text("# Root\n\n" + section(root_paths), encoding="utf-8")
    (split_dir / "AGS-03_MAPS.md").write_text("# Maps\n\n" + section(maps_paths), encoding="utf-8")
    (split_dir / "AGS-04_CONTEXT.md").write_text("# Context\n\n" + section(context_paths), encoding="utf-8")
    (split_dir / "AGS-05_SKILLS.md").write_text("# Skills\n\n" + section(skills_paths), encoding="utf-8")
    (split_dir / "AGS-06_CONTRACTS.md").write_text("# Contracts\n\n" + section(contracts_paths), encoding="utf-8")
    (split_dir / "AGS-07_SYSTEM.md").write_text(
        "# System\n\n" + section([*cortex_paths, *memory_paths, *tools_paths, *github_paths, *meta_paths]),
        encoding="utf-8",
    )


def write_split_pack_catalytic_dpt(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope) -> None:
    split_dir = pack_dir / "COMBINED" / "SPLIT"
    split_dir.mkdir(parents=True, exist_ok=True)

    def section(paths: Sequence[str]) -> str:
        out_lines: List[str] = []
        for rel in paths:
            src = pack_dir / rel
            if not src.exists():
                continue
            text = read_text(src)
            fence = choose_fence(text)
            out_lines.append(f"## `{rel}`")
            out_lines.append("")
            out_lines.append(fence)
            out_lines.append(text.rstrip("\n"))
            out_lines.append(fence)
            out_lines.append("")
        return "\n".join(out_lines).rstrip() + "\n"

    def is_cdpt(path: str) -> bool:
        return path.startswith("repo/CATALYTIC-DPT/")

    cdpt_paths = [p for p in included_repo_paths if is_cdpt(p)]
    cdpt_root = [p for p in cdpt_paths if p.count("/") == 2]

    agents_paths = [p for p in cdpt_root if p.endswith("/AGENTS.md")]
    readme_paths = [p for p in cdpt_root if p.endswith("/README.md")]
    roadmap_paths = [p for p in cdpt_root if "ROADMAP" in p.upper()]
    changelog_paths = [p for p in cdpt_root if p.endswith("/CHANGELOG.md")]
    architecture_docs = [p for p in cdpt_root if p.endswith(".md") and p not in {*agents_paths, *readme_paths, *roadmap_paths, *changelog_paths}]

    config_paths = [p for p in cdpt_root if p.endswith(".json")] + [p for p in cdpt_paths if "/SCHEMAS/" in p or "/MCP/" in p]
    testbench_paths = [p for p in cdpt_paths if "/TESTBENCH/" in p or "/FIXTURES/" in p]

    docs_paths = [
        *agents_paths,
        *readme_paths,
        *roadmap_paths,
        *changelog_paths,
        *architecture_docs,
    ]
    docs_paths = sorted(set(docs_paths))

    config_paths = sorted(set(config_paths) - set(docs_paths))
    testbench_paths = sorted(set(testbench_paths) - set(docs_paths))

    system_paths = sorted(set(cdpt_paths) - set(docs_paths) - set(config_paths) - set(testbench_paths))

    meta_dir = pack_dir / "meta"
    meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()]) if meta_dir.exists() else []

    (split_dir / f"{scope.file_prefix}-00_INDEX.md").write_text(
        "\n".join(
            [
                f"# {scope.title} Pack Index",
                "",
                "This directory contains a generated snapshot intended for LLM handoff.",
                "",
                "## Read order",
                "1) `repo/CATALYTIC-DPT/AGENTS.md`",
                "2) `repo/CATALYTIC-DPT/README.md`",
                "3) `repo/CATALYTIC-DPT/ROADMAP_V2.1.md`",
                "4) `repo/CATALYTIC-DPT/swarm_config.json`",
                "5) `repo/CATALYTIC-DPT/CHANGELOG.md`",
                "6) `meta/ENTRYPOINTS.md` and `meta/CONTEXT.txt`",
                "",
                "## Notes",
                f"- If `--combined` is enabled, `COMBINED/` contains `{scope.file_prefix}-FULL-COMBINED-*` and `{scope.file_prefix}-FULL-TREEMAP-*` outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / f"{scope.file_prefix}-01_DOCS.md").write_text("# Docs\n\n" + section(docs_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-02_CONFIG.md").write_text("# Config\n\n" + section(config_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-03_TESTBENCH.md").write_text("# Testbench\n\n" + section(testbench_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-04_SYSTEM.md").write_text("# System\n\n" + section([*system_paths, *meta_paths]), encoding="utf-8")


def write_split_pack(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope) -> None:
    if scope.key == SCOPE_AGS.key:
        write_split_pack_ags(pack_dir, included_repo_paths)
    elif scope.key == SCOPE_CATALYTIC_DPT.key:
        write_split_pack_catalytic_dpt(pack_dir, included_repo_paths, scope=scope)
    else:
        raise ValueError(f"Unsupported scope for split pack: {scope.key}")


def write_split_pack_lite(pack_dir: Path, *, scope: PackScope) -> None:
    """
    Write a discussion-first SPLIT set alongside the full SPLIT docs.

    This does not affect what FULL includes/copies under repo/**; it is derived
    documentation intended for fast navigation and lower token load.
    """
    split_dir = pack_dir / "COMBINED" / "SPLIT_LITE"
    split_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, text: str) -> None:
        path.write_text(text.rstrip() + "\n", encoding="utf-8")

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-00_INDEX.md",
            "\n".join(
                [
                    "# AGS Pack Index (SPLIT_LITE)",
                    "",
                    "This directory contains a compressed, discussion-first map of the pack (pointers + indexes).",
                    "",
                    "## Read order",
                    "1) `repo/AGENTS.md`",
                    "2) `repo/README.md`",
                    "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md`",
                    "4) `repo/MAPS/ENTRYPOINTS.md`",
                    "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/*/SKILL.md`",
                    "6) `repo/CORTEX/query.py` and `repo/TOOLS/critic.py`",
                    "7) `meta/PACK_INFO.json` (and `meta/REPO_STATE.json` if present)",
                    "8) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`",
                    "",
                ]
            ),
        )
    else:
        write(
            split_dir / f"{scope.file_prefix}-00_INDEX.md",
            "\n".join(
                [
                    f"# {scope.title} Pack Index (SPLIT_LITE)",
                    "",
                    "This directory contains a compressed, discussion-first map of the pack (pointers + indexes).",
                    "",
                    "## Read order",
                    "1) `repo/CATALYTIC-DPT/AGENTS.md`",
                    "2) `repo/CATALYTIC-DPT/README.md`",
                    "3) `repo/CATALYTIC-DPT/ROADMAP_V2.1.md`",
                    "4) `repo/CATALYTIC-DPT/swarm_config.json`",
                    "5) `meta/PACK_INFO.json` (and `meta/REPO_STATE.json` if present)",
                    "6) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`",
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-01_CANON.md",
            "\n".join(
                [
                    "# Canon (SPLIT_LITE)",
                    "",
                    "See `repo/CANON/*` (canonical rules).",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-02_ROOT.md",
            "\n".join(
                [
                    "# Root (SPLIT_LITE)",
                    "",
                    "- `repo/AGENTS.md` (agent procedure)",
                    "- `repo/README.md` (orientation)",
                    "- `repo/.gitignore` (generated artifacts exclusions)",
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-03_MAPS.md",
            "\n".join(
                [
                    "# Maps (SPLIT_LITE)",
                    "",
                    "- Core navigation: `repo/MAPS/ENTRYPOINTS.md`",
                    "- Data flow: `repo/MAPS/DATA_FLOW.md`",
                    "",
                    "## Repo File Tree",
                    "",
                    "See `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`.",
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        # Skills summary table from the repo snapshot (if present in the pack).
        skills_root = pack_dir / "repo" / "SKILLS"
        skill_names = sorted([p.name for p in skills_root.iterdir() if p.is_dir() and not p.name.startswith("_")]) if skills_root.exists() else []
        rows = ["| Skill | Contract | Entrypoint (repo path, may be omitted in LITE) |", "|---|---|---|"]
        for name in skill_names:
            contract = f"`repo/SKILLS/{name}/SKILL.md`"
            entry = f"`repo/SKILLS/{name}/run.py`" if (skills_root / name / "run.py").exists() else f"`repo/SKILLS/{name}/`"
            rows.append(f"| `{name}` | {contract} | {entry} |")
        write(
            split_dir / "AGS-05_SKILLS.md",
            "\n".join(
                [
                    "# Skills (SPLIT_LITE)",
                    "",
                    "LITE ships manifests + pointers; implementations are accessed on demand.",
                    "",
                    "In LITE, `repo/SKILLS/*/SKILL.md` is the required interface. `run.py` / `validate.py` may be omitted.",
                    "If you need implementation details, load them from a FULL (or TEST) pack or from the repo filesystem.",
                    "",
                    "## Skills Table",
                    "",
                    *rows,
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-04_CONTEXT.md",
            "\n".join(
                [
                    "# Context (SPLIT_LITE)",
                    "",
                    "- Decisions: `repo/CONTEXT/decisions/`",
                    "- Preferences: `repo/CONTEXT/preferences/`",
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-06_CONTRACTS.md",
            "\n".join(
                [
                    "# Contracts (SPLIT_LITE)",
                    "",
                    "- Runner: `repo/CONTRACTS/runner.py`",
                    "- LITE may omit fixtures; fixtures live in FULL (or TEST) packs.",
                    "- In LITE, use `meta/FILE_TREE.txt` / `meta/FILE_INDEX.json` for navigation, and `meta/FIXTURE_INDEX.json` if present.",
                    "",
                ]
            ),
        )

    if scope.key == SCOPE_AGS.key:
        write(
            split_dir / "AGS-07_SYSTEM.md",
            "\n".join(
                [
                    "# System (SPLIT_LITE)",
                    "",
                    "LITE is laws + maps + indexes + pointers. Raw code and fixtures may be omitted in LITE.",
                    "When you need full implementation bodies, load them from a FULL (or TEST) pack or from the repo filesystem.",
                    "",
                    "- Cortex query interface: `repo/CORTEX/query.py`",
                    "- Governance critic: `repo/TOOLS/critic.py`",
                    "- MCP server: `repo/MCP/server.py`",
                    "- Packer engine: `repo/MEMORY/LLM_PACKER/Engine/packer.py`",
                    "",
                    "## Meta inventories",
                    "",
                    "- `meta/PACK_INFO.json` (pack metadata)",
                    "- `meta/REPO_STATE.json` (hash inventory; if present)",
                    "- `meta/FILE_TREE.txt` / `meta/FILE_INDEX.json` (navigation)",
                    "",
                ]
            ),
        )

def write_start_here(pack_dir: Path, *, scope: PackScope) -> None:
    if scope.key == SCOPE_AGS.key:
        text = "\n".join(
            [
                "# START HERE",
                "",
                "This snapshot is meant to be shared with any LLM to continue work on the Agent Governance System (AGS) repository.",
                "",
                "## Read order",
                "1) `repo/AGENTS.md` (procedural operating contract)",
                "2) `repo/README.md` and `repo/CONTEXT/archive/planning/INDEX.md` (orientation + planning)",
                "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md` (authority)",
                "4) `repo/MAPS/ENTRYPOINTS.md` (where to change what)",
                "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/` (execution and fixtures)",
                "6) `meta/ENTRYPOINTS.md` (snapshot-specific pointers)",
                "",
                "## Notes",
                "- `BUILD` contents are not included. Only a file tree inventory is captured in `meta/BUILD_TREE.txt`.",
                "- Research under `repo/CONTEXT/research/` is non-binding and opt-in.",
                f"- If `--combined` is enabled, see `COMBINED/{scope.file_prefix}-FULL-COMBINED-*` and `COMBINED/{scope.file_prefix}-FULL-TREEMAP-*`.",
                "",
            ]
        )
    else:
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
                "6) `COMBINED/SPLIT/*` (chunked snapshot)",
                "7) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json` (navigation)",
                "",
                "## Notes",
                f"- If `--combined` is enabled, see `COMBINED/{scope.file_prefix}-FULL-COMBINED-*` and `COMBINED/{scope.file_prefix}-FULL-TREEMAP-*`.",
                "",
            ]
        )
    
    # Add provenance to START_HERE.md
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header, add_header_to_content
        header = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=text
        )
        text = add_header_to_content(text, header, file_type="md")
    except ImportError:
        pass

    (pack_dir / "meta" / "START_HERE.md").write_text(text, encoding="utf-8")


def write_entrypoints(pack_dir: Path, *, scope: PackScope) -> None:
    if scope.key == SCOPE_AGS.key:
        text = "\n".join(
            [
                "# Snapshot Entrypoints",
                "",
                "Key entrypoints for modifying and verifying this repository:",
                "",
                "- `repo/AGENTS.md`",
                "- `repo/README.md`",
                "- `repo/CONTEXT/archive/planning/INDEX.md`",
                "- `repo/CANON/CONTRACT.md`",
                "- `repo/CANON/INVARIANTS.md`",
                "- `repo/CANON/VERSIONING.md`",
                "- `repo/MAPS/ENTRYPOINTS.md`",
                "- `repo/CONTRACTS/runner.py`",
                "- `repo/MEMORY/packer.py`",
                "- `repo/CORTEX/query.py`",
                "",
                "Notes:",
                "- `BUILD` contents are not included. Only `meta/BUILD_TREE.txt` is captured.",
                "- Research under `repo/CONTEXT/research/` has no authority.",
                f"- If `--combined` is enabled, see `COMBINED/{scope.file_prefix}-FULL-COMBINED-*` and `COMBINED/{scope.file_prefix}-FULL-TREEMAP-*`.",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "# Snapshot Entrypoints",
                "",
                f"Key entrypoints for `{scope.title}`:",
                "",
                "- `repo/CATALYTIC-DPT/AGENTS.md`",
                "- `repo/CATALYTIC-DPT/README.md`",
                "- `repo/CATALYTIC-DPT/ROADMAP_V2.1.md`",
                "- `repo/CATALYTIC-DPT/swarm_config.json`",
                "- `repo/CATALYTIC-DPT/CHANGELOG.md`",
                "- `repo/CATALYTIC-DPT/TESTBENCH/`",
                "",
                "Notes:",
                f"- If `--combined` is enabled, see `COMBINED/{scope.file_prefix}-FULL-COMBINED-*` and `COMBINED/{scope.file_prefix}-FULL-TREEMAP-*`.",
                "",
            ]
        )
    
    # Add provenance to ENTRYPOINTS.md
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header, add_header_to_content
        header = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=text
        )
        text = add_header_to_content(text, header, file_type="md")
    except ImportError:
        pass

    (pack_dir / "meta" / "ENTRYPOINTS.md").write_text(text, encoding="utf-8")


def write_build_tree(pack_dir: Path, project_root: Path) -> None:
    build_dir = project_root / "BUILD"
    tree_path = pack_dir / "meta" / "BUILD_TREE.txt"
    if not build_dir.exists():
        tree_path.write_text("BUILD does not exist.\n", encoding="utf-8")
        return
    paths = [
        p.relative_to(project_root).as_posix()
        for p in sorted(build_dir.rglob("*"))
        if p.is_file()
    ]
    tree_path.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")


def write_pack_file_tree_and_index(pack_dir: Path) -> None:
    all_files = [p for p in pack_dir.rglob("*") if p.is_file()]
    rel_paths = sorted(p.relative_to(pack_dir).as_posix() for p in all_files)

    (pack_dir / "meta" / "FILE_TREE.txt").write_text(
        "\n".join(rel_paths) + ("\n" if rel_paths else ""),
        encoding="utf-8",
    )

    file_index: List[Dict[str, Any]] = []
    for p in sorted(all_files, key=lambda x: x.relative_to(pack_dir).as_posix()):
        rel = p.relative_to(pack_dir).as_posix()
        size = p.stat().st_size
        file_index.append(
            {
                "path": rel,
                "bytes": size,
                "sha256": hash_file(p) if size <= 2 * 1024 * 1024 else None,
            }
        )
    write_json(pack_dir / "meta" / "FILE_INDEX.json", file_index)


def write_context_report(pack_dir: Path, *, scope: PackScope) -> Tuple[int, List[str]]:
    """
    Write CONTEXT.txt with token estimates per file and warnings.
    
    Returns:
        (total_tokens, warnings): Total estimated tokens and any warnings.
    """
    warnings: List[str] = []
    
    total_bytes = 0
    total_tokens = 0
    
    # Categorize tokens for smarter warnings and readability
    category_map = {
        "CANON": [],
        "CONTEXT": [],
        "SKILLS": [],
        "CORTEX": [],
        "TOOLS": [],
        "META": [],
        "REPO_ROOT": [],
        "COMBINED": [],
        "OTHER": []
    }
    
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(pack_dir).as_posix()
        size = path.stat().st_size
        tokens = estimate_file_tokens(path)
        
        entry = (rel, size, tokens)
        total_bytes += size
        total_tokens += tokens
        
        # Sort into categories
        if rel.startswith("repo/CANON/"): category_map["CANON"].append(entry)
        elif rel.startswith("repo/CONTEXT/"): category_map["CONTEXT"].append(entry)
        elif rel.startswith("repo/SKILLS/"): category_map["SKILLS"].append(entry)
        elif rel.startswith("repo/CORTEX/"): category_map["CORTEX"].append(entry)
        elif rel.startswith("repo/TOOLS/"): category_map["TOOLS"].append(entry)
        elif rel.startswith("meta/"): category_map["META"].append(entry)
        elif rel.startswith("repo/") and "/" not in rel[5:]: category_map["REPO_ROOT"].append(entry)
        elif rel.startswith("COMBINED/"): category_map["COMBINED"].append(entry)
        else: category_map["OTHER"].append(entry)

    # "Repo+Meta" is the common baseline payload (excludes COMBINED/* outputs).
    repo_meta_tokens = sum(t for rel, _, t in category_map["META"] if rel.startswith("meta/")) + sum(
        t for rel, _, t in category_map["CANON"] + category_map["CONTEXT"] + category_map["SKILLS"] + category_map["CORTEX"] + category_map["TOOLS"] + category_map["REPO_ROOT"] + category_map["OTHER"]
        if rel.startswith("repo/")
    )

    split_tokens = sum(t for rel, _, t in category_map["COMBINED"] if rel.startswith("COMBINED/SPLIT/"))
    split_lite_tokens = sum(t for rel, _, t in category_map["COMBINED"] if rel.startswith("COMBINED/SPLIT_LITE/"))

    combined_files = [
        (rel, tokens)
        for rel, _, tokens in category_map["COMBINED"]
        if rel.startswith(f"COMBINED/{scope.file_prefix}-FULL-COMBINED-") or rel.startswith(f"COMBINED/{scope.file_prefix}-FULL-TREEMAP-")
    ]
    combined_files.sort(key=lambda it: it[0])

    # "Effective" for legacy reporting kept as repo+meta only (no COMBINED/*).
    effective_tokens = repo_meta_tokens
    
    lines: List[str] = [
        f"# {scope.file_prefix} Pack Context Report",
        "",
        "## Payload Token Counts",
        "",
        "These counts are reported per output payload (not summed across all pack outputs).",
        "",
        f"- `repo/` + `meta/` (baseline): {repo_meta_tokens:,} tokens",
        f"- `COMBINED/SPLIT/**` (sum): {split_tokens:,} tokens",
        f"- `COMBINED/SPLIT_LITE/**` (sum): {split_lite_tokens:,} tokens",
    ]

    if combined_files:
        lines.extend(["", "Combined single-file payloads:"])
        for rel, tokens in combined_files:
            lines.append(f"- `{rel}`: {tokens:,} tokens")

    lines.extend(
        [
            "",
            "## Category Summary",
            "",
            f"{'Category':<15} {'Files':>8} {'Tokens':>12} {'% Baseline':>12}",
            "-" * 55,
        ]
    )
    
    for cat, files in category_map.items():
        if not files: continue
        cat_tokens = sum(t for _, _, t in files)
        percent = (cat_tokens / repo_meta_tokens * 100) if repo_meta_tokens > 0 and cat not in ("COMBINED",) else 0
        percent_str = f"{percent:>11.1f}%" if cat != "COMBINED" else "N/A"
        lines.append(f"{cat:<15} {len(files):>8} {cat_tokens:>12,} {percent_str}")

    lines.extend([
        "-" * 55,
        f"{'BASELINE':<15} {'-':>8} {repo_meta_tokens:>12,} {'100.0%':>12}",
        f"{'TOTAL (ALL)':<15} {sum(len(f) for f in category_map.values()):>8} {total_tokens:>12,} {'-':>12}",
        "",
        "## Detailed Breakdown",
        ""
    ])
    
    # Detail sections (only important ones or if not too many)
    for cat in ["CANON", "CONTEXT", "SKILLS", "CORTEX", "TOOLS", "META", "REPO_ROOT"]:
        files = category_map[cat]
        if not files: continue
        
        lines.append(f"### {cat} ({sum(t for _, _, t in files):,} tokens)")
        # If too many files, only show top ones to reduce bloat
        sorted_files = sorted(files, key=lambda x: x[2], reverse=True)
        
        display_limit = 10 if cat != "CANON" else 20
        for i, (rel, _, tokens) in enumerate(sorted_files):
            if i >= display_limit:
                lines.append(f"  ... and {len(files) - display_limit} more files")
                break
            
            # Use short name, but prefix if generic
            filename = rel.split("/")[-1]
            if filename in ["run.py", "validate.py", "SKILL.md", "expected.json", "input.json"]:
                parts = rel.split("/")
                if len(parts) >= 3:
                    filename = f"{parts[-2]}/{filename}"
            
            lines.append(f"- {filename:<45} {tokens:>10,} tokens")
        lines.append("")

    # Add warnings (based on single payload size, not pack-wide totals)
    lines.append("## Status")
    payload_candidates: List[Tuple[str, int]] = [
        ("repo/+meta/ baseline", repo_meta_tokens),
        ("COMBINED/SPLIT/** (sum)", split_tokens),
        ("COMBINED/SPLIT_LITE/** (sum)", split_lite_tokens),
        *[(rel, tokens) for rel, tokens in combined_files],
    ]
    max_name, max_tokens = max(payload_candidates, key=lambda it: it[1]) if payload_candidates else ("(none)", 0)

    if max_tokens > TOKEN_LIMIT_CRITICAL:
        warning = f"[!] CRITICAL: Largest single payload ({max_name}) is {max_tokens:,} tokens (> {TOKEN_LIMIT_CRITICAL:,})."
        warnings.append(warning)
        lines.append(warning)
    elif max_tokens > TOKEN_LIMIT_WARNING:
        warning = f"[!] WARNING: Largest single payload ({max_name}) is {max_tokens:,} tokens (> {TOKEN_LIMIT_WARNING:,})."
        warnings.append(warning)
        lines.append(warning)
    else:
        lines.append(f"[OK] Largest single payload ({max_name}) is {max_tokens:,} tokens (within limits).")
    
    lines.append("")

    report_text = "\n".join(lines)
    
    # Add provenance to CONTEXT.txt
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header, add_header_to_content
        header = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=report_text
        )
        report_text = add_header_to_content(report_text, header, file_type="md")
    except ImportError:
        pass

    (pack_dir / "meta" / "CONTEXT.txt").write_text(report_text, encoding="utf-8")
    return effective_tokens, warnings


def print_payload_token_counts(pack_dir: Path) -> None:
    """
    Print per-payload token counts to stdout.

    This mirrors the `## Payload Token Counts` section in `meta/CONTEXT.txt`.
    """
    report_path = pack_dir / "meta" / "CONTEXT.txt"
    if not report_path.exists():
        return
    text = read_text(report_path)
    lines = text.splitlines()
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Payload Token Counts":
            start_idx = idx
            break
    if start_idx is None:
        return
    out: List[str] = []
    for line in lines[start_idx:]:
        if out and line.startswith("## "):
            break
        out.append(line)
    if not out:
        return
    print("\n".join(out).rstrip() + "\n")


def copy_repo_files(
    pack_dir: Path,
    project_root: Path,
    included_paths: Sequence[str],
) -> None:
    for rel in included_paths:
        src = project_root / rel
        if not src.exists() or not src.is_file():
            continue
        dst = pack_dir / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def ensure_under_packs_root(out_dir: Path) -> Path:
    packs_root = PACKS_ROOT.resolve()
    out_dir_resolved = out_dir.resolve()
    try:
        out_dir_resolved.relative_to(packs_root)
    except ValueError as exc:
        raise ValueError(
            f"OutDir must be under MEMORY/LLM_PACKER/_packs/. Received: {out_dir}"
        ) from exc
    return out_dir_resolved


def default_stamp_for_out_dir(out_dir: Path) -> str:
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})", out_dir.name)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def compute_treemap_text(
    pack_dir: Path,
    *,
    stamp: str,
    include_combined_paths: bool,
    scope: PackScope,
) -> str:
    base_paths = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())

    if not include_combined_paths:
        return build_pack_tree_text(base_paths, extra_paths=[])

    combined_md_rel = f"COMBINED/{scope.file_prefix}-FULL-COMBINED-{stamp}.md"
    combined_txt_rel = f"COMBINED/{scope.file_prefix}-FULL-COMBINED-{stamp}.txt"
    treemap_md_rel = f"COMBINED/{scope.file_prefix}-FULL-TREEMAP-{stamp}.md"
    treemap_txt_rel = f"COMBINED/{scope.file_prefix}-FULL-TREEMAP-{stamp}.txt"

    return build_pack_tree_text(
        base_paths,
        extra_paths=[combined_md_rel, combined_txt_rel, treemap_md_rel, treemap_txt_rel],
    )


def append_repo_tree_to_split_maps(pack_dir: Path, *, tree_text: str, scope: PackScope) -> None:
    if scope.key == SCOPE_AGS.key:
        split_target = pack_dir / "COMBINED" / "SPLIT" / "AGS-03_MAPS.md"
    else:
        split_target = pack_dir / "COMBINED" / "SPLIT" / f"{scope.file_prefix}-00_INDEX.md"
    if not split_target.exists():
        return
    existing = read_text(split_target).rstrip("\n")
    updated = "\n".join(
        [
            existing,
            "",
            "## Repo File Tree",
            "",
            "```",
            tree_text.rstrip("\n"),
            "```",
            "",
        ]
    )
    split_target.write_text(updated, encoding="utf-8")


def write_combined_outputs(pack_dir: Path, *, stamp: str, scope: PackScope) -> None:
    combined_dir = pack_dir / "COMBINED"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_md_rel = f"COMBINED/{scope.file_prefix}-FULL-COMBINED-{stamp}.md"
    combined_txt_rel = f"COMBINED/{scope.file_prefix}-FULL-COMBINED-{stamp}.txt"
    treemap_md_rel = f"COMBINED/{scope.file_prefix}-FULL-TREEMAP-{stamp}.md"
    treemap_txt_rel = f"COMBINED/{scope.file_prefix}-FULL-TREEMAP-{stamp}.txt"

    tree_text = compute_treemap_text(pack_dir, stamp=stamp, include_combined_paths=True, scope=scope)
    tree_md = "\n".join(["# Pack Tree", "", "```", tree_text.rstrip("\n"), "```", ""]) + "\n"

    # Add provenance to treemap outputs
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header, add_header_to_content
        
        # MD treemap
        header_md = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=tree_md
        )
        tree_md = add_header_to_content(tree_md, header_md, file_type="md")
        
        # TXT treemap
        header_txt = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=tree_text
        )
        tree_text = add_header_to_content(tree_text, header_txt, file_type="md")
    except ImportError:
        pass

    (pack_dir / treemap_txt_rel).write_text(tree_text, encoding="utf-8")
    (pack_dir / treemap_md_rel).write_text(tree_md, encoding="utf-8")

    combined_md_lines = [f"# {scope.file_prefix} FULL COMBINED", ""]
    combined_txt_lines = [f"{scope.file_prefix} FULL COMBINED", ""]

    base_paths = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    for rel in base_paths:
        if rel.startswith("COMBINED/"):
            continue
            
        abs_path = pack_dir / rel
        if not abs_path.exists() or not abs_path.is_file():
            continue
        text = read_text(abs_path)
        size = abs_path.stat().st_size
        combined_md_lines.append(build_combined_md_block(rel, text, size))
        combined_txt_lines.append(build_combined_txt_block(rel, text, size))

    md_content = "\n".join(combined_md_lines).rstrip() + "\n"
    txt_content = "\n".join(combined_txt_lines).rstrip() + "\n"
    
    # Add provenance to combined outputs
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header, add_header_to_content
        
        # MD provenance
        header_md = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=md_content
        )
        md_content = add_header_to_content(md_content, header_md, file_type="md")
        
        # TXT provenance
        header_txt = generate_header(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            output_content=txt_content
        )
        txt_content = add_header_to_content(txt_content, header_txt, file_type="md") # txt uses md-style header here
    except ImportError:
        pass

    (pack_dir / combined_md_rel).write_text(md_content, encoding="utf-8")
    (pack_dir / combined_txt_rel).write_text(txt_content, encoding="utf-8")


def write_provenance_manifest(pack_dir: Path) -> None:
    """Generate meta/PROVENANCE.json for the entire pack."""
    meta_dir = pack_dir / "meta"
    
    # Files to include in the manifest
    targets = {
        "meta/FILE_INDEX.json": pack_dir / "meta" / "FILE_INDEX.json",
        "meta/PACK_INFO.json": pack_dir / "meta" / "PACK_INFO.json",
        "meta/REPO_STATE.json": pack_dir / "meta" / "REPO_STATE.json",
        "meta/BUILD_TREE.txt": pack_dir / "meta" / "BUILD_TREE.txt",
        "meta/FILE_TREE.txt": pack_dir / "meta" / "FILE_TREE.txt",
        "meta/CONTEXT.txt": pack_dir / "meta" / "CONTEXT.txt",
    }

    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_manifest, hash_content
        
        manifest = generate_manifest(
            generator="MEMORY/LLM_PACKER/Engine/packer.py",
            target_files=targets,
            inputs=["CANON/", "CONTEXT/decisions/", "SKILLS/", "CORTEX/"],
        )
        
        # Add a self-checksum (excluding the checksum field itself)
        # We handle this by generating the checksum of the sorted JSON
        canonical_json = json.dumps(manifest, sort_keys=True)
        manifest["provenance"]["checksum"] = hash_content(canonical_json)
        
        write_json(meta_dir / "PROVENANCE.json", manifest)
    except ImportError:
        pass


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
) -> Path:
    scope = SCOPES.get(scope_key)
    if not scope:
        raise ValueError(f"Unknown scope: {scope_key}. Choices: {', '.join(sorted(SCOPES.keys()))}")
    if scope.key != SCOPE_AGS.key and profile != "full":
        raise ValueError(f"Only --profile full is supported for scope={scope.key} (received: {profile})")

    manifest, omitted = build_state_manifest(PROJECT_ROOT, scope=scope)
    digest = manifest_digest(manifest)

    if out_dir is None:
        out_dir = PACKS_ROOT / f"llm-pack-{scope.key}-{digest[:12]}"
    out_dir = ensure_under_packs_root(out_dir)

    baseline_path = baseline_path_for_scope(scope)
    baseline = load_baseline(baseline_path)
    baseline_files_by_path = {f["path"]: f for f in (baseline or {}).get("files", [])}
    current_files_by_path = {f["path"]: f for f in manifest.get("files", [])}

    anchors = set(scope.anchors)

    if mode == "delta" and baseline is not None:
        changed = []
        for path, entry in current_files_by_path.items():
            prev = baseline_files_by_path.get(path)
            if prev is None or prev.get("hash") != entry.get("hash") or prev.get("size") != entry.get("size"):
                changed.append(path)
        deleted = sorted(set(baseline_files_by_path.keys()) - set(current_files_by_path.keys()))
        include_paths = sorted(set(changed) | anchors)
    else:
        include_paths = sorted(set(current_files_by_path.keys()) | anchors)
        deleted = []

    omitted_paths_for_lite: List[str] = []
    if profile == "lite":
        lite_include: List[str] = []
        for rel in include_paths:
            if rel.endswith((".cmd", ".ps1")):
                omitted_paths_for_lite.append(rel)
                continue
            if "/fixtures/" in rel:
                omitted_paths_for_lite.append(rel)
                continue
            if "/_runs/" in rel or "/_generated/" in rel:
                omitted_paths_for_lite.append(rel)
                continue
            if rel.startswith("CONTEXT/research/") or rel.startswith("CONTEXT/archive/"):
                omitted_paths_for_lite.append(rel)
                continue
            if "/_packs/" in rel and rel.startswith("MEMORY/"):
                omitted_paths_for_lite.append(rel)
                continue

            if rel == "AGENTS.md":
                lite_include.append(rel)
            elif rel == "README.md":
                lite_include.append(rel)
            elif rel.startswith("CANON/"):
                lite_include.append(rel)
            elif rel.startswith("MAPS/"):
                lite_include.append(rel)
            elif rel == "CONTRACTS/runner.py":
                lite_include.append(rel)
            elif rel == "CORTEX/query.py":
                lite_include.append(rel)
            elif rel == "TOOLS/critic.py":
                lite_include.append(rel)
            elif rel.startswith("SKILLS/") and (rel.endswith("/SKILL.md") or rel.endswith("/version.json")):
                lite_include.append(rel)
            else:
                omitted_paths_for_lite.append(rel)
        include_paths = sorted(set(lite_include))
    elif profile != "full":
        raise ValueError(f"Unknown profile: {profile}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "repo").mkdir(parents=True, exist_ok=True)

    repo_pack_paths = [f"repo/{p}" for p in include_paths]

    copy_repo_files(out_dir, PROJECT_ROOT, include_paths)
    write_json(out_dir / "meta" / "REPO_OMITTED_BINARIES.json", omitted)
    write_json(out_dir / "meta" / "REPO_STATE.json", manifest)
    write_json(
        out_dir / "meta" / "PACK_INFO.json",
        {
            "scope": scope.key,
            "mode": mode,
            **({"profile": profile} if profile != "full" else {}),
            "canon_version": manifest.get("canon_version"),
            "grammar_version": manifest.get("grammar_version"),
            "repo_digest": digest,
            "included_paths": include_paths,
            "deleted_paths": deleted,
        },
    )
    write_build_tree(out_dir, PROJECT_ROOT)
    write_start_here(out_dir, scope=scope)
    write_entrypoints(out_dir, scope=scope)
    write_split_pack(out_dir, repo_pack_paths, scope=scope)
    if split_lite:
        write_split_pack_lite(out_dir, scope=scope)

    effective_stamp = stamp or default_stamp_for_out_dir(out_dir)
    tree_text = compute_treemap_text(out_dir, stamp=effective_stamp, include_combined_paths=bool(combined), scope=scope)
    append_repo_tree_to_split_maps(out_dir, tree_text=tree_text, scope=scope)

    if combined:
        write_combined_outputs(out_dir, stamp=effective_stamp, scope=scope)

    if profile == "lite" and scope.key == SCOPE_AGS.key:
        write_lite_indexes(
            out_dir,
            project_root=PROJECT_ROOT,
            include_paths=include_paths,
            omitted_paths=omitted_paths_for_lite,
            files_by_path=current_files_by_path,
        )

    write_pack_file_tree_and_index(out_dir)
    
    # Generate token context report with warnings
    total_tokens, token_warnings = write_context_report(out_dir, scope=scope)
    print_payload_token_counts(out_dir)
    for warning in token_warnings:
        color = ANSI_RED if "CRITICAL" in warning else ANSI_YELLOW
        print(f"{color}{warning}{ANSI_RESET}")

    # Generate PROVENANCE.json manifest
    write_provenance_manifest(out_dir)

    write_json(baseline_path, manifest)

    if zip_enabled:
        archive_dir = PACKS_ROOT / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        zip_path = archive_dir / f"{out_dir.name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_dir)

    return out_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create memory/LLM packs under MEMORY/LLM_PACKER/_packs/."
    )
    parser.add_argument(
        "--scope",
        choices=tuple(sorted(SCOPES.keys())),
        default=SCOPE_AGS.key,
        help="What to pack: default is the full AGS repo; catalytic-dpt packs only CATALYTIC-DPT/.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "delta"),
        default="full",
        help="Pack mode: full includes all included text files; delta includes only changes since last baseline plus anchors.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "lite"),
        default="full",
        help="Pack profile: full is record-keeping; lite is discussion-first (contracts + interfaces + symbolic indexes).",
    )
    parser.add_argument(
        "--split-lite",
        action="store_true",
        help="Also write COMBINED/SPLIT_LITE/** alongside COMBINED/SPLIT/** in the same pack.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for the pack, relative to the repo root and under MEMORY/LLM_PACKER/_packs/.",
    )
    parser.add_argument("--combined", action="store_true", help="Write COMBINED/FULL-COMBINED-* and COMBINED/FULL-TREEMAP-* outputs.")
    parser.add_argument(
        "--stamp",
        default="",
        help="Stamp string for COMBINED output filenames. Defaults to a timestamp or to one parsed from the out_dir name.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Write a zip archive under MEMORY/LLM_PACKER/_packs/archive/.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    pack_dir = make_pack(
        scope_key=args.scope,
        mode=args.mode,
        profile=args.profile,
        split_lite=bool(args.split_lite),
        out_dir=out_dir,
        combined=bool(args.combined),
        stamp=args.stamp or None,
        zip_enabled=bool(args.zip),
    )
    print(f"Pack created: {pack_dir}")
