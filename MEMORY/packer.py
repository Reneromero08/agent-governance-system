#!/usr/bin/env python3

"""
Packer for AGS memory packs.

This script produces two related artifacts:

1) A repository state manifest (hashes + sizes) used to compare snapshots and drive delta
   packs.
2) A shareable "LLM pack" directory with curated entrypoints, indices and optional combined
   markdown suitable for handoff to another model.

All outputs are written under `MEMORY/LLM-PACKER-1.1/_packs/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = Path(__file__).resolve().parent
LLM_PACKER_DIR = MEMORY_DIR / "LLM-PACKER-1.1"
PACKS_ROOT = LLM_PACKER_DIR / "_packs"
STATE_DIR = PACKS_ROOT / "_state"
BASELINE_PATH = STATE_DIR / "baseline.json"

INCLUDE_DIRS = (
    "CANON",
    "CONTEXT",
    "MAPS",
    "SKILLS",
    "CONTRACTS",
    "MEMORY",
    "CORTEX",
    "TOOLS",
    ".github",
)

ROOT_FILES = (
    "README.md",
    "ROADMAP.md",
    "LICENSE",
    "AGENTS.md",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
)

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

EXCLUDED_DIR_PARTS = {
    ".git",
    "BUILD",
    "_runs",
    "_generated",
    "_packs",
    "Original",
    "ORIGINAL",
    "__pycache__",
    "node_modules",
}

CANON_VERSION_FILE = PROJECT_ROOT / "CANON" / "VERSIONING.md"

def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_canon_version() -> str:
    if not CANON_VERSION_FILE.exists():
        return "unknown"
    text = read_text(CANON_VERSION_FILE)
    match = re.search(r"canon_version:\s*([0-9]+\\.[0-9]+\\.[0-9]+)", text)
    return match.group(1) if match else "unknown"


def is_text_path(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext:
        return ext in TEXT_EXTENSIONS
    return path.name in TEXT_BASENAMES


def is_excluded_rel_path(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if parts & EXCLUDED_DIR_PARTS:
        return True
    # Allow `.github/` but avoid other hidden folders by default.
    if any(part.startswith(".") and part != ".github" for part in rel_path.parts):
        return True
    return False


def iter_repo_candidates(project_root: Path) -> Iterable[Path]:
    for directory in INCLUDE_DIRS:
        base = project_root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(project_root)
            if is_excluded_rel_path(rel):
                continue
            yield path

    for file_name in ROOT_FILES:
        path = project_root / file_name
        if path.exists() and path.is_file():
            yield path


def build_state_manifest(project_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    canon_version = read_canon_version()
    files: List[Dict[str, Any]] = []
    omitted: List[Dict[str, Any]] = []

    seen: set[str] = set()
    for abs_path in iter_repo_candidates(project_root):
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
    manifest: Dict[str, Any] = {"canon_version": canon_version, "files": files}
    return manifest, omitted


def manifest_digest(manifest: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for entry in manifest.get("files", []):
        line = f"{entry['hash']} {entry['size']} {entry['path']}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def load_baseline() -> Optional[Dict[str, Any]]:
    if not BASELINE_PATH.exists():
        return None
    try:
        return json.loads(read_text(BASELINE_PATH))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def write_split_pack(pack_dir: Path, included_repo_paths: Sequence[str]) -> None:
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

    (split_dir / "00_INDEX.md").write_text(
        "\n".join(
            [
                "# AGS Pack Index",
                "",
                "This directory contains a generated snapshot of the repository intended for LLM handoff.",
                "",
                "## Read order",
                "1) `repo/AGENTS.md`",
                "2) `repo/README.md` and `repo/ROADMAP.md`",
                "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md`",
                "4) `repo/MAPS/ENTRYPOINTS.md`",
                "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/`",
                "6) `meta/ENTRYPOINTS.md`",
                "",
                "## Notes",
                "- `BUILD/` contents are not included. Only a file tree inventory is captured in `meta/BUILD_TREE.txt`.",
                "- Research under `repo/CONTEXT/research/` is non-binding and opt-in.",
                "- If `--combined` is enabled, `COMBINED/` contains `FULL-COMBINED-*` and `FULL-TREEMAP-*` outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / "01_CANON.md").write_text("# Canon\n\n" + section(canon_paths), encoding="utf-8")
    (split_dir / "02_ROOT.md").write_text("# Root\n\n" + section(root_paths), encoding="utf-8")
    (split_dir / "03_MAPS.md").write_text("# Maps\n\n" + section(maps_paths), encoding="utf-8")
    (split_dir / "04_CONTEXT.md").write_text("# Context\n\n" + section(context_paths), encoding="utf-8")
    (split_dir / "05_SKILLS.md").write_text("# Skills\n\n" + section(skills_paths), encoding="utf-8")
    (split_dir / "06_CONTRACTS.md").write_text("# Contracts\n\n" + section(contracts_paths), encoding="utf-8")
    (split_dir / "07_SYSTEM.md").write_text(
        "# System\n\n" + section([*cortex_paths, *memory_paths, *tools_paths, *github_paths]),
        encoding="utf-8",
    )


def write_start_here(pack_dir: Path) -> None:
    text = "\n".join(
        [
            "# START HERE",
            "",
            "This snapshot is meant to be shared with any LLM to continue work on the Agent Governance System (AGS) repository.",
            "",
            "## Read order",
            "1) `repo/AGENTS.md` (procedural operating contract)",
            "2) `repo/README.md` and `repo/ROADMAP.md` (orientation)",
            "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md` (authority)",
            "4) `repo/MAPS/ENTRYPOINTS.md` (where to change what)",
            "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/` (execution and fixtures)",
            "6) `meta/ENTRYPOINTS.md` (snapshot-specific pointers)",
            "",
            "## Notes",
            "- `BUILD/` contents are not included. Only a file tree inventory is captured in `meta/BUILD_TREE.txt`.",
            "- Research under `repo/CONTEXT/research/` is non-binding and opt-in.",
            "- If `--combined` is enabled, see `COMBINED/FULL-COMBINED-*` and `COMBINED/FULL-TREEMAP-*`.",
            "",
        ]
    )
    (pack_dir / "meta" / "START_HERE.md").write_text(text, encoding="utf-8")


def write_entrypoints(pack_dir: Path) -> None:
    text = "\n".join(
        [
            "# Snapshot Entrypoints",
            "",
            "Key entrypoints for modifying and verifying this repository:",
            "",
            "- `repo/AGENTS.md`",
            "- `repo/README.md`",
            "- `repo/ROADMAP.md`",
            "- `repo/CANON/CONTRACT.md`",
            "- `repo/CANON/INVARIANTS.md`",
            "- `repo/CANON/VERSIONING.md`",
            "- `repo/MAPS/ENTRYPOINTS.md`",
            "- `repo/CONTRACTS/runner.py`",
            "- `repo/MEMORY/packer.py`",
            "- `repo/CORTEX/query.py`",
            "",
            "Notes:",
            "- `BUILD/` contents are not included. Only `meta/BUILD_TREE.txt` is captured.",
            "- Research under `repo/CONTEXT/research/` has no authority.",
            "- If `--combined` is enabled, see `COMBINED/FULL-COMBINED-*` and `COMBINED/FULL-TREEMAP-*`.",
            "",
        ]
    )
    (pack_dir / "meta" / "ENTRYPOINTS.md").write_text(text, encoding="utf-8")


def write_build_tree(pack_dir: Path, project_root: Path) -> None:
    build_dir = project_root / "BUILD"
    tree_path = pack_dir / "meta" / "BUILD_TREE.txt"
    if not build_dir.exists():
        tree_path.write_text("BUILD/ does not exist.\n", encoding="utf-8")
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
            f"OutDir must be under MEMORY/LLM-PACKER-1.1/_packs/. Received: {out_dir}"
        ) from exc
    return out_dir_resolved


def default_stamp_for_out_dir(out_dir: Path) -> str:
    match = re.search(r"(\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})", out_dir.name)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def compute_treemap_text(pack_dir: Path, *, stamp: str, include_combined_paths: bool) -> str:
    base_paths = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())

    if not include_combined_paths:
        return build_pack_tree_text(base_paths, extra_paths=[])

    combined_md_rel = f"COMBINED/FULL-COMBINED-{stamp}.md"
    combined_txt_rel = f"COMBINED/FULL-COMBINED-{stamp}.txt"
    treemap_md_rel = f"COMBINED/FULL-TREEMAP-{stamp}.md"
    treemap_txt_rel = f"COMBINED/FULL-TREEMAP-{stamp}.txt"

    return build_pack_tree_text(
        base_paths,
        extra_paths=[combined_md_rel, combined_txt_rel, treemap_md_rel, treemap_txt_rel],
    )


def append_repo_tree_to_split_maps(pack_dir: Path, *, tree_text: str) -> None:
    split_maps_path = pack_dir / "COMBINED" / "SPLIT" / "03_MAPS.md"
    if not split_maps_path.exists():
        return
    existing = read_text(split_maps_path).rstrip("\n")
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
    split_maps_path.write_text(updated, encoding="utf-8")


def write_combined_outputs(pack_dir: Path, *, stamp: str) -> None:
    combined_dir = pack_dir / "COMBINED"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_md_rel = f"COMBINED/FULL-COMBINED-{stamp}.md"
    combined_txt_rel = f"COMBINED/FULL-COMBINED-{stamp}.txt"
    treemap_md_rel = f"COMBINED/FULL-TREEMAP-{stamp}.md"
    treemap_txt_rel = f"COMBINED/FULL-TREEMAP-{stamp}.txt"

    tree_text = compute_treemap_text(pack_dir, stamp=stamp, include_combined_paths=True)

    (pack_dir / treemap_txt_rel).write_text(tree_text, encoding="utf-8")
    (pack_dir / treemap_md_rel).write_text(
        "\n".join(["# Pack Tree", "", "```", tree_text.rstrip("\n"), "```", ""]) + "\n",
        encoding="utf-8",
    )

    combined_md_lines = ["# FULL COMBINED", ""]
    combined_txt_lines = ["FULL COMBINED", ""]

    base_paths = sorted(p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*") if p.is_file())
    for rel in base_paths:
        abs_path = pack_dir / rel
        if not abs_path.exists() or not abs_path.is_file():
            continue
        text = read_text(abs_path)
        size = abs_path.stat().st_size
        combined_md_lines.append(build_combined_md_block(rel, text, size))
        combined_txt_lines.append(build_combined_txt_block(rel, text, size))

    (pack_dir / combined_md_rel).write_text("\n".join(combined_md_lines).rstrip() + "\n", encoding="utf-8")
    (pack_dir / combined_txt_rel).write_text("\n".join(combined_txt_lines).rstrip() + "\n", encoding="utf-8")


def make_pack(
    *,
    mode: str,
    out_dir: Optional[Path],
    combined: bool,
    stamp: Optional[str],
    zip_enabled: bool,
) -> Path:
    manifest, omitted = build_state_manifest(PROJECT_ROOT)
    digest = manifest_digest(manifest)

    if out_dir is None:
        out_dir = PACKS_ROOT / f"llm-pack-{digest[:12]}"
    out_dir = ensure_under_packs_root(out_dir)

    baseline = load_baseline()
    baseline_files_by_path = {f["path"]: f for f in (baseline or {}).get("files", [])}
    current_files_by_path = {f["path"]: f for f in manifest.get("files", [])}

    anchors = {
        "AGENTS.md",
        "README.md",
        "ROADMAP.md",
        "CANON/CONTRACT.md",
        "CANON/INVARIANTS.md",
        "CANON/VERSIONING.md",
        "MAPS/ENTRYPOINTS.md",
        "CONTRACTS/runner.py",
        "MEMORY/packer.py",
    }

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
            "mode": mode,
            "canon_version": manifest.get("canon_version"),
            "repo_digest": digest,
            "included_paths": include_paths,
            "deleted_paths": deleted,
        },
    )
    write_build_tree(out_dir, PROJECT_ROOT)
    write_start_here(out_dir)
    write_entrypoints(out_dir)
    write_split_pack(out_dir, repo_pack_paths)

    effective_stamp = stamp or default_stamp_for_out_dir(out_dir)
    tree_text = compute_treemap_text(out_dir, stamp=effective_stamp, include_combined_paths=bool(combined))
    append_repo_tree_to_split_maps(out_dir, tree_text=tree_text)

    if combined:
        write_combined_outputs(out_dir, stamp=effective_stamp)

    write_pack_file_tree_and_index(out_dir)

    write_json(BASELINE_PATH, manifest)

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
        description="Create AGS memory/LLM packs under MEMORY/LLM-PACKER-1.1/_packs/."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "delta"),
        default="full",
        help="Pack mode: full includes all included text files; delta includes only changes since last baseline plus anchors.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for the pack, relative to the repo root and under MEMORY/LLM-PACKER-1.1/_packs/.",
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
        help="Write a zip archive under MEMORY/LLM-PACKER-1.1/_packs/archive/.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    pack_dir = make_pack(
        mode=args.mode,
        out_dir=out_dir,
        combined=bool(args.combined),
        stamp=args.stamp or None,
        zip_enabled=bool(args.zip),
    )
    print(f"Pack created: {pack_dir}")
