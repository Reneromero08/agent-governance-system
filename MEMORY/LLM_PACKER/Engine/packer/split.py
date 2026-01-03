"""
SPLIT output generation (Phase 1).

Output target: pack_dir/SPLIT/ 
FORBIDDEN: Any reference to COMBINED/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_LAB, read_text

def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()

def choose_fence(text: str) -> str:
    """Choose a fence that doesn't conflict with existing fences in text."""
    if "````" in text:
        return "`````"
    if "```" in text:
        return "````"
    return "```"

def write_split_pack(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope) -> None:
    """Dispatch split pack writing based on scope."""
    if scope.key == SCOPE_AGS.key:
        write_split_pack_ags(pack_dir, included_repo_paths)
    elif scope.key == SCOPE_LAB.key:
        write_split_pack_lab(pack_dir, included_repo_paths, scope=scope)
    else:
        raise ValueError(f"Unsupported scope for split pack: {scope.key}")

def write_split_pack_ags(pack_dir: Path, included_repo_paths: Sequence[str]) -> None:
    # Target: SPLIT/ directory
    split_dir = pack_dir / "SPLIT"
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

    # Group paths (6-bucket structure + optional root files)
    law_paths = [p for p in included_repo_paths if p.startswith("repo/LAW/")]
    capability_paths = [p for p in included_repo_paths if p.startswith("repo/CAPABILITY/")]
    navigation_paths = [p for p in included_repo_paths if p.startswith("repo/NAVIGATION/")]
    direction_paths = [p for p in included_repo_paths if p.startswith("repo/DIRECTION/")]
    thought_paths = [p for p in included_repo_paths if p.startswith("repo/THOUGHT/")]
    memory_paths = [p for p in included_repo_paths if p.startswith("repo/MEMORY/")]
    root_paths = [p for p in included_repo_paths if p.startswith("repo/") and p.count("/") == 1]
    github_paths = [p for p in included_repo_paths if p.startswith("repo/.github/")]

    meta_dir = pack_dir / "meta"
    meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()]) if meta_dir.exists() else []

    canon_contract = rel_posix("LAW", "CANON", "CONTRACT.md")
    canon_invariants = rel_posix("LAW", "CANON", "INVARIANTS.md")
    canon_versioning = rel_posix("LAW", "CANON", "VERSIONING.md")
    maps_entrypoints = rel_posix("NAVIGATION", "MAPS", "ENTRYPOINTS.md")
    contracts_runner = rel_posix("LAW", "CONTRACTS", "runner.py")
    skills_dir = rel_posix("CAPABILITY", "SKILLS")
    cortex_dir = rel_posix("NAVIGATION", "CORTEX")
    tools_dir = rel_posix("CAPABILITY", "TOOLS")

    # Write Index (NO COMBINED references)
    (split_dir / "AGS-00_INDEX.md").write_text(
        "\n".join(
            [
                "# AGS Pack Index",
                "",
                "This directory contains a generated snapshot of the repository intended for LLM handoff.",
                "",
                "## Read order",
                "1) `repo/AGENTS.md`",
                "2) `repo/README.md`",
                f"3) `repo/{canon_contract}` and `repo/{canon_invariants}` and `repo/{canon_versioning}`",
                f"4) `repo/{maps_entrypoints}`",
                f"5) `repo/{contracts_runner}` and `repo/{skills_dir}/`",
                f"6) `repo/{cortex_dir}/` and `repo/{tools_dir}/`",
                "7) `meta/ENTRYPOINTS.md` and `meta/PACK_INFO.json` (Snapshot specific)",
                "",
                "## Notes",
                "- `BUILD` contents are excluded.",
                "- Single-file bundles available in `FULL/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / "AGS-01_LAW.md").write_text("# LAW\n\n" + section(law_paths), encoding="utf-8")
    (split_dir / "AGS-02_CAPABILITY.md").write_text("# CAPABILITY\n\n" + section(capability_paths), encoding="utf-8")
    (split_dir / "AGS-03_NAVIGATION.md").write_text("# NAVIGATION\n\n" + section(navigation_paths), encoding="utf-8")
    direction_body = section(direction_paths)
    if direction_body.strip():
        (split_dir / "AGS-04_DIRECTION.md").write_text("# DIRECTION\n\n" + direction_body, encoding="utf-8")
    thought_body = section(thought_paths)
    if thought_body.strip():
        (split_dir / "AGS-05_THOUGHT.md").write_text("# THOUGHT\n\n" + thought_body, encoding="utf-8")
    (split_dir / "AGS-06_MEMORY.md").write_text("# MEMORY\n\n" + section(memory_paths), encoding="utf-8")
    (split_dir / "AGS-07_ROOT_FILES.md").write_text(
        "# ROOT_FILES\n\n" + section([*root_paths, *github_paths, *meta_paths]),
        encoding="utf-8",
    )

def write_split_pack_lab(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope) -> None:
    split_dir = pack_dir / "SPLIT"
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

    def is_lab(path: str) -> bool:
        return path.startswith("repo/THOUGHT/LAB/")

    meta_dir = pack_dir / "meta"
    meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()]) if meta_dir.exists() else []

    lab_paths = [p for p in included_repo_paths if is_lab(p)]
    docs_paths = sorted([p for p in lab_paths if p.lower().endswith((".md", ".txt"))])
    system_paths = sorted([p for p in lab_paths if p not in set(docs_paths)])

    (split_dir / "LAB-00_INDEX.md").write_text(
        "\n".join(
            [
                "# LAB Pack Index",
                "",
                "This directory contains a generated snapshot intended for LLM handoff.",
                "",
                "## Read order",
                "1) `repo/THOUGHT/LAB/`",
                "2) `meta/ENTRYPOINTS.md` and `meta/PACK_INFO.json`",
                "",
                "## Notes",
                "- See `FULL/` for single-file bundles.",
                "- This scope is volatile and may change without notice.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / "LAB-01_DOCS.md").write_text("# Docs\n\n" + section(docs_paths), encoding="utf-8")
    (split_dir / "LAB-02_SYSTEM.md").write_text("# System\n\n" + section([*system_paths, *meta_paths]), encoding="utf-8")
