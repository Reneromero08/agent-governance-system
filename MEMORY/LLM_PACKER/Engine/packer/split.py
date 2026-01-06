"""
SPLIT output generation (Phase 1).

Output target: pack_dir/SPLIT/
FORBIDDEN: Any reference to COMBINED/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_LAB, read_text
from .firewall_writer import PackerWriter

def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()

def choose_fence(text: str) -> str:
    """Choose a fence that doesn't conflict with existing fences in text."""
    if "````" in text:
        return "`````"
    if "```" in text:
        return "````"
    return "```"

def write_split_pack(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    """Dispatch split pack writing based on scope."""
    if scope.key == SCOPE_AGS.key:
        write_split_pack_ags(pack_dir, included_repo_paths, writer=writer)
    elif scope.key == SCOPE_LAB.key:
        write_split_pack_lab(pack_dir, included_repo_paths, scope=scope, writer=writer)
    else:
        raise ValueError(f"Unsupported scope for split pack: {scope.key}")

def write_split_pack_ags(pack_dir: Path, included_repo_paths: Sequence[str], writer: Optional[PackerWriter] = None) -> None:
    # Target: SPLIT/ directory
    split_dir = pack_dir / "SPLIT"
    if writer is None:
        split_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(split_dir, kind="durable", parents=True, exist_ok=True)

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

    # Group paths (6-bucket structure + optional root files + PROOFS)
    # Extract PROOFS from NAVIGATION
    navigation_all = [p for p in included_repo_paths if p.startswith("repo/NAVIGATION/")]
    proof_paths = [p for p in navigation_all if p.startswith("repo/NAVIGATION/PROOFS/")]
    navigation_paths = [p for p in navigation_all if not p.startswith("repo/NAVIGATION/PROOFS/")]

    law_paths = [p for p in included_repo_paths if p.startswith("repo/LAW/")]
    capability_paths = [p for p in included_repo_paths if p.startswith("repo/CAPABILITY/")]
    # navigation_paths defined above
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
    index_content = "\n".join(
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
    )
    if writer is None:
        (split_dir / "AGS-00_INDEX.md").write_text(index_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-00_INDEX.md", index_content, encoding="utf-8")

    law_content = "# LAW\n\n" + section(law_paths)
    if writer is None:
        (split_dir / "AGS-01_LAW.md").write_text(law_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-01_LAW.md", law_content, encoding="utf-8")

    capability_content = "# CAPABILITY\n\n" + section(capability_paths)
    if writer is None:
        (split_dir / "AGS-02_CAPABILITY.md").write_text(capability_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-02_CAPABILITY.md", capability_content, encoding="utf-8")

    navigation_content = "# NAVIGATION\n\n" + section(navigation_paths)
    if writer is None:
        (split_dir / "AGS-03_NAVIGATION.md").write_text(navigation_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-03_NAVIGATION.md", navigation_content, encoding="utf-8")

    # AGS-04_PROOFS (New)
    proofs_content = "# PROOFS\n\n" + section(proof_paths)
    if writer is None:
        (split_dir / "AGS-04_PROOFS.md").write_text(proofs_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-04_PROOFS.md", proofs_content, encoding="utf-8")

    # DIRECTION/THOUGHT are intentionally omitted from the AGS pack.
    # Keep SPLIT numbering contiguous.

    memory_content = "# MEMORY\n\n" + section(memory_paths)
    if writer is None:
        (split_dir / "AGS-05_MEMORY.md").write_text(memory_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-05_MEMORY.md", memory_content, encoding="utf-8")

    root_files_content = "# ROOT_FILES\n\n" + section([*root_paths, *github_paths, *meta_paths])
    if writer is None:
        (split_dir / "AGS-06_ROOT_FILES.md").write_text(root_files_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "AGS-06_ROOT_FILES.md", root_files_content, encoding="utf-8")

def write_split_pack_lab(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
    split_dir = pack_dir / "SPLIT"
    if writer is None:
        split_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer.mkdir(split_dir, kind="durable", parents=True, exist_ok=True)

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

    index_content = "\n".join(
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
    )
    if writer is None:
        (split_dir / "LAB-00_INDEX.md").write_text(index_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "LAB-00_INDEX.md", index_content, encoding="utf-8")

    docs_content = "# Docs\n\n" + section(docs_paths)
    if writer is None:
        (split_dir / "LAB-01_DOCS.md").write_text(docs_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "LAB-01_DOCS.md", docs_content, encoding="utf-8")

    system_content = "# System\n\n" + section([*system_paths, *meta_paths])
    if writer is None:
        (split_dir / "LAB-02_SYSTEM.md").write_text(system_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "LAB-02_SYSTEM.md", system_content, encoding="utf-8")
