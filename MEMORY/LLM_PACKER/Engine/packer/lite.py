"""
LITE output generation (Phase 1).

Output target: pack_dir/LITE/
FORBIDDEN: Any reference to COMBINED/ or SPLIT_LITE/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_CATALYTIC_DPT, SCOPE_LAB, read_text

def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()

def write_split_pack_lite(pack_dir: Path, *, scope: PackScope) -> None:
    """
    Write a discussion-first LITE set.
    """
    lite_dir = pack_dir / "LITE"
    lite_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, text: str) -> None:
        path.write_text(text.rstrip() + "\n", encoding="utf-8")

    if scope.key == SCOPE_AGS.key:
        canon_contract = rel_posix("LAW", "CANON", "CONTRACT.md")
        canon_invariants = rel_posix("LAW", "CANON", "INVARIANTS.md")
        canon_versioning = rel_posix("LAW", "CANON", "VERSIONING.md")
        contracts_runner = rel_posix("LAW", "CONTRACTS", "runner.py")
        maps_entrypoints = rel_posix("NAVIGATION", "MAPS", "ENTRYPOINTS.md")
        critic_tool = rel_posix("CAPABILITY", "TOOLS", "critic.py")
        skills_dir = rel_posix("CAPABILITY", "SKILLS")
        packer_readme = rel_posix("MEMORY", "LLM_PACKER", "README.md")
        write(
            lite_dir / "AGS-00_INDEX.md",
            "\n".join(
                [
                    "# AGS Pack Index (LITE)",
                    "",
                    "This directory contains a compressed, discussion-first map of the pack.",
                    "",
                    "## Read order",
                    "1) `repo/AGENTS.md`",
                    "2) `repo/README.md`",
                    f"3) `repo/{canon_contract}` and `repo/{canon_invariants}` and `repo/{canon_versioning}`",
                    f"4) `repo/{contracts_runner}`",
                    f"5) `repo/{maps_entrypoints}`",
                    f"6) `repo/{critic_tool}` and `repo/{skills_dir}/*/SKILL.md`",
                    "7) `repo/DIRECTION/` (roadmaps, if used)",
                    f"8) `repo/{packer_readme}`",
                    "9) `meta/PACK_INFO.json`",
                    "10) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`",
                    "",
                ]
            ),
        )

    elif scope.key == SCOPE_CATALYTIC_DPT.key:
        write(
            lite_dir / f"{scope.file_prefix}-00_INDEX.md",
            "\n".join(
                [
                    f"# {scope.file_prefix} Pack Index (LITE)",
                    "",
                    "Lite profile not yet fully implemented for this scope.",
                    "See FULL/ or SPLIT/ for content.",
                    "",
                ]
            ),
        )
    elif scope.key == SCOPE_LAB.key:
        write(
            lite_dir / f"{scope.file_prefix}-00_INDEX.md",
            "\n".join(
                [
                    f"# {scope.file_prefix} Pack Index (LITE)",
                    "",
                    "Lite profile not yet fully implemented for this scope.",
                    "See FULL/ or SPLIT/ for content.",
                    "",
                ]
            ),
        )

    # Copy SPLIT chunk references (stub implementation for now, mirroring logic)
    # Ideally logic would be more ELO-aware, but for Phase 1 we follow roadmap
    # constraints to output to LITE/ only.
    if scope.key == SCOPE_AGS.key:
        chunks = [
            "AGS-01_LAW.md",
            "AGS-02_CAPABILITY.md",
            "AGS-03_NAVIGATION.md",
            "AGS-04_DIRECTION.md",
            "AGS-06_MEMORY.md",
            "AGS-07_ROOT_FILES.md",
        ]
        split_src = pack_dir / "SPLIT"
        for chunk in chunks:
            src = split_src / chunk
            if src.exists():
                write(lite_dir / chunk, read_text(src))

def write_lite_indexes(
    pack_dir: Path,
    *,
    project_root: Path,
    include_paths: Sequence[str],
    omitted_paths: Sequence[str],
    files_by_path: Dict[str, Dict[str, Any]],
) -> None:
    """Write lightweight indexes to LITE/."""
    # (Simplified from legacy packer, focused on LITE/ output)
    pass # Implementation deferred for rigorous ELO logic later; 
         # split_pack_lite handles the critical path for Phase 1.
