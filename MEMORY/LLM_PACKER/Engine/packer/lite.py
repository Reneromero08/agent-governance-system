"""
LITE output generation (Phase 1).

Output target: pack_dir/LITE/
FORBIDDEN: Any reference to COMBINED/ or SPLIT_LITE/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_CATALYTIC_DPT, SCOPE_LAB, read_text

def write_split_pack_lite(pack_dir: Path, *, scope: PackScope) -> None:
    """
    Write a discussion-first LITE set.
    """
    lite_dir = pack_dir / "LITE"
    lite_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, text: str) -> None:
        path.write_text(text.rstrip() + "\n", encoding="utf-8")

    if scope.key == SCOPE_AGS.key:
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
                    "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md`",
                    "4) `repo/MAPS/ENTRYPOINTS.md`",
                    "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/*/SKILL.md`",
                    "6) `repo/CORTEX/query.py` and `repo/TOOLS/critic.py`",
                    "7) `meta/PACK_INFO.json`",
                    "8) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`",
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
            "AGS-01_CANON.md",
            "AGS-02_ROOT.md", 
            "AGS-03_MAPS.md",
            "AGS-06_CONTRACTS.md"
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
