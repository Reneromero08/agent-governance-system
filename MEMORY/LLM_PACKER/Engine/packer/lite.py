"""
LITE output generation (Phase 1).

Output target: pack_dir/LITE/
FORBIDDEN: Any reference to COMBINED/ or SPLIT_LITE/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import json
from .core import PackScope, SCOPE_AGS, SCOPE_LAB, read_text, PROJECT_ROOT
from .proofs import get_lite_proof_summary

def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()

def write_split_pack_lite(pack_dir: Path, *, scope: PackScope, project_root: Path) -> None:
    """
    Write a discussion-first LITE index.

    P.2 contract: LITE must be manifest-only (no repo file bodies). This module
    only writes human-readable index stubs; manifests and CAS refs are generated
    in `core.py`.
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

        # LITE PROOFS Summary (AGS Only)
        lite_proofs = get_lite_proof_summary(project_root)
        (lite_dir / "PROOFS.json").write_text(json.dumps(lite_proofs, indent=2) + "\n", encoding="utf-8")

    elif scope.key == SCOPE_LAB.key:
        write(
            lite_dir / "LAB-00_INDEX.md",
            "\n".join(
                [
                    "# LAB Pack Index (LITE)",
                    "",
                    "This directory contains a compressed, discussion-first map of the pack.",
                    "",
                    "## Read order",
                    "1) `repo/THOUGHT/LAB/`",
                    "2) `meta/PACK_INFO.json`",
                    "3) `meta/FILE_TREE.txt` and `meta/FILE_INDEX.json`",
                    "",
                ]
            ),
        )

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
