"""
SPLIT output generation (Phase 1).

Output target: pack_dir/SPLIT/
FORBIDDEN: Any reference to COMBINED/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_LAB, SCOPE_CAT_CAS, read_text
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
    elif scope.key == SCOPE_CAT_CAS.key:
        write_split_pack_cat_cas(pack_dir, included_repo_paths, scope=scope, writer=writer)
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


def write_split_pack_cat_cas(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope, writer: Optional[PackerWriter] = None) -> None:
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

    def is_track(path: str, track_dir: str) -> bool:
        rel = path.replace("\\", "/")
        if rel.startswith("repo/"):
            rel = rel[5:]
        return rel.startswith(f"{track_dir}/") or rel == track_dir

    meta_dir = pack_dir / "meta"
    meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()]) if meta_dir.exists() else []

    # Track 1: Foundations
    t1 = sorted(p for p in included_repo_paths if is_track(p, "1_foundations"))
    # Track 2: Substrate Expansion
    t2 = sorted(p for p in included_repo_paths if is_track(p, "2_substrate_expansion"))
    # Track 3: Physics / Complexity
    t3 = sorted(p for p in included_repo_paths if is_track(p, "3_physics_complexity"))
    # Track 4: Holographic
    t4 = sorted(p for p in included_repo_paths if is_track(p, "4_holographic"))
    # Track 5: Topological Proofs
    t5 = sorted(p for p in included_repo_paths if is_track(p, "5_topological_proofs"))
    # Track 6: Frontier Phases
    t6 = sorted(p for p in included_repo_paths if is_track(p, "6_frontier_phases"))
    # Track 7: Decoder
    t7 = sorted(p for p in included_repo_paths if is_track(p, "7_decoder"))
    # Docs, _lib, workspace, root files
    infra = sorted(p for p in included_repo_paths if not any(
        is_track(p, d) for d in ("1_foundations", "2_substrate_expansion",
            "3_physics_complexity", "4_holographic", "5_topological_proofs",
            "6_frontier_phases", "7_decoder")
    ))

    index_content = "\n".join(
        [
            "# CAT_CAS Pack Index",
            "",
            "This directory contains a generated snapshot of the CAT_CAS lab intended for LLM handoff.",
            "",
            "## Read order",
            "1) `CAT-08_DOCS_AND_INFRA.md` (AGENTS.md, README.md, MANIFESTO.md, MASTER_REPORT.md, CAT_CAS_OS.md, PRIMER.md, _lib/)",
            "2) `CAT-01_FOUNDATIONS.md` - Track 1: Reversible Computing & Landauer Basics (Exps 01-05)",
            "3) `CAT-02_SUBSTRATE.md` - Track 2: Catalytic Memory & Inference Substrate (Exps 06-13)",
            "4) `CAT-03_COMPLEXITY.md` - Track 3: Limits, Factorization, NP, Temporal (Exps 14-24)",
            "5) `CAT-04_HOLOGRAPHIC.md` - Track 4: Lattice/Crypto, Graphs, Wormholes, MERA (Exps 25-33)",
            "6) `CAT-05_TOPOLOGICAL.md` - Track 5: Zeta/RH, Halting Oracles, ToE (Exps 34-41)",
            "7) `CAT-06_FRONTIER.md` - Track 6: Limits to Emergence Chain (Exps 42-48)",
            "8) `CAT-07_DECODER.md` - Track 7: Decoder Theory + Physical Substrate (Exps 49-50)",
            "",
            "## Notes",
            "- Single-file bundles available in `FULL/`.",
            "- `meta/` contains PACK_INFO.json, REPO_STATE.json, and FILE_TREE.txt.",
            "",
        ]
    )
    if writer is None:
        (split_dir / "CAT-00_INDEX.md").write_text(index_content, encoding="utf-8")
    else:
        writer.write_text(split_dir / "CAT-00_INDEX.md", index_content, encoding="utf-8")

    sections = [
        ("CAT-01_FOUNDATIONS.md", "# Track 1: Foundations (Reversible Computing & Landauer)\n\n", t1),
        ("CAT-02_SUBSTRATE.md", "# Track 2: Substrate Expansion (Catalytic Memory & Inference)\n\n", t2),
        ("CAT-03_COMPLEXITY.md", "# Track 3: Physics / Complexity (Limits, Factorization, NP, Temporal)\n\n", t3),
        ("CAT-04_HOLOGRAPHIC.md", "# Track 4: Holographic (Lattice/Crypto, Graphs, Wormholes, MERA)\n\n", t4),
        ("CAT-05_TOPOLOGICAL.md", "# Track 5: Topological Proofs (Zeta/RH, Halting Oracles, ToE)\n\n", t5),
        ("CAT-06_FRONTIER.md", "# Track 6: Frontier Phases (Limits to Emergence Chain)\n\n", t6),
        ("CAT-07_DECODER.md", "# Track 7: Decoder (Decoder Theory + Physical Substrate)\n\n", t7),
        ("CAT-08_DOCS_AND_INFRA.md", "# Docs, Infrastructure & Meta\n\n", infra + meta_paths),
    ]

    for filename, header, paths in sections:
        content = header + section(paths)
        if writer is None:
            (split_dir / filename).write_text(content, encoding="utf-8")
        else:
            writer.write_text(split_dir / filename, content, encoding="utf-8")
