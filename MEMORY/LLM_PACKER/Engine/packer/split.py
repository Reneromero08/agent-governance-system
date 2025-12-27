"""
SPLIT output generation (Phase 1).

Output target: pack_dir/SPLIT/ 
FORBIDDEN: Any reference to COMBINED/ in output paths or documentation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .core import PackScope, SCOPE_AGS, SCOPE_CATALYTIC_DPT, SCOPE_LAB, read_text

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
    elif scope.key == SCOPE_CATALYTIC_DPT.key:
        write_split_pack_catalytic_dpt(pack_dir, included_repo_paths, scope=scope)
    elif scope.key == SCOPE_LAB.key:
        write_split_pack_catalytic_dpt_lab(pack_dir, included_repo_paths, scope=scope)
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

    # Group paths
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

    meta_dir = pack_dir / "meta"
    meta_paths = sorted([f"meta/{p.name}" for p in meta_dir.iterdir() if p.is_file()]) if meta_dir.exists() else []

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
                "2) `repo/README.md` and `repo/CONTEXT/archive/planning/INDEX.md`",
                "3) `repo/CANON/CONTRACT.md` and `repo/CANON/INVARIANTS.md` and `repo/CANON/VERSIONING.md`",
                "4) `repo/MAPS/ENTRYPOINTS.md`",
                "5) `repo/CONTRACTS/runner.py` and `repo/SKILLS/`",
                "6) `repo/CORTEX/` and `repo/TOOLS/`",
                "7) `meta/ENTRYPOINTS.md` and `meta/CONTEXT.txt` (Snapshot specific)",
                "",
                "## Notes",
                "- `BUILD` contents are excluded.",
                "- Research under `repo/CONTEXT/research/` is non-binding and opt-in.",
                "- Single-file bundles available in `FULL/`.",
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

    docs_paths = sorted(set([*agents_paths, *readme_paths, *roadmap_paths, *changelog_paths, *architecture_docs]))
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
                "- See `FULL/` for single-file bundles.",
                "- LAB is packed separately into `LAB/` inside the same bundle.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / f"{scope.file_prefix}-01_DOCS.md").write_text("# Docs\n\n" + section(docs_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-02_CONFIG.md").write_text("# Config\n\n" + section(config_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-03_TESTBENCH.md").write_text("# Testbench\n\n" + section(testbench_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-04_SYSTEM.md").write_text("# System\n\n" + section([*system_paths, *meta_paths]), encoding="utf-8")

def write_split_pack_catalytic_dpt_lab(pack_dir: Path, included_repo_paths: Sequence[str], *, scope: PackScope) -> None:
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
        return path.startswith("repo/CATALYTIC-DPT/LAB/")

    lab_paths = [p for p in included_repo_paths if is_lab(p)]
    lab_root = [p for p in lab_paths if p.count("/") == 3]

    docs_paths = [p for p in lab_root if p.endswith(".md")]
    research_paths = [p for p in lab_paths if "/RESEARCH/" in p]
    commonsense_paths = [p for p in lab_paths if "/COMMONSENSE/" in p]
    mcp_paths = [p for p in lab_paths if "/MCP/" in p]

    # Everything else goes to system
    used = set(docs_paths + research_paths + commonsense_paths + mcp_paths)
    system_paths = [p for p in lab_paths if p not in used]

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
                "1) `repo/CATALYTIC-DPT/LAB/ROADMAP_PATCH_SEMIOTIC.md`",
                "2) `repo/CATALYTIC-DPT/LAB/COMMONSENSE/`",
                "3) `repo/CATALYTIC-DPT/LAB/MCP/`",
                "4) `repo/CATALYTIC-DPT/LAB/RESEARCH/`",
                "5) `meta/ENTRYPOINTS.md`",
                "",
                "## Notes",
                "- `BUILD` contents are excluded.",
                "- Single-file bundles available in `FULL/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (split_dir / f"{scope.file_prefix}-01_DOCS.md").write_text("# Docs\n\n" + section(docs_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-02_COMMONSENSE.md").write_text("# Commonsense\n\n" + section(commonsense_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-03_MCP.md").write_text("# MCP\n\n" + section(mcp_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-04_RESEARCH.md").write_text("# Research\n\n" + section(research_paths), encoding="utf-8")
    (split_dir / f"{scope.file_prefix}-05_SYSTEM.md").write_text("# System\n\n" + section([*system_paths, *meta_paths]), encoding="utf-8")
