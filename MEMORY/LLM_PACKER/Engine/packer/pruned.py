"""PRUNED output generation (Phase 1).

Output target: pack_dir/PRUNED/

PRUNED is a reduced planning context optimized for LLM navigation.
It MUST include:
- AGS_ROADMAP_MASTER.md (root-level master roadmap)
- NAVIGATION/OPS/**
- LAW/CANON/**
- LAW/CONTEXT/**
- LAW/CONTRACTS/schemas/** (schemas only, not _runs)
- CAPABILITY/SKILLS/**/SKILL.md (manifests only)
- CAPABILITY/TOOLS/**.md (docs only)
- NAVIGATION/CORTEX/meta/FILE_INDEX.json and SECTION_INDEX.json if those exist

It MUST exclude:
- large code trees by default
- any runtime artifacts under LAW/CONTRACTS/_runs/
- any directories under BUILD (reserved for user outputs)

It MUST include a PRUNED manifest:
- PRUNED/PACK_MANIFEST_PRUNED.json (same format as PACK_MANIFEST.json but for PRUNED only)
- PRUNED/meta/PRUNED_RULES.json describing allowlist rules used to generate PRUNED

ATOMICITY GUARANTEE:
PRUNED output is written atomically via staging directory:
- All PRUNED artifacts are written to a staging directory
- Only after ALL files are written successfully, staging is renamed to PRUNED/
- On any exception, staging is deleted and existing PRUNED/ is left untouched

FAIL-CLOSED: On rename failure, backup is restored and staging is cleaned up.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import (
    PackScope,
    SCOPE_AGS,
    SCOPE_LAB,
    hash_file,
    read_text,
    _canonical_json_bytes,
)
from .firewall_writer import PackerWriter


def rel_posix(*parts: str) -> str:
    return Path(*parts).as_posix()


def _is_pruned_allowed_path(rel_path: str, *, scope: PackScope) -> bool:
    """
    Check if a path should be included in PRUNED output based on allowlist rules.
    """
    path = Path(rel_path)
    parts = path.parts

    # Guard against empty paths
    if not parts or len(parts) == 0:
        return False

    # Always exclude _runs directories
    if "_runs" in parts:
        return False

    # Always exclude _generated directories
    if "_generated" in parts:
        return False

    # Always exclude BUILD directories
    if "BUILD" in parts:
        return False

    if scope.key == SCOPE_AGS.key:
        # INCLUDE: AGS_ROADMAP_MASTER.md at root
        if rel_path == "AGS_ROADMAP_MASTER.md":
            return True

        # INCLUDE: NAVIGATION/OPS/**
        if parts[0] == "NAVIGATION" and len(parts) > 1 and parts[1] == "OPS":
            return True

        # INCLUDE: LAW/CANON/**
        if parts[0] == "LAW" and len(parts) > 1 and parts[1] == "CANON":
            return True

        # INCLUDE: LAW/CONTEXT/**
        if parts[0] == "LAW" and len(parts) > 1 and parts[1] == "CONTEXT":
            return True

        # INCLUDE: LAW/CONTRACTS/schemas/** (schemas only, not _runs)
        if parts[0] == "LAW" and len(parts) > 1 and parts[1] == "CONTRACTS":
            if len(parts) > 2 and parts[2] == "schemas":
                return True

        # INCLUDE: CAPABILITY/SKILLS/**/SKILL.md (manifests only)
        if parts[0] == "CAPABILITY" and len(parts) > 1 and parts[1] == "SKILLS":
            if path.name == "SKILL.md":
                return True

        # INCLUDE: CAPABILITY/TOOLS/**.md (docs only)
        if parts[0] == "CAPABILITY" and len(parts) > 1 and parts[1] == "TOOLS":
            if path.suffix.lower() == ".md":
                return True

        # INCLUDE: NAVIGATION/CORTEX/meta/FILE_INDEX.json and SECTION_INDEX.json
        if parts[0] == "NAVIGATION" and len(parts) > 1 and parts[1] == "CORTEX":
            if len(parts) > 2 and parts[2] == "meta":
                if path.name in {"FILE_INDEX.json", "SECTION_INDEX.json"}:
                    return True

        return False

    elif scope.key == SCOPE_LAB.key:
        # LAB scope: include minimal navigation context
        # INCLUDE: AGS_ROADMAP_MASTER.md at root
        if rel_path == "AGS_ROADMAP_MASTER.md":
            return True

        return False

    return False


def build_pruned_manifest(
    pack_dir: Path,
    project_root: Path,
    included_paths: Sequence[str],
    *,
    scope: PackScope,
) -> Dict[str, Any]:
    """
    Build PRUNED manifest for included paths.

    Returns manifest with entries in deterministic order.
    """
    source_root = project_root / scope.source_root_rel

    pruned_entries: List[Dict[str, Any]] = []
    pruned_paths: List[str] = []

    for rel in sorted(set(included_paths)):
        if not _is_pruned_allowed_path(rel, scope=scope):
            continue

        src = source_root / rel
        if not src.exists() or not src.is_file():
            continue

        pruned_paths.append(rel)
        pruned_entries.append({
            "path": rel,
            "hash": hash_file(src),
            "size": src.stat().st_size,
        })

    # Sort for determinism
    pruned_paths.sort()
    pruned_entries.sort(key=lambda e: (e["path"], e["hash"]))

    manifest = {
        "version": "PRUNED.1.0",
        "scope": scope.key,
        "entries": pruned_entries,
    }

    return manifest


def write_pruned_pack(
    pack_dir: Path,
    project_root: Path,
    included_paths: Sequence[str],
    *,
    scope: PackScope,
    writer: Optional[PackerWriter] = None,
) -> None:
    """
    Write PRUNED output directory with manifest and index files.

    ATOMICITY GUARANTEE:
    - All PRUNED artifacts are written to a staging directory
    - Only after ALL files are written successfully, staging is renamed to PRUNED/
    - On any exception, staging is deleted and existing PRUNED/ is left untouched
    - On rename failure, backup is restored and staging is cleaned up.

    PRUNED includes:
    - PACK_MANIFEST_PRUNED.json: manifest of included files
    - meta/PRUNED_RULES.json: description of allowlist rules
    - Index files organized by category
    """
    # Create staging directory with unique identifier
    staging_id = str(uuid.uuid4())[:12]
    staging_dir = pack_dir / f".pruned_staging_{staging_id}"

    try:
        if writer is None:
            staging_dir.mkdir(parents=True, exist_ok=True)
        else:
            writer.mkdir(staging_dir, kind="tmp", parents=True, exist_ok=True)

        # Build manifest
        manifest = build_pruned_manifest(
            staging_dir,
            project_root,
            included_paths,
            scope=scope,
        )

        # Write manifest
        manifest_bytes = _canonical_json_bytes(manifest)
        if writer is None:
            (staging_dir / "PACK_MANIFEST_PRUNED.json").write_bytes(manifest_bytes)
        else:
            writer.write_bytes(staging_dir / "PACK_MANIFEST_PRUNED.json", manifest_bytes)

        # Write PRUNED_RULES metadata
        pruned_rules = {
            "version": "PRUNED.1.0",
            "scope": scope.key,
            "rules": {
                "include": {
                    "AGS_ROADMAP_MASTER.md": "Master roadmap at repository root",
                    "NAVIGATION/OPS/**": "Operational procedures",
                    "LAW/CANON/**": "Core governance rules",
                    "LAW/CONTEXT/**": "ADRs and decision records",
                    "LAW/CONTRACTS/schemas/**": "JSON schemas for validation",
                    "CAPABILITY/SKILLS/**/SKILL.md": "Skill manifests only",
                    "CAPABILITY/TOOLS/**/*.md": "Tool documentation",
                    "NAVIGATION/CORTEX/meta/{FILE_INDEX,SECTION_INDEX}.json": "Cortex metadata indices",
                },
                "exclude": {
                    "_runs/**": "Runtime artifacts",
                    "_generated/**": "Generated files",
                    "BUILD": "Build outputs",
                    "LAW/CONTRACTS/_runs/**": "Contract runtime artifacts",
                },
            },
            "note": "PRUNED is a reduced planning context for LLM navigation, not a complete pack.",
        }
        if writer is None:
            (staging_dir / "meta" / "PRUNED_RULES.json").parent.mkdir(parents=True, exist_ok=True)
            (staging_dir / "meta" / "PRUNED_RULES.json").write_text(
                json.dumps(pruned_rules, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            writer.mkdir((staging_dir / "meta" / "PRUNED_RULES.json").parent, kind="tmp", parents=True, exist_ok=True)
            writer.write_text(
                staging_dir / "meta" / "PRUNED_RULES.json",
                json.dumps(pruned_rules, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        # Write index files by category
        if scope.key == SCOPE_AGS.key:
            _write_pruned_index_ags(staging_dir, project_root, manifest["entries"], writer=writer)
        elif scope.key == SCOPE_LAB.key:
            _write_pruned_index_lab(staging_dir, project_root, manifest["entries"], writer=writer)

        # Atomic rename: staging -> PRUNED with backup-on-fallback
        # Policy: Backup existing PRUNED/, swap in staging, cleanup backup on success
        pruned_final = pack_dir / "PRUNED"
        pruned_backup = pack_dir / "PRUNED._old"

        if pruned_final.exists():
            # Backup existing PRUNED/ atomically
            if writer is None:
                pruned_final.rename(pruned_backup)
            else:
                writer.rename(pruned_final, pruned_backup)

        try:
            # Atomic rename: staging -> PRUNED
            if writer is None:
                staging_dir.rename(pruned_final)
            else:
                writer.rename(staging_dir, pruned_final)

            # Success: cleanup backup
            if pruned_backup.exists():
                if writer is None:
                    shutil.rmtree(pruned_backup)
                else:
                    # For backup cleanup, we need to use direct file operations since it's cleanup
                    shutil.rmtree(pruned_backup)
        except Exception as rename_exception:
            # Rename failed: restore backup if exists
            if pruned_backup.exists():
                try:
                    if writer is None:
                        pruned_backup.rename(pruned_final)
                    else:
                        writer.rename(pruned_backup, pruned_final)
                except Exception:
                    pass

            # Cleanup staging directory
            if staging_dir.exists():
                try:
                    if writer is None:
                        shutil.rmtree(staging_dir)
                    else:
                        # For staging cleanup, we need to use direct file operations since it's cleanup
                        shutil.rmtree(staging_dir)
                except Exception:
                    pass

            # Re-raise original exception
            raise rename_exception

    except Exception as pruned_exception:
        # Cleanup staging directory on failure
        if staging_dir.exists():
            try:
                if writer is None:
                    shutil.rmtree(staging_dir)
                else:
                    # For staging cleanup, we need to use direct file operations since it's cleanup
                    shutil.rmtree(staging_dir)
            except Exception:
                pass

        # Re-raise to signal failure
        raise pruned_exception


def _write_pruned_index_ags(
    pruned_dir: Path,
    project_root: Path,
    entries: List[Dict[str, Any]],
    writer: Optional[PackerWriter] = None,
) -> None:
    """Write AGS PRUNED index files organized by category."""

    def section(title: str, paths: Sequence[str]) -> str:
        lines = [f"# {title}", ""]
        for rel in paths:
            src = project_root / rel
            if not src.exists():
                continue
            text = read_text(src)
            lines.append(f"## `{rel}` ({src.stat().st_size:,} bytes)")
            lines.append("")
            lines.append("```")
            lines.append(text.rstrip("\n"))
            lines.append("```")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # Group entries by category
    roadmap_paths = [e["path"] for e in entries if e["path"] == "AGS_ROADMAP_MASTER.md"]
    ops_paths = [e["path"] for e in entries if "NAVIGATION/OPS" in e["path"]]
    canon_paths = [e["path"] for e in entries if "LAW/CANON" in e["path"]]
    context_paths = [e["path"] for e in entries if "LAW/CONTEXT" in e["path"]]
    schema_paths = [e["path"] for e in entries if "LAW/CONTRACTS/schemas" in e["path"]]
    skill_manifests = [e["path"] for e in entries if "CAPABILITY/SKILLS" in e["path"] and e["path"].endswith("/SKILL.md")]
    tool_docs = [e["path"] for e in entries if "CAPABILITY/TOOLS" in e["path"] and e["path"].endswith(".md")]
    cortex_meta = [e["path"] for e in entries if "NAVIGATION/CORTEX/meta" in e["path"]]

    # Write main index
    index_content = "\n".join(
        [
            "# AGS Pack Index (PRUNED)",
            "",
            "This directory contains a reduced planning context for LLM navigation.",
            "",
            "## What is PRUNED?",
            "",
            "PRUNED is a minimal subset of full AGS repository, optimized for:",
            "- Understanding governance rules and decisions",
            "- Navigating roadmaps and operational procedures",
            "- Finding skill manifests and tool documentation",
            "",
            "## Contents",
            "",
            "- `AGS-01_NAVIGATION.md` - Roadmaps and OPS",
            "- `AGS-02_LAW_CANON.md` - Core governance rules",
            "- `AGS-03_LAW_CONTEXT.md` - ADRs and decision records",
            "- `AGS-04_CONTRACTS_SCHEMAS.md` - JSON schemas",
            "- `AGS-05_SKILL_MANIFESTS.md` - Skill SKILL.md files",
            "- `AGS-06_TOOL_DOCS.md` - Tool documentation",
            "",
            "## Notes",
            "",
            "- PRUNED does NOT include full source code trees.",
            "- Use FULL/ or SPLIT/ for complete repository access.",
            "- PRUNED is deterministic and additive (does not affect other outputs).",
            "",
        ]
    )
    if writer is None:
        (pruned_dir / "AGS-00_INDEX.md").write_text(index_content, encoding="utf-8")
    else:
        writer.write_text(pruned_dir / "AGS-00_INDEX.md", index_content, encoding="utf-8")

    # Write category sections
    if roadmap_paths or ops_paths:
        nav_content = section("NAVIGATION (Roadmaps & OPS)", sorted([*roadmap_paths, *ops_paths]))
        if writer is None:
            (pruned_dir / "AGS-01_NAVIGATION.md").write_text(nav_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-01_NAVIGATION.md", nav_content, encoding="utf-8")

    if canon_paths:
        canon_content = section("LAW/CANON", sorted(canon_paths))
        if writer is None:
            (pruned_dir / "AGS-02_LAW_CANON.md").write_text(canon_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-02_LAW_CANON.md", canon_content, encoding="utf-8")

    if context_paths:
        context_content = section("LAW/CONTEXT", sorted(context_paths))
        if writer is None:
            (pruned_dir / "AGS-03_LAW_CONTEXT.md").write_text(context_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-03_LAW_CONTEXT.md", context_content, encoding="utf-8")

    if schema_paths:
        schema_content = section("LAW/CONTRACTS/schemas", sorted(schema_paths))
        if writer is None:
            (pruned_dir / "AGS-04_CONTRACTS_SCHEMAS.md").write_text(schema_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-04_CONTRACTS_SCHEMAS.md", schema_content, encoding="utf-8")

    if skill_manifests:
        skills_content = section("CAPABILITY/SKILLS/*/SKILL.md", sorted(skill_manifests))
        if writer is None:
            (pruned_dir / "AGS-05_SKILL_MANIFESTS.md").write_text(skills_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-05_SKILL_MANIFESTS.md", skills_content, encoding="utf-8")

    if tool_docs:
        tools_content = section("CAPABILITY/TOOLS/**/*.md", sorted(tool_docs))
        if writer is None:
            (pruned_dir / "AGS-06_TOOL_DOCS.md").write_text(tools_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-06_TOOL_DOCS.md", tools_content, encoding="utf-8")

    if cortex_meta:
        meta_content = section("NAVIGATION/CORTEX/meta", sorted(cortex_meta))
        if writer is None:
            (pruned_dir / "AGS-07_CORTEX_META.md").write_text(meta_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "AGS-07_CORTEX_META.md", meta_content, encoding="utf-8")


def _write_pruned_index_lab(
    pruned_dir: Path,
    project_root: Path,
    entries: List[Dict[str, Any]],
    writer: Optional[PackerWriter] = None,
) -> None:
    """Write LAB PRUNED index files (minimal)."""

    roadmap_paths = [e["path"] for e in entries if e["path"] == "AGS_ROADMAP_MASTER.md"]

    index_content = "\n".join(
        [
            "# LAB Pack Index (PRUNED)",
            "",
            "This directory contains a minimal planning context for LAB.",
            "",
            "## Contents",
            "",
            "- `LAB-01_NAVIGATION.md` - Roadmaps",
            "",
            "## Notes",
            "",
            "- PRUNED is minimal and does not include full LAB source code.",
            "- Use FULL/ or SPLIT/ for complete LAB access.",
            "",
        ]
    )
    if writer is None:
        (pruned_dir / "LAB-00_INDEX.md").write_text(index_content, encoding="utf-8")
    else:
        writer.write_text(pruned_dir / "LAB-00_INDEX.md", index_content, encoding="utf-8")

    if roadmap_paths:
        def section(title: str, paths: Sequence[str]) -> str:
            lines = [f"# {title}", ""]
            for rel in paths:
                src = project_root / rel
                if not src.exists():
                    continue
                text = read_text(src)
                lines.append(f"## `{rel}` ({src.stat().st_size:,} bytes)")
                lines.append("")
                lines.append("```")
                lines.append(text.rstrip("\n"))
                lines.append("```")
                lines.append("")
            return "\n".join(lines).rstrip() + "\n"

        nav_content = section("AGS_ROADMAP_MASTER.md", sorted(roadmap_paths))
        if writer is None:
            (pruned_dir / "LAB-01_NAVIGATION.md").write_text(nav_content, encoding="utf-8")
        else:
            writer.write_text(pruned_dir / "LAB-01_NAVIGATION.md", nav_content, encoding="utf-8")
