#!/usr/bin/env python3

"""
Runtime helpers for skills.

Ensures a skill's required_canon_version range matches the current canon version.
"""

import re
from pathlib import Path
from typing import Iterable, Optional, Tuple


Version = Tuple[int, int, int]


def _parse_version(value: str) -> Version:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", value.strip())
    if not match:
        raise ValueError(f"Invalid version: {value}")
    return tuple(int(part) for part in match.groups())


def _parse_constraints(range_str: str) -> Iterable[Tuple[str, Version]]:
    cleaned = range_str.strip().strip('"').strip("'")
    if not cleaned:
        return []
    constraints = []
    for token in cleaned.split():
        match = re.match(r"^(>=|<=|>|<|==|=)?(\d+\.\d+\.\d+)$", token)
        if not match:
            raise ValueError(f"Invalid range token: {token}")
        op = match.group(1) or "=="
        constraints.append((op, _parse_version(match.group(2))))
    return constraints


def _satisfies(version: Version, constraints: Iterable[Tuple[str, Version]]) -> bool:
    for op, bound in constraints:
        if op == ">=" and not (version >= bound):
            return False
        if op == ">" and not (version > bound):
            return False
        if op == "<=" and not (version <= bound):
            return False
        if op == "<" and not (version < bound):
            return False
        if op in ("==", "=") and not (version == bound):
            return False
    return True


def _read_required_range(skill_dir: Path) -> Optional[str]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    lines = skill_md.read_text(errors="ignore").splitlines()[:50]
    for line in lines:
        match = re.search(r"^\*\*required_canon_version:\*\*\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    return None


def _read_canon_version(project_root: Path) -> Optional[str]:
    versioning = project_root / "CANON" / "VERSIONING.md"
    if not versioning.exists():
        return None
    content = versioning.read_text(errors="ignore")
    match = re.search(r"canon_version:\s*(\d+\.\d+\.\d+)", content)
    return match.group(1) if match else None


def ensure_canon_compat(skill_dir: Path) -> bool:
    project_root = skill_dir.resolve().parents[1]
    required_range = _read_required_range(skill_dir)
    if not required_range:
        print(f"[skill] Missing required_canon_version in {skill_dir / 'SKILL.md'}")
        return False

    canon_version_str = _read_canon_version(project_root)
    if not canon_version_str:
        print(f"[skill] Missing canon_version in {project_root / 'CANON' / 'VERSIONING.md'}")
        return False

    try:
        constraints = _parse_constraints(required_range)
        canon_version = _parse_version(canon_version_str)
    except ValueError as exc:
        print(f"[skill] Version parsing error: {exc}")
        return False

    if not _satisfies(canon_version, constraints):
        print(
            "[skill] Canon version not supported: "
            f"{canon_version_str} not in {required_range}"
        )
        return False

    return True


__all__ = ["ensure_canon_compat"]
