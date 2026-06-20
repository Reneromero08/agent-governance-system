#!/usr/bin/env python3
"""Verify every run manifest below a runs root without trusting manifest paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def verify(root: Path) -> list[str]:
    errors: list[str] = []
    manifests = sorted(root.glob("*/run_manifest.json"))
    for manifest_path in manifests:
        run = manifest_path.parent
        try:
            manifest = load_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{run.name}: invalid run_manifest.json: {exc}")
            continue
        files = manifest.get("files")
        if not isinstance(files, dict):
            errors.append(f"{run.name}: invalid files table")
            continue
        if "run_manifest.json" in files:
            errors.append(f"{run.name}: manifest includes itself")
        run_root = run.resolve()
        for name, binding in files.items():
            if not isinstance(name, str) or not isinstance(binding, dict):
                errors.append(f"{run.name}: invalid binding {name!r}")
                continue
            path = (run / name).resolve()
            try:
                path.relative_to(run_root)
            except ValueError:
                errors.append(f"{run.name}: invalid path {name}")
                continue
            if not path.is_file():
                errors.append(f"{run.name}: missing {name}")
                continue
            if path.stat().st_size != binding.get("size"):
                errors.append(f"{run.name}: size {name}")
            if sha256_file(path) != binding.get("sha256"):
                errors.append(f"{run.name}: sha256 {name}")
    if not manifests:
        errors.append("no run manifests")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    errors = verify(args.root)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"RUN_MANIFESTS_VERIFIED count={len(list(args.root.glob('*/run_manifest.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
