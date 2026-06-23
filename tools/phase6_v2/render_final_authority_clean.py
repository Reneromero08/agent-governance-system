#!/usr/bin/env python3
"""Render final Phase 6 V2 authority files with diff-clean Markdown."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import render_final_authority as render


def normalize(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    render.patch_roadmap()
    render.patch_navigation()
    render.patch_chiral()
    render.write_architecture_review()
    render.patch_work_package()
    render.patch_bindings()

    for path in (
        render.ROADMAP,
        render.NAVIGATION,
        render.CHIRAL,
        render.ARCH_REVIEW,
        render.WORK_PACKAGE,
    ):
        normalize(path)

    render.regenerate_inventory()

    allowed = {
        render.ROADMAP.relative_to(render.ROOT).as_posix(),
        render.NAVIGATION.relative_to(render.ROOT).as_posix(),
        render.CHIRAL.relative_to(render.ROOT).as_posix(),
        render.ARCH_REVIEW.relative_to(render.ROOT).as_posix(),
        render.WORK_PACKAGE.relative_to(render.ROOT).as_posix(),
        render.BINDINGS.relative_to(render.ROOT).as_posix(),
        render.INVENTORY.relative_to(render.ROOT).as_posix(),
        render.VERIFICATION.relative_to(render.ROOT).as_posix(),
    }
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=render.ROOT, text=True
    ).splitlines())
    if changed != allowed:
        raise RuntimeError(f"unexpected changed paths: {sorted(changed ^ allowed)}")
    subprocess.run(["git", "diff", "--check"], cwd=render.ROOT, check=True)

    manifest = {
        "status": render.STATUS,
        "changed_paths": sorted(changed),
        "sha256": {
            rel: hashlib.sha256((render.ROOT / rel).read_bytes()).hexdigest()
            for rel in sorted(changed)
        },
    }
    runner_temp = Path(subprocess.check_output(
        ["bash", "-lc", "printf %s \"${RUNNER_TEMP:-/tmp}\""], text=True
    ).strip())
    (runner_temp / "phase6_v2_authority_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
