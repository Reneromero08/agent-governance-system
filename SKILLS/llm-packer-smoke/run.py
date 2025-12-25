#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOOLS.skill_runtime import ensure_canon_compat

PACKER_SCRIPT = PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "Engine" / "packer.py"
PACKS_ROOT = PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"
RUNS_ROOT = PROJECT_ROOT / "CONTRACTS" / "_runs"


def resolve_out_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_under_packs(path: Path) -> None:
    packs_root = PACKS_ROOT.resolve()
    try:
        path.resolve().relative_to(packs_root)
    except ValueError as exc:
        raise ValueError(f"out_dir must be under MEMORY/LLM_PACKER/_packs/: {path}") from exc


def ensure_runner_writes_under_runs(path: Path) -> None:
    runs_root = RUNS_ROOT.resolve()
    try:
        path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"runner output must be under CONTRACTS/_runs/: {path}") from exc


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        config = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    out_dir_raw = str(config.get("out_dir", "MEMORY/LLM_PACKER/_packs/fixture-smoke"))
    combined = bool(config.get("combined", False))
    zip_enabled = bool(config.get("zip", False))
    mode = str(config.get("mode", "full"))
    profile = str(config.get("profile", "full"))
    scope = str(config.get("scope", "ags"))
    stamp = str(config.get("stamp", "fixture-smoke"))
    split_lite = bool(config.get("split_lite", False))

    out_dir = resolve_out_dir(out_dir_raw)
    ensure_under_packs(out_dir)
    ensure_runner_writes_under_runs(output_path)
    if not PACKER_SCRIPT.exists():
        print(f"Missing packer script at {PACKER_SCRIPT}")
        return 1

    args = [
        sys.executable,
        str(PACKER_SCRIPT),
        "--scope",
        scope,
        "--mode",
        mode,
        "--profile",
        profile,
        "--out-dir",
        out_dir.relative_to(PROJECT_ROOT).as_posix(),
    ]
    if stamp:
        args.extend(["--stamp", stamp])
    if zip_enabled:
        args.append("--zip")
    if combined:
        args.append("--combined")
    if split_lite:
        args.append("--split-lite")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return result.returncode

    required = [
        "meta/START_HERE.md",
        "meta/ENTRYPOINTS.md",
        "meta/FILE_TREE.txt",
        "meta/FILE_INDEX.json",
        "meta/REPO_OMITTED_BINARIES.json",
        "meta/REPO_STATE.json",
        "meta/PACK_INFO.json",
        "meta/BUILD_TREE.txt",
        "meta/CONTEXT.txt",
    ]
    if scope == "ags":
        required.extend(
            [
                "COMBINED/SPLIT/AGS-00_INDEX.md",
                "COMBINED/SPLIT/AGS-01_CANON.md",
                "COMBINED/SPLIT/AGS-02_ROOT.md",
                "COMBINED/SPLIT/AGS-03_MAPS.md",
                "COMBINED/SPLIT/AGS-04_CONTEXT.md",
                "COMBINED/SPLIT/AGS-05_SKILLS.md",
                "COMBINED/SPLIT/AGS-06_CONTRACTS.md",
                "COMBINED/SPLIT/AGS-07_SYSTEM.md",
            ]
        )
    elif scope == "catalytic-dpt":
        required.extend(
            [
                "COMBINED/SPLIT/CATALYTIC-DPT-00_INDEX.md",
                "COMBINED/SPLIT/CATALYTIC-DPT-01_DOCS.md",
                "COMBINED/SPLIT/CATALYTIC-DPT-02_CONFIG.md",
                "COMBINED/SPLIT/CATALYTIC-DPT-03_TESTBENCH.md",
                "COMBINED/SPLIT/CATALYTIC-DPT-04_SYSTEM.md",
                "COMBINED/SPLIT/CATALYTIC-DPT-05_LAB.md",
            ]
        )
    else:
        print(f"Unknown scope in fixture: {scope}")
        return 1
    if split_lite:
        if scope == "ags":
            required.extend(
                [
                    "COMBINED/SPLIT_LITE/AGS-00_INDEX.md",
                    "COMBINED/SPLIT_LITE/AGS-01_CANON.md",
                    "COMBINED/SPLIT_LITE/AGS-02_ROOT.md",
                    "COMBINED/SPLIT_LITE/AGS-03_MAPS.md",
                    "COMBINED/SPLIT_LITE/AGS-04_CONTEXT.md",
                    "COMBINED/SPLIT_LITE/AGS-05_SKILLS.md",
                    "COMBINED/SPLIT_LITE/AGS-06_CONTRACTS.md",
                    "COMBINED/SPLIT_LITE/AGS-07_SYSTEM.md",
                ]
            )
        else:
            required.extend(
                [
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-00_INDEX.md",
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-01_DOCS.md",
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-02_CONFIG.md",
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-03_TESTBENCH.md",
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-04_SYSTEM.md",
                    "COMBINED/SPLIT_LITE/CATALYTIC-DPT-05_LAB.md",
                ]
            )
    if profile == "lite":
        required.extend(
            [
                "meta/LITE_ALLOWLIST.json",
                "meta/LITE_OMITTED.json",
                "meta/LITE_START_HERE.md",
                "meta/SKILL_INDEX.json",
                "meta/FIXTURE_INDEX.json",
                "meta/CODEBOOK.md",
                "meta/CODE_SYMBOLS.json",
            ]
        )
    if combined:
        prefix = "AGS" if scope == "ags" else "CATALYTIC-DPT"
        required.extend(
            [
                f"COMBINED/{prefix}-FULL-COMBINED-{stamp}.md",
                f"COMBINED/{prefix}-FULL-COMBINED-{stamp}.txt",
                f"COMBINED/{prefix}-FULL-TREEMAP-{stamp}.md",
                f"COMBINED/{prefix}-FULL-TREEMAP-{stamp}.txt",
            ]
        )
    missing = [p for p in required if not (out_dir / p).exists()]
    if missing:
        print("Packer output missing required files:")
        for p in missing:
            print(f"- {p}")
        return 1

    start_here_text = (out_dir / "meta/START_HERE.md").read_text(encoding="utf-8", errors="replace")
    entrypoints_text = (out_dir / "meta/ENTRYPOINTS.md").read_text(encoding="utf-8", errors="replace")
    if scope == "ags":
        required_mentions = [
            "`repo/AGENTS.md`",
            "`repo/README.md`",
            "`repo/CONTEXT/archive/planning/INDEX.md`",
        ]
    else:
        required_mentions = [
            "`repo/CATALYTIC-DPT/AGENTS.md`",
            "`repo/CATALYTIC-DPT/README.md`",
            "`repo/CATALYTIC-DPT/ROADMAP_V2.1.md`",
        ]
    for mention in required_mentions:
        if mention not in start_here_text:
            print(f"START_HERE.md missing required mention: {mention}")
            return 1
        if mention not in entrypoints_text:
            print(f"ENTRYPOINTS.md missing required mention: {mention}")
            return 1

    if scope == "ags":
        maps_text = (out_dir / "COMBINED" / "SPLIT" / "AGS-03_MAPS.md").read_text(encoding="utf-8", errors="replace")
        if "## Repo File Tree" not in maps_text or "PACK/" not in maps_text:
            print("AGS-03_MAPS.md missing embedded repo file tree")
            return 1
    else:
        index_text = (out_dir / "COMBINED" / "SPLIT" / "CATALYTIC-DPT-00_INDEX.md").read_text(encoding="utf-8", errors="replace")
        if "## Repo File Tree" not in index_text or "PACK/" not in index_text:
            print("CATALYTIC-DPT-00_INDEX.md missing embedded repo file tree")
            return 1

    if profile == "lite":
        # Ensure excluded content is not copied into repo/** in the generated pack.
        excluded_markers = [
            "/fixtures/",
            "/_runs/",
            "/_generated/",
            "/CONTEXT/archive/",
            "/CONTEXT/research/",
        ]
        tree_text = (out_dir / "meta/FILE_TREE.txt").read_text(encoding="utf-8", errors="replace")
        for marker in excluded_markers:
            if f"repo{marker}" in tree_text:
                print(f"LITE pack unexpectedly contains excluded content: repo{marker}")
                return 1
        if ".cmd" in tree_text or ".ps1" in tree_text:
            print("LITE pack unexpectedly contains OS wrapper files (*.cmd/*.ps1)")
            return 1

    output_payload = {
        "pack_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
        "stamp": stamp,
        "verified": required,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
