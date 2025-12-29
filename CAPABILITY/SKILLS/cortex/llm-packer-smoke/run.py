#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

PACKER_MODULE = "MEMORY.LLM_PACKER.Engine.packer"
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

    out_dir_raw = str(config.get("out_dir", "MEMORY/LLM_PACKER/_packs/_system/fixtures/fixture-smoke"))
    combined = bool(config.get("combined", False))
    zip_enabled = bool(config.get("zip", False))
    mode = str(config.get("mode", "full"))
    profile = str(config.get("profile", "full"))
    scope = str(config.get("scope", "ags"))
    stamp = str(config.get("stamp", "fixture-smoke"))
    split_lite = bool(config.get("split_lite", False))
    allow_duplicate_hashes = config.get("allow_duplicate_hashes", None)

    out_dir = resolve_out_dir(out_dir_raw)
    ensure_under_packs(out_dir)
    ensure_runner_writes_under_runs(output_path)

    args = [
        sys.executable,
        "-m",
        PACKER_MODULE,
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
    if allow_duplicate_hashes is True:
        args.append("--allow-duplicate-hashes")
    elif allow_duplicate_hashes is False:
        args.append("--disallow-duplicate-hashes")
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
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
        "meta/PROVENANCE.json",
    ]
    if scope == "ags":
        required.extend(
            [
                "SPLIT/AGS-00_INDEX.md",
                "SPLIT/AGS-01_CANON.md",
                "SPLIT/AGS-02_ROOT.md",
                "SPLIT/AGS-03_MAPS.md",
                "SPLIT/AGS-04_CONTEXT.md",
                "SPLIT/AGS-05_SKILLS.md",
                "SPLIT/AGS-06_CONTRACTS.md",
                "SPLIT/AGS-07_SYSTEM.md",
            ]
        )
    elif scope == "catalytic-dpt":
        required.extend(
            [
                "SPLIT/CATALYTIC-DPT-00_INDEX.md",
                "SPLIT/CATALYTIC-DPT-01_DOCS.md",
                "SPLIT/CATALYTIC-DPT-02_CONFIG.md",
                "SPLIT/CATALYTIC-DPT-03_TESTBENCH.md",
                "SPLIT/CATALYTIC-DPT-04_SYSTEM.md",
            ]
        )
    elif scope == "lab":
        required.extend(
            [
                "SPLIT/CATALYTIC-DPT-LAB-00_INDEX.md",
                "SPLIT/CATALYTIC-DPT-LAB-01_DOCS.md",
                "SPLIT/CATALYTIC-DPT-LAB-02_COMMONSENSE.md",
                "SPLIT/CATALYTIC-DPT-LAB-03_MCP.md",
                "SPLIT/CATALYTIC-DPT-LAB-04_RESEARCH.md",
                "SPLIT/CATALYTIC-DPT-LAB-05_ARCHIVE.md",
                "SPLIT/CATALYTIC-DPT-LAB-06_SYSTEM.md",
            ]
        )
    else:
        print(f"Unknown scope in fixture: {scope}")
        return 1

    if split_lite or profile == "lite":
        if scope == "ags":
            required.append("LITE/AGS-00_INDEX.md")
        elif scope == "catalytic-dpt":
            required.append("LITE/CATALYTIC-DPT-00_INDEX.md")
        else:
            required.append("LITE/CATALYTIC-DPT-LAB-00_INDEX.md")

    if combined:
        required.extend(
            [
                f"FULL/{'AGS' if scope == 'ags' else ('CATALYTIC-DPT' if scope == 'catalytic-dpt' else 'CATALYTIC-DPT-LAB')}-FULL-{stamp}.md",
                f"FULL/{'AGS' if scope == 'ags' else ('CATALYTIC-DPT' if scope == 'catalytic-dpt' else 'CATALYTIC-DPT-LAB')}-FULL-TREEMAP-{stamp}.md",
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
        start_here_mentions = ["`repo/AGENTS.md`", "`repo/README.md`"]
        entrypoints_mentions = ["`repo/AGENTS.md`", "`repo/README.md`"]
    elif scope == "catalytic-dpt":
        start_here_mentions = ["`repo/CATALYTIC-DPT/AGENTS.md`", "`repo/CATALYTIC-DPT/README.md`"]
        entrypoints_mentions = ["`repo/CATALYTIC-DPT/AGENTS.md`", "`repo/CATALYTIC-DPT/README.md`"]
    else:
        start_here_mentions = ["`repo/CATALYTIC-DPT/LAB/`"]
        entrypoints_mentions = ["`repo/CATALYTIC-DPT/LAB/`"]

    for mention in start_here_mentions:
        if mention not in start_here_text:
            print(f"START_HERE.md missing required mention: {mention}")
            return 1
    for mention in entrypoints_mentions:
        if mention not in entrypoints_text:
            print(f"ENTRYPOINTS.md missing required mention: {mention}")
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
