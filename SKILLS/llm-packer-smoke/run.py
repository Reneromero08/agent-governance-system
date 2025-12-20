#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKER_SCRIPT = PROJECT_ROOT / "MEMORY" / "packer.py"
PACKS_ROOT = PROJECT_ROOT / "MEMORY" / "LLM-PACKER-1.0" / "_packs"
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
        raise ValueError(f"out_dir must be under MEMORY/_packs/: {path}") from exc


def ensure_runner_writes_under_runs(path: Path) -> None:
    runs_root = RUNS_ROOT.resolve()
    try:
        path.resolve().relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"runner output must be under CONTRACTS/_runs/: {path}") from exc


def main(input_path: Path, output_path: Path) -> int:
    try:
        config = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    out_dir_raw = str(config.get("out_dir", "MEMORY/LLM-PACKER-1.0/_packs/fixture-smoke"))
    combined = bool(config.get("combined", False))
    zip_enabled = bool(config.get("zip", False))
    mode = str(config.get("mode", "full"))

    out_dir = resolve_out_dir(out_dir_raw)
    ensure_under_packs(out_dir)
    ensure_runner_writes_under_runs(output_path)
    if not PACKER_SCRIPT.exists():
        print(f"Missing packer script at {PACKER_SCRIPT}")
        return 1

    args = [
        sys.executable,
        str(PACKER_SCRIPT),
        "--mode",
        mode,
        "--out-dir",
        out_dir.relative_to(PROJECT_ROOT).as_posix(),
    ]
    if zip_enabled:
        args.append("--zip")
    if combined:
        args.append("--combined")
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
        "COMBINED/SPLIT/00_INDEX.md",
        "COMBINED/SPLIT/01_CANON.md",
    ]
    missing = [p for p in required if not (out_dir / p).exists()]
    if missing:
        print("Packer output missing required files:")
        for p in missing:
            print(f"- {p}")
        return 1

    start_here_text = (out_dir / "meta/START_HERE.md").read_text(encoding="utf-8", errors="replace")
    entrypoints_text = (out_dir / "meta/ENTRYPOINTS.md").read_text(encoding="utf-8", errors="replace")
    required_mentions = [
        "`repo/AGENTS.md`",
        "`repo/README.md`",
        "`repo/ROADMAP.md`",
    ]
    for mention in required_mentions:
        if mention not in start_here_text:
            print(f"START_HERE.md missing required mention: {mention}")
            return 1
        if mention not in entrypoints_text:
            print(f"ENTRYPOINTS.md missing required mention: {mention}")
            return 1

    output_payload = {
        "pack_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
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
