#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

PACKER_MODULE = "MEMORY.LLM_PACKER.Engine.packer"
PACKS_ROOT = PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"
RUNS_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"


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
        raise ValueError(f"runner output must be under LAW/CONTRACTS/_runs/: {path}") from exc


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        config = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    out_dir_raw = str(config.get("out_dir", "MEMORY/LLM_PACKER/_packs/_system/fixtures/fixture-smoke"))
    project_root_raw = config.get("project_root")
    p2_runs_dir_raw = config.get("p2_runs_dir")
    combined = bool(config.get("combined", False))
    zip_enabled = bool(config.get("zip", False))
    mode = str(config.get("mode", "full"))
    profile = str(config.get("profile", "full"))
    scope = str(config.get("scope", "ags"))
    stamp = str(config.get("stamp", "fixture-smoke"))
    split_lite = bool(config.get("split_lite", False))
    allow_duplicate_hashes = config.get("allow_duplicate_hashes", None)
    emit_pruned = bool(config.get("emit_pruned", False))
    assert_archive_excluded = bool(config.get("assert_archive_excluded", False))

    out_dir = resolve_out_dir(out_dir_raw)
    ensure_under_packs(out_dir)
    ensure_runner_writes_under_runs(output_path)

    project_root = None
    if project_root_raw:
        project_root = resolve_out_dir(str(project_root_raw))

    p2_runs_dir = None
    if p2_runs_dir_raw:
        p2_runs_dir = resolve_out_dir(str(p2_runs_dir_raw))
        ensure_runner_writes_under_runs(p2_runs_dir / "RUN_ROOTS.json")

    args = [
        sys.executable,
        "-u",
        "-m",
        PACKER_MODULE,
    ]
    if project_root is not None:
        args.extend(["--project-root", project_root.relative_to(PROJECT_ROOT).as_posix()])
    if p2_runs_dir is not None:
        args.extend(["--p2-runs-dir", p2_runs_dir.relative_to(PROJECT_ROOT).as_posix()])
    args.extend([
        "--scope",
        scope,
        "--mode",
        mode,
        "--profile",
        profile,
        "--out-dir",
        out_dir.relative_to(PROJECT_ROOT).as_posix(),
    ])
    # Fixtures should stay fast/deterministic; proof regeneration is a repo-wide gate that can be run separately.
    args.append("--skip-proofs")
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
    if not emit_pruned:
        args.append("--no-emit-pruned")
    result = subprocess.run(args, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
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
                "SPLIT/AGS-01_LAW.md",
                "SPLIT/AGS-02_CAPABILITY.md",
                "SPLIT/AGS-03_NAVIGATION.md",
                "SPLIT/AGS-04_PROOFS.md",
                "SPLIT/AGS-05_MEMORY.md",
                "SPLIT/AGS-06_ROOT_FILES.md",
            ]
        )
    elif scope == "lab":
        required.extend(
            [
                "SPLIT/LAB-00_INDEX.md",
                "SPLIT/LAB-01_DOCS.md",
                "SPLIT/LAB-02_SYSTEM.md",
            ]
        )
    else:
        print(f"Unknown scope in fixture: {scope}")
        return 1

    if split_lite or profile == "lite":
        if scope == "ags":
            required.append("LITE/AGS-00_INDEX.md")
            required.append("LITE/PROOFS.json")
        else:
            required.append("LITE/LAB-00_INDEX.md")

    if emit_pruned:
        pruned_files = [
            "PRUNED/PACK_MANIFEST_PRUNED.json",
            "PRUNED/meta/PRUNED_RULES.json",
        ]
        pruned_dir = out_dir / "PRUNED"
        if pruned_dir.exists():
            required.extend(pruned_files)

    if combined:
        required.extend(
            [
                f"FULL/{'AGS' if scope == 'ags' else 'LAB'}-FULL-{stamp}.md",
                f"FULL/{'AGS' if scope == 'ags' else 'LAB'}-FULL-TREEMAP-{stamp}.md",
            ]
        )
    missing = [p for p in required if not (out_dir / p).exists()]
    if missing:
        print("Packer output missing required files:")
        for p in missing:
            print(f"- {p}")
        return 1

    if not emit_pruned:
        pruned_dir = out_dir / "PRUNED"
        if pruned_dir.exists():
            print("PRUNED directory exists when emit_pruned is OFF")
            return 1
    else:
        pruned_dir = out_dir / "PRUNED"
        if not pruned_dir.exists():
            print("WARNING: PRUNED directory not created (packer --emit-pruned not yet wired)")
            print("  PRUNED validation will be added when packer implements emit-pruned")
        else:
            pruned_required = ["PRUNED/PACK_MANIFEST_PRUNED.json", "PRUNED/meta/PRUNED_RULES.json"]
            pruned_missing = [p for p in pruned_required if not (out_dir / p).exists()]
            if pruned_missing:
                print("PRUNED output missing required files:")
                for p in pruned_missing:
                    print(f"- {p}")
                return 1

    archive_excluded = None
    if assert_archive_excluded and scope == "ags":
        repo_state_path = out_dir / "meta/REPO_STATE.json"
        try:
            repo_state = json.loads(repo_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error reading REPO_STATE.json for archive check: {exc}")
            return 1
        entries = repo_state.get("files", [])
        has_archive = any(e.get("path", "").startswith("MEMORY/ARCHIVE/") for e in entries)
        if has_archive:
            print("REPO_STATE.json includes MEMORY/ARCHIVE entries when they should be excluded")
            return 1
        archive_excluded = True

    start_here_text = (out_dir / "meta/START_HERE.md").read_text(encoding="utf-8", errors="replace")
    entrypoints_text = (out_dir / "meta/ENTRYPOINTS.md").read_text(encoding="utf-8", errors="replace")
    if scope == "ags":
        start_here_mentions = ["`repo/AGENTS.md`", "`repo/README.md`"]
        entrypoints_mentions = ["`repo/AGENTS.md`", "`repo/README.md`"]
    else:
        start_here_mentions = ["`repo/THOUGHT/LAB/`"]
        entrypoints_mentions = ["`repo/THOUGHT/LAB/`"]

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
        "emit_pruned": emit_pruned,
    }
    if archive_excluded is not None:
        output_payload["archive_excluded"] = archive_excluded
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1
        
    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "BUILD"
        ]
    )
    writer.open_commit_gate()

    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
