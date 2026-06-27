#!/usr/bin/env python3
"""Reproduce the frozen combined plan and all compiled sessions."""
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent
PACKAGE = FRONTIER / "combined_observability_campaign"
RATIFICATION = HERE / "PROJECT_OWNER_RATIFICATION.json"
BINDING = HERE / "COMBINED_CAMPAIGN_BINDING.json"
sys.path.insert(0, str(PACKAGE))

from generate_campaign_plan import generate, sha256_file, verify  # noqa: E402
from compile_session_schedule import write_session  # noqa: E402


def git(repo: Path, *args: str) -> int:
    return subprocess.run(("git", *args), cwd=repo, check=False).returncode


def compare_dirs(left: Path, right: Path) -> bool:
    names = sorted({p.name for p in left.iterdir()} | {p.name for p in right.iterdir()})
    return all((left / name).is_file() and (right / name).is_file() and filecmp.cmp(left / name, right / name, shallow=False) for name in names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    binding = json.loads(BINDING.read_text())
    if binding.get("active_authority") is False:
        auth = binding.get("authorization", {})
        for field in (
            "campaign_implementation_authorized",
            "physical_acquisition_authorized_after_preflight",
            "physical_acquisition_executed",
            "hardware_ran",
            "authorization_artifact_created",
            "calibration_authorized",
            "scientific_acquisition_authorized",
            "restoration_authorized",
            "target_coupling_authorized",
            "orientation_recovery_authorized",
            "small_wall_authorized",
            "phase6b6_entered",
        ):
            if auth.get(field) is not False:
                raise SystemExit(f"inactive campaign binding authorizes {field}")
        if binding.get("owner_decision") != "APPROVED_FOR_INTEGRATION":
            raise SystemExit("inactive campaign binding owner decision mismatch")
        if binding.get("next_boundary") != "PHASE6B6_REQUIRES_SEPARATE_FUTURE_AUTHORITY":
            raise SystemExit("inactive campaign binding boundary mismatch")
        print("COMBINED_CAMPAIGN_BINDING_INACTIVE_NO_AUTHORITY")
        return 0
    source = binding["plan_source_commit"]
    if git(repo, "merge-base", "--is-ancestor", source, "HEAD") != 0:
        raise SystemExit("frozen source is not an ancestor of HEAD")
    frozen_inputs = [
        PACKAGE / "campaign_orders.py",
        PACKAGE / "campaign_plan.py",
        PACKAGE / "generate_campaign_plan.py",
        PACKAGE / "CAMPAIGN_CONTRACT.md",
        PACKAGE / "ANALYSIS_CONTRACT.md",
        RATIFICATION,
    ]
    frozen_rel = [str(path.relative_to(repo)) for path in frozen_inputs]
    if git(repo, "diff", "--quiet", source, "HEAD", "--", *frozen_rel) != 0:
        raise SystemExit("authorized package changed after frozen source")
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        repeat = tmp / "repeat"
        generate(output, source, RATIFICATION)
        generate(repeat, source, RATIFICATION)
        if not compare_dirs(output, repeat):
            raise SystemExit("double generation differs")
        errors = verify(output)
        if errors:
            raise SystemExit(f"plan verification failed: {errors}")
        if sha256_file(output / "campaign_plan.json") != binding["campaign_plan"]["sha256"]:
            raise SystemExit("campaign plan hash mismatch")
        if sha256_file(output / "campaign_manifest.json") != binding["campaign_manifest_sha256"]:
            raise SystemExit("campaign manifest hash mismatch")
        sessions = tmp / "sessions"; sessions.mkdir()
        plan_path = output / "campaign_plan.json"
        for route in ("v4s5", "v2s3"):
            for seed in range(6):
                write_session(plan_path, f"{route}_seed{seed}", sessions / f"{route}_seed{seed}")
        again = tmp / "again"
        write_session(plan_path, "v4s5_seed4", again)
        if not compare_dirs(sessions / "v4s5_seed4", again):
            raise SystemExit("session compilation is not deterministic")
    print("COMBINED_CAMPAIGN_BINDING_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
