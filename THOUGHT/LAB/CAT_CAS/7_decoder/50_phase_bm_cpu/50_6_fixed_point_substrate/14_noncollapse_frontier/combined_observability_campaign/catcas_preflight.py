#!/usr/bin/env python3
"""Read-only preflight for the authorized combined campaign on catcas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from generate_campaign_plan import verify


def command(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout.strip()


def first_k10temp() -> str | None:
    for name in sorted(Path("/sys/class/hwmon").glob("hwmon*/name")):
        try:
            if name.read_text().strip() == "k10temp":
                candidate = name.parent / "temp1_input"
                if candidate.is_file():
                    return str(candidate)
        except OSError:
            pass
    return None


def cpu_flags() -> set[str]:
    flags: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags.update(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    return flags


def inspect(plan_dir: Path, repo_root: Path, output_root: Path, min_free_gb: float) -> dict[str, Any]:
    manifest = json.loads((plan_dir / "campaign_manifest.json").read_text())
    plan = json.loads((plan_dir / "campaign_plan.json").read_text())
    rc_head, head = command("git", "rev-parse", "HEAD", cwd=repo_root)
    rc_status, status = command("git", "status", "--short", cwd=repo_root)
    flags = cpu_flags()
    cpus = os.cpu_count() or 0
    msr = {str(core): os.access(f"/dev/cpu/{core}/msr", os.R_OK) for core in range(min(cpus, 6))}
    cpufreq = {
        str(core): all(Path(f"/sys/devices/system/cpu/cpu{core}/cpufreq/{name}").is_file() for name in ("scaling_min_freq", "scaling_max_freq"))
        for core in range(min(cpus, 6))
    }
    usage = shutil.disk_usage(output_root.parent if output_root.parent.exists() else repo_root)
    plan_errors = verify(plan_dir)
    checks = {
        "running_as_root": os.geteuid() == 0,
        "cpu_count_at_least_6": cpus >= 6,
        "constant_tsc": "constant_tsc" in flags,
        "nonstop_tsc": "nonstop_tsc" in flags,
        "k10temp_available": first_k10temp() is not None,
        "msr_readable_cores_0_5": len(msr) == 6 and all(msr.values()),
        "cpufreq_controls_cores_0_5": len(cpufreq) == 6 and all(cpufreq.values()),
        "free_space_sufficient": usage.free >= int(min_free_gb * 1024**3),
        "repo_head_resolved": rc_head == 0,
        "repo_clean": rc_status == 0 and status == "",
        "repo_head_matches_plan": rc_head == 0 and head == manifest.get("source_commit") == plan.get("source_commit"),
        "plan_manifest_valid": not plan_errors,
        "output_path_unused": not output_root.exists(),
        "restoration_not_authorized": plan.get("restoration_authorized") is False and manifest.get("restoration_authorized") is False,
    }
    return {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_PREFLIGHT_V1",
        "host": os.uname().nodename,
        "repo_root": str(repo_root),
        "repo_head": head if rc_head == 0 else None,
        "plan_dir": str(plan_dir),
        "plan_sha256": manifest.get("campaign_plan", {}).get("sha256"),
        "output_root": str(output_root),
        "cpu_count": cpus,
        "k10temp_path": first_k10temp(),
        "msr_readable": msr,
        "cpufreq_controls": cpufreq,
        "free_bytes": usage.free,
        "minimum_free_gb": min_free_gb,
        "plan_validation_errors": plan_errors,
        "checks": checks,
        "acquisition_ready": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()
    report = inspect(args.plan_dir.resolve(), args.repo_root.resolve(), args.output_root.resolve(), args.min_free_gb)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["acquisition_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
