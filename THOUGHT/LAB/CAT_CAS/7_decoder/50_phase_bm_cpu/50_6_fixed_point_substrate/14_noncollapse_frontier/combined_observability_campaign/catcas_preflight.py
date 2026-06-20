#!/usr/bin/env python3
"""Read-only engineering/acquisition preflight for the Phase 6 combined campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from generate_campaign_plan import verify

ROUTES = {"v4s5": (4, 5), "v2s3": (2, 3)}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value)) and set(value) != {"0"}


def safe_bundle_path(root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


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
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def verify_bound_files(root: Path, bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    files = bundle.get("files")
    if not isinstance(files, dict) or not files:
        return ["source bundle has no file bindings"]
    for relative, binding in sorted(files.items()):
        path = safe_bundle_path(root, relative)
        if path is None:
            errors.append(f"invalid bundle path: {relative}")
            continue
        if not isinstance(binding, dict):
            errors.append(f"invalid binding object: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing bound file: {relative}")
            continue
        if path.stat().st_size != binding.get("size"):
            errors.append(f"size mismatch: {relative}")
        if sha256_file(path) != binding.get("sha256"):
            errors.append(f"sha256 mismatch: {relative}")
    return errors


def session_bundles_valid(root: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if len(directories) != 12:
        errors.append(f"expected 12 compiled sessions, found {len(directories)}")
    for directory in directories:
        try:
            manifest = json.loads((directory / "session_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("schema_id") != "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1":
                errors.append(f"{directory.name}: bad manifest schema")
                continue
            root_resolved = directory.resolve()
            for name, binding in manifest.get("files", {}).items():
                path = (directory / name).resolve()
                try:
                    path.relative_to(root_resolved)
                except ValueError:
                    errors.append(f"{directory.name}: path escape {name}")
                    continue
                if not path.is_file():
                    errors.append(f"{directory.name}: missing {name}")
                    continue
                if path.stat().st_size != binding.get("size"):
                    errors.append(f"{directory.name}: size mismatch {name}")
                if sha256_file(path) != binding.get("sha256"):
                    errors.append(f"{directory.name}: sha mismatch {name}")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{directory.name}: invalid session bundle: {exc}")
    return not errors, errors


def evidence_valid(bundle_root: Path, bundle: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    evidence = bundle.get("evidence")
    if not isinstance(evidence, dict):
        return False, ["source bundle missing evidence map"]

    for key in ("strict_test_log", "sanitizer_test_log"):
        path = safe_bundle_path(bundle_root, evidence.get(key))
        if path is None or not path.is_file():
            errors.append(f"missing evidence file: {key}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "OK" not in text or "FAILED" in text or "ERROR" in text:
            errors.append(f"test evidence not clean: {key}")

    validation_path = safe_bundle_path(bundle_root, evidence.get("validation_report"))
    if validation_path is None or not validation_path.is_file():
        errors.append("missing validation report")
    else:
        try:
            report = json.loads(validation_path.read_text(encoding="utf-8"))
            if report.get("schema_id") != "CAT_CAS_PHASE6_VALIDATION_EVIDENCE_V1":
                errors.append("unexpected validation report schema")
            if report.get("sessions_expected") != 12:
                errors.append("validation report expected-session count is not 12")
            if report.get("sessions_passed") != 12 or report.get("all_pass") is not True:
                errors.append("validation report does not prove 12/12 pass")
            if len(report.get("records", [])) != 12:
                errors.append("validation report record count is not 12")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid validation report: {exc}")
    return not errors, errors


def authorization_valid(path: Path | None, *, bundle_path: Path,
                        bundle: dict[str, Any], output_root: Path) -> tuple[bool, dict[str, Any], list[str]]:
    if path is None:
        return False, {}, ["acquisition authorization artifact not supplied"]
    errors: list[str] = []
    try:
        authorization = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, {}, [f"invalid authorization artifact: {exc}"]

    if authorization.get("schema_id") != "CAT_CAS_PHASE6_ACQUISITION_AUTHORIZATION_V1":
        errors.append("unexpected authorization schema")
    if authorization.get("acquisition_authorized") is not True:
        errors.append("acquisition_authorized is not true")
    if authorization.get("restoration_authorized") is not False:
        errors.append("restoration must remain unauthorized")
    if authorization.get("executor_commit") != bundle.get("executor_commit"):
        errors.append("authorization executor commit mismatch")
    if authorization.get("campaign_plan_sha256") != bundle.get("campaign_plan_sha256"):
        errors.append("authorization campaign plan mismatch")
    if authorization.get("source_bundle_sha256") != sha256_file(bundle_path):
        errors.append("authorization source bundle mismatch")
    if authorization.get("authorized_output_root") != str(output_root):
        errors.append("authorization output root mismatch")
    if not authorization.get("authorized_by"):
        errors.append("authorization missing authorized_by")
    return not errors, authorization, errors


def inspect(plan_dir: Path, bundle_root: Path, output_root: Path,
            min_free_gb: float, authorization_path: Path | None = None,
            target_facts_path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((plan_dir / "campaign_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((plan_dir / "campaign_plan.json").read_text(encoding="utf-8"))
    bundle_path = bundle_root / "source_bundle.json"
    if not bundle_path.is_file():
        raise ValueError("preflight requires source_bundle.json")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    runner = bundle_root / "combined_pdn_runner"
    binding_path = bundle_root / "COMBINED_CAMPAIGN_BINDING.json"
    schedules = bundle_root / "compiled_sessions"
    binding = json.loads(binding_path.read_text(encoding="utf-8")) if binding_path.is_file() else {}

    target_facts: dict[str, Any] = {}
    if target_facts_path is not None:
        target_facts_path = target_facts_path.resolve()
        try:
            target_facts_path.relative_to(bundle_root.resolve())
        except ValueError as exc:
            raise ValueError("target facts must be inside the sealed bundle") from exc
        target_facts = json.loads(target_facts_path.read_text(encoding="utf-8"))
        if target_facts.get("schema_id") != "CAT_CAS_PHASE6_TARGET_HOST_FACTS_V1":
            raise ValueError("unexpected target host facts schema")
    flags = set(target_facts.get("cpu_flags", [])) if target_facts else cpu_flags()
    cpus = int(target_facts.get("cpu_count", 0)) if target_facts else (os.cpu_count() or 0)
    msr = target_facts.get("msr_readable", {}) if target_facts else {
        str(core): os.access(f"/dev/cpu/{core}/msr", os.R_OK)
        for core in range(min(cpus, 6))}
    cpufreq = target_facts.get("cpufreq_controls", {}) if target_facts else {
        str(core): all(os.access(
            f"/sys/devices/system/cpu/cpu{core}/cpufreq/{name}", os.R_OK | os.W_OK)
            for name in ("scaling_min_freq", "scaling_max_freq"))
        for core in range(min(cpus, 6))}
    usage = shutil.disk_usage(output_root.parent if output_root.parent.exists() else bundle_root)
    plan_errors = verify(plan_dir)
    plan_hash = sha256_file(plan_dir / "campaign_plan.json")
    manifest_hash = sha256_file(plan_dir / "campaign_manifest.json")
    route_ok = (all(victim != sender and victim < cpus and sender < cpus
                    for victim, sender in ROUTES.values())
                and {item.get("route") for item in plan.get("sessions", [])} == set(ROUTES))

    bound_file_errors = verify_bound_files(bundle_root, bundle)
    sessions_ok, session_errors = session_bundles_valid(schedules)
    evidence_ok, evidence_errors = evidence_valid(bundle_root, bundle)
    authorization_ok, authorization, authorization_errors = authorization_valid(
        authorization_path, bundle_path=bundle_path, bundle=bundle, output_root=output_root)

    executor_commit = bundle.get("executor_commit")
    engineering_checks = {
        "bundle_schema_valid": bundle.get("schema_id") == "CAT_CAS_PHASE6_EXECUTOR_SOURCE_BUNDLE_V1",
        "bundle_class_is_non_authorizing": bundle.get("artifact_class") == "ENGINEERING_BUNDLE_NOT_ACQUISITION_AUTHORIZATION",
        "bundle_files_hash_bound": not bound_file_errors,
        "executor_commit_valid": valid_commit(executor_commit),
        "executor_exists_and_executable": runner.is_file() and os.access(runner, os.X_OK),
        "executor_binary_hash_valid": runner.is_file() and sha256_file(runner) == bundle.get("executor_sha256"),
        "plan_sources_agree": manifest.get("source_commit") == plan.get("source_commit"),
        "plan_manifest_valid": not plan_errors,
        "plan_hash_matches_bundle": plan_hash == bundle.get("campaign_plan_sha256"),
        "campaign_manifest_hash_matches_bundle": manifest_hash == bundle.get("campaign_manifest_sha256"),
        "canonical_plan_binding_valid": plan_hash == binding.get("campaign_plan", {}).get("sha256") and manifest_hash == binding.get("campaign_manifest_sha256"),
        "all_twelve_schedules_compiled": sessions_ok,
        "test_and_validation_evidence_valid": evidence_ok,
        "output_path_unused": not output_root.exists(),
        "restoration_not_authorized": plan.get("restoration_authorized") is False and manifest.get("restoration_authorized") is False and bundle.get("restoration_authorized") is False,
    }
    host_checks = {
        "running_as_root": target_facts.get("running_as_root") is True if target_facts else os.geteuid() == 0,
        "cpu_count_at_least_6": cpus >= 6,
        "route_cores_online_and_distinct": route_ok,
        "constant_tsc": "constant_tsc" in flags,
        "nonstop_tsc": "nonstop_tsc" in flags,
        "k10temp_available": target_facts.get("k10temp_available") is True if target_facts else first_k10temp() is not None,
        "msr_readable_cores_0_5": len(msr) == 6 and all(msr.values()),
        "cpufreq_controls_readable_writable_cores_0_5": len(cpufreq) == 6 and all(cpufreq.values()),
        "free_space_sufficient": int(target_facts.get("free_bytes", 0)) >= int(min_free_gb * 1024**3) if target_facts else usage.free >= int(min_free_gb * 1024**3),
        "sender_cleanup_verified": target_facts.get("sender_cleanup_verified") is True if target_facts else True,
        "host_control_state_restored": target_facts.get("host_control_state_restored") is True if target_facts else True,
    }
    engineering_ready = all(engineering_checks.values()) and all(host_checks.values())
    acquisition_ready = engineering_ready and authorization_ok

    return {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_PREFLIGHT_V3",
        "host": target_facts.get("host", os.uname().nodename if hasattr(os, "uname") else "local"),
        "bundle_root": str(bundle_root),
        "source_bundle_sha256": sha256_file(bundle_path),
        "executor_commit": executor_commit,
        "plan_source_commit": manifest.get("source_commit"),
        "plan_dir": str(plan_dir),
        "plan_sha256": plan_hash,
        "campaign_manifest_sha256": manifest_hash,
        "output_root": str(output_root),
        "cpu_count": cpus,
        "k10temp_path": target_facts.get("k10temp_path") if target_facts else first_k10temp(),
        "msr_readable": msr,
        "cpufreq_controls": cpufreq,
        "free_bytes": usage.free,
        "minimum_free_gb": min_free_gb,
        "plan_validation_errors": plan_errors,
        "bound_file_errors": bound_file_errors,
        "session_bundle_errors": session_errors,
        "evidence_errors": evidence_errors,
        "authorization_errors": authorization_errors,
        "authorization": authorization,
        "engineering_checks": engineering_checks,
        "host_checks": host_checks,
        "engineering_ready": engineering_ready,
        "acquisition_authorized": authorization_ok,
        "acquisition_ready": acquisition_ready,
        "scientific_acquisition_started": False,
        "physical_carrier_restoration_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--bundle-root", "--repo-root", dest="bundle_root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--engineering-only", action="store_true",
                        help="return success for engineering readiness even without acquisition authorization")
    parser.add_argument("--target-facts", type=Path,
                        help="sealed CAT_CAS host facts for local engineering-only preflight")
    args = parser.parse_args()
    try:
        report = inspect(
            args.plan_dir.resolve(), args.bundle_root.resolve(), args.output_root.resolve(),
            args.min_free_gb, args.authorization.resolve() if args.authorization else None,
            args.target_facts.resolve() if args.target_facts else None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.engineering_only:
        return 0 if report["engineering_ready"] else 2
    return 0 if report["acquisition_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
