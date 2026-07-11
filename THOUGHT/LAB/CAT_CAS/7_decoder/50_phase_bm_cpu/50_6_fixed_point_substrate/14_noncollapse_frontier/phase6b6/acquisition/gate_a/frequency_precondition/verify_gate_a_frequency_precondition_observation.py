#!/usr/bin/env python3
"""Verify the sealed read-only Gate A frequency observation.

The immutable integrated source and sealed observation commit are the null
baseline for these custody checks. This verifier opens no network connection,
performs no target operation, and exposes no frequency-control surface.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[10]
INTEGRATED_MAIN = "dde70523d639d34defa37f1b55982e789ea52c18"
EVIDENCE_COMMIT = "f617e288d93eb20d3845356b4c233d15d8cf5c2d"
EVIDENCE_TREE = "067aec7c91438ef99fe533279db561884f6f9fac"
PROBE_BLOB = "296e4ee9624fb5a5685148423e337bff6c0c4cec"
PROBE_SHA256 = "03138d8c30078940001b018df4cea250dca6e7ec4b65122f72c74a5ef4a50762"
PROBE_PATH = HERE / "gate_a_frequency_precondition_probe.py"
EVIDENCE_ROOT = (
    HERE.parents[2]
    / "evidence"
    / "gate_a_frequency_precondition_observation_dde70523_01"
)
EXPECTED_FILES = {
    "HOST_COMMAND.json",
    "PROBE_RECEIPT.json",
    "PROBE_SOURCE.py",
    "README.md",
    "RESULT.json",
    "SOURCE_BINDING.json",
    "STDERR.txt",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_git_custody() -> dict[str, Any]:
    evidence_rel = rel(EVIDENCE_ROOT)
    require(
        run("git", "merge-base", "--is-ancestor", INTEGRATED_MAIN, EVIDENCE_COMMIT, check=False).returncode == 0,
        "integrated main is not an ancestor of evidence commit",
    )
    require(
        run("git", "rev-parse", f"{EVIDENCE_COMMIT}^").stdout.strip() == INTEGRATED_MAIN,
        "evidence commit parent mismatch",
    )
    sealed = run("git", "rev-parse", f"{EVIDENCE_COMMIT}:{evidence_rel}").stdout.strip()
    current = run("git", "rev-parse", f"HEAD:{evidence_rel}").stdout.strip()
    require(sealed == EVIDENCE_TREE, "sealed evidence tree mismatch")
    require(current == EVIDENCE_TREE, "current evidence tree changed")
    require(
        run("git", "diff", "--quiet", EVIDENCE_COMMIT, "HEAD", "--", evidence_rel, check=False).returncode == 0,
        "evidence changed after sealing",
    )
    status = run("git", "status", "--porcelain=v1", "--untracked-files=all", "--", evidence_rel, check=False)
    require(status.returncode == 0 and status.stdout == "", "evidence differs from HEAD")
    return {"status": "SEALED_EVIDENCE_TREE_EXACT", "tree_sha1": EVIDENCE_TREE}


def verify_inventory() -> dict[str, Any]:
    inventory_path = EVIDENCE_ROOT / "FINAL_INVENTORY.json"
    inventory = load(inventory_path)
    require(inventory["self_excluded"] is True, "inventory must exclude itself")
    require(inventory["file_count"] == len(EXPECTED_FILES), "inventory file count mismatch")
    entries = inventory["files"]
    require(isinstance(entries, list), "inventory files must be a list")
    by_path = {entry["path"]: entry for entry in entries}
    require(set(by_path) == EXPECTED_FILES, f"inventory path set mismatch: {sorted(by_path)}")
    physical = {path.name for path in EVIDENCE_ROOT.iterdir() if path.is_file()}
    require(physical == EXPECTED_FILES | {"FINAL_INVENTORY.json"}, f"physical file set mismatch: {sorted(physical)}")
    for name, entry in by_path.items():
        path = EVIDENCE_ROOT / name
        require(path.stat().st_size == entry["size_bytes"], f"size mismatch: {name}")
        require(sha256(path) == entry["sha256"], f"SHA-256 mismatch: {name}")
    return {
        "status": "FINAL_INVENTORY_CLOSES",
        "retained_file_count": len(physical),
        "inventory_sha256": sha256(inventory_path),
    }


def verify_source_binding() -> dict[str, Any]:
    binding = load(EVIDENCE_ROOT / "SOURCE_BINDING.json")
    require(binding["integrated_main_commit"] == INTEGRATED_MAIN, "integrated source binding mismatch")
    require(binding["pr_42_source_head"] == "fa95603b12afd5a1cbb61254efe855bf1aaa23e8", "PR source binding mismatch")
    require(binding["source_review_comment_id"] == "4948537605", "review comment binding mismatch")
    require(binding["source_review_status"] == "GATE_A_FREQUENCY_PRECONDITION_REVIEW_EXACT", "review status mismatch")
    probe = binding["probe"]
    require(probe["git_blob"] == PROBE_BLOB, "probe blob binding mismatch")
    require(probe["sha256"] == PROBE_SHA256, "probe SHA-256 binding mismatch")
    require(probe["byte_size"] == PROBE_PATH.stat().st_size == 11894, "probe byte-size mismatch")
    copied = EVIDENCE_ROOT / "PROBE_SOURCE.py"
    require(copied.read_bytes() == PROBE_PATH.read_bytes(), "retained probe differs from integrated source")
    require(sha256(copied) == PROBE_SHA256, "retained probe SHA-256 mismatch")
    require(run("git", "rev-parse", f"HEAD:{rel(PROBE_PATH)}").stdout.strip() == PROBE_BLOB, "integrated probe blob mismatch")
    require(binding["target"] == "root@192.168.137.100", "target binding mismatch")
    require(binding["sample_count"] == 200 and binding["sample_interval_ms"] == 10, "sampling binding mismatch")
    require(binding["required_frequency_khz"] == 1_600_000, "required frequency binding mismatch")
    return {"status": "PROBE_SOURCE_AND_REVIEW_BINDING_EXACT", "probe_sha256": PROBE_SHA256}


def verify_host_command() -> dict[str, Any]:
    command = load(EVIDENCE_ROOT / "HOST_COMMAND.json")
    require(command["command_array"] == [
        "ssh.exe", "root@192.168.137.100", "python3", "-", "--sample-count", "200", "--interval-ms", "10"
    ], "host command array mismatch")
    require(command["target_contact_count"] == 1, "target contact count mismatch")
    require(command["ssh_invocation_count"] == 1, "SSH invocation count mismatch")
    require(command["scp_invocation_count"] == 0, "SCP invocation count mismatch")
    require(command["retry_count"] == 0, "retry count mismatch")
    require(command["process_started"] is True and command["timed_out"] is False, "process start/timeout mismatch")
    require(command["return_code"] == 0 and command["process_failure"] is None, "SSH process result mismatch")
    require(command["stdin"]["sha256"] == PROBE_SHA256 and command["stdin"]["size_bytes"] == 11894, "stdin custody mismatch")
    require(command["stdout"]["sha256"] == sha256(EVIDENCE_ROOT / "PROBE_RECEIPT.json"), "stdout custody mismatch")
    require(command["stderr"]["sha256"] == sha256(EVIDENCE_ROOT / "STDERR.txt"), "stderr custody mismatch")
    require((EVIDENCE_ROOT / "STDERR.txt").read_bytes() == b"", "stderr must be empty")
    return {"status": "ONE_READ_ONLY_SSH_CONTACT_EXACT", "elapsed_seconds": command["elapsed_monotonic_seconds"]}


def verify_receipt() -> dict[str, Any]:
    receipt = load(EVIDENCE_ROOT / "PROBE_RECEIPT.json")
    require(receipt["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PRECONDITION_OBSERVATION_V1", "receipt schema mismatch")
    require(receipt["status"] == "FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED", "receipt status mismatch")
    require(receipt["failure"] == "REQUIRED_FREQUENCY_NEVER_OBSERVED_ON_BOTH_CORES", "receipt failure mismatch")
    require(receipt["observation_mode"] == "IDLE_READ_ONLY", "observation mode mismatch")
    require(receipt["cores"] == [4, 5], "core binding mismatch")
    require(receipt["required_frequency_khz"] == 1_600_000, "required frequency mismatch")
    require(receipt["sample_count_requested"] == 200 and receipt["sample_interval_ms"] == 10, "sampling configuration mismatch")
    for key in ("control_writes", "frequency_writes", "voltage_writes", "msr_reads", "msr_writes", "network_operations"):
        require(receipt[key] == 0, f"nonzero prohibited operation count: {key}")
    metadata = {str(item["core"]): item for item in receipt["policy_metadata"]}
    require(set(metadata) == {"4", "5"}, "policy metadata core set mismatch")
    for core in ("4", "5"):
        item = metadata[core]
        require(item["files"]["scaling_driver"]["parsed"] == "acpi-cpufreq", f"driver mismatch: core {core}")
        require(item["files"]["scaling_governor"]["parsed"] == "schedutil", f"governor mismatch: core {core}")
        require(item["files"]["cpuinfo_min_freq"]["parsed"] == 800000, f"cpuinfo minimum mismatch: core {core}")
        require(item["files"]["cpuinfo_max_freq"]["parsed"] == 3200000, f"cpuinfo maximum mismatch: core {core}")
        require(item["files"]["scaling_min_freq"]["parsed"] == 800000, f"scaling minimum mismatch: core {core}")
        require(item["files"]["scaling_max_freq"]["parsed"] == 3200000, f"scaling maximum mismatch: core {core}")
        require(item["required_frequency_supported"] is True, f"required frequency unsupported: core {core}")
        require(item["identity"]["resolved_path"] == f"/sys/devices/system/cpu/cpufreq/policy{core}", f"policy identity mismatch: core {core}")
    samples = receipt["samples"]
    require(len(samples) == 200, "sample count mismatch")
    previous_monotonic = -1
    previous_utc = -1
    for index, sample in enumerate(samples):
        require(sample["index"] == index, f"sample index mismatch: {index}")
        require(sample["frequency_khz"] == {"4": 800000, "5": 800000}, f"frequency sample mismatch: {index}")
        require(sample["pair_exact"] is False, f"unexpected exact pair: {index}")
        require(sample["monotonic_ns"] > previous_monotonic, f"monotonic time not increasing: {index}")
        require(sample["utc_ns"] > previous_utc, f"UTC time not increasing: {index}")
        previous_monotonic = sample["monotonic_ns"]
        previous_utc = sample["utc_ns"]
    summary = receipt["summary"]
    require(summary["pair_exact_sample_count"] == 0, "paired exact count mismatch")
    require(summary["longest_consecutive_exact_pairs"] == 0, "longest exact run mismatch")
    require(summary["all_pairs_exact"] is False and summary["any_pair_exact"] is False, "pair summary mismatch")
    for core in ("4", "5"):
        observed = summary["per_core"][core]
        require(observed["minimum_khz"] == observed["maximum_khz"] == 800000, f"per-core range mismatch: {core}")
        require(observed["unique_khz"] == [800000], f"per-core unique set mismatch: {core}")
        require(observed["exact_sample_count"] == 0 and observed["ever_exact"] is False, f"per-core exact summary mismatch: {core}")
    return {"status": "FLAT_800MHZ_OBSERVATION_EXACT", "sample_count": 200}


def verify_result() -> dict[str, Any]:
    result = load(EVIDENCE_ROOT / "RESULT.json")
    require(result["status"] == "FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED", "result status mismatch")
    require(result["failure"] == "REQUIRED_FREQUENCY_NEVER_OBSERVED_ON_BOTH_CORES", "result failure mismatch")
    require(result["receipt_sha256"] == sha256(EVIDENCE_ROOT / "PROBE_RECEIPT.json"), "result receipt binding mismatch")
    require(result["target_contacts"] == 1 and result["ssh_invocations"] == 1, "result contact count mismatch")
    require(result["scp_invocations"] == 0 and result["retry_count"] == 0, "result retry/SCP mismatch")
    require(result["smoke_executions"] == 0, "smoke execution unexpectedly recorded")
    for key in ("control_writes", "frequency_writes", "voltage_writes", "msr_accesses", "msr_reads", "msr_writes"):
        require(result[key] == 0, f"nonzero prohibited result count: {key}")
    require(result["next_boundary"] == "SEPARATE_FREQUENCY_PREPARATION_RESTORATION_DESIGN", "next boundary mismatch")
    return {"status": "RESULT_SUMMARY_EXACT", "next_boundary": result["next_boundary"]}


def verify_resting_state() -> dict[str, Any]:
    review = HERE / "verify_gate_a_frequency_precondition_review.py"
    completed = run(sys.executable, str(review), check=False)
    require(completed.returncode == 0, f"integrated review verifier failed: {completed.stderr}")
    active = HERE.parent / "adapter" / "GATE_A_EXECUTION_AUTHORITY.json"
    require(not active.exists(), "active Gate A authority exists")
    return {
        "status": "NO_ACTIVE_AUTHORITY__NO_SMOKE_AUTHORIZED",
        "frequency_preparation_writes_authorized": False,
        "third_gate_a_attempt_authorized": False,
        "gate_b_authorized": False,
    }


def main() -> int:
    try:
        result = {
            "status": "GATE_A_FREQUENCY_PRECONDITION_OBSERVATION_VALID",
            "git_custody": verify_git_custody(),
            "inventory": verify_inventory(),
            "source_binding": verify_source_binding(),
            "host_command": verify_host_command(),
            "receipt": verify_receipt(),
            "result": verify_result(),
            "resting_state": verify_resting_state(),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, subprocess.CalledProcessError, VerifyError) as exc:
        print(f"GATE_A_FREQUENCY_PRECONDITION_OBSERVATION_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
