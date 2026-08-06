#!/usr/bin/env python3
"""Run frozen-source qualifiers for V3 candidates I-O."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/ags-audio-source-7c79414-v3")
FRONTIER = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/"
    "cat_cas_phase_frontier"
)
SOURCE_CAT = SOURCE / FRONTIER
RAW_LOG = ROOT / "raw_logs" / "source_reproduction_v3"
RAW_OUT = ROOT / "raw_outputs" / "source_reproduction_v3"


QUALIFIERS = {
    "I": "qualify_cyclotomic_ht_projective_orbit_obstruction.sh",
    "J": "qualify_phase_vm_root_bisimulation.sh",
    "K": "qualify_four_rotor_necklace_multi_port_tt.sh",
    "L": "qualify_f17_cubic_chain_period17_height_lower_bound.sh",
    "M": "qualify_wilczek_zee_nonabelian_phase_frame.sh",
    "N": "qualify_two_shared_latent_descriptor_catvm.sh",
    "O": "qualify_f17_cubic_chain_transfer.sh",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def git(args: list[str], cwd: Path = SOURCE) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def collect_files(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        out.append(
            {
                "path": str(item.relative_to(path)),
                "sha256": file_sha256(item),
                "bytes": item.stat().st_size,
            }
        )
    return out


def next_absent_dir(base: str) -> Path:
    out = RAW_OUT / base
    suffix = 0
    while out.exists():
        suffix += 1
        out = RAW_OUT / f"{base}_retry{suffix}"
    out.mkdir(parents=True)
    return out


def run_qualifier(candidate: str, label: str, out_dir: Path, timeout: int = 1800) -> dict[str, Any]:
    qualifier = SOURCE_CAT / QUALIFIERS[candidate]
    log_prefix = RAW_LOG / f"candidate_{candidate.lower()}_{label}"
    RAW_LOG.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(qualifier), str(out_dir)]
    start = time.monotonic_ns()
    proc = subprocess.run(
        command,
        cwd=SOURCE_CAT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONHASHSEED": "0"},
        timeout=timeout,
    )
    elapsed_ns = time.monotonic_ns() - start
    stdout = log_prefix.with_suffix(".stdout")
    stderr = log_prefix.with_suffix(".stderr")
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return {
        "candidate": candidate,
        "label": label,
        "qualifier": str(qualifier.relative_to(SOURCE)),
        "command": command,
        "cwd": str(SOURCE_CAT),
        "return_code": proc.returncode,
        "elapsed_ns": elapsed_ns,
        "stdout_path": str(stdout.relative_to(ROOT)),
        "stderr_path": str(stderr.relative_to(ROOT)),
        "stdout_sha256": file_sha256(stdout),
        "stderr_sha256": file_sha256(stderr),
        "stdout_bytes": stdout.stat().st_size,
        "stderr_bytes": stderr.stat().st_size,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "generated_files": collect_files(out_dir),
    }


def missing_arg_control(candidate: str) -> dict[str, Any]:
    qualifier = SOURCE_CAT / QUALIFIERS[candidate]
    proc = subprocess.run(
        ["bash", str(qualifier)],
        cwd=SOURCE_CAT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    stdout = RAW_LOG / f"candidate_{candidate.lower()}_missing_arg.stdout"
    stderr = RAW_LOG / f"candidate_{candidate.lower()}_missing_arg.stderr"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return {
        "candidate": candidate,
        "control": "missing_evidence_dir_argument",
        "return_code": proc.returncode,
        "stdout_path": str(stdout.relative_to(ROOT)),
        "stderr_path": str(stderr.relative_to(ROOT)),
        "stdout_sha256": file_sha256(stdout),
        "stderr_sha256": file_sha256(stderr),
        "stdout_bytes": stdout.stat().st_size,
        "stderr_bytes": stderr.stat().st_size,
    }


def stale_result_control(candidate: str) -> dict[str, Any]:
    out_dir = next_absent_dir(f"candidate_{candidate.lower()}_stale_preexisting")
    stale = out_dir / "result.json"
    stale.write_text(
        json.dumps({"result": "PASS", "stale_preexisting_control": True}) + "\n",
        encoding="utf-8",
    )
    before = file_sha256(stale)
    run = run_qualifier(candidate, "stale_preexisting", out_dir)
    after_exists = stale.exists()
    after = file_sha256(stale) if after_exists else None
    return {
        "candidate": candidate,
        "control": "preexisting_stale_result_json",
        "run": run,
        "stale_hash_before": before,
        "result_hash_after": after,
        "stale_result_overwritten_or_rejected": before != after,
    }


def executable_state() -> dict[str, Any]:
    out = {}
    for candidate, qualifier in QUALIFIERS.items():
        path = SOURCE_CAT / qualifier
        out[candidate] = {
            "path": str(path.relative_to(SOURCE)),
            "mode_octal": oct(path.stat().st_mode & 0o777),
            "executable": os.access(path, os.X_OK),
            "sha256": file_sha256(path),
            "git_object_mode": git(["ls-tree", "HEAD", "--", str(FRONTIER / qualifier)]).split()[0],
        }
    return out


def normalized_signature(run: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in run["generated_files"]:
        if item["path"] == "SHA256SUMS":
            continue
        if item["path"].endswith((".stderr", ".stderr.txt", "stderr.txt")):
            records.append(item)
            continue
        records.append(item)
    return records


def classify(runs: list[dict[str, Any]], missing: list[dict[str, Any]], stale: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    miss = {item["candidate"]: item for item in missing}
    stale_by = {item["candidate"]: item for item in stale}
    for candidate in QUALIFIERS:
        pair = [run for run in runs if run["candidate"] == candidate]
        if len(pair) != 2:
            out[candidate] = "SOURCE_EVIDENCE_INCOMPLETE"
        elif any(run["return_code"] != 0 for run in pair):
            out[candidate] = "SOURCE_NOT_REPRODUCED"
        elif miss[candidate]["return_code"] == 0:
            out[candidate] = "SOURCE_QUALIFIER_FAIL_OPEN"
        elif not stale_by[candidate]["stale_result_overwritten_or_rejected"]:
            out[candidate] = "SOURCE_QUALIFIER_FAIL_OPEN"
        else:
            sigs = [json_sha256(normalized_signature(run)) for run in pair]
            out[candidate] = "SOURCE_REPRODUCED" if sigs[0] == sigs[1] else "SOURCE_REPRODUCED_WITH_RUN_LOCAL_DIFFERENCES"
    return out


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Source Reproduction Report V3",
        "",
        "Status: source package reproduction only. Scientific verification is separate.",
        "",
        f"Scientific source SHA: `{payload['scientific_source_commit']}`",
        f"Source worktree: `{payload['source_worktree']}`",
        f"Source status before: `{payload['source_status_before'] or 'clean'}`",
        f"Source status after: `{payload['source_status_after'] or 'clean'}`",
        "",
        "## Classifications",
        "",
    ]
    for candidate, classification in payload["classifications"].items():
        lines.append(f"- Candidate {candidate}: `{classification}`")
    lines.extend(["", "## Runs", ""])
    for run in payload["runs"]:
        lines.append(
            f"- {run['candidate']} {run['label']}: rc={run['return_code']}, files={len(run['generated_files'])}, stdout={run['stdout_sha256']}, stderr={run['stderr_sha256']}"
        )
    lines.extend(["", "## Controls", ""])
    for control in payload["missing_arg_controls"]:
        lines.append(f"- {control['candidate']} missing-argument control: rc={control['return_code']}")
    for control in payload["stale_result_controls"]:
        lines.append(
            f"- {control['candidate']} stale-result control: overwritten_or_rejected={control['stale_result_overwritten_or_rejected']}, rc={control['run']['return_code']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    RAW_LOG.mkdir(parents=True, exist_ok=True)
    before = git(["status", "--short"])
    runs = []
    for candidate in QUALIFIERS:
        for index in (1, 2):
            out_dir = next_absent_dir(f"candidate_{candidate.lower()}_run{index}")
            runs.append(run_qualifier(candidate, f"run{index}", out_dir))
    missing = [missing_arg_control(candidate) for candidate in QUALIFIERS]
    stale = [stale_result_control(candidate) for candidate in QUALIFIERS]
    after = git(["status", "--short"])
    payload = {
        "schema_version": "audio_noncollapse_v3_source_reproduction",
        "created_utc": utc_now(),
        "canonical": False,
        "small_wall_crossed": False,
        "scientific_source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_tree": git(["rev-parse", "HEAD^{tree}"]),
        "source_worktree": str(SOURCE),
        "source_status_before": before,
        "source_status_after": after,
        "executable_state": executable_state(),
        "runs": runs,
        "missing_arg_controls": missing,
        "stale_result_controls": stale,
    }
    payload["classifications"] = classify(runs, missing, stale)
    payload["sha256"] = json_sha256(payload)
    (ROOT / "SOURCE_REPRODUCTION_DATA_V3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "SOURCE_REPRODUCTION_REPORT_V3.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(json.dumps({"classifications": payload["classifications"], "runs": len(runs), "controls": len(missing) + len(stale)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
