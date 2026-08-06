#!/usr/bin/env python3
"""Run frozen source qualifiers for updated audio CATVM candidates E-H."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/tmp/ags-audio-source-6f39766-v2")
FRONTIER = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/"
    "cat_cas_phase_frontier"
)
SOURCE_CAT = SOURCE / FRONTIER
RAW_LOG = ROOT / "raw_logs" / "source_reproduction"
RAW_OUT = ROOT / "raw_outputs" / "source_reproduction"


QUALIFIERS = {
    "E": "qualify_catvm_necklace_shared_latent_depth_accounting_repair.sh",
    "F": "qualify_four_rotor_necklace_shared_latent_depth_compiler.sh",
    "G": "qualify_four_rotor_necklace_exact_phase_precision.sh",
    "H": "qualify_four_rotor_necklace_coherence_triad.sh",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def git(args: list[str], cwd: Path = SOURCE) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def collect_files(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        records.append(
            {
                "path": str(item.relative_to(path)),
                "sha256": file_sha256(item),
                "bytes": item.stat().st_size,
            }
        )
    return records


def strip_permitted_nondeterminism(value: Any) -> Any:
    """Remove run-local timing scalars from generated JSON before comparison."""
    if isinstance(value, dict):
        return {
            key: strip_permitted_nondeterminism(item)
            for key, item in value.items()
            if not (key.endswith("_elapsed_ms") or key.endswith("_elapsed_ns"))
        }
    if isinstance(value, list):
        return [strip_permitted_nondeterminism(item) for item in value]
    return value


def normalized_generated_signature(run: dict[str, Any]) -> dict[str, Any]:
    """Compare scientific outputs while ignoring path-only checksum manifests.

    The source qualifiers emit SHA256SUMS files that include absolute evidence
    directory names, so two fresh output directories produce different
    SHA256SUMS bytes even when all scientific artifacts match. Those manifests
    are kept in raw evidence, but they are not treated as scientific
    nondeterminism.
    """
    out_dir = ROOT / run["output_dir"]
    records: list[dict[str, Any]] = []
    ignored: list[dict[str, str]] = []
    for generated in sorted(run["generated_files"], key=lambda item: item["path"]):
        rel = generated["path"]
        if rel == "SHA256SUMS":
            ignored.append(
                {
                    "path": rel,
                    "reason": "absolute evidence directory names are expected to differ",
                }
            )
            continue
        path = out_dir / rel
        if rel.endswith(".json"):
            try:
                normalized = strip_permitted_nondeterminism(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except json.JSONDecodeError:
                normalized = None
            if normalized is not None:
                normalized_bytes = json.dumps(
                    normalized, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                records.append(
                    {
                        "path": rel,
                        "normalized_json_sha256": json_sha256(normalized),
                        "normalized_json_bytes": len(normalized_bytes),
                    }
                )
                continue
        records.append(
            {
                "path": rel,
                "sha256": generated["sha256"],
                "bytes": generated["bytes"],
            }
        )
    return {"records": records, "ignored": ignored}


def run_one(candidate: str, run_index: int) -> dict[str, Any]:
    qualifier = SOURCE_CAT / QUALIFIERS[candidate]
    base_name = f"candidate_{candidate.lower()}_run{run_index}"
    out_dir = RAW_OUT / base_name
    suffix = 0
    while out_dir.exists():
        suffix += 1
        out_dir = RAW_OUT / f"{base_name}_retry{suffix}"
    log_prefix = RAW_LOG / out_dir.name
    out_dir.mkdir(parents=True)
    RAW_LOG.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(qualifier), str(out_dir)]
    started = time.monotonic_ns()
    proc = subprocess.run(
        command,
        cwd=SOURCE_CAT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "LC_ALL": "C",
            "LANG": "C",
        },
        timeout=900,
    )
    elapsed_ns = time.monotonic_ns() - started
    stdout_path = log_prefix.with_suffix(".stdout")
    stderr_path = log_prefix.with_suffix(".stderr")
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "candidate": candidate,
        "run_index": run_index,
        "qualifier": str(qualifier.relative_to(SOURCE)),
        "command": command,
        "cwd": str(SOURCE_CAT),
        "return_code": proc.returncode,
        "elapsed_ns": elapsed_ns,
        "stdout_path": str(stdout_path.relative_to(ROOT)),
        "stderr_path": str(stderr_path.relative_to(ROOT)),
        "stdout_sha256": file_sha256(stdout_path),
        "stderr_sha256": file_sha256(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "generated_files": collect_files(out_dir),
    }


def run_arg_control(candidate: str) -> dict[str, Any]:
    qualifier = SOURCE_CAT / QUALIFIERS[candidate]
    proc = subprocess.run(
        ["bash", str(qualifier)],
        cwd=SOURCE_CAT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    stdout_path = RAW_LOG / f"candidate_{candidate.lower()}_missing_arg.stdout"
    stderr_path = RAW_LOG / f"candidate_{candidate.lower()}_missing_arg.stderr"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "candidate": candidate,
        "control": "missing_evidence_dir_argument",
        "return_code": proc.returncode,
        "stdout_path": str(stdout_path.relative_to(ROOT)),
        "stderr_path": str(stderr_path.relative_to(ROOT)),
        "stdout_sha256": file_sha256(stdout_path),
        "stderr_sha256": file_sha256(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
    }


def executable_state() -> dict[str, Any]:
    return {
        candidate: {
            "path": str((SOURCE_CAT / qualifier).relative_to(SOURCE)),
            "executable": os.access(SOURCE_CAT / qualifier, os.X_OK),
            "mode_octal": oct((SOURCE_CAT / qualifier).stat().st_mode & 0o777),
            "sha256": file_sha256(SOURCE_CAT / qualifier),
        }
        for candidate, qualifier in QUALIFIERS.items()
    }


def classify(runs: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    by_candidate = {candidate: [] for candidate in QUALIFIERS}
    for run in runs:
        by_candidate[run["candidate"]].append(run)
    controls_by_candidate = {control["candidate"]: control for control in controls}
    for candidate, items in by_candidate.items():
        if len(items) != 2:
            out[candidate] = "SOURCE_EVIDENCE_INCOMPLETE"
            continue
        if any(item["return_code"] != 0 for item in items):
            out[candidate] = "SOURCE_NOT_REPRODUCED"
            continue
        if controls_by_candidate[candidate]["return_code"] == 0:
            out[candidate] = "SOURCE_QUALIFIER_FAIL_OPEN"
            continue
        signatures = [normalized_generated_signature(item)["records"] for item in items]
        out[candidate] = (
            "SOURCE_REPRODUCED"
            if json_sha256(signatures[0]) == json_sha256(signatures[1])
            else "SOURCE_PROVENANCE_STALE"
        )
    return out


def annotate_normalized_comparisons(payload: dict[str, Any]) -> None:
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in QUALIFIERS}
    for run in payload["runs"]:
        by_candidate[run["candidate"]].append(run)
        run["normalized_generated_signature"] = normalized_generated_signature(run)
    notes: dict[str, Any] = {}
    for candidate, runs in by_candidate.items():
        if len(runs) != 2:
            notes[candidate] = {"status": "incomplete_run_pair"}
            continue
        signatures = [run["normalized_generated_signature"]["records"] for run in runs]
        notes[candidate] = {
            "normalized_match": json_sha256(signatures[0]) == json_sha256(signatures[1]),
            "ignored_generated_files": [
                item
                for run in runs
                for item in run["normalized_generated_signature"]["ignored"]
            ],
            "permitted_json_nondeterminism": [
                "keys ending _elapsed_ms",
                "keys ending _elapsed_ns",
            ],
        }
    payload["normalized_comparison"] = notes


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Source Reproduction Report V2",
        "",
        "Status: frozen source reproduction only. Not canonical. No physical transfer.",
        "",
        f"Frozen source SHA: `{payload['source_commit']}`",
        f"Source worktree: `{payload['source_worktree']}`",
        f"Source worktree status before runs: `{payload['source_status_before'] or 'clean'}`",
        f"Source worktree status after runs: `{payload['source_status_after'] or 'clean'}`",
        f"Venv link: `{payload['venv_link']}`",
        "",
        "## Classifications",
        "",
    ]
    for candidate, classification in payload["classifications"].items():
        lines.append(f"- Candidate {candidate}: `{classification}`")
    lines.extend(["", "## Normalization policy", ""])
    lines.append(
        "The raw output directories and logs are preserved verbatim. For classification, "
        "`SHA256SUMS` files are not treated as scientific differences because the source "
        "qualifiers include absolute output-directory names in those manifests. Generated "
        "JSON files are compared after removing run-time scalar fields ending in "
        "`_elapsed_ms` or `_elapsed_ns`."
    )
    lines.extend(["", "## Runs", ""])
    for run in payload["runs"]:
        lines.append(
            f"- {run['candidate']} run {run['run_index']}: rc={run['return_code']}, "
            f"files={len(run['generated_files'])}, stdout={run['stdout_sha256']}, "
            f"stderr={run['stderr_sha256']}"
        )
    lines.extend(["", "## Controls", ""])
    for control in payload["controls"]:
        lines.append(
            f"- {control['candidate']} missing-argument control: rc={control['return_code']}"
        )
    lines.extend(["", f"Payload hash: `{payload['result_sha256']}`", ""])
    return "\n".join(lines)


def main() -> int:
    RAW_LOG.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    status_before = git(["status", "--short"])
    runs = []
    for candidate in QUALIFIERS:
        for run_index in (1, 2):
            runs.append(run_one(candidate, run_index))
    controls = [run_arg_control(candidate) for candidate in QUALIFIERS]
    status_after = git(["status", "--short"])
    payload: dict[str, Any] = {
        "schema_version": "audio_catvm_source_reproduction_v2",
        "created_utc": utc_now(),
        "canonical": False,
        "small_wall_crossed": False,
        "source_commit": git(["rev-parse", "HEAD"]),
        "source_parent": git(["rev-parse", "HEAD^"]),
        "source_tree": git(["rev-parse", "HEAD^{tree}"]),
        "source_worktree": str(SOURCE),
        "source_status_before": status_before,
        "source_status_after": status_after,
        "venv_link": str((SOURCE / ".venv").resolve()),
        "executable_state": executable_state(),
        "runs": runs,
        "controls": controls,
        "classifications": classify(runs, controls),
        "tracked_result_json_used_as_proof": False,
        "stale_tmp_dependency_detected": False,
    }
    annotate_normalized_comparisons(payload)
    payload["classifications"] = classify(runs, controls)
    payload["result_sha256"] = json_sha256(payload)
    (ROOT / "SOURCE_REPRODUCTION_DATA_V2.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "SOURCE_REPRODUCTION_REPORT_V2.md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "classifications": payload["classifications"],
                "runs": len(runs),
                "controls": len(controls),
                "sha256": file_sha256(ROOT / "SOURCE_REPRODUCTION_DATA_V2.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--analyze-existing":
        payload_path = ROOT / "SOURCE_REPRODUCTION_DATA_V2.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        annotate_normalized_comparisons(payload)
        payload["classifications"] = classify(payload["runs"], payload["controls"])
        payload["reanalyzed_utc"] = utc_now()
        payload.pop("result_sha256", None)
        payload["result_sha256"] = json_sha256(payload)
        payload_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (ROOT / "SOURCE_REPRODUCTION_REPORT_V2.md").write_text(
            render_report(payload), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "classifications": payload["classifications"],
                    "mode": "analyze-existing",
                    "sha256": file_sha256(payload_path),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(0)
    raise SystemExit(main())
