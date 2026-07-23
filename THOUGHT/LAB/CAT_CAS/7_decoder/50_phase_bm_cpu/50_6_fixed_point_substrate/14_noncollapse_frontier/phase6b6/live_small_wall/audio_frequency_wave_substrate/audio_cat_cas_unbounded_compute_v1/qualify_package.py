"""Focused final qualification for the compact toroidal path-sum package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from external_verifier import verify as external_verify
from invariant_probe import run as reproduce_invariants
from phase_path_engine import (
    canonical_bytes,
    engine_fingerprint,
    sha256_bytes,
    source_no_smuggle,
)
from prospective_runner import run as reproduce_raw


HERE = Path(__file__).resolve().parent
BASE_HEAD = "ebbf1e64ccffb23d2d801ff147c75ab927da7ff4"
PREDECESSOR_PATHS = (
    (
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
        "50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/"
        "live_small_wall/audio_frequency_wave_substrate/"
        "audio_phase_native_computer_v1"
    ),
    (
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
        "50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/"
        "live_small_wall/audio_frequency_wave_substrate/"
        "audio_catalytic_waveform_ising_v3"
    ),
    (
        "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
        "50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/"
        "live_small_wall/audio_frequency_wave_substrate/"
        "audio_catalytic_waveform_ising_v3_six_site"
    ),
)


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_manifest() -> dict[str, Any]:
    manifest = load("PACKAGE_MANIFEST.json")
    records: list[dict[str, Any]] = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "PACKAGE_MANIFEST.json" or "__pycache__" in path.parts:
            continue
        payload = path.read_bytes()
        records.append(
            {
                "bytes": len(payload),
                "path": path.relative_to(HERE).as_posix(),
                "sha256": sha256_bytes(payload),
            }
        )
    if records != manifest["files"]:
        raise RuntimeError("package manifest file records drift")
    if sha256_bytes(canonical_bytes(records)) != manifest["content_root"]:
        raise RuntimeError("package manifest content root drift")
    return manifest


def qualify() -> dict[str, Any]:
    python_sources = sorted(HERE.glob("*.py"))
    for path in python_sources:
        compile(path.read_bytes(), str(path), "exec")
    if not source_no_smuggle()["passed"]:
        raise RuntimeError("native no-smuggle qualification failed")
    contract = load("PROSPECTIVE_CONTRACT.json")
    if contract["engine_fingerprint"] != engine_fingerprint():
        raise RuntimeError("engine fingerprint drift")
    raw_committed = load("PROSPECTIVE_RAW_RESULTS.json")
    if reproduce_raw() != raw_committed:
        raise RuntimeError("fresh-process prospective reproduction drift")
    external_committed = load("EXTERNAL_VERIFICATION.json")
    if external_verify() != external_committed:
        raise RuntimeError("external adjudication reproduction drift")
    invariant_committed = load("INVARIANT_RESULTS.json")
    if reproduce_invariants() != invariant_committed:
        raise RuntimeError("relational invariant reproduction drift")
    final = load("FINAL_RESULTS.json")
    if final["decision"] != (
        "CAT_CAS_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_VERIFIED"
    ):
        raise RuntimeError("final decision drift")
    manifest = verify_manifest()
    caches = [
        path
        for path in HERE.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    if caches:
        raise RuntimeError("generated Python cache exists in final package")
    predecessor_custody: dict[str, bool] = {}
    for path in PREDECESSOR_PATHS:
        before = git_text("rev-parse", f"{BASE_HEAD}:{path}")
        current = git_text("rev-parse", f"HEAD:{path}")
        predecessor_custody[path.rsplit("/", 1)[-1]] = before == current
    if not all(predecessor_custody.values()):
        raise RuntimeError("frozen predecessor package changed")
    return {
        "decision": final["decision"],
        "external_cases": final["external_acceptance"]["accepted"],
        "manifest_content_root": manifest["content_root"],
        "manifest_files": manifest["file_count_excluding_manifest"],
        "predecessor_custody": predecessor_custody,
        "python_sources_compiled": len(python_sources),
        "raw_reproduction": "PASS",
        "source_no_smuggle": "PASS",
        "status": "PASS",
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), sort_keys=True))
