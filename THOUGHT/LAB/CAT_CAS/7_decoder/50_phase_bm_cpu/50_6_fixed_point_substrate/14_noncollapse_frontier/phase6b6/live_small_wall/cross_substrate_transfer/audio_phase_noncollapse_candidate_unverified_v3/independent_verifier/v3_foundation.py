#!/usr/bin/env python3
"""Create V3 source-freeze and import foundation files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKTREE = Path.cwd()
SOURCE = Path("/tmp/ags-audio-source-7c79414-v3")
FRONTIER_REL = Path(
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/"
    "14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate"
)
SOURCE_FRONTIER = SOURCE / FRONTIER_REL
SOURCE_CAT = SOURCE_FRONTIER / "cat_cas_phase_frontier"
SNAPSHOT = ROOT / "source_snapshot"

SCIENTIFIC_HEAD = "7c79414f2beb34c29bf0d63783a6effea26c65ed"
PROMPT_RECONCILED_HEAD = "1251cb3a34669fa6b00eddcd66a2228b8f7f9731"
LAST_V2_SOURCE_HEAD = "6f39766e9cf622e2e41d178f8131bd8777b6cd1d"
COMPLETED_V2_BRANCH_HEAD = "25e58a26395549c703d430f0e37410db2b7f54a0"


CANDIDATE_FILES: dict[str, list[str]] = {
    "navigation": [
        "AUDIO_SIDE_QUEST_ROADMAP.md",
        "AUTONOMOUS_LANE_STATE.json",
        "CLAIM_AUTHORITY_REGISTRY.json",
        "cat_cas_phase_frontier/CURRENT_FINDINGS.md",
    ],
    "I_projective_orbit": [
        "cat_cas_phase_frontier/CYCLOTOMIC_HT_PROJECTIVE_ORBIT_RESULTS.json",
        "cat_cas_phase_frontier/cyclotomic_ht_projective_orbit_obstruction.cpp",
        "cat_cas_phase_frontier/cyclotomic_ht_projective_orbit_oracle.py",
        "cat_cas_phase_frontier/qualify_cyclotomic_ht_projective_orbit_obstruction.sh",
    ],
    "J_phase_vm_bisimulation": [
        "cat_cas_phase_frontier/PHASE_VM_ROOT_BISIMULATION_RESULTS.json",
        "cat_cas_phase_frontier/phase_vm_root_bisimulation.c",
        "cat_cas_phase_frontier/phase_vm_root_bisimulation_oracle.py",
        "cat_cas_phase_frontier/qualify_phase_vm_root_bisimulation.sh",
        "cat_cas_phase_frontier/streaming_phase_vm.c",
    ],
    "K_multi_port_tt": [
        "cat_cas_phase_frontier/MULTI_PORT_TT_RESULTS.json",
        "cat_cas_phase_frontier/MULTI_PORT_TT_DENSE_ORACLE_RESULTS.json",
        "cat_cas_phase_frontier/MULTI_PORT_TT_CLEANROOM_REVIEW.md",
        "cat_cas_phase_frontier/four_rotor_necklace_multi_port_tt_phase.py",
        "cat_cas_phase_frontier/four_rotor_necklace_multi_port_dense_oracle.py",
        "cat_cas_phase_frontier/qualify_four_rotor_necklace_multi_port_tt.sh",
    ],
    "L_period17_height": [
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_ORACLE_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_PROVENANCE.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_HEIGHT_LOWER_BOUND_INDEPENDENT_REVIEW.md",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_height_lower_bound.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_height_lower_bound_oracle.py",
        "cat_cas_phase_frontier/qualify_f17_cubic_chain_period17_height_lower_bound.sh",
        "cat_cas_phase_frontier/f17_cubic_chain_adaptive_gauge.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_cyclotomic_module.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_executed_recurrence.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_unit_height_reduction.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_unit_height_reduction_oracle.py",
    ],
    "M_wilczek_zee": [
        "cat_cas_phase_frontier/WILCZEK_ZEE_NONABELIAN_PHASE_FRAME_RESULTS.json",
        "cat_cas_phase_frontier/wilczek_zee_nonabelian_phase_frame.c",
        "cat_cas_phase_frontier/wilczek_zee_nonabelian_phase_frame_oracle.py",
        "cat_cas_phase_frontier/qualify_wilczek_zee_nonabelian_phase_frame.sh",
    ],
    "N_two_port_catvm": [
        "cat_cas_phase_frontier/TWO_SHARED_LATENT_DESCRIPTOR_RESULTS.json",
        "cat_cas_phase_frontier/TWO_SHARED_LATENT_DESCRIPTOR_PROGRAMS.json",
        "cat_cas_phase_frontier/TWO_SHARED_LATENT_DESCRIPTOR_CLEANROOM_REVIEW.md",
        "cat_cas_phase_frontier/catvm_necklace_two_shared_latent_descriptor_protocol.py",
        "cat_cas_phase_frontier/catvm_necklace_two_shared_latent_descriptor_service.cpp",
        "cat_cas_phase_frontier/catvm_necklace_two_shared_latent_descriptor_controller.py",
        "cat_cas_phase_frontier/two_shared_latent_descriptor_oracle.py",
        "cat_cas_phase_frontier/qualify_two_shared_latent_descriptor_catvm.sh",
        "cat_cas_phase_frontier/TWO_SHARED_LATENT_PHASE_RESULTS.json",
        "cat_cas_phase_frontier/TWO_SHARED_LATENT_PHASE_CLEANROOM_REVIEW.md",
        "cat_cas_phase_frontier/four_rotor_necklace_two_shared_latent_phase.cpp",
        "cat_cas_phase_frontier/catvm_necklace_two_shared_latent_service.cpp",
        "cat_cas_phase_frontier/catvm_necklace_two_shared_latent_controller.py",
        "cat_cas_phase_frontier/two_shared_latent_phase_oracle.py",
        "cat_cas_phase_frontier/qualify_two_shared_latent_phase.sh",
        "cat_cas_phase_frontier/four_rotor_necklace_generator_phase.cpp",
        "cat_cas_phase_frontier/four_rotor_bosonic_givens_phase.cpp",
        "cat_cas_phase_frontier/four_rotor_necklace_orbit_phase.cpp",
        "cat_cas_phase_frontier/catvm_necklace_shared_latent_protocol.py",
    ],
    "O_cubic_chain_transfer": [
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_TRANSFER_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_TRANSFER_ORACLE_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_TRANSFER_PROVENANCE.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_TRANSFER_INDEPENDENT_REVIEW.md",
        "cat_cas_phase_frontier/f17_cubic_chain_transfer_closure.py",
        "cat_cas_phase_frontier/f17_cubic_chain_transfer_oracle.py",
        "cat_cas_phase_frontier/qualify_f17_cubic_chain_transfer.sh",
    ],
    "post_1251_discovered_pi_content_delta": [
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_ORACLE_RESULTS.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_PROVENANCE.json",
        "cat_cas_phase_frontier/F17_CUBIC_CHAIN_PERIOD17_PI_CONTENT_RECURRENCE_INDEPENDENT_REVIEW.md",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_pi_content_recurrence.py",
        "cat_cas_phase_frontier/f17_cubic_chain_period17_pi_content_recurrence_oracle.py",
        "cat_cas_phase_frontier/qualify_f17_cubic_chain_period17_pi_content_recurrence.sh",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def git(args: list[str], cwd: Path = SOURCE) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def write_json(name: str, payload: dict[str, Any]) -> None:
    payload.setdefault("created_utc", utc_now())
    payload.setdefault("canonical", False)
    payload.setdefault("small_wall_crossed", False)
    payload["sha256"] = json_sha256(payload)
    (ROOT / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_record(rel: str, group: str) -> dict[str, Any]:
    source = SOURCE_FRONTIER / rel
    if not source.exists():
        raise FileNotFoundError(source)
    target = SNAPSHOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    stat = source.stat()
    return {
        "candidate_group": group,
        "source_relative_path": str(FRONTIER_REL / rel),
        "snapshot_relative_path": str(target.relative_to(ROOT)),
        "sha256": file_sha256(source),
        "bytes": stat.st_size,
        "mode_octal": oct(stat.st_mode & 0o777),
        "executable": os.access(source, os.X_OK),
        "git_object_mode": git(["ls-tree", "HEAD", "--", str(FRONTIER_REL / rel)]).split()[0],
    }


def render_claims() -> str:
    return """# Claims Under Test V3

Status: imported as unverified V3 candidates. Not canonical. No Small Wall
promotion. No physical Family 10h transfer.

Scientific source freeze: `7c79414f2beb34c29bf0d63783a6effea26c65ed`.

The prompt-nominated reconciled head was `1251cb3a34669fa6b00eddcd66a2228b8f7f9731`.
Inspection found a later scientific code commit, `7c79414f2beb34c29bf0d63783a6effea26c65ed`,
followed by documentation/reconciliation commit `9165ef74d9750dec63f5ed2a3a1c69f156f23135`.

Candidates:

- I: exact HT infinite projective orbit finite-quotient obstruction.
- J: root-locked finite phase VM classical bisimulation obstruction.
- K: multi-port tensor-train full-rank/dense-baseline obstruction.
- L: exact F17 period-17 boundary-height lower-bound obstruction.
- M: non-Abelian Wilczek-Zee shared-frame holonomy mechanism.
- N: descriptor-compiled two-shared-latent-port CATVM custody.
- O: F17 cubic-chain topology-factorized reversible transfer.

Discovered post-1251 scientific delta:

- F17 period-17 pi-content recurrence package. It is recorded in the source
  receipt as a later scientific delta but is not promoted to a numbered
  candidate unless it becomes necessary for Candidate L scope.
"""


def render_plan() -> str:
    return """# Verification Plan V3

Evidence policy: hashes and modes are provenance receipts. Scientific
classification must come from independent behavior, theorem reconstruction,
controls, transfer tests, resource laws, and strongest compact baselines.

Plan:

1. Freeze source and import only I-O plus dependencies.
2. Investment-gate all candidates.
3. Run source qualifiers twice where practical, preserving raw logs.
4. Independently reconstruct:
   - I theorem using exact algebra.
   - J finite transition semantics and symbolic bisimulation.
   - K rank/baseline with alternate matricizations where practical.
   - L valuation/lower-bound scope and compact indexed baselines.
   - M matrix holonomy and noncommutator behavior.
   - N protocol framing/custody with inherited oversized-packet checks.
   - O exact path transfer and dynamic-program baselines.
5. Build branch-local obstruction and mechanism transfer harnesses independent
   of audio arithmetic.
6. Record Family 10h physical and counterfactual relevance separately.
"""


def main() -> int:
    if git(["rev-parse", "HEAD"]) != SCIENTIFIC_HEAD:
        raise RuntimeError("source worktree is not at scientific V3 head")
    records = []
    for group, files in CANDIDATE_FILES.items():
        for rel in files:
            records.append(source_record(rel, group))

    moving_head = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/codex/audio-frequency-wave-substrate"],
        cwd=WORKTREE,
        text=True,
    ).split()[0]
    post_science = git(["rev-list", "--reverse", f"{SCIENTIFIC_HEAD}..origin/codex/audio-frequency-wave-substrate"], cwd=WORKTREE)
    post_science_commits = [line for line in post_science.splitlines() if line]
    post_1251 = git(["rev-list", "--reverse", f"{PROMPT_RECONCILED_HEAD}..origin/codex/audio-frequency-wave-substrate"], cwd=WORKTREE)
    post_1251_commits = [line for line in post_1251.splitlines() if line]
    source_delta = []
    for commit in post_1251_commits:
        files = git(["diff-tree", "--no-commit-id", "--name-status", "-r", commit], cwd=WORKTREE).splitlines()
        source_delta.append(
            {
                "commit": commit,
                "subject": git(["log", "-1", "--format=%s", commit], cwd=WORKTREE),
                "files": files,
                "scientific_code_or_result_added": any(
                    line.startswith("A\t")
                    and "cat_cas_phase_frontier/" in line
                    and not line.endswith(".md")
                    for line in files
                ),
            }
        )

    environment = {
        "python": subprocess.check_output([str(SOURCE / ".venv/bin/python"), "--version"], text=True).strip(),
        "gcc": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
        "g++": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "worktree_python": subprocess.check_output([str(WORKTREE / ".venv/bin/python"), "--version"], text=True).strip(),
    }
    try:
        environment["numpy"] = subprocess.check_output(
            [str(SOURCE / ".venv/bin/python"), "-c", "import numpy; print(numpy.__version__)"],
            text=True,
        ).strip()
        environment["scipy"] = subprocess.check_output(
            [str(SOURCE / ".venv/bin/python"), "-c", "import scipy; print(scipy.__version__)"],
            text=True,
        ).strip()
        environment["sympy"] = subprocess.check_output(
            [str(SOURCE / ".venv/bin/python"), "-c", "import sympy; print(sympy.__version__)"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        environment["dependency_probe_error"] = str(exc)

    receipt = {
        "schema_version": "audio_noncollapse_v3_source_receipt",
        "source_worktree": str(SOURCE),
        "source_status": git(["status", "--short"]),
        "scientific_source_commit": SCIENTIFIC_HEAD,
        "scientific_source_parent": git(["rev-parse", "HEAD^"]),
        "scientific_source_tree": git(["rev-parse", "HEAD^{tree}"]),
        "prompt_reconciled_source_head": PROMPT_RECONCILED_HEAD,
        "last_independently_verified_audio_source_head": LAST_V2_SOURCE_HEAD,
        "completed_v2_branch_head": COMPLETED_V2_BRANCH_HEAD,
        "moving_audio_branch_head_at_freeze": moving_head,
        "post_scientific_head_commits": post_science_commits,
        "post_1251_delta": source_delta,
        "freeze_decision": (
            "Frozen at 7c79414f because inspection found a post-1251 scientific code/result commit; "
            "current moving branch 9165ef74 is documentation/reconciliation after that scientific commit."
        ),
        "environment": environment,
        "copied_file_count": len(records),
    }
    write_json("SOURCE_RECEIPT_V3.json", receipt)
    write_json(
        "SOURCE_DELTA_MANIFEST_V3.json",
        {
            "schema_version": "audio_noncollapse_v3_delta_manifest",
            "source_frontier": str(FRONTIER_REL),
            "copied_files": records,
            "not_imported": "No full branch import; no autonomous lane state treated as proof.",
        },
    )
    write_json(
        "SOURCE_COMMIT_GRAPH.json",
        {
            "schema_version": "audio_noncollapse_v3_commit_graph",
            "last_v2_source": LAST_V2_SOURCE_HEAD,
            "prompt_reconciled_head": PROMPT_RECONCILED_HEAD,
            "scientific_freeze": SCIENTIFIC_HEAD,
            "moving_branch_head": moving_head,
            "post_1251_delta": source_delta,
            "post_scientific_head_commits": post_science_commits,
        },
    )
    write_json(
        "V1_V2_PREDECESSOR_BINDING.json",
        {
            "schema_version": "audio_noncollapse_v3_predecessor_binding",
            "v1": {
                "A": "REJECTED_SOURCE_DEFECT",
                "B": "REJECTED_FIXTURE_SPECIALIZATION",
                "C": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_OBSTRUCTION",
                "D": "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL",
            },
            "v2": {
                "E": "REJECTED_SOURCE_DEFECT",
                "F": "INDEPENDENTLY_VERIFIED_FAMILY_SCOPED_REMATERIALIZATION",
                "G": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_PRECISION_OBSTRUCTION",
                "H": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_COHERENCE_OBSTRUCTION",
            },
            "do_not_reopen_except_regression_controls": True,
        },
    )
    (ROOT / "CLAIMS_UNDER_TEST_V3.md").write_text(render_claims(), encoding="utf-8")
    (ROOT / "VERIFICATION_PLAN_V3.md").write_text(render_plan(), encoding="utf-8")
    print(json.dumps({"copied_files": len(records), "moving_head": moving_head, "scientific_head": SCIENTIFIC_HEAD}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
