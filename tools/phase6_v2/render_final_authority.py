#!/usr/bin/env python3
"""Render the final Phase 6 V2 authority state without touching source or contracts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_COMMIT = "329e5a0c1a56595fbfb17cc16c41fcb33faff5a2"
GENERATED_COMMIT = "70d5aa893db7d93baa86a56d5b1ed128730c2ef3"
INCOMPLETE_EVIDENCE_COMMIT = "339c9fb85aff2578c51d5f8e9cee7e99e768d136"
INCOMPLETE_CORRECTION_COMMIT = "f524e0230ed46b56b93dffe6b37f446d6602df0c"
RAW_EVIDENCE_COMMIT = "c81b543ffb74644a35aa97605ca47fa9ec89c76c"
COMMAND_EVIDENCE_COMMIT = "f531ac8016c9c95141ed1c0ec180bcd01370d346"
PLAN_SHA256 = "f6d7cd314a0b614d80f520f92df13b0bd52f222e4b2fd7b53c09229bc49df48d"
BUNDLE_SHA256 = "5764f7d391e16624e1a5861f9b7056ad4bbfe8227b2a2731da965626f200ee2b"
STATUS = "PHASE6_V2_ENGINEERING_QUALIFICATION_COMPLETE__INDEPENDENT_PR_REVIEW_NEXT"

ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())

ROADMAP = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/PHASE6_ROADMAP.md"
NAVIGATION = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/PHASE6_NAVIGATION.md"
CHIRAL = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md"
ARCH_REVIEW = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/ARCHITECTURE_REVIEW.md"
WORK_PACKAGE = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/V2_FINAL_QUALIFICATION_WORK_PACKAGE.md"
EVIDENCE_ROOT = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/evidence"
BINDINGS = EVIDENCE_ROOT / "FINAL_BINDINGS.json"
INVENTORY = EVIDENCE_ROOT / "EVIDENCE_INVENTORY.sha256"
VERIFICATION = EVIDENCE_ROOT / "EVIDENCE_INVENTORY_VERIFICATION.txt"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_roadmap() -> None:
    text = ROADMAP.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "**Current head:** `PHASE6B5D_V2_CALIBRATION_ARCHITECTURE_DRAFT__GATE_R_STILL_NEXT`  ",
        f"**Current head:** `{STATUS}`  ",
        "roadmap status",
    )
    text = replace_once(
        text,
        "**Immediate engineering gate:** close V2 qualification Q1 through Q4 and complete independent PR review.  ",
        "**Immediate engineering gate:** complete independent PR review; V2 qualification Q1 through Q4 are closed.  ",
        "roadmap gate",
    )
    old = """- [ ] close strict source and contract defects listed in the V2 engineering addendum Q1
- [ ] pass the updated V2 GitHub CI lane at the exact final head
- [ ] pass strict compile, equivalence, sanitizers, and full no-write gate on the Linux target at the exact final head
- [ ] regenerate generated contracts and the evidence inventory from the final source head
- [ ] synchronize PR description and all authority documents with final hashes and logs
- [ ] complete independent merge review"""
    new = """- [x] close strict source and contract defects listed in the V2 engineering addendum Q1
- [x] pass the updated V2 GitHub CI lane at the exact final head
- [x] pass strict compile, equivalence, sanitizers, and full no-write gate on the Linux target at the exact final head
- [x] regenerate generated contracts and the evidence inventory from the final source head
- [x] synchronize PR description and all authority documents with final hashes and logs
- [ ] complete independent merge review"""
    text = replace_once(text, old, new, "roadmap checklist")
    anchor = "The current V2 plan uses ascending tone order and qualifies the physical interface. It is not the selected reversed/randomized tone-order scientific control."
    binding = f"""{anchor}

Final V2 engineering qualification binding:

```text
source commit: {SOURCE_COMMIT}
generated contract commit: {GENERATED_COMMIT}
raw evidence closure: {RAW_EVIDENCE_COMMIT}
command evidence closure: {COMMAND_EVIDENCE_COMMIT}
plan SHA-256: {PLAN_SHA256}
source-bundle SHA-256: {BUNDLE_SHA256}
unique functional test cases: 78
total unittest executions: 194
Windows tested SHA: {GENERATED_COMMIT}
Linux tested SHA: {GENERATED_COMMIT}
independent PR review: pending
Gate R: pending
Phase 6B.6: not entered
```
"""
    text = replace_once(text, anchor, binding.rstrip(), "roadmap binding")
    ROADMAP.write_text(text, encoding="utf-8")


def patch_navigation() -> None:
    text = NAVIGATION.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "**Current state:** `PHASE6B5D_V2_CALIBRATION_ARCHITECTURE_DRAFT__GATE_R_STILL_NEXT`  ",
        f"**Current state:** `{STATUS}`  ",
        "navigation status",
    )
    text = replace_once(
        text,
        "| 14 | `noncollapse_frontier` | Active Phase 6B architecture and physical ladder | Phase 6B.5D frozen; V2 qualification draft; Gate R still next |",
        "| 14 | `noncollapse_frontier` | Active Phase 6B architecture and physical ladder | Phase 6B.5D frozen; V2 engineering qualification complete; independent PR review next; Gate R still next |",
        "navigation table",
    )
    marker = "## Status at a glance"
    block = f"""## Current V2 authority binding

```text
status: {STATUS}
source: {SOURCE_COMMIT}
generated contracts: {GENERATED_COMMIT}
raw evidence: {RAW_EVIDENCE_COMMIT}
command evidence: {COMMAND_EVIDENCE_COMMIT}
plan SHA-256: {PLAN_SHA256}
source-bundle SHA-256: {BUNDLE_SHA256}
independent PR review: pending
Gate R: pending
Phase 6B.6: not entered
hardware_ran: false
authorization_artifact_created: false
```

---

{marker}"""
    text = replace_once(text, marker, block, "navigation binding")
    NAVIGATION.write_text(text, encoding="utf-8")


def patch_chiral() -> None:
    text = CHIRAL.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "**Status:** `V2_CALIBRATION_ARCHITECTURE_DRAFT__GATE_R_STILL_NEXT`  ",
        f"**Status:** `{STATUS}`  ",
        "chiral status",
    )
    text = replace_once(
        text,
        "**Immediate engineering gate:** V2 qualification Q1 through Q4 and independent PR review  ",
        "**Immediate engineering gate:** independent PR review; V2 qualification Q1 through Q4 are complete  ",
        "chiral gate",
    )
    old = """- [ ] strict command-line and authorization parser closure
- [ ] mechanical runtime/plan threshold binding
- [ ] exact capture-quality and schema closure
- [ ] immutable run-root directory and symlink rejection
- [ ] exact final-head GitHub CI pass
- [ ] exact final-head target-toolchain and sanitizer pass
- [ ] regenerated contracts and final evidence inventory
- [ ] independent merge review"""
    new = """- [x] strict command-line and authorization parser closure
- [x] mechanical runtime/plan threshold binding
- [x] exact capture-quality and schema closure
- [x] immutable run-root directory and symlink rejection
- [x] exact final-head GitHub CI pass
- [x] exact final-head target-toolchain and sanitizer pass
- [x] regenerated contracts and final evidence inventory
- [ ] independent merge review"""
    text = replace_once(text, old, new, "chiral checklist")
    text = replace_once(
        text,
        "- [ ] V2 engineering lane closed cleanly or explicitly excluded from the integration decision",
        "- [x] V2 engineering lane closed cleanly for independent integration review",
        "chiral Gate R prerequisite",
    )
    anchor = "The current V2 schedule is ascending-order engineering calibration. It is not the reversed/randomized tone-order scientific control."
    binding = f"""{anchor}

Final engineering qualification binding:

```text
source commit: {SOURCE_COMMIT}
generated contract commit: {GENERATED_COMMIT}
raw evidence closure: {RAW_EVIDENCE_COMMIT}
command evidence closure: {COMMAND_EVIDENCE_COMMIT}
plan SHA-256: {PLAN_SHA256}
source-bundle SHA-256: {BUNDLE_SHA256}
unique functional test cases: 78
total unittest executions: 194
independent PR review: pending
Gate R: pending
Phase 6B.6: not entered
```
"""
    text = replace_once(text, anchor, binding.rstrip(), "chiral binding")
    CHIRAL.write_text(text, encoding="utf-8")


def write_architecture_review() -> None:
    content = f"""# Phase 6 V2 Architectural Review

**Status:** `{STATUS}`  
**Authority:** `../../PHASE6_V2_ENGINEERING_QUALIFICATION_ADDENDUM_2026-06-22.md`  
**Execution packet:** `../V2_FINAL_QUALIFICATION_WORK_PACKAGE.md`  
**Independent PR review:** pending  
**Gate R:** pending  
**Phase 6B.6:** not entered  
**Hardware calibration executed:** false  
**Scientific acquisition authorized:** false

## Final bound object

```text
source repair:
{SOURCE_COMMIT}

generated contracts:
{GENERATED_COMMIT}

raw evidence closure:
{RAW_EVIDENCE_COMMIT}

command evidence closure:
{COMMAND_EVIDENCE_COMMIT}

plan SHA-256:
{PLAN_SHA256}

source-bundle SHA-256:
{BUNDLE_SHA256}
```

Historical provenance is retained:

```text
b7563e5f: retained source provenance
93f28c5d: superseded generated-contract object
{INCOMPLETE_EVIDENCE_COMMIT}: incomplete evidence provenance
{INCOMPLETE_CORRECTION_COMMIT}: incomplete evidence-correction provenance
```

## Qualification closure

The committed source and generated contracts passed:

- strict C compilation with `-Wall -Wextra -Werror`;
- V2 runner contracts;
- C/Python waveform equivalence;
- Slot2 primitive identity;
- direct capture-quality rejection testing;
- exact plan/runtime threshold identity;
- strict authorization and numeric parsing;
- same-byte analyzer custody;
- immutable run-root and symlink rejection;
- V2 calibration-contract and analyzer tests;
- ASan and UBSan target lanes;
- deterministic contract regeneration;
- the canonical full no-write repository gate;
- exact-head Windows and Phenom II Linux qualification.

Machine-derived execution counts:

```text
unique functional test cases: 78
capture-quality subset recheck: 1
ASan reexecutions: 38
UBSan reexecutions: 38
Windows focused executions: 39
total unittest executions: 194
all exit codes zero: true
```

The evidence inventory is committed at:

```text
combined_observability_campaign/v2/evidence/EVIDENCE_INVENTORY.sha256
```

Its independent verification record reports `PASSED` for 26 entries.

## Preserved scientific boundary

```text
V1:
PERMANENT_RETROSPECTIVE_NEGATIVE_ADJUDICATION
NO_STABLE_PREDICTIVE_OPERATOR
PRISTINE_FINAL_TEST_HYGIENE_NOT_PROVEN

T48 carrier:
TRANSFER_EQUIVARIANCE_SUPPORTED under a minimal C0 receiver chart
STRICT_CARRIER_CLOSURE_PARTIAL

V2:
ENGINEERING_QUALIFICATION_COMPLETE

Gate R:
PENDING

Phase 6B.6:
NOT ENTERED

physical restoration:
NOT ESTABLISHED

target coupling:
NOT ESTABLISHED

fold-odd invariant:
NOT ESTABLISHED

Small Wall crossing:
NOT ESTABLISHED
```

The V2 schedule is ascending-order engineering calibration. It is not the proposed reversed/randomized tone-order scientific control.

## Authorization boundary

```text
hardware_ran=false
authorization_artifact_created=false
calibration_authorized=false
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

## Next legitimate action

Independent review of PR #21 is next. Gate R remains a separate project-owner decision. No physical acquisition, restoration experiment, target-coupling experiment, or Small Wall execution is authorized by this qualification.
"""
    ARCH_REVIEW.write_text(content, encoding="utf-8")


def patch_work_package() -> None:
    text = WORK_PACKAGE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "**Status:** `CONNECTOR_AUDIT_COMPLETE__LOCAL_MATERIALIZATION_AND_LINUX_REQUIRED`  ",
        f"**Status:** `{STATUS}`  ",
        "work package status",
    )
    text = replace_once(
        text,
        "**PR:** `#21`, draft and unmerged  ",
        f"""**PR:** `#21`, draft and unmerged  
**Final source commit:** `{SOURCE_COMMIT}`  
**Final generated-contract commit:** `{GENERATED_COMMIT}`  
**Raw evidence closure:** `{RAW_EVIDENCE_COMMIT}`  
**Command evidence closure:** `{COMMAND_EVIDENCE_COMMIT}`  
**Plan SHA-256:** `{PLAN_SHA256}`  
**Source-bundle SHA-256:** `{BUNDLE_SHA256}`  """,
        "work package binding metadata",
    )
    marker = "---\n\n## 1. Corrected repository state"
    closure = f"""---

## Completion record

This work package is retained as the execution and audit history for a completed engineering qualification.

```text
status: {STATUS}
source commit: {SOURCE_COMMIT}
generated-contract commit: {GENERATED_COMMIT}
raw evidence closure: {RAW_EVIDENCE_COMMIT}
command evidence closure: {COMMAND_EVIDENCE_COMMIT}
unique functional test cases: 78
total unittest executions: 194
independent PR review: pending
Gate R: pending
Phase 6B.6: not entered
```

No hardware calibration or scientific acquisition ran. All scientific authorization fields remain false.

---

## 1. Corrected repository state"""
    text = replace_once(text, marker, closure, "work package closure")
    WORK_PACKAGE.write_text(text, encoding="utf-8")


def patch_bindings() -> None:
    data = json.loads(BINDINGS.read_text(encoding="utf-8"))
    data["prior_incomplete_evidence_correction_commit"] = INCOMPLETE_CORRECTION_COMMIT
    data["raw_evidence_closure_commit"] = RAW_EVIDENCE_COMMIT
    data["command_evidence_closure_commit"] = COMMAND_EVIDENCE_COMMIT
    data["authority_status"] = {
        "state": STATUS,
        "independent_pr_review": "pending",
        "gate_r": "pending",
        "phase6b6": "not_entered",
    }
    provenance = data.setdefault("provenance", {})
    provenance["f524e023"] = "incomplete evidence-correction provenance"
    provenance["c81b543f"] = "raw evidence persistence and count correction"
    provenance["f531ac80"] = "command evidence closure"
    BINDINGS.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def regenerate_inventory() -> None:
    files = sorted(
        p for p in EVIDENCE_ROOT.rglob("*")
        if p.is_file() and p != INVENTORY
    )
    rels = [p.relative_to(ROOT).as_posix() for p in files]
    verification_text = "\n".join(
        ["Evidence inventory verification"]
        + [f"OK: {rel}" for rel in rels]
        + [f"Entries: {len(rels)}", "Verification: PASSED", ""]
    )
    VERIFICATION.write_text(verification_text, encoding="utf-8")

    files = sorted(
        p for p in EVIDENCE_ROOT.rglob("*")
        if p.is_file() and p != INVENTORY
    )
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    INVENTORY.write_text("\n".join(rows) + "\n", encoding="utf-8")

    seen = set()
    previous = ""
    for row in INVENTORY.read_text(encoding="utf-8").splitlines():
        digest, rel = row.split("  ", 1)
        if rel in seen or rel < previous:
            raise RuntimeError("inventory duplicate or ordering failure")
        seen.add(rel)
        previous = rel
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"inventory path missing: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"inventory digest mismatch: {rel}")


def main() -> int:
    patch_roadmap()
    patch_navigation()
    patch_chiral()
    write_architecture_review()
    patch_work_package()
    patch_bindings()
    regenerate_inventory()

    allowed = {
        ROADMAP.relative_to(ROOT).as_posix(),
        NAVIGATION.relative_to(ROOT).as_posix(),
        CHIRAL.relative_to(ROOT).as_posix(),
        ARCH_REVIEW.relative_to(ROOT).as_posix(),
        WORK_PACKAGE.relative_to(ROOT).as_posix(),
        BINDINGS.relative_to(ROOT).as_posix(),
        INVENTORY.relative_to(ROOT).as_posix(),
        VERIFICATION.relative_to(ROOT).as_posix(),
    }
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, text=True
    ).splitlines())
    if changed != allowed:
        raise RuntimeError(f"unexpected changed paths: {sorted(changed ^ allowed)}")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)

    manifest = {
        "status": STATUS,
        "changed_paths": sorted(changed),
        "sha256": {
            rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            for rel in sorted(changed)
        },
    }
    out = Path(subprocess.check_output(
        ["bash", "-lc", "printf %s \"${RUNNER_TEMP:-/tmp}\""], text=True
    ).strip()) / "phase6_v2_authority_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
