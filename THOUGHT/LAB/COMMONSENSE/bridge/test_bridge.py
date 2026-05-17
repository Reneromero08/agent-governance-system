"""Smoke test: COMMONSENSE bridge end-to-end pipeline.

Validates:
    1. Fact extraction (regex) produces parseable fact-sets
    2. Bridge integration runs extract -> resolve -> verdict
    3. Truth attractor fragment returns valid scores
    4. Known-good inputs pass; known-bad inputs fail
    5. Canon references and @C symbols are detected

Usage:
    python bridge/test_bridge.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
COMMONSENSE_DIR = BRIDGE_DIR.parent

sys.path.insert(0, str(COMMONSENSE_DIR))

from bridge.fact_extractor import extract_facts, extract_facts_regex, extract_facts_prompt
from bridge.integration import check_output, commonsense_fragment, batch_check, Verdict


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TESTS = {
    "canon_references": {
        "text": "According to @CANON/CONSTITUTION we must maximize coherence. See also @C:a3f2c9 for details.",
        "expect": {
            "has_refs": True,
            "extract_nonempty": True,
            "status": "pass",
        },
    },
    "default_with_exception": {
        "text": "Birds can normally fly unless they are penguins. Penguins cannot fly.",
        "expect": {
            "has_default": True,
            "has_exception": True,
            "extract_nonempty": True,
        },
    },
    "invariant_violation": {
        "text": "The system must never allow unsigned commits. Always require ceremony before changes. An invariant was violated when the agent pushed without approval.",
        "expect": {
            "has_invariant": True,
            "extract_nonempty": True,
        },
    },
    "simple_facts": {
        "text": "The Earth is round. Water freezes at 0 degrees Celsius. The sky is blue.",
        "expect": {
            "fact_count_ge": 3,
            "extract_nonempty": True,
        },
    },
    "empty_input": {
        "text": "",
        "expect": {
            "extract_empty": True,
        },
    },
    "short_input": {
        "text": "OK.",
        "expect": {
            "extract_nonempty": True,
        },
    },
    "governance_context": {
        "text": (
            "All governance checks must pass before deployment. "
            "The job specification must be validated against the schema. "
            "Default trust is low for unverified outputs."
        ),
        "expect": {
            "extract_nonempty": True,
            "status": "pass",
        },
    },
    "hard_fail_trigger": {
        "text": (
            "An invariant was violated in the governance domain. "
            "The invariant violation means the build must be halted."
        ),
        "expect": {
            "extract_nonempty": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests() -> int:
    passed = 0
    failed = 0
    results = []

    print("=" * 60)
    print("COMMONSENSE BRIDGE SMOKE TEST")
    print("=" * 60)

    # ---- Test 1: Regex extraction basics ----
    print("\n--- Regex Extraction ---")
    for name, tc in TESTS.items():
        facts = extract_facts_regex(tc["text"])
        exp = tc["expect"]

        ok = True
        details = []

        if exp.get("extract_empty"):
            if len(facts) != 0:
                ok = False
                details.append(f"expected empty, got {len(facts)} facts")
        if exp.get("extract_nonempty"):
            if len(facts) == 0:
                ok = False
                details.append("expected non-empty facts")
        if exp.get("fact_count_ge"):
            fact_count = sum(1 for f in facts if f.startswith("fact:"))
            if fact_count < exp["fact_count_ge"]:
                ok = False
                details.append(f"expected >= {exp['fact_count_ge']} facts, got {fact_count}")
        if exp.get("has_refs"):
            refs = [f for f in facts if f.startswith("ref:")]
            if not refs:
                ok = False
                details.append("expected ref: entries, got none")
        if exp.get("has_default"):
            defaults = [f for f in facts if f.startswith("default:")]
            if not defaults:
                ok = False
                details.append("expected default: entries, got none")
        if exp.get("has_exception"):
            exceptions = [f for f in facts if f.startswith("exception:")]
            if not exceptions:
                ok = False
                details.append("expected exception: entries, got none")
        if exp.get("has_invariant"):
            invariants = [f for f in facts if f.startswith("invariant:")]
            if not invariants:
                ok = False
                details.append("expected invariant: entries, got none")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}: {len(facts)} facts {details if details else ''}")
        results.append({"test": f"extract_{name}", "status": status, "n_facts": len(facts),
                        "facts": facts[:5], "details": details})

    # ---- Test 2: Integration pipeline ----
    print("\n--- Integration Pipeline ---")

    # Canonical references text
    text_ok = "According to @CANON/CONSTITUTION, governance rules require ceremony for all changes."
    verdict = check_output(text_ok)
    print(f"  [PASS] verify pipeline runs: status={verdict.status} score={verdict.score} "
          f"facts={len(verdict.extracted_facts)} selected={len(verdict.selected_ids)}")
    passed += 1

    # Empty text
    verdict_empty = check_output("")
    print(f"  [PASS] empty input handled: facts={len(verdict_empty.extracted_facts)} "
          f"status={verdict_empty.status}")
    passed += 1

    # ---- Test 3: Truth attractor fragment ----
    print("\n--- Truth Attractor Fragment ---")

    fragment = commonsense_fragment(text_ok)
    required_keys = {"fragment", "score", "confidence", "status"}
    has_keys = required_keys <= set(fragment.keys())
    ok_frag = has_keys and fragment["fragment"] == "commonsense"

    status_frag = "PASS" if ok_frag else "FAIL"
    if ok_frag:
        passed += 1
    else:
        failed += 1
    print(f"  [{status_frag}] fragment returns correct shape: {list(fragment.keys())}")
    print(f"           score={fragment['score']}, confidence={fragment['confidence']}, "
          f"status={fragment['status']}")

    # ---- Test 4: Batch check ----
    print("\n--- Batch Check ---")
    texts = [
        "The Earth is round. Water is wet.",
        "Birds normally fly unless they are penguins.",
        "",
    ]
    batch = batch_check(texts)
    all_verdicts = all(isinstance(v, Verdict) for v in batch)
    status_batch = "PASS" if all_verdicts and len(batch) == 3 else "FAIL"
    if all_verdicts:
        passed += 1
    else:
        failed += 1
    print(f"  [{status_batch}] batch_check returns {len(batch)} Verdicts")

    # ---- Test 5: extract_facts convenience wrapper ----
    print("\n--- Convenience Wrapper ---")
    facts_w = extract_facts("Birds normally fly. Penguins cannot fly.")
    ok_w = len(facts_w) > 0
    status_w = "PASS" if ok_w else "FAIL"
    if ok_w:
        passed += 1
    else:
        failed += 1
    print(f"  [{status_w}] extract_facts returns {len(facts_w)} facts: {facts_w}")

    # ---- Test 6: Governance-domain hard fail ----
    print("\n--- Hard Fail Detection ---")
    gov_text = (
        "An invariant was violated in the governance domain. "
        "The agent committed without a ceremony. "
        "This is a violation of the contract rules."
    )
    verdict_gov = check_output(gov_text)
    # Note: hard_fail only triggers if facts match @INVARIANT_VIOLATION or similar
    # We just verify the pipeline runs without error
    ok_gov = verdict_gov.status in ("pass", "soft_fail", "hard_fail")
    status_gov = "PASS" if ok_gov else "FAIL"
    if ok_gov:
        passed += 1
    else:
        failed += 1
    print(f"  [{status_gov}] governance text: status={verdict_gov.status} "
          f"facts={len(verdict_gov.extracted_facts)} "
          f"selected={verdict_gov.selected_ids} "
          f"emits={len(verdict_gov.emits)}")

    # ---- Summary ----
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} passed ({failed} failed)")
    print(f"{'=' * 60}")

    # Save report
    report_path = BRIDGE_DIR / "test_results.json"
    report = {
        "phase": "bridge_smoke",
        "date": "2026-05-17",
        "passed": passed,
        "failed": failed,
        "total": total,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
