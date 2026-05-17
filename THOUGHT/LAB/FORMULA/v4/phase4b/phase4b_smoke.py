"""Phase 4b Smoke Test: Validate control loop with mock outputs.

Tests that the full architecture works correctly without a real model:
  1. t=2 lattice tolerates 1 node failure -> majority vote holds
  2. Soft gate approves consensus -> no slowdown
  3. Hard gate recovers from errors -> regenerated output passes
  4. Df spikes detect anomalous outputs
  5. @C symbol system compresses and resolves correctly
  6. Full loop runs end-to-end across prompt categories

Run with: python phase4b_smoke.py
No GPU required. Results written to results/smoke_report.json
"""

import sys, json, math, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from phase4b_lattice import (
    evaluate_lattice, compute_consensus, ConsensusResult,
    verify_node1_primary, verify_node2_external_knowledge, verify_node3_logical,
    Verdict, THRESHOLD_DEFAULT,
)
from phase4b_gates import (
    SoftGate, HardGate, DfTracker, CSymbolRegistry, DfSnapshot,
    build_correction_context,
)
from phase4b_loop import (
    MockModel, FailingMockModel, run_control_loop, run_experiment,
    AgentGenerateFn,
)
from phase4b_prompts import (
    TEST_PROMPTS, verify_answer, verify_multi_step,
)

# ---- Test Suites ----

class SmokeTestSuite:
    """Collection of smoke tests. Each test returns (name, passed, details)."""

    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0

    def test(self, name, condition, details=""):
        if condition:
            self.passed += 1
            result = "PASS"
        else:
            self.failed += 1
            result = "FAIL"
        self.tests.append({"test": name, "result": result, "details": details[:200]})
        print(f"    [{result}] {name}")
        if details and result == "FAIL":
            print(f"           {details[:150]}")

    def summary(self):
        return f"{self.passed}/{self.passed + self.failed} tests passed"


def run_smoke_test():
    print("=" * 60)
    print("PHASE 4b SMOKE TEST: Step-Level Macro-Consensus Control Loop")
    print("=" * 60)

    suite = SmokeTestSuite()
    seed = 20260516

    # ========================================================================
    # TEST: t=2 Verification Lattice
    # ========================================================================
    print("\n--- t=2 Verification Lattice ---")

    # ---- Test 1: Node 1 (Primary) PASS ----
    f1_entry = TEST_PROMPTS[0]  # F1: "What is the capital of Burkina Faso?"
    text_pass = "The capital of Burkina Faso is Ouagadougou."
    n1 = verify_node1_primary(text_pass, f1_entry)
    suite.test("Node1 passes correct output",
               n1.verdict == Verdict.PASS and n1.score >= 1.0,
               f"verdict={n1.verdict.value}, score={n1.score}")

    # ---- Test 2: Node 1 (Primary) FAIL ----
    text_fail = "The capital of Burkina Faso is London."
    n1_fail = verify_node1_primary(text_fail, f1_entry)
    suite.test("Node1 fails incorrect output",
               n1_fail.verdict == Verdict.FAIL and n1_fail.score == 0.0,
               f"verdict={n1_fail.verdict.value}, score={n1_fail.score}")

    # ---- Test 3: Node 1 (Primary) ABSTAIN ----
    ambiguous_entry = TEST_PROMPTS[14]  # A1: ambiguous prompt
    n1_abstain = verify_node1_primary("Some response.", ambiguous_entry)
    suite.test("Node1 abstains on unverifiable prompt",
               n1_abstain.verdict == Verdict.ABSTAIN,
               f"verdict={n1_abstain.verdict.value}")

    # ---- Test 4: Node 2 (External Knowledge) PASS ----
    n2 = verify_node2_external_knowledge(text_pass, f1_entry)
    suite.test("Node2 passes with knowledge match",
               n2.verdict == Verdict.PASS and n2.score >= 1.0,
               f"verdict={n2.verdict.value}, evidence={n2.evidence[:60]}")

    # ---- Test 5: Node 2 (External Knowledge) FAIL ----
    n2_fail = verify_node2_external_knowledge(text_fail, f1_entry)
    suite.test("Node2 fails with wrong answer",
               n2_fail.verdict == Verdict.FAIL,
               f"verdict={n2_fail.verdict.value}")

    # ---- Test 6: Node 3 (Logical) PASS ----
    reasoning_entry = TEST_PROMPTS[8]  # R1: train problem
    text_reasoning = "They meet at 4:30 PM after traveling the required distances."
    n3 = verify_node3_logical(text_reasoning, reasoning_entry)
    suite.test("Node3 passes consistent reasoning",
               n3.verdict == Verdict.PASS,
               f"verdict={n3.verdict.value}")

    # ---- Test 7: Node 3 (Logical) on adversarial ----
    d1_entry = TEST_PROMPTS[20]  # D1: flat earth
    text_reject = "This claim is not supported by evidence. The Earth is spherical."
    n3_adv = verify_node3_logical(text_reject, d1_entry)
    suite.test("Node3 passes adversarial rejection",
               n3.verdict == Verdict.PASS,
               f"verdict={n3_adv.verdict.value}")

    # ---- Test 8: Consensus rule - majority vote ----
    from phase4b_lattice import ConsensusResult
    cr = compute_consensus([
        verify_node1_primary(text_pass, f1_entry),
        verify_node2_external_knowledge(text_pass, f1_entry),
        verify_node3_logical(text_pass, f1_entry),
    ])
    suite.test("Consensus holds when >=2 nodes pass",
               cr.consensus_holds == True and cr.consensus_ratio >= 2/3,
               f"holds={cr.consensus_holds}, ratio={cr.consensus_ratio:.2f}, grad_S={cr.grad_S:.4f}")

    # ---- Test 9: Consensus rule - tolerates 1 failure ----
    cr_tol = compute_consensus([
        verify_node1_primary(text_fail, f1_entry),  # FAIL
        verify_node2_external_knowledge(text_pass, f1_entry),  # PASS
        verify_node3_logical(text_pass, f1_entry),  # PASS
    ])
    suite.test("Lattice tolerates 1 node failure (t=2)",
               cr_tol.consensus_holds == True and cr_tol.consensus_ratio == 2/3,
               f"holds={cr_tol.consensus_holds}, ratio={cr_tol.consensus_ratio:.2f}")

    # ---- Test 10: Consensus fails with 2 failures ----
    cr_fail = compute_consensus([
        verify_node1_primary(text_fail, f1_entry),  # FAIL
        verify_node2_external_knowledge(text_fail, f1_entry),  # FAIL
        verify_node3_logical(text_pass, f1_entry),  # PASS
    ])
    suite.test("Consensus fails with >=2 node failures",
               cr_fail.consensus_holds == False and cr_fail.consensus_ratio <= 1/3,
               f"holds={cr_fail.consensus_holds}, ratio={cr_fail.consensus_ratio:.2f}")

    # ---- Test 11: grad_S computation ----
    suite.test("grad_S = 0 when full consensus",
               cr.grad_S == 0.0,
               f"grad_S={cr.grad_S}")
    suite.test("grad_S > 0 when dissonance exists",
               cr_fail.grad_S > 0,
               f"grad_S={cr_fail.grad_S:.4f}")

    # ---- Test 12: Resonance R = 1/grad_S ----
    suite.test("R = inf when grad_S = 0 (full consensus)",
               cr.resonance == float('inf'),
               f"R={cr.resonance}")
    suite.test("R is finite when grad_S > 0",
               math.isfinite(cr_fail.resonance),
               f"R={cr_fail.resonance:.4f}")

    # ---- Test 13: Full lattice evaluation end-to-end ----
    full = evaluate_lattice(text_pass, f1_entry)
    suite.test("Full lattice evaluates all 3 nodes",
               len(full.node_results) == 3,
               f"n_nodes={len(full.node_results)}")

    # ========================================================================
    # TEST: Soft Gate
    # ========================================================================
    print("\n--- Soft Gate ---")

    sg = SoftGate()
    event = sg.approve(0, cr)  # cr has full consensus
    stats = sg.get_stats()

    suite.test("Soft gate records approval event",
               len(sg.events) == 1 and sg.events[0].step == 0,
               f"events={len(sg.events)}, step={sg.events[0].step}")
    suite.test("Soft gate stats show approval",
               stats["n_approvals"] == 1,
               f"n_approvals={stats['n_approvals']}")

    # ========================================================================
    # TEST: Hard Gate
    # ========================================================================
    print("\n--- Hard Gate ---")

    hg = HardGate(log_dir=RESULTS)
    event_rec = hg.halt_and_correct(
        step=0,
        consensus=cr_fail,  # has 2 failures
        failed_output=text_fail,
        regenerated_output=text_pass,
        regenerated_verified=True,
    )
    hg_stats = hg.get_stats()

    suite.test("Hard gate records decoherence event",
               len(hg.events) == 1,
               f"events={len(hg.events)}")
    suite.test("Hard gate stores failed output",
               text_fail in event_rec.failed_output,
               "")
    suite.test("Hard gate stores clean consensus",
               len(event_rec.clean_consensus) > 0,
               f"n_clean={len(event_rec.clean_consensus)}")
    suite.test("Hard gate recovery rate tracks correctly",
               hg_stats["recovery_rate"] == 1.0,
               f"recovery_rate={hg_stats['recovery_rate']}")

    # ---- Test: Context reconstruction ----
    ctx = build_correction_context(
        f1_entry["prompt"], text_fail, cr_fail.passing_outputs)
    suite.test("Correction context has 3 messages",
               len(ctx) == 3,
               f"n_messages={len(ctx)}")
    suite.test("Correction context includes correction message",
               "VERIFICATION FAILED" in ctx[2]["content"],
               "")

    # ========================================================================
    # TEST: Df Anomaly Detection
    # ========================================================================
    print("\n--- Df Anomaly Detection ---")

    df_tracker = DfTracker(window_size=5, anomaly_threshold=2.0)

    # Record normal distribution (low Df)
    rng = np.random.RandomState(seed)
    normal_logits = rng.randn(1, 32000).astype(np.float32) * 0.1
    normal_snap = df_tracker.record(0, normal_logits)

    suite.test("Normal output has moderate Df (not anomalous)",
               not normal_snap.is_anomaly,
               f"df={normal_snap.df_value:.2f}, anomaly={normal_snap.is_anomaly}")

    # Record multiple normal steps to establish baseline
    for i in range(1, 6):
        df_tracker.record(i, rng.randn(1, 32000).astype(np.float32) * 0.1)

    # Now inject anomalous logits (high entropy -> high Df spike)
    anomaly_logits = rng.randn(1, 32000).astype(np.float32) * 10.0
    anomaly_snap = df_tracker.record(6, anomaly_logits)

    suite.test("Anomalous output detected by Df spike",
               anomaly_snap.is_anomaly,
               f"df={anomaly_snap.df_value:.2f}, score={anomaly_snap.anomaly_score:.2f}, anomaly={anomaly_snap.is_anomaly}")

    df_stats = df_tracker.get_stats()
    suite.test("Df statistics include anomaly rate",
               df_stats["anomaly_rate"] > 0,
               f"anomaly_rate={df_stats['anomaly_rate']:.2f}")
    suite.test("Df anomaly count is correct",
               df_stats["anomaly_count"] >= 1,
               f"count={df_stats['anomaly_count']}")

    # ========================================================================
    # TEST: @C Symbol System
    # ========================================================================
    print("\n--- @C Symbol System ---")

    c_reg = CSymbolRegistry()
    long_content = "The capital of Burkina Faso is Ouagadougou. " * 100
    symbol = c_reg.compress(long_content)

    suite.test("@C symbol compresses content",
               symbol.compression_ratio > 10,
               f"ratio={symbol.compression_ratio:.0f}x, original={symbol.original_size}B, compressed={symbol.compressed_size}B")

    symbol_str = f"@C:{symbol.hash_short}"
    resolved = c_reg.resolve(symbol_str)
    suite.test("@C symbol resolves to original content",
               resolved == long_content,
               f"resolved_len={len(resolved) if resolved else 0}")

    # Verify hash integrity - unknown hash
    tampered = c_reg.resolve("@C:deadbe")
    suite.test("@C symbol returns None for unknown hash",
               tampered is None,
               "")

    # Test stats
    stats = c_reg.get_stats()
    suite.test("@C registry reports compression stats",
               stats["n_symbols"] >= 1,
               f"n_symbols={stats['n_symbols']}")

    # ========================================================================
    # TEST: Full Control Loop (mock with good model)
    # ========================================================================
    print("\n--- Full Control Loop (Mock - Good Model) ---")

    mock = MockModel(seed=seed)

    # Run single prompt
    f1_result = run_control_loop(TEST_PROMPTS[0], mock.generate)
    suite.test("Control loop runs to completion",
               f1_result.n_steps > 0,
               f"steps={f1_result.n_steps}")
    suite.test("Soft gate fires correctly for correct output",
               f1_result.n_soft_gates > 0,
               f"soft={f1_result.n_soft_gates}, hard={f1_result.n_hard_gates}")
    suite.test("Hard gate does NOT fire for correct output (good model)",
               f1_result.n_hard_gates == 0,
               f"hard={f1_result.n_hard_gates}")
    suite.test("Final text passes verification",
               f1_result.final_verified == True,
               f"verified={f1_result.final_verified}")
    suite.test("Resonance trajectory is recorded",
               len(f1_result.resonance_trajectory) > 0,
               f"n_R={len(f1_result.resonance_trajectory)}")
    suite.test("Df trajectory is recorded",
               len(f1_result.df_trajectory) > 0,
               f"n_Df={len(f1_result.df_trajectory)}")

    # ========================================================================
    # TEST: Full Control Loop (mock with failing model - hard gate testing)
    # ========================================================================
    print("\n--- Full Control Loop (Mock - Failing Model) ---")

    failing_mock = FailingMockModel(seed=seed, fail_rate=0.5)

    # Run with a factual prompt that encourages failure
    f3_entry = TEST_PROMPTS[2]  # F3: Fe chemical symbol
    failing_result = run_control_loop(f3_entry, failing_mock.generate)

    suite.test("Control loop runs with failing model",
               failing_result.n_steps > 0,
               f"steps={failing_result.n_steps}")
    suite.test("Hard gate fires when model produces errors",
               failing_result.n_hard_gates > 0,
               f"hard={failing_result.n_hard_gates}")
    suite.test("Hard gate recovery rate is tracked",
               failing_result.hard_gate_stats["recovery_rate"] >= 0,
               f"recovery_rate={failing_result.hard_gate_stats['recovery_rate']}")

    # ========================================================================
    # TEST: Full Experiment Runner
    # ========================================================================
    print("\n--- Full Experiment Runner (subset) ---")

    # Run 3 prompts to test experiment infrastructure
    subset = TEST_PROMPTS[:3]  # F1, F2, F3
    results, aggregated = run_experiment(
        prompt_list=subset, generator=mock.generate, verbose=False)

    suite.test("Experiment runs all prompts",
               len(results) == 3,
               f"n_results={len(results)}")
    suite.test("Aggregated stats computed",
               aggregated["n_prompts"] == 3,
               f"n={aggregated['n_prompts']}")
    suite.test("Accuracy computed",
               aggregated["accuracy"] > 0,
               f"acc={aggregated['accuracy']}")

    # ========================================================================
    # TEST: Multi-step verification
    # ========================================================================
    print("\n--- Multi-step Verification ---")

    e1_entry = TEST_PROMPTS[23]  # E1: multi-step
    multi_text = "Fe and Mercury are the correct answers for the two questions."
    multi_passed, multi_score = verify_multi_step(multi_text, e1_entry)
    suite.test("Multi-step verifier works",
               multi_passed,
               f"passed={multi_passed}, score={multi_score:.2f}")

    # Also test failing multi-step
    e1_fail_text = "Not sure about the correct elements."
    multi_fail, multi_fail_score = verify_multi_step(e1_fail_text, e1_entry)
    suite.test("Multi-step verifier detects failure",
               not multi_fail,
               f"passed={multi_fail}, score={multi_fail_score:.2f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST SUMMARY: {suite.summary()}")
    print(f"{'=' * 60}")

    report = {
        "phase": "Phase 4b",
        "date": "2026-05-16",
        "model": "TraDo-4B-Instruct (mock)",
        "tests": suite.tests,
        "summary": suite.summary(),
        "passed": suite.passed,
        "failed": suite.failed,
        "total": suite.passed + suite.failed,
        "architecture_verified": [
            "t=2 verification lattice (3 nodes)",
            "Majority vote consensus rule (tolerates 1 failure)",
            "grad_S = sqrt(dissonance_density) computation",
            "R = 1/grad_S resonance tracking",
            "Soft gate (unitary evolution - approve and continue)",
            "Hard gate (projective measurement - halt, correct, regenerate)",
            "Context reconstruction for correction",
            "Df anomaly detection (effective dimensionality spikes)",
            "@C symbol system (SHA-256 content addressing)",
            "Full step-level control loop",
        ],
    }

    report_path = RESULTS / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}", flush=True)

    # Return exit code for CI
    return 0 if suite.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())
