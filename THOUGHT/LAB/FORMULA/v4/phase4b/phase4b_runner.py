"""Phase 4b: Step-Level Macro-Consensus Runner — Epistemic Truth Attractor

Runs all four experimental conditions on the same prompts:
    1. CONTROL: TraDo-4B, T=0.7, no lattice, no verification. Single generation.
    2. VALUES_LATTICE: t=2 lattice with values constitution C (equal weights).
    3. EPISTEMIC_LATTICE: t=2 lattice with epistemic C_epistemic (calibrated weights).
    4. EPISTEMIC_NO_COMMONSENSE: Same as EPISTEMIC_LATTICE but COMMONSENSE
       replaced with a second factual verification fragment.

Each condition runs on the same 26 test prompts after calibration on 12
separate calibration prompts.
"""

import argparse, json, math, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Path setup
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Try importing facts cassette for hard gate recovery
try:
    import sys as _sys
    _cassette_path = str(Path(__file__).resolve().parents[5] / "THOUGHT" / "LAB" / "TINY_COMPRESS" / "llm-spectral" / "auto_feedback")
    if _cassette_path not in _sys.path:
        _sys.path.insert(0, _cassette_path)
    from facts_cassette import FactsCassette
    FACTS_CASSETTE_AVAILABLE = True
except Exception:
    FACTS_CASSETTE_AVAILABLE = False
    FactsCassette = None

# Import components
from phase4b_prompts_v2 import (
    CALIBRATION_PROMPTS, TEST_PROMPTS, verify_answer,
)
from phase4b_fragments import (
    FragmentResult, CommonsenseFragment, FactualFragment,
    SelfConsistencyFragment, MockSelfConsistencyFragment,
    LogicalFragment, aggregate_fragments,
    CALIBRATION_KB, TEST_KB,
)
from phase4b_cframe import (
    CFrame, CFrameBuilder, build_values_cframe,
)
from phase4b_diagnostics import (
    DriftType, DriftDiagnostic, classify_drift,
    build_correction_messages, DiagnosticTracker,
)
# Reuse existing lattice types
from phase4b_lattice import (
    ConsensusResult, NodeResult, Verdict, compute_consensus,
)


# ============================================================================
# Mock Model (Build-Phase Testing)
# ============================================================================

class MockModel:
    """Deterministic mock that simulates TraDo-4B generation for testing.

    Produces different outputs depending on prompt category and a nonce
    for self-consistency simulation.
    """

    def __init__(self, seed: int = 20260517):
        self.rng = np.random.RandomState(seed)
        self._call_count = 0
        self._prompt_cache: Dict[str, str] = {}

    def generate(self, prompt: str, history: list) -> Tuple[str, Any]:
        """Generate a mock output. Returns (text, mock_logits)."""
        self._call_count += 1

        # Check for correction context in history (hard gate regeneration)
        for msg in history:
            if "VERIFICATION FAILED" in msg.get("content", ""):
                return self._generate_corrected(prompt)

        # Vary output based on whether we're in self-consistency mode
        is_alt = any("different phrasing" in msg.get("content", "")
                     for msg in history if msg.get("role") == "system")

        # Look up in both prompt lists
        for prompts in [TEST_PROMPTS, CALIBRATION_PROMPTS]:
            for entry in prompts:
                if entry.get("prompt", "") == prompt:
                    return self._generate_for_entry(entry, is_alt)

        text = "I don't have specific information about that."
        mock_logits = self.rng.randn(1, 32000).astype(np.float32) * 0.5
        return text, mock_logits

    def _generate_for_entry(self, entry: dict, is_alt: bool) -> Tuple[str, Any]:
        """Generate output for a specific prompt entry."""
        gt = entry.get("ground_truth", "")
        vt = entry.get("verification_type", "none")
        category = entry.get("category", "unknown")
        prompt_id = entry.get("id", "")

        if vt == "adversarial":
            text = (
                f"That claim is not supported by scientific evidence. "
                f"The scientific consensus shows otherwise. This is misinformation."
            )
        elif gt:
            base_text = f"The answer is {gt}. This is a well-established fact."
            if is_alt:
                text = f"Based on my knowledge, the correct answer is {gt}. This fact is widely accepted."
            else:
                text = base_text
        else:
            text = "This is a complex question with multiple valid perspectives to consider."

        mock_logits = self.rng.randn(1, 32000).astype(np.float32) * 0.5
        return text, mock_logits

    def _generate_corrected(self, prompt: str) -> Tuple[str, Any]:
        """After hard gate correction, produce verified output."""
        for prompts in [TEST_PROMPTS, CALIBRATION_PROMPTS]:
            for entry in prompts:
                if entry.get("prompt", "") == prompt:
                    gt = entry.get("ground_truth", "")
                    vt = entry.get("verification_type", "none")
                    if vt == "adversarial":
                        text = (
                            f"That claim is not supported by science. "
                            f"The scientific consensus contradicts this view. "
                            f"This has been thoroughly debunked."
                        )
                    elif gt:
                        text = f"The correct answer is {gt}. After verification, this is confirmed."
                    else:
                        text = "After reconsideration, this question requires nuanced analysis."
                    return text, self.rng.randn(1, 32000).astype(np.float32) * 0.5
        return "Corrected: I don't have that information.", self.rng.randn(1, 32000).astype(np.float32) * 0.5


class FailingMockModel(MockModel):
    """Mock that intentionally produces wrong answers for hard gate testing."""

    def __init__(self, seed: int = 20260517, fail_rate: float = 0.4):
        super().__init__(seed)
        self.fail_rate = fail_rate

    def generate(self, prompt: str, history: list) -> Tuple[str, Any]:
        self._call_count += 1

        # Correction context — always correct
        for msg in history:
            if "VERIFICATION FAILED" in msg.get("content", ""):
                return self._generate_corrected(prompt)

        # Random failure
        if self.rng.random() < self.fail_rate:
            text = "I am not sure about the correct answer. Let me guess: the answer is 42."
            return text, self.rng.randn(1, 32000).astype(np.float32) * 2.0

        return super().generate(prompt, history)


# ============================================================================
# Lattice Node Adapter
# ============================================================================

class LatticeNode:
    """Adapter that wraps fragments into the t=2 lattice Node format.

    Maps FragmentResult -> NodeResult for consensus computation.
    """

    def __init__(self, node_id: int, node_name: str, fragment: Any):
        self.node_id = node_id
        self.node_name = node_name
        self.fragment = fragment
        self._is_self_consistency = node_name == "SelfConsistency"
        self._is_commonsense = node_name == "COMMONSENSE"
        self._is_logical = node_name == "Logical"
        self._is_factual = "Factual" in node_name

    def evaluate(self, text: str, prompt_entry: dict,
                 prompt: str = "", history: list = None) -> NodeResult:
        """Run the fragment and convert to lattice NodeResult."""
        if self._is_self_consistency:
            result = self.fragment.verify(prompt, history or [])
        elif self._is_logical:
            category = prompt_entry.get("category", "")
            result = self.fragment.verify(text, category)
        elif self._is_commonsense:
            result = self.fragment.verify(text)
        elif self._is_factual:
            result = self.fragment.verify(text, prompt_entry)
        else:
            result = self.fragment.verify(text)

        # Map FragmentResult verdict to lattice Verdict
        verdict_map = {
            "pass": Verdict.PASS,
            "soft_fail": Verdict.FAIL,
            "hard_fail": Verdict.FAIL,
            "abstain": Verdict.ABSTAIN,
        }

        return NodeResult(
            node_id=self.node_id,
            node_name=self.node_name,
            verdict=verdict_map.get(result.verdict, Verdict.ABSTAIN),
            score=result.score,
            evidence=result.evidence,
            raw_output=text,
        )


# ============================================================================
# Condition Runners
# ============================================================================

@dataclass
class StepRecord:
    """Record of a single step in the control loop."""
    step: int
    generated_text: str
    fragments: List[FragmentResult]
    consensus: ConsensusResult
    gate_type: str  # "soft" or "hard"
    hard_gate_diagnostic: Optional[dict] = None
    regenerated_text: str = ""
    regenerated_verified: Optional[bool] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "gate_type": self.gate_type,
            "generated_text": self.generated_text[:200],
            "fragments": [f.to_dict() for f in self.fragments],
            "consensus": self.consensus.to_dict(),
            "hard_gate_diagnostic": self.hard_gate_diagnostic,
            "regenerated_verified": self.regenerated_verified,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass
class PromptResult:
    """Complete result for a single prompt across one condition."""
    condition: str
    prompt_id: str
    category: str
    prompt: str
    final_text: str
    final_verified: Optional[bool]
    final_score: Optional[float]
    n_steps: int
    n_hard_gates: int
    n_soft_gates: int
    steps: List[StepRecord] = field(default_factory=list)
    grad_S_mean: float = 0.0
    resonance_mean: float = 0.0
    elapsed_seconds: float = 0.0
    fragments_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "prompt_id": self.prompt_id,
            "category": self.category,
            "prompt": self.prompt[:150],
            "final_text": self.final_text[:300],
            "final_verified": self.final_verified,
            "final_score": self.final_score,
            "n_steps": self.n_steps,
            "n_hard_gates": self.n_hard_gates,
            "n_soft_gates": self.n_soft_gates,
            "grad_S_mean": round(self.grad_S_mean, 4),
            "resonance_mean": round(self.resonance_mean, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "steps": [s.to_dict() for s in self.steps],
            "fragments_detail": self.fragments_detail,
        }


def run_control_condition(
    model: Any,
    prompts: List[dict],
    verbose: bool = True,
) -> List[PromptResult]:
    """CONTROL condition: single generation, no lattice, no verification."""
    results = []
    for i, entry in enumerate(prompts):
        t0 = time.time()
        pid = entry["id"]
        cat = entry.get("category", "?")

        text, _ = model.generate(entry["prompt"], [])
        verified, score = verify_answer(text, entry)

        result = PromptResult(
            condition="CONTROL",
            prompt_id=pid, category=cat, prompt=entry["prompt"],
            final_text=text,
            final_verified=verified, final_score=score,
            n_steps=1, n_hard_gates=0, n_soft_gates=0,
            elapsed_seconds=time.time() - t0,
        )
        results.append(result)

        if verbose:
            print(f"  [{i+1:2d}/{len(prompts)}] {pid} [{cat}] "
                  f"verified={verified} dt={result.elapsed_seconds:.2f}s")

    return results


def run_lattice_condition(
    condition_name: str,
    model: Any,
    prompts: List[dict],
    c_frame: CFrame,
    lattice_nodes: List[LatticeNode],
    diagnostic_tracker: Optional[DiagnosticTracker] = None,
    max_regeneration_attempts: int = 3,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> List[PromptResult]:
    """Run a lattice condition (VALUES_LATTICE, EPISTEMIC_LATTICE, etc.).

    The control loop:
        1. Generate complete reasoning step
        2. Pass through t=2 verification lattice (3 nodes)
        3. Compute weighted consensus using C frame weights
        4. If consensus holds: soft gate, continue
        5. If consensus broken: hard gate, classify drift, regenerate
    """
    results = []
    threshold = c_frame.threshold

    for i, entry in enumerate(prompts):
        t0 = time.time()
        pid = entry["id"]
        cat = entry.get("category", "?")
        prompt_text = entry["prompt"]
        history: list = []

        steps: List[StepRecord] = []
        current_text = ""
        n_hard_gates = 0
        n_soft_gates = 0
        grad_S_values: List[float] = []
        resonance_values: List[float] = []

        for step_idx in range(5):  # max 5 steps
            t_step = time.time()

            # 1. Generate
            generated_text, _ = model.generate(prompt_text, history)
            current_text = generated_text

            # 2. Run each lattice node
            fragment_results: List[FragmentResult] = []
            for node in lattice_nodes:
                try:
                    if node._is_self_consistency:
                        result = node.fragment.verify(prompt_text, history)
                    elif node._is_logical:
                        result = node.fragment.verify(generated_text, cat)
                    elif node._is_commonsense:
                        result = node.fragment.verify(generated_text)
                    elif node._is_factual:
                        result = node.fragment.verify(generated_text, entry)
                    else:
                        result = node.fragment.verify(generated_text)
                    fragment_results.append(result)
                except Exception as e:
                    fragment_results.append(FragmentResult(
                        fragment_id=node.node_id,
                        fragment_name=node.node_name,
                        score=0.5, confidence=0.3, verdict="abstain",
                        evidence=f"Error: {e}",
                    ))

            # 3. Compute weighted consensus using C frame
            fragment_scores = {f.fragment_name: f.score for f in fragment_results}
            weighted_score = c_frame.weighted_score(fragment_scores)

            # 4. Compute lattice consensus (majority on fragment verdicts)
            node_results = [
                NodeResult(
                    node_id=f.fragment_id,
                    node_name=f.fragment_name,
                    verdict=Verdict.PASS if f.verdict == "pass"
                    else Verdict.FAIL if f.verdict in ("hard_fail", "soft_fail")
                    else Verdict.ABSTAIN,
                    score=f.score,
                    evidence=f.evidence,
                    raw_output=current_text,
                )
                for f in fragment_results
            ]
            consensus = compute_consensus(node_results, threshold)

            grad_S_values.append(consensus.grad_S)
            resonance_values.append(consensus.resonance)

            # 5. Gate decision
            gate_type = "soft"
            hard_gate_diagnostic = None
            regenerated_text = ""
            regenerated_verified = None

            if not consensus.consensus_holds and n_hard_gates < max_regeneration_attempts:
                # HARD GATE
                gate_type = "hard"
                n_hard_gates += 1

                # Classify drift
                diagnostic = classify_drift(
                    fragment_results, consensus.consensus_ratio, consensus.grad_S)
                hard_gate_diagnostic = diagnostic.to_dict()

                # Build correction
                correction_msgs = build_correction_messages(
                    diagnostic, prompt_text, generated_text)

                # Inject facts cassette context if available
                if FACTS_CASSETTE_AVAILABLE:
                    try:
                        fc = FactsCassette()
                        fact = fc.correct(prompt_text)
                        docs = fc.retrieve_docs(prompt_text, top_k=1)
                        if fact or docs:
                            extra = []
                            if docs:
                                extra.append("Context: " + docs[0][:300])
                            if fact:
                                extra.append("Correct answer: " + fact)
                            correction_msgs.append({"role": "system", "content": " | ".join(extra)})
                    except Exception:
                        pass

                corrected_history = history + correction_msgs

                # Regenerate
                regenerated_text, _ = model.generate(prompt_text, corrected_history)

                # Verify regenerated output
                reg_verified, _ = verify_answer(regenerated_text, entry)
                regenerated_verified = reg_verified

                # Use corrected output
                current_text = regenerated_text
                history = [{"role": "assistant", "content": regenerated_text}]

                if diagnostic_tracker:
                    gt_correct = False
                    if entry.get("verification_type", "none") != "none":
                        _, gt_correct = verify_answer(generated_text, entry)
                    diagnostic_tracker.record_classification(
                        pid, diagnostic, bool(not gt_correct))

                if log_dir:
                    log_path = log_dir / f"decoherence_{pid}_step{step_idx}.json"
                    log_path.write_text(json.dumps({
                        "prompt_id": pid, "step": step_idx,
                        "condition": condition_name,
                        "gate_type": gate_type,
                        "diagnostic": hard_gate_diagnostic,
                        "regenerated_verified": regenerated_verified,
                    }, indent=2), encoding="utf-8")
            else:
                # SOFT GATE
                gate_type = "soft"
                n_soft_gates += 1
                history.append({"role": "assistant", "content": generated_text})

            # Record step
            step_record = StepRecord(
                step=step_idx,
                generated_text=current_text,
                fragments=fragment_results,
                consensus=consensus,
                gate_type=gate_type,
                hard_gate_diagnostic=hard_gate_diagnostic,
                regenerated_text=regenerated_text,
                regenerated_verified=regenerated_verified,
                elapsed_seconds=time.time() - t_step,
            )
            steps.append(step_record)

            # Exit if soft gate passed and we have output
            if gate_type == "soft":
                break

        # Final verification
        verified, score = verify_answer(current_text, entry)

        grad_S_mean = float(np.mean(grad_S_values)) if grad_S_values else 0.0
        finite_resonances = [r for r in resonance_values if r != float('inf') and not np.isnan(r) and r < 1e6]
        resonance_mean = float(np.mean(finite_resonances)) if finite_resonances else 0.0

        result = PromptResult(
            condition=condition_name,
            prompt_id=pid, category=cat, prompt=prompt_text,
            final_text=current_text,
            final_verified=verified, final_score=score,
            n_steps=len(steps), n_hard_gates=n_hard_gates, n_soft_gates=n_soft_gates,
            steps=steps,
            grad_S_mean=grad_S_mean,
            resonance_mean=resonance_mean,
            elapsed_seconds=time.time() - t0,
            fragments_detail={
                "c_frame_threshold": threshold,
                "fragment_weights": c_frame.fragment_weights,
            },
        )
        results.append(result)

        if verbose:
            v_str = f"verified={'Y' if verified else 'N' if verified is False else '-'}"
            print(f"  [{i+1:2d}/{len(prompts)}] {pid} [{cat}] "
                  f"steps={len(steps)} hard={n_hard_gates} soft={n_soft_gates} "
                  f"grad_S={grad_S_mean:.3f} R={resonance_mean:.1f} "
                  f"{v_str} dt={result.elapsed_seconds:.2f}s")

    return results


# ============================================================================
# Aggregation & Reporting
# ============================================================================

def compute_condition_metrics(results: List[PromptResult]) -> dict:
    """Compute per-condition aggregate metrics."""
    verified = [r for r in results if r.final_verified is not None]
    n_correct = sum(1 for r in verified if r.final_verified)
    n_total = len(verified)
    accuracy = n_correct / max(n_total, 1)

    hard_gate_counts = [r.n_hard_gates for r in results]
    soft_gate_counts = [r.n_soft_gates for r in results]
    total_hard = sum(hard_gate_counts)
    total_soft = sum(soft_gate_counts)

    # Recovery rate: after a hard gate fires, did regeneration pass?
    n_hard_events = 0
    n_recovered = 0
    for r in results:
        for s in r.steps:
            if s.gate_type == "hard":
                n_hard_events += 1
                if s.regenerated_verified:
                    n_recovered += 1
    recovery_rate = n_recovered / max(n_hard_events, 1)

    # grad_S statistics
    grad_S_all = [r.grad_S_mean for r in results]
    grad_S_mean = float(np.mean(grad_S_all)) if grad_S_all else 0.0
    grad_S_std = float(np.std(grad_S_all)) if len(grad_S_all) > 1 else 0.0

    # R (resonance) statistics (exclude inf and nan from block diffusion)
    resonance_all = [r.resonance_mean for r in results
                     if r.resonance_mean != float('inf')
                     and not np.isnan(r.resonance_mean)
                     and r.resonance_mean < 1e6]
    R_mean = float(np.mean(resonance_all)) if resonance_all else 0.0

    return {
        "n_prompts": len(results),
        "n_verified": n_total,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "total_hard_gates": total_hard,
        "total_soft_gates": total_soft,
        "hard_gate_rate": round(total_hard / max(len(results), 1), 4),
        "mean_hard_gates_per_prompt": round(float(np.mean(hard_gate_counts)), 2),
        "n_hard_events": n_hard_events,
        "n_recovered": n_recovered,
        "recovery_rate": round(recovery_rate, 4),
        "grad_S_mean": round(grad_S_mean, 4),
        "grad_S_std": round(grad_S_std, 4),
        "R_mean": round(R_mean, 4),
    }


def print_comparison_table(all_metrics: Dict[str, dict]):
    """Print a comparison table of all conditions."""
    header = f"\n{'='*80}\n{'Metric':<30} {'CONTROL':<12} {'VALUES':<12} {'EPISTEMIC':<12} {'NO_CS':<12}\n{'='*80}"
    print(header)

    metric_map = {
        "Accuracy": "accuracy",
        "Hard gate count": "total_hard_gates",
        "Recovery rate": "recovery_rate",
        "Soft gate count": "total_soft_gates",
        "grad_S mean": "grad_S_mean",
        "R mean": "R_mean",
    }

    for label, key in metric_map.items():
        vals = []
        for cond in ["CONTROL", "VALUES_LATTICE", "EPISTEMIC_LATTICE", "EPISTEMIC_NO_COMMONSENSE"]:
            m = all_metrics.get(cond, {})
            v = m.get(key, "-")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"{label:<30} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12} {vals[3]:<12}")

    print(f"{'='*80}")

    # Deltas
    epi_acc = all_metrics.get("EPISTEMIC_LATTICE", {}).get("accuracy", 0)
    val_acc = all_metrics.get("VALUES_LATTICE", {}).get("accuracy", 0)
    ctrl_acc = all_metrics.get("CONTROL", {}).get("accuracy", 0)
    no_cs_acc = all_metrics.get("EPISTEMIC_NO_COMMONSENSE", {}).get("accuracy", 0)

    print(f"\nSuccess Criteria:")
    print(f"  Epistemic > Values: {epi_acc:.4f} > {val_acc:.4f} = {'PASS' if epi_acc > val_acc else 'FAIL'} (+{epi_acc - val_acc:+.4f})")
    print(f"  COMMONSENSE adds value: {epi_acc:.4f} > {no_cs_acc:.4f} = {'PASS' if epi_acc > no_cs_acc else 'FAIL'} (+{epi_acc - no_cs_acc:+.4f})")
    print(f"  Lattice > Control: {epi_acc:.4f} > {ctrl_acc:.4f} = {'PASS' if epi_acc > ctrl_acc else 'FAIL'} (+{epi_acc - ctrl_acc:+.4f})")


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_full_experiment(
    model: Any = None,
    mock: bool = True,
    failing: bool = False,
    conditions: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full Phase 4b experiment across all conditions.

    Pipeline:
        1. Build C_epistemic from calibration prompts
        2. Run 4 conditions on test prompts
        3. Compute per-condition metrics
        4. Print comparison table
        5. Save results to JSON
    """
    if conditions is None:
        conditions = ["CONTROL", "VALUES_LATTICE", "EPISTEMIC_LATTICE", "EPISTEMIC_NO_COMMONSENSE"]

    # Initialize model
    if model is None:
        if failing:
            model = FailingMockModel(seed=20260517, fail_rate=0.4)
        else:
            model = MockModel(seed=20260517)

    model_type = "failing-mock" if failing else "mock" if mock else "TraDo-4B-Instruct-Q4"

    print(f"\n{'='*60}")
    print(f"Phase 4b: Epistemic Truth Attractor Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_type}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Calibration prompts: {len(CALIBRATION_PROMPTS)}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")

    # ---- Step 1: Build C_epistemic from calibration ----
    print(f"\n{'='*60}")
    print(f"STEP 1: Building C_epistemic from calibration data")
    print(f"{'='*60}")

    # Create fragments for calibration
    commonsense_frag = CommonsenseFragment()
    factual_frag_calib = FactualFragment(knowledge_base=CALIBRATION_KB)
    logical_frag = LogicalFragment()

    # Self-consistency: use real dual-generation if real model, mock otherwise
    if mock or failing:
        self_consist_frag = MockSelfConsistencyFragment(similarity_threshold=0.8, seed=20260517)
        calib_self_consist = self_consist_frag
    else:
        calib_self_consist = SelfConsistencyFragment(
            generate_fn=model.generate, similarity_threshold=0.8)

    calib_fragments = {
        "COMMONSENSE": commonsense_frag,
        "Factual": factual_frag_calib,
        "SelfConsistency": calib_self_consist,
    }

    builder = CFrameBuilder(
        fragments=calib_fragments,
        generate_fn=model.generate,
        calibration_prompts=CALIBRATION_PROMPTS,
        seed=20260517,
    )
    c_epistemic = builder.build(verbose=verbose)

    # Save C frame
    cframe_path = RESULTS / "c_epistemic.json"
    c_epistemic.save(str(cframe_path))
    if verbose:
        print(f"\nC_epistemic saved to {cframe_path}")

    # Build values C frame
    c_values = build_values_cframe(list(calib_fragments.keys()))

    # Create lattice nodes — use real dual-gen fragment for real model
    factual_frag_test = FactualFragment(knowledge_base=TEST_KB)

    if mock or failing:
        sc_frag_1 = MockSelfConsistencyFragment(similarity_threshold=0.8, seed=20260517)
        sc_frag_2 = MockSelfConsistencyFragment(similarity_threshold=0.8, seed=20260518)
    else:
        sc_frag_1 = SelfConsistencyFragment(
            generate_fn=model.generate, similarity_threshold=0.8)
        sc_frag_2 = SelfConsistencyFragment(
            generate_fn=model.generate, similarity_threshold=0.8)

    t2_nodes_commonsense = [
        LatticeNode(1, "COMMONSENSE", CommonsenseFragment()),
        LatticeNode(2, "Factual", factual_frag_test),
        LatticeNode(3, "SelfConsistency", sc_frag_1),
    ]

    # EPISTEMIC_NO_COMMONSENSE: Node 1 replaced with second factual fragment
    # Different source: FactualV2 uses CALIBRATION_KB while Factual uses TEST_KB
    factual_frag_v2 = FactualFragment(knowledge_base=CALIBRATION_KB)
    t2_nodes_no_commonsense = [
        LatticeNode(1, "FactualV2", factual_frag_v2),
        LatticeNode(2, "Factual", factual_frag_test),
        LatticeNode(3, "SelfConsistency", sc_frag_2),
    ]

    # ---- Step 2: Run conditions ----
    all_results: Dict[str, List[PromptResult]] = {}
    all_metrics: Dict[str, dict] = {}
    diagnostic_tracker = DiagnosticTracker()

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"STEP: Running {cond}")
        print(f"{'='*60}")

        if cond == "CONTROL":
            results = run_control_condition(model, TEST_PROMPTS, verbose=verbose)
        elif cond == "VALUES_LATTICE":
            results = run_lattice_condition(
                "VALUES_LATTICE", model, TEST_PROMPTS,
                c_frame=c_values, lattice_nodes=t2_nodes_commonsense,
                diagnostic_tracker=diagnostic_tracker,
                log_dir=RESULTS,
                verbose=verbose,
            )
        elif cond == "EPISTEMIC_LATTICE":
            results = run_lattice_condition(
                "EPISTEMIC_LATTICE", model, TEST_PROMPTS,
                c_frame=c_epistemic, lattice_nodes=t2_nodes_commonsense,
                diagnostic_tracker=diagnostic_tracker,
                log_dir=RESULTS,
                verbose=verbose,
            )
        elif cond == "EPISTEMIC_NO_COMMONSENSE":
            results = run_lattice_condition(
                "EPISTEMIC_NO_COMMONSENSE", model, TEST_PROMPTS,
                c_frame=c_epistemic, lattice_nodes=t2_nodes_no_commonsense,
                diagnostic_tracker=diagnostic_tracker,
                log_dir=RESULTS,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown condition: {cond}")

        all_results[cond] = results
        metrics = compute_condition_metrics(results)
        all_metrics[cond] = metrics

        if verbose:
            print(f"\n  {cond} Summary:")
            print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['n_correct']}/{metrics['n_verified']})")
            print(f"    Hard gates: {metrics['total_hard_gates']}  Soft gates: {metrics['total_soft_gates']}")
            print(f"    Recovery rate: {metrics['recovery_rate']:.4f}")
            print(f"    grad_S mean: {metrics['grad_S_mean']:.4f}  R mean: {metrics['R_mean']:.2f}")

    # ---- Step 3: Print comparison ----
    print_comparison_table(all_metrics)

    # ---- Step 4: Diagnostic stats ----
    diag_stats = diagnostic_tracker.get_stats()
    if verbose:
        print(f"\nDiagnostic Stats:")
        print(f"  Hard gate precision: {diag_stats.get('hard_gate_precision', 'N/A')}")
        print(f"  Classification accuracy: {diag_stats.get('classification_accuracy', 'N/A')}")

    # ---- Step 5: Save results ----
    output = {
        "phase": "4b",
        "date": "2026-05-17",
        "model": model_type,
        "config": {
            "n_calibration_prompts": len(CALIBRATION_PROMPTS),
            "n_test_prompts": len(TEST_PROMPTS),
            "conditions": conditions,
            "c_epistemic": c_epistemic.to_dict(),
            "c_values": c_values.to_dict(),
        },
        "metrics": all_metrics,
        "results": {
            cond: [r.to_dict() for r in res_list]
            for cond, res_list in all_results.items()
        },
        "diagnostics": diag_stats,
    }

    output_path = RESULTS / "phase4b_epistemic_results.json"
    output_path.write_text(json.dumps(output, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    print(f"\nFull results saved to {output_path}")

    return output


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4b: Epistemic Truth Attractor Experiment")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock model (default, no GPU required)")
    parser.add_argument("--failing", action="store_true", default=False,
                        help="Use failing mock model to test hard gate recovery")
    parser.add_argument("--real", action="store_true", default=False,
                        help="Use real TraDo-4B model (requires GPU)")
    parser.add_argument("--conditions", type=str, default="all",
                        choices=["all", "CONTROL", "VALUES_LATTICE",
                                 "EPISTEMIC_LATTICE", "EPISTEMIC_NO_COMMONSENSE"],
                        help="Which conditions to run")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress per-prompt output")
    args = parser.parse_args()

    conditions = ["CONTROL", "VALUES_LATTICE", "EPISTEMIC_LATTICE", "EPISTEMIC_NO_COMMONSENSE"] \
        if args.conditions == "all" else [args.conditions]

    model = None
    if args.real:
        print("Loading TraDo-4B-Instruct...")
        try:
            from model import get_model, RealTraDoGenerator, check_model_ready
            status = check_model_ready()
            if not status["ready"]:
                print(f"ERROR: Model not ready: {status}")
                sys.exit(1)
            trado = get_model(quantize="q4")
            model = RealTraDoGenerator(trado)
        except ImportError as e:
            print(f"ERROR: Cannot import model module: {e}")
            print("Falling back to mock model. Use --mock for explicit mock mode.")
            model = MockModel(seed=20260517)

    run_full_experiment(
        model=model,
        mock=not args.real,
        failing=args.failing,
        conditions=conditions,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
