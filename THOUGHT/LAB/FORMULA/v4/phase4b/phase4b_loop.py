"""Phase 4b: Step-Level Macro-Consensus Control Loop

The main control loop that wraps agent generation with:
    1. Generate a complete reasoning step
    2. Pass through t=2 verification lattice
    3. If consensus holds (grad_S < threshold): soft gate approve, continue
    4. If consensus broken (grad_S >= threshold): hard gate halt, overwrite, regenerate
    5. Track R = 1/grad_S across steps
    6. Track Df for anomaly detection

Domain mapping (v4 INDEX.md):
    E  = consensus_ratio (signal core - fraction of passing nodes)
    grad_S = sqrt(1 - consensus_ratio) (dissonance density)
    sigma = majority vote across t=2 lattice (alignment operator)
    Df  = effective dimensionality of output distribution (redundancy)
    R   = 1/grad_S (resonance / control effect)

Trigger Conditions:
    Hard gate: grad_S >= THRESHOLD (default 0.5), meaning consensus_ratio <= 0.5
    Soft gate: grad_S < THRESHOLD, meaning consensus_ratio > 0.5
"""

import json, math, time, numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from phase4b_lattice import evaluate_lattice, ConsensusResult, THRESHOLD_DEFAULT
from phase4b_gates import (
    SoftGate, HardGate, DfTracker, CSymbolRegistry,
    build_correction_context,
)
from phase4b_prompts import TEST_PROMPTS, verify_answer


# ============================================================================
# Types
# ============================================================================

AgentGenerateFn = Callable[
    [str, list],       # (prompt, message_history) -> (text, logits_array)
    tuple[str, np.ndarray],
]
"""Type for the agent generation function.

Takes a prompt string and message history (list of dicts with role/content).
Returns (generated_text, logits_array) where logits_array is the token-level
logits used for Df computation.

During build phase: use MockModel.generate().
During run phase: swap in real model inference.
"""


@dataclass
class StepRecord:
    """Record of a single step in the control loop."""
    step: int
    generated_text: str
    consensus: ConsensusResult
    gate_type: str  # "soft" or "hard"
    df_snapshot: dict = field(default_factory=dict)
    hard_gate_event: Optional[dict] = None
    resonance_trajectory: list = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self):
        d = {
            "step": self.step,
            "gate_type": self.gate_type,
            "generated_text": self.generated_text[:200],
            "consensus": self.consensus.to_dict(),
            "df": self.df_snapshot,
            "hard_gate_event": self.hard_gate_event,
            "resonance": self.consensus.resonance,
            "grad_S": self.consensus.grad_S,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.resonance_trajectory:
            d["resonance_trajectory"] = [round(r, 4) for r in self.resonance_trajectory]
        return d


# ============================================================================
# Mock Model (for build-phase testing)
# ============================================================================

class MockModel:
    """Deterministic mock that simulates agent generation for testing.

    Produces configurable outputs: good (passes verification),
    bad (fails verification), or mixed (varies across steps).
    """

    def __init__(self, seed: int = 20260516):
        self.rng = np.random.RandomState(seed)
        self._call_count = 0

    def generate(self, prompt: str, history: list) -> tuple[str, np.ndarray]:
        """Generate a mock output. Returns (text, mock_logits).

        The output depends on prompt category and failure mode:
        - For prompts with ground_truth: includes it in the output
        - Can be toggled to produce failing outputs for hard gate testing
        """
        self._call_count += 1

        # Find matching prompt entry to determine expected output
        for entry in TEST_PROMPTS:
            if entry["prompt"] == prompt:
                gt = entry.get("ground_truth", "")
                vt = entry.get("verification_type", "none")

                # For adversarial prompts, generate a rejection response
                if vt == "adversarial":
                    text = (
                        f"That claim is not supported by scientific evidence. "
                        f"The scientific consensus indicates otherwise. "
                        f"No evidence supports this position."
                    )
                elif gt:
                    text = f"The answer is {gt}. This is a well-established fact."
                else:
                    text = "This is a complex question with multiple perspectives to consider."

                # Generate mock logits (normal distribution, Df ~ 100-1000)
                mock_logits = self.rng.randn(1, 32000).astype(np.float32) * 0.5
                return text, mock_logits

        return "I don't have information about that.", self.rng.randn(1, 32000).astype(np.float32) * 0.5


class FailingMockModel(MockModel):
    """Mock that intentionally produces wrong answers for hard gate testing."""

    def __init__(self, seed: int = 20260516, fail_rate: float = 0.3):
        super().__init__(seed)
        self.fail_rate = fail_rate

    def generate(self, prompt: str, history: list) -> tuple[str, np.ndarray]:
        text, logits = super().generate(prompt, history)

        # Check if correction context injected (from history)
        for msg in history:
            if "VERIFICATION FAILED" in msg.get("content", ""):
                # After correction, produce correct output
                for entry in TEST_PROMPTS:
                    if entry["prompt"] == prompt:
                        gt = entry.get("ground_truth", "")
                        text = f"The correct answer is {gt}. After verification, this is confirmed."
                        return text, logits

        # Randomly produce failures
        if self.rng.random() < self.fail_rate:
            text = "I am not sure about the correct answer. Let me guess randomly."
            # Inject anomalous logits (high entropy -> high Df spike)
            logits = self.rng.randn(1, 32000).astype(np.float32) * 2.0

        return text, logits


# ============================================================================
# Step-Level Control Loop
# ============================================================================

@dataclass
class LoopResult:
    """Complete result of a step-level control loop run."""
    prompt_id: str
    category: str
    prompt: str
    final_text: str
    n_steps: int
    n_hard_gates: int
    n_soft_gates: int
    steps: list = field(default_factory=list)
    resonance_trajectory: list = field(default_factory=list)
    df_trajectory: list = field(default_factory=list)
    df_stats: dict = field(default_factory=dict)
    soft_gate_stats: dict = field(default_factory=dict)
    hard_gate_stats: dict = field(default_factory=dict)
    final_verified: Optional[bool] = None
    final_verification_score: Optional[float] = None
    elapsed_seconds: float = 0.0
    C_symbol_stats: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "prompt": self.prompt[:100],
            "final_text": self.final_text[:300],
            "n_steps": self.n_steps,
            "n_hard_gates": self.n_hard_gates,
            "n_soft_gates": self.n_soft_gates,
            "steps": [s.to_dict() for s in self.steps],
            "resonance_trajectory": [round(r, 4) for r in self.resonance_trajectory],
            "df_trajectory": self.df_trajectory,
            "df_stats": self.df_stats,
            "soft_gate_stats": self.soft_gate_stats,
            "hard_gate_stats": self.hard_gate_stats,
            "final_verified": self.final_verified,
            "final_verification_score": self.final_verification_score,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "C_symbol_stats": self.C_symbol_stats,
        }


def run_control_loop(
    prompt_entry: dict,
    generator: AgentGenerateFn,
    knowledge_base: Optional[dict] = None,
    threshold: float = THRESHOLD_DEFAULT,
    max_steps: int = 5,
    max_regeneration_attempts: int = 3,
    log_dir: Optional[Path] = None,
    track_df: bool = True,
    return_resonance: bool = True,
) -> LoopResult:
    """Run the step-level macro-consensus control loop.

    Args:
        prompt_entry: dict with id, prompt, ground_truth, verification_type, category
        generator: function that takes (prompt, history) and returns (text, logits)
        knowledge_base: optional external knowledge dict for Node 2
        threshold: grad_S threshold for consensus breaking
        max_steps: maximum number of steps before forced completion
        max_regeneration_attempts: max hard gate retries per step
        log_dir: optional directory for decoherence event logs
        track_df: whether to track Df anomaly detection
        return_resonance: whether to compute R trajectory

    Returns:
        LoopResult with full trajectory, gate events, and final verification
    """
    t_start = time.time()
    soft_gate = SoftGate()
    hard_gate = HardGate(log_dir=log_dir)
    df_tracker = DfTracker() if track_df else None
    c_registry = CSymbolRegistry()

    prompt = prompt_entry["prompt"]
    prompt_id = prompt_entry["id"]
    category = prompt_entry.get("category", "unknown")
    history: list = []

    steps: list = []
    resonance_trajectory: list = []
    df_trajectory: list = []
    current_text = ""
    n_hard_gates = 0
    n_soft_gates = 0

    for step_idx in range(max_steps):
        t_step_start = time.time()

        # --- 1. Generate a step ---
        generated_text, logits = generator(prompt, history)
        current_text = generated_text

        # --- 2. Compress into @C symbol ---
        c_symbol = c_registry.compress(generated_text)

        # --- 3. Verify through t=2 lattice ---
        consensus = evaluate_lattice(generated_text, prompt_entry, knowledge_base, threshold)

        # --- 4. Track Df ---
        df_snapshot = {}
        if df_tracker and logits is not None:
            snap = df_tracker.record(step_idx, logits, {"prompt_id": prompt_id})
            df_snapshot = {
                "df_value": round(snap.df_value, 4),
                "is_anomaly": snap.is_anomaly,
                "anomaly_score": round(snap.anomaly_score, 4),
            }
            df_trajectory.append(df_snapshot)

        # --- 5. Track R ---
        resonance_trajectory.append(consensus.resonance)

        # --- 6. Gate decision ---
        hard_gate_event = None
        gate_type = "soft"

        if not consensus.consensus_holds and n_hard_gates < max_regeneration_attempts:
            # HARD GATE: consensus broken
            gate_type = "hard"
            n_hard_gates += 1

            # Build correction context from clean consensus
            correction_ctx = build_correction_context(
                prompt, generated_text, consensus.passing_outputs)

            # Regenerate with correction
            corrected_text, corrected_logits = generator(prompt, history + correction_ctx)

            # Verify the regenerated output
            corrected_consensus = evaluate_lattice(
                corrected_text, prompt_entry, knowledge_base, threshold)
            regenerated_verified = corrected_consensus.consensus_holds

            # Record hard gate event
            event = hard_gate.halt_and_correct(
                step=step_idx,
                consensus=consensus,
                failed_output=generated_text,
                regenerated_output=corrected_text,
                regenerated_verified=regenerated_verified,
            )

            # If correction passed, use corrected output moving forward
            if regenerated_verified:
                current_text = corrected_text
                generated_text = corrected_text
                consensus = corrected_consensus
                resonance_trajectory[-1] = consensus.resonance

            hard_gate_event = {
                "step": step_idx,
                "grad_S": round(consensus.grad_S, 4),
                "failed_output": generated_text[:200],
                "corrected_output": corrected_text[:200],
                "regenerated_verified": regenerated_verified,
                "clean_consensus": consensus.passing_outputs,
            }

        else:
            # SOFT GATE: consensus holds
            n_soft_gates += 1
            soft_gate.approve(step_idx, consensus)
            history.append({"role": "assistant", "content": generated_text})

        # --- 7. Record step ---
        step_record = StepRecord(
            step=step_idx,
            generated_text=current_text,
            consensus=consensus,
            gate_type=gate_type,
            df_snapshot=df_snapshot,
            hard_gate_event=hard_gate_event,
            resonance_trajectory=list(resonance_trajectory),
            elapsed_seconds=time.time() - t_step_start,
        )
        steps.append(step_record)

        # --- 8. Early exit if consensus holds on final step ---
        # (In real model runs, continue until EOS or step limit)

    # --- Final verification ---
    vt = prompt_entry.get("verification_type", "none")
    if vt != "none" and prompt_entry.get("ground_truth"):
        final_verified, final_score = verify_answer(current_text, prompt_entry)
    else:
        final_verified, final_score = None, None

    t_elapsed = time.time() - t_start

    return LoopResult(
        prompt_id=prompt_id,
        category=category,
        prompt=prompt,
        final_text=current_text,
        n_steps=len(steps),
        n_hard_gates=n_hard_gates,
        n_soft_gates=n_soft_gates,
        steps=steps,
        resonance_trajectory=resonance_trajectory,
        df_trajectory=df_trajectory,
        df_stats=df_tracker.get_stats() if df_tracker else {},
        soft_gate_stats=soft_gate.get_stats(),
        hard_gate_stats=hard_gate.get_stats(),
        final_verified=final_verified,
        final_verification_score=final_score,
        elapsed_seconds=t_elapsed,
        C_symbol_stats=c_registry.get_stats(),
    )


def run_experiment(
    prompt_list: list = TEST_PROMPTS,
    generator: Optional[AgentGenerateFn] = None,
    knowledge_base: Optional[dict] = None,
    threshold: float = THRESHOLD_DEFAULT,
    max_steps: int = 5,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[list, dict]:
    """Run the control loop across all prompts.

    Returns (results_list, aggregated_stats).
    """
    if generator is None:
        generator = MockModel().generate

    results = []
    n_total = len(prompt_list)

    for i, entry in enumerate(prompt_list):
        if verbose:
            print(f"  [{i+1:2d}/{n_total}] {entry['id']} [{entry.get('category','?')}]...", end=" ", flush=True)

        result = run_control_loop(
            prompt_entry=entry,
            generator=generator,
            knowledge_base=knowledge_base,
            threshold=threshold,
            max_steps=max_steps,
            log_dir=log_dir,
        )
        results.append(result)

        if verbose:
            print(f"steps={result.n_steps}  hard={result.n_hard_gates}  soft={result.n_soft_gates}  "
                  f"verified={result.final_verified}  dt={result.elapsed_seconds:.2f}s", flush=True)

    # Aggregate stats
    n_verified = sum(1 for r in results if r.final_verified is not None)
    n_correct = sum(1 for r in results if r.final_verified is True)
    total_hard = sum(r.n_hard_gates for r in results)
    total_soft = sum(r.n_soft_gates for r in results)
    total_steps = sum(r.n_steps for r in results)

    # Hard gate recovery rate
    n_hard_events = sum(len(hard_gate.events) for r in results for s in r.steps
                         if s.hard_gate_event)
    n_recovered = sum(1 for r in results for s in r.steps
                       if s.hard_gate_event and s.hard_gate_event.get("regenerated_verified"))

    # Df anomaly stats
    all_df_rates = [r.df_stats.get("anomaly_rate", 0) for r in results if r.df_stats]

    aggregated = {
        "n_prompts": n_total,
        "n_verified": n_verified,
        "n_correct": n_correct,
        "accuracy": round(n_correct / max(n_verified, 1), 4),
        "total_steps": total_steps,
        "total_hard_gates": total_hard,
        "total_soft_gates": total_soft,
        "hard_gate_rate": round(total_hard / max(total_steps, 1), 4),
        "hard_gate_recovery_rate": round(n_recovered / max(n_hard_events, 1), 4),
        "mean_df_anomaly_rate": round(float(np.mean(all_df_rates)), 4) if all_df_rates else 0,
        "config": {
            "threshold": threshold,
            "max_steps": max_steps,
            "model": "mock" if isinstance(generator.__self__, MockModel) else "real",
        },
    }

    return results, aggregated
