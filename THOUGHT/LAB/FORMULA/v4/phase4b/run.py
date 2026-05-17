"""Phase 4b: Step-Level Macro-Consensus Deployment

**Model:** TraDo-4B-Instruct (dLLM, Q4 on RTX 3060 12GB)
**Date:** 2026-05-16

Run conditions:
  CONTROL: no loop, no verification, standard temperature
  VERIFY-ONLY: verification without gating (passive monitoring)
  CYBERNETIC: full control loop with soft + hard gates

Build phase: use --mock flag to test without model.
Model run: python run.py --condition CYBERNETIC
"""

import argparse, json, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

from phase4b_prompts import TEST_PROMPTS, verify_answer
from phase4b_lattice import evaluate_lattice
from phase4b_loop import (
    run_control_loop, run_experiment,
    MockModel, FailingMockModel,
)
from phase4b_gates import (
    DfTracker, CSymbolRegistry,
)
from phase4b_smoke import run_smoke_test


def run_model_condition(condition: str, model, prompt_list, max_steps: int = 5):
    """Run a specific condition (CONTROL, VERIFY-ONLY, CYBERNETIC).

    CONTROL: standard generation, no loop, no gating
    VERIFY-ONLY: generation with verification monitoring, no gating
    CYBERNETIC: full control loop with soft + hard gates
    """
    results = []

    for i, entry in enumerate(prompt_list):
        pid = entry["id"]
        cat = entry["category"]
        t0 = time.time()

        if condition == "CONTROL":
            # Standard generation: no verification, no gating
            text, _ = model.generate(entry["prompt"], [])
            vt = entry.get("verification_type", "none")
            final_verified, final_score = None, None
            if vt != "none" and entry.get("ground_truth"):
                final_verified, final_score = verify_answer(text, entry)

            result = {
                "condition": condition,
                "prompt_id": pid, "category": cat,
                "generated_text": text,
                "final_verified": final_verified,
                "final_score": final_score,
                "n_steps": 1, "n_hard_gates": 0, "n_soft_gates": 0,
                "elapsed_seconds": time.time() - t0,
                "resonance_trajectory": [],
                "df_trajectory": [],
            }

        elif condition == "VERIFY-ONLY":
            # Verify-only: passive monitoring, no gating
            text, logits = model.generate(entry["prompt"], [])
            consensus = evaluate_lattice(text, entry)
            df_tracker = DfTracker()
            if logits is not None:
                df_tracker.record(0, logits, {"prompt_id": pid})

            vt = entry.get("verification_type", "none")
            final_verified, final_score = None, None
            if vt != "none" and entry.get("ground_truth"):
                final_verified, final_score = verify_answer(text, entry)

            result = {
                "condition": condition,
                "prompt_id": pid, "category": cat,
                "generated_text": text,
                "final_verified": final_verified,
                "final_score": final_score,
                "consensus": consensus.to_dict(),
                "n_steps": 1, "n_hard_gates": 0, "n_soft_gates": 0,
                "elapsed_seconds": time.time() - t0,
                "resonance_trajectory": [consensus.resonance],
                "df_stats": df_tracker.get_stats(),
                "df_trajectory": [{"df_value": round(s.df_value, 4), "is_anomaly": s.is_anomaly}
                                  for s in df_tracker.history],
            }

        elif condition == "CYBERNETIC":
            # Full control loop with gating
            loop_result = run_control_loop(
                entry, model.generate,
                max_steps=max_steps, log_dir=RESULTS)

            result = {
                "condition": condition,
                **loop_result.to_dict(),
            }
        else:
            raise ValueError(f"Unknown condition: {condition}")

        results.append(result)

        print(f"  [{i+1:2d}/{len(prompt_list)}] {pid} [{cat}] "
              f"verified={result.get('final_verified')}  "
              f"hard={result.get('n_hard_gates', 0)}  "
              f"dt={result.get('elapsed_seconds', 0):.1f}s", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4b: Step-Level Macro-Consensus Deployment")
    parser.add_argument("--condition", choices=["CONTROL", "VERIFY-ONLY", "CYBERNETIC", "ALL"],
                       default="ALL", help="Experiment condition to run")
    parser.add_argument("--mock", action="store_true", default=False,
                       help="Use mock model (no GPU required)")
    parser.add_argument("--failing", action="store_true", default=False,
                       help="Use failing mock model to test hard gate recovery")
    parser.add_argument("--smoke", action="store_true", default=False,
                       help="Run smoke test only")
    parser.add_argument("--max-steps", type=int, default=5,
                       help="Maximum steps per prompt in CYBERNETIC condition")
    parser.add_argument("--prompts", type=str, default="all",
                       choices=["all", "factual", "reasoning", "adversarial", "multi_step"],
                       help="Filter prompts by category")
    args = parser.parse_args()

    if args.smoke:
        sys.exit(run_smoke_test())

    # Filter prompts
    prompt_list = TEST_PROMPTS
    if args.prompts != "all":
        prompt_list = [p for p in TEST_PROMPTS if p.get("category") == args.prompts]

    print(f"Phase 4b: Step-Level Macro-Consensus Deployment")
    print(f"{'=' * 60}")
    print(f"Condition: {args.condition}")
    print(f"Model: {'mock' if args.mock else 'TraDo-4B-Instruct (Q4)'}")
    print(f"Prompts: {len(prompt_list)} ({args.prompts})")
    print(f"Max steps: {args.max_steps}")
    print()

    # Initialize model
    if args.failing:
        model = FailingMockModel(seed=20260516, fail_rate=0.5)
        print("WARNING: Using failing mock model (tests hard gate recovery)")
    elif args.mock:
        model = MockModel(seed=20260516)
        print("Using mock model (no GPU required)")
    else:
        print("Loading TraDo-4B-Instruct...")
        from model import get_model, RealTraDoGenerator, check_model_ready
        status = check_model_ready()
        if not status["ready"]:
            print(f"ERROR: Model not ready: {status}")
            sys.exit(1)
        print(f"  Shards: {status['n_shards']}, Size: {status['total_size_gb']}GB")
        trado = get_model(quantize="q4")
        generator = RealTraDoGenerator(trado)
        model = generator
        print(f"  Hidden dim: {trado.hidden_dim}")
        print("Model loaded and ready for experiment.")

    print()

    # Run conditions
    conditions = ["CONTROL", "VERIFY-ONLY", "CYBERNETIC"] if args.condition == "ALL" else [args.condition]
    all_results = {}

    for cond in conditions:
        print(f"\n{'=' * 60}")
        print(f"CONDITION: {cond}")
        print(f"{'=' * 60}")

        t0 = time.time()
        results = run_model_condition(cond, model, prompt_list, args.max_steps)
        elapsed = time.time() - t0

        all_results[cond] = results

        # Summary
        verified = [r for r in results if r.get("final_verified") is not None]
        correct = sum(1 for r in verified if r.get("final_verified"))
        acc = correct / max(len(verified), 1)
        total_hard = sum(r.get("n_hard_gates", 0) for r in results)
        total_soft = sum(r.get("n_soft_gates", 0) for r in results)

        print(f"\n  {cond} Summary:")
        print(f"    Accuracy: {acc:.3f} ({correct}/{len(verified)})")
        print(f"    Hard gates: {total_hard}  Soft gates: {total_soft}")
        print(f"    Elapsed: {elapsed:.1f}s")

    # Save combined results
    combined = {
        "phase": "4b",
        "date": "2026-05-16",
        "model": "mock" if args.mock else "TraDo-4B-Instruct-Q4",
        "config": {
            "max_steps": args.max_steps,
            "conditions": conditions,
            "n_prompts": len(prompt_list),
            "prompt_filter": args.prompts,
        },
        "results": all_results,
    }

    result_path = RESULTS / "phase4b_all_results.json"
    result_path.write_text(json.dumps(combined, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    print(f"\nResults saved to {result_path}")


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    main()
