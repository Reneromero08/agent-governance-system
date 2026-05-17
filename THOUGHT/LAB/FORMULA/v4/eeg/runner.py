"""
EEG Phase Coherence Validation Runner.

Orchestrates all three tests in sequence:
  1. Eureka Synchronization (PLV spike at insight)
  2. Symbolic Resonance (High-sigma > low-sigma PLV)
  3. Flow State Phase Transition (Theta-gamma PAC in flow)

Usage:
  python runner.py                 # All tests, synthetic mode
  python runner.py --task 1        # Only Task 1
  python runner.py --mode real --data-root <path>  # Real data mode (NYI)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure runner can find task modules
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from task1_insight.run import run_eureka_test
from task2_symbols.run import run_symbols_test
from task3_flow.run import run_flow_test
from utils import write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="EEG Phase Coherence Validation Runner for v4 Wave Mechanics"
    )
    p.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                   help="Data mode (default: synthetic)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Root directory for OpenNeuro datasets (real mode)")
    p.add_argument("--output-root", type=str, default=None,
                   help="Root output directory (default: ./results)")
    p.add_argument("--task", type=int, choices=[1, 2, 3], default=0,
                   help="Run single task (1/2/3) or all (0)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = args.output_root or str(_here / "results")
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    print("=" * 60)
    print("EEG Phase Coherence Validation -- v4 Wave Mechanics")
    print(f"Mode: {args.mode}, Seed: {args.seed}")
    print(f"Output root: {output_root}")
    print("=" * 60)

    all_results = {}
    all_pass = True

    tasks_to_run = [args.task] if args.task else [1, 2, 3]

    for task_id in tasks_to_run:
        print(f"\n{'#' * 60}")
        print(f"# RUNNING TASK {task_id}")
        print(f"{'#' * 60}")

        task_output = os.path.join(output_root, f"task{task_id}_{ts}")

        try:
            if task_id == 1:
                results = run_eureka_test(
                    mode=args.mode,
                    data_dir=args.data_root,
                    output_dir=task_output,
                    seed=args.seed,
                )
            elif task_id == 2:
                results = run_symbols_test(
                    mode=args.mode,
                    data_dir=args.data_root,
                    output_dir=task_output,
                    seed=args.seed,
                )
            elif task_id == 3:
                results = run_flow_test(
                    mode=args.mode,
                    data_dir=args.data_root,
                    output_dir=task_output,
                    seed=args.seed,
                )
            else:
                raise ValueError(f"Unknown task: {task_id}")

            all_results[f"task{task_id}"] = results
            task_pass = results.get("overall_pass", False)
            all_pass = all_pass and task_pass

            print(f"\nTASK {task_id}: {'PASS' if task_pass else 'FAIL'}")

        except Exception as exc:
            print(f"\nTASK {task_id}: ERROR -- {exc}")
            import traceback
            traceback.print_exc()
            all_results[f"task{task_id}"] = {"error": str(exc), "overall_pass": False}
            all_pass = False

    # -------------------------------------------------------------------
    # Cross-test summary: does R scale with sigma and D_f?
    # -------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"\nCROSS-TEST SUMMARY")
    for t in [1, 2, 3]:
        key = f"task{t}"
        if key in all_results:
            status = "PASS" if all_results[key].get("overall_pass", False) else "FAIL"
        else:
            status = "skipped"
        name = {1: "Eureka/PLV", 2: "Symbols/PLV", 3: "Flow/PAC"}[t]
        print(f"  Task {t} ({name}):    {status}")
    print(f"  All run tests:         {'PASS' if all_pass else 'FAIL'}")

    # Write summary receipt
    summary = {
        "timestamp_utc": ts,
        "mode": args.mode,
        "seed": args.seed,
        "tasks_run": tasks_to_run,
        "all_pass": all_pass,
        "task_results": {
            k: {"overall_pass": v.get("overall_pass", False) if isinstance(v, dict) else False}
            for k, v in all_results.items()
        },
    }
    summary_path = os.path.join(output_root, f"summary_{ts}.json")
    os.makedirs(output_root, exist_ok=True)
    write_json(summary_path, summary)
    print(f"\nSummary: {summary_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
