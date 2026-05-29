#!/usr/bin/env python3
"""
Parallel Cipher Marathon Runner
================================
Runs N independent cipher marathon tests in parallel using multiprocessing.
Takes advantage of LM Studio/Ollama's parallel request handling.
"""

import sys
import argparse
import multiprocessing as mp
from pathlib import Path
import time

CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_cipher_marathon import CipherStressRunner

def run_single_test(args_tuple):
    """Run a single test instance."""
    run_id, args = args_tuple
    print(f"[Run {run_id}] Starting...")

    # Create runner with unique output
    runner = CipherStressRunner(args)
    runner.run()

    print(f"[Run {run_id}] Complete!")
    return run_id

def main():
    parser = argparse.ArgumentParser(description="Run parallel cipher marathon tests")
    parser.add_argument("--num-runs", "-n", type=int, default=8, help="Number of parallel runs")
    parser.add_argument("--url", default="http://10.5.0.2:1234")
    parser.add_argument("--model", default="liquid/lfm2.5-1.2b")
    parser.add_argument("--threshold", type=float, default=1.2)
    parser.add_argument("--output-dir", default="examples/test_chats")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"PARALLEL CIPHER MARATHON - {args.num_runs} runs")
    print(f"=" * 60)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print()

    start = time.time()

    # Create args for each run
    run_args = [(i+1, args) for i in range(args.num_runs)]

    # Run in parallel
    with mp.Pool(processes=args.num_runs) as pool:
        results = pool.map(run_single_test, run_args)

    duration = time.time() - start
    print()
    print(f"=" * 60)
    print(f"ALL {args.num_runs} RUNS COMPLETE")
    print(f"Total time: {duration:.1f}s")
    print(f"Average per run: {duration/args.num_runs:.1f}s")
    print(f"=" * 60)

if __name__ == "__main__":
    main()
