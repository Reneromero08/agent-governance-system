#!/usr/bin/env python3
"""Run MEDIUM level tasks."""

import sys
import os

scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from task_benchmarks import run_level

if __name__ == "__main__":
    results = run_level("medium")

    completed = sum(1 for r in results if r["status"] == "completed")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'='*70}")
    print(f"MEDIUM LEVEL SUMMARY: {completed}/{len(results)} completed, {errors} errors")
    print(f"{'='*70}")
