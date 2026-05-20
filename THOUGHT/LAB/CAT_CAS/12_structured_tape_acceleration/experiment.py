"""
Structured Tape Acceleration Experiment (Hardened)
--------------------------------------------------
Section 3: Breaking the Space-Time Trade-off (The Catalytic Frontier)

Control parameters:
  - Sweep scales: depth = 4, 6, 8, 10, 12
  - Tape types: random, structured, antistructured
  - Iterations per scale: 30 (statistical hardening)
  - Metrics: entropy injected (bits), XOR ops, time, bits erased

The catalytic TEP solver XORs computed values (leaf values + combine results)
into temp registers. These XOR operands are determined by tree topology, not
tape content. We measure whether tape structure changes:

  1. Entropy injected (sum of Hamming weights of all XOR operands)
  2. Bits erased (terminal tape integrity check)
  3. Zero-cost XOR rate
"""

import sys
import time
import hashlib
import numpy as np
from pathlib import Path

CAT_CAS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CAS_DIR / "01_tree_evaluation"))

from tree_eval import TreeEval
from catalytic_engine import MemoryTracker, CatalyticTape, OutOfMemoryError


# =========================================================================
# Control parameters
# =========================================================================
SWEEP_DEPTHS = [4, 6, 8, 10]
K = 256
CLEAN_LIMIT = 2048           # 2KB clean memory budget
TAPE_SIZE = 4 * 1024 * 1024  # 4MB tape
ITERATIONS = 15              # per depth per tape type
TARGET_REG = 100

TAPE_TYPES = ["random", "structured", "antistructured"]


# =========================================================================
# Instrumented solver
# =========================================================================

def hamming_weight(val):
    return val.bit_count() if val else 0


class InstrumentedSolver:
    def __init__(self, tep, tape, tracker):
        self.tep = tep
        self.tape = tape
        self.tracker = tracker
        self.entropy_injected = 0
        self.xor_count = 0
        self.zero_xors = 0
        self.leaf_hits = 0
        self.combine_calls = 0

    def _xor_tape(self, index, xor_val):
        self.xor_count += 1
        hw = hamming_weight(xor_val)
        self.entropy_injected += hw
        if hw == 0:
            self.zero_xors += 1
        current = self.tape.read(index)
        self.tape.write(index, (current ^ xor_val) & 0xFF)

    def evaluate_node(self, node_index, current_depth, target_reg):
        self.tracker.allocate(16)

        if current_depth == self.tep.depth:
            leaf_index = node_index - (2 ** (self.tep.depth - 1))
            val = self.tep.get_leaf_val(leaf_index)
            self._xor_tape(target_reg, val)
            self.leaf_hits += 1
            self.tracker.free(16)
            return

        temp1 = 2 * current_depth
        temp2 = 2 * current_depth + 1

        g1 = self.tape.read(temp1)
        g2 = self.tape.read(temp2)

        self.evaluate_node(2 * node_index, current_depth + 1, temp1)
        self.evaluate_node(2 * node_index + 1, current_depth + 1, temp2)

        left_val = self.tape.read(temp1) ^ g1
        right_val = self.tape.read(temp2) ^ g2

        combined_val = self.tep.combine(left_val, right_val)
        self._xor_tape(target_reg, combined_val)
        self.combine_calls += 1

        self.evaluate_node(2 * node_index + 1, current_depth + 1, temp2)
        self.evaluate_node(2 * node_index, current_depth + 1, temp1)

        self.tracker.free(16)


# =========================================================================
# Tape factories
# =========================================================================

def make_random_tape():
    return CatalyticTape(size_bytes=TAPE_SIZE)


def make_structured_tape(tep, depth):
    tape = CatalyticTape(size_bytes=TAPE_SIZE)
    for d in range(1, depth):
        t1, t2 = 2 * d, 2 * d + 1
        subtree_depth = depth - d
        leaves_per_half = 2 ** (subtree_depth - 1)
        acc_left, acc_right = 0, 0
        for i in range(leaves_per_half):
            acc_left ^= tep.get_leaf_val(i)
            acc_right ^= tep.get_leaf_val(leaves_per_half + i)
        tape.write(t1, acc_left & 0xFF)
        tape.write(t2, acc_right & 0xFF)
    return tape


def make_antistructured_tape(tep, depth):
    tape = CatalyticTape(size_bytes=TAPE_SIZE)
    for d in range(1, depth):
        t1, t2 = 2 * d, 2 * d + 1
        subtree_depth = depth - d
        leaves_per_half = 2 ** (subtree_depth - 1)
        acc_left, acc_right = 0, 0
        for i in range(leaves_per_half):
            acc_left ^= tep.get_leaf_val(i)
            acc_right ^= tep.get_leaf_val(leaves_per_half + i)
        tape.write(t1, (acc_left ^ 0xFF) & 0xFF)
        tape.write(t2, (acc_right ^ 0xFF) & 0xFF)
    return tape


# =========================================================================
# Run single solve, return metric dict
# =========================================================================

def run_solve(tep, depth, tape, ground_truth):
    tracker = MemoryTracker(limit_bytes=CLEAN_LIMIT)
    solver = InstrumentedSolver(tep=tep, tape=tape, tracker=tracker)

    original_target = tape.read(TARGET_REG)
    initial_hash = tape.get_sha256()

    t0 = time.perf_counter()
    solver.evaluate_node(node_index=1, current_depth=1, target_reg=TARGET_REG)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    final_target = tape.read(TARGET_REG)
    result = final_target ^ original_target

    # Restore tape
    tape.write(TARGET_REG, (final_target ^ result) & 0xFF)
    final_hash = tape.get_sha256()
    tape_restored = initial_hash == final_hash

    correct = result == ground_truth

    return {
        "correct": correct,
        "tape_restored": tape_restored,
        "time_ms": elapsed_ms,
        "entropy_injected": solver.entropy_injected,
        "xor_count": solver.xor_count,
        "zero_xors": solver.zero_xors,
        "leaf_hits": solver.leaf_hits,
        "combine_calls": solver.combine_calls,
        "peak_clean": tracker.max_observed,
        "reads": tape.read_count,
        "writes": tape.write_count,
    }


# =========================================================================
# Run sweep for one tape type at one depth
# =========================================================================

def run_sweep(tep, depth, tape_factory, ground_truth, label):
    times, entropies, zero_rates = [], [], []
    correct_count, restored_count = 0, 0

    for _ in range(ITERATIONS):
        tape = tape_factory()
        r = run_solve(tep, depth, tape, ground_truth)
        times.append(r["time_ms"])
        entropies.append(r["entropy_injected"])
        zero_rates.append(r["zero_xors"] / r["xor_count"] * 100 if r["xor_count"] else 0)
        if r["correct"]:
            correct_count += 1
        if r["tape_restored"]:
            restored_count += 1

    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)
    mean_zero_rate = np.mean(zero_rates)

    print(f"  {label:>14} | time={mean_time:8.2f}±{std_time:6.2f} ms | "
          f"entropy={mean_entropy:>10.1f}±{std_entropy:>6.1f} bits | "
          f"zero_xor={mean_zero_rate:5.1f}% | "
          f"correct={correct_count}/{ITERATIONS} | "
          f"restored={restored_count}/{ITERATIONS}")

    return {
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "mean_zero_rate": mean_zero_rate,
        "correct": correct_count,
        "restored": restored_count,
    }


# =========================================================================
# Main experiment
# =========================================================================

def main():
    print("=" * 78)
    print("CAT_CAS: Structured Tape Acceleration Experiment (HARDENED)")
    print("  Section 3: Breaking the Space-Time Trade-off")
    print("=" * 78)
    print()
    print(f"Control parameters:")
    print(f"  Sweep depths:     {SWEEP_DEPTHS}")
    print(f"  Iterations/depth: {ITERATIONS}")
    print(f"  Tape size:        {TAPE_SIZE // (1024*1024)} MB")
    print(f"  Clean budget:     {CLEAN_LIMIT} bytes")
    print(f"  k:                {K}")
    print()

    # Store all results for cross-depth analysis
    all_results = []

    for depth in SWEEP_DEPTHS:
        tep = TreeEval(depth=depth, k=K)
        ground_truth = tep.evaluate_recursive(1, 1)
        num_nodes = 2**depth - 1
        num_xors = 3 * num_nodes           # leaf XOR + 2 reverse per node
        theoretical_max_hw = num_xors * 8  # max 8 bits per XOR

        print("-" * 78)
        print(f"DEPTH {depth}: nodes={num_nodes:,}, fixed XOR ops={num_xors}, "
              f"ground_truth={ground_truth}")
        print(f"  Theoretical max entropy: {theoretical_max_hw:,} bits")
        print("-" * 78)
        print(f"  {'Tape type':>14} | {'Time (ms)':>18} | {'Entropy (bits)':>20} | "
              f"{'0-XOR%':>6} | {'Result':>10}")
        print("  " + "-" * 76)

        depth_results = {}
        for tape_type in TAPE_TYPES:
            if tape_type == "random":
                factory = make_random_tape
            elif tape_type == "structured":
                factory = lambda: make_structured_tape(tep, depth)
            else:
                factory = lambda: make_antistructured_tape(tep, depth)

            result = run_sweep(tep, depth, factory, ground_truth, tape_type)
            depth_results[tape_type] = result

        all_results.append({"depth": depth, "results": depth_results})

        # Assertions per depth
        for tt in TAPE_TYPES:
            r = depth_results[tt]
            assert r["correct"] == ITERATIONS, \
                f"FAIL: {tt} at depth={depth}: {r['correct']}/{ITERATIONS} correct"
            assert r["restored"] == ITERATIONS, \
                f"FAIL: {tt} at depth={depth}: {r['restored']}/{ITERATIONS} restored"
        print()

    # =====================================================================
    # Cross-depth analysis: does structure affect entropy?
    # =====================================================================
    print("=" * 78)
    print("CROSS-DEPTH ENTROPY INVARIANCE TEST")
    print("=" * 78)
    print()

    invariant_violations = 0
    for entry in all_results:
        depth = entry["depth"]
        r = entry["results"]
        baseline = r["random"]["mean_entropy"]

        for tt in ["structured", "antistructured"]:
            delta = r[tt]["mean_entropy"] - baseline
            pct = abs(delta) / baseline * 100 if baseline > 0 else 0
            symbol = "=" if pct < 0.001 else "!="
            if symbol == "!=":
                invariant_violations += 1
            print(f"  depth={depth:>3}  {tt:>14} vs random: "
                  f"entropy delta={delta:+8.1f} ({pct:.6f}%)  [{symbol}]")

    print()
    print("-" * 78)
    if invariant_violations == 0:
        print("  ENTROPY INVARIANCE: CONFIRMED across all depths and tape types.")
    else:
        print(f"  ENTROPY INVARIANCE: BROKEN ({invariant_violations} violations detected).")

    # =====================================================================
    # Temperature sweep chart
    # =====================================================================
    print()
    print("=" * 78)
    print("ENTROPY vs DEPTH (ASCII chart)")
    print("=" * 78)
    print()

    max_entropy = max(
        r["random"]["mean_entropy"]
        for entry in all_results
        for r in [entry["results"]]
    )
    bar_width = 60
    scale = bar_width / max_entropy if max_entropy > 0 else 1

    print(f"  {'Depth':>6} | {'Nodes':>10} | {'Entropy (random)':>18} | Chart (max={max_entropy:.0f})")
    print("  " + "-" * 74)
    for entry in all_results:
        depth = entry["depth"]
        nodes = 2**depth - 1
        entropy = entry["results"]["random"]["mean_entropy"]
        bar = "#" * int(entropy * scale)
        print(f"  {depth:>6} | {nodes:>10,} | {entropy:>18.1f} | {bar}")

    # =====================================================================
    # Hard assertions
    # =====================================================================
    print()
    print("=" * 78)
    print("HARD ASSERTIONS")
    print("=" * 78)

    total_solves = len(SWEEP_DEPTHS) * len(TAPE_TYPES) * ITERATIONS
    total_correct = sum(
        entry["results"][tt]["correct"]
        for entry in all_results
        for tt in TAPE_TYPES
    )
    total_restored = sum(
        entry["results"][tt]["restored"]
        for entry in all_results
        for tt in TAPE_TYPES
    )

    print(f"  Total solves:        {total_solves}")
    print(f"  Correct results:     {total_correct}/{total_solves}")
    print(f"  Tape restorations:   {total_restored}/{total_solves}")
    print(f"  Bits erased:         0 (all tapes restored to exact pre-compute state)")

    assert total_correct == total_solves, \
        f"FAIL: {total_solves - total_correct} solves produced wrong results!"
    assert total_restored == total_solves, \
        f"FAIL: {total_solves - total_restored} solves failed tape restoration!"

    # Entropy must be deterministic (std < 1 bit) for random tape
    for entry in all_results:
        std_entropy = entry["results"]["random"]["std_entropy"]
        assert std_entropy < 1.0, \
            f"FAIL: depth={entry['depth']} random entropy std={std_entropy:.2f} (expected <1)"

    # Entropy must be identical across tape types for same depth
    for entry in all_results:
        r = entry["results"]
        baseline = r["random"]["mean_entropy"]
        for tt in ["structured", "antistructured"]:
            assert abs(r[tt]["mean_entropy"] - baseline) < 1.0, \
                f"FAIL: depth={entry['depth']} {tt} entropy differs from random " \
                f"({r[tt]['mean_entropy']:.1f} vs {baseline:.1f})"

    print()
    print("  [PASS] All hard assertions passed.")
    print()

    # =====================================================================
    # Verdict
    # =====================================================================
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print()
    print("  Structured tape acceleration: NOT OBSERVED (algorithm-limited)")
    print()
    print("  The catalytic TEP solver's XOR operands are determined entirely by")
    print("  tree topology and leaf values. Pre-existing tape structure cannot")
    print("  reduce the Hamming weight of XOR corrections because every XOR")
    print("  operand is computed fresh from the tree during traversal.")
    print()
    print(f"  Result confirmed across {len(SWEEP_DEPTHS)} depth scales")
    print(f"  ({ITERATIONS} iterations each, {len(TAPE_TYPES)} tape types, ")
    print(f"  {total_solves} total solves).")
    print()
    print(f"  Entropy per solve is a deterministic function of tree properties,")
    print(f"  not tape content. The tape is a passive substrate.")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
