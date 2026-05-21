"""
Hypothesis Test: Catalytic Period-Finding for Integer Factorization
===================================================================
Hypothesis: Catalytic borrowed space can compute the period-finding step
of Shor's algorithm (f(x) = a^x mod N, find period r) classically,
using dirty tape as workspace, without qubits.

Method:
  1. Build a dirty catalytic tape (random data) using CatalyticTape engine
  2. Reversibly compute a^x mod N values onto the tape (XOR encoding)
  3. Detect period r where a^r ≡ 1 (mod N)
  4. Uncompute all tape writes to restore original dirty data
  5. Use period r to factor N via Shor's classical post-processing:
     gcd(a^(r/2) ± 1, N) yields factors
  6. Verify tape is byte-for-byte restored (catalytic property)

Measurements:
  - Wall-clock time per factorization
  - Number of tape operations (reads + writes)
  - Clean RAM usage (tracked via MemoryTracker)
  - Tape integrity (SHA-256 pre/post match)
  - Scaling behavior as N grows

Uses the same CatalyticTape and MemoryTracker from 01_tree_evaluation.
"""

import sys
import time
import math
from pathlib import Path

CAT_CAS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CAS_DIR / "01_tree_evaluation"))

from catalytic_engine import MemoryTracker, CatalyticTape


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def next_prime(n):
    if n < 2:
        return 2
    c = n if is_prime(n) else n + 1
    while not is_prime(c):
        c += 1
    return c


def catalytic_period_find(a, N, tape, tracker):
    """
    Find the period r of f(x) = a^x mod N using catalytic space.

    The dirty tape stores XOR-encoded values of a^x mod N.
    After period detection, all writes are reversed to restore the tape.

    Clean RAM usage:
      - 8 bytes: loop counter x (int64)
      - 8 bytes: current value a^x mod N (int64)
      - 8 bytes: XOR temp for encode/decode (int64)
      - 8 bytes: period result (int64)
      Total: 32 bytes clean workspace

    Returns (period_or_None, tape_ops).
    """
    tracker.allocate(32)  # 4 x int64 clean registers

    ops_before = tape.read_count + tape.write_count

    # Forward pass: reversibly encode a^x mod N onto dirty tape
    # Each value is stored as tape[x] ^= (a^x mod N) — self-inverse
    period = None
    steps = 0
    current = 1  # a^0 mod N

    # Max search depth: period r divides phi(N) < N
    max_x = min(N, tape.size_bytes)

    for x in range(1, max_x):
        current = (current * a) % N

        # Reversible XOR-encode onto tape
        old_val = tape.read(x)
        tape.write(x, old_val ^ (current & 0xFF))
        steps += 1

        if current == 1:
            period = x
            break

    # Reverse pass: undo all XOR writes to restore tape
    # Replay the same sequence and XOR again (self-inverse)
    rev_current = 1
    for x in range(1, steps + 1):
        rev_current = (rev_current * a) % N
        old_val = tape.read(x)
        tape.write(x, old_val ^ (rev_current & 0xFF))

    ops_after = tape.read_count + tape.write_count
    total_ops = ops_after - ops_before

    tracker.free(32)
    return period, total_ops


def factor_catalytic(N, tape, tracker):
    """
    Factor semiprime N using catalytic period-finding + Shor's post-processing.

    Returns (p, q, period, base_a, total_ops, time_s).
    """
    if N % 2 == 0:
        return 2, N // 2, 1, 2, 0, 0.0

    t0 = time.perf_counter()
    total_ops = 0

    for a in range(2, N):
        g = math.gcd(a, N)
        if g > 1:
            elapsed = time.perf_counter() - t0
            return g, N // g, 0, a, total_ops, elapsed

        period, ops = catalytic_period_find(a, N, tape, tracker)
        total_ops += ops

        if period is None:
            continue
        if period % 2 != 0:
            continue

        half_pow = pow(a, period // 2, N)
        if half_pow == N - 1:
            continue

        p = math.gcd(half_pow + 1, N)
        q = math.gcd(half_pow - 1, N)

        if 1 < p < N:
            elapsed = time.perf_counter() - t0
            return p, N // p, period, a, total_ops, elapsed
        if 1 < q < N:
            elapsed = time.perf_counter() - t0
            return q, N // q, period, a, total_ops, elapsed

    elapsed = time.perf_counter() - t0
    return None, None, None, None, total_ops, elapsed


def make_semiprimes():
    """Build semiprimes at target bit sizes."""
    targets = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]
    results = []
    for bits in targets:
        half = bits // 2
        p = next_prime(1 << (half - 1))
        q = next_prime(p + 2)
        if q == p:
            q = next_prime(q + 1)
        N = p * q
        actual_bits = N.bit_length()
        results.append((N, actual_bits, p, q))
    return results


def main():
    print("=" * 78)
    print("HYPOTHESIS TEST: Catalytic Period-Finding for Integer Factorization")
    print("  Using CatalyticTape engine from CAT_CAS/01_tree_evaluation")
    print("=" * 78)
    print()
    print("METHOD")
    print("-" * 40)
    print("  1. XOR-encode a^x mod N onto dirty catalytic tape (forward pass)")
    print("  2. Detect period r where a^r = 1 mod N")
    print("  3. XOR-decode to restore tape (reverse pass)")
    print("  4. Factor N via gcd(a^(r/2) +/- 1, N)")
    print("  5. Verify tape SHA-256 integrity (catalytic property)")
    print()
    print("QUESTION: Does factoring time scale polynomially with bit size?")
    print("  Shor's (quantum) = O(n^3)  |  Classical NFS = O(exp(n^(1/3)))")
    print()

    # Initialize catalytic infrastructure
    TAPE_SIZE = 8 * 1024 * 1024  # 8 MB tape for larger periods
    tape = CatalyticTape(size_bytes=TAPE_SIZE)
    tracker = MemoryTracker(limit_bytes=256)

    initial_hash = tape.get_sha256()
    print(f"  Tape Size:        {TAPE_SIZE // 1024} KB")
    print(f"  Tape SHA-256:     {initial_hash[:32]}...")
    print(f"  Clean RAM Limit:  {tracker.limit_bytes} bytes")
    print()

    semiprimes = make_semiprimes()

    print(f"{'Bits':>5} | {'N':>16} | {'Factors':>20} | {'a':>4} | "
          f"{'Period':>10} | {'Tape Ops':>12} | {'Time (s)':>10} | "
          f"{'Tape OK':>7} | {'Status':>6}")
    print("-" * 110)

    results = []
    timeout_hit = False

    for N, bits, known_p, known_q in semiprimes:
        if timeout_hit:
            print(f"{bits:>5} | {N:>16} | {'SKIPPED':>20} | {'--':>4} | "
                  f"{'--':>10} | {'--':>12} | {'--':>10} | "
                  f"{'--':>7} | {'SKIP':>6}")
            results.append((bits, N, 0, 0, 0, 0.0, False))
            continue

        # Record tape state before this factorization
        pre_hash = tape.get_sha256()

        p, q, period, base_a, ops, elapsed = factor_catalytic(N, tape, tracker)

        # Verify tape restoration
        post_hash = tape.get_sha256()
        tape_ok = (post_hash == pre_hash)

        if p is not None and q is not None:
            factors_str = f"{min(p,q)} x {max(p,q)}"
            correct = (min(p, q) == min(known_p, known_q) and
                       max(p, q) == max(known_p, known_q))
            status = "OK" if correct else "WRONG"
            period_str = str(period) if period else "gcd"
            a_str = str(base_a) if base_a else "--"
        else:
            factors_str = "FAILED"
            status = "FAIL"
            period_str = "--"
            a_str = "--"
            correct = False

        tape_str = "YES" if tape_ok else "NO"

        print(f"{bits:>5} | {N:>16} | {factors_str:>20} | {a_str:>4} | "
              f"{period_str:>10} | {ops:>12,} | {elapsed:>10.4f} | "
              f"{tape_str:>7} | {status:>6}")

        results.append((bits, N, period if period else 0, ops, base_a, elapsed, correct))

        sys.stdout.flush()

        if elapsed > 60.0:
            timeout_hit = True

    # ---- Final tape integrity check ----
    final_hash = tape.get_sha256()
    tape_fully_restored = (final_hash == initial_hash)

    print()
    print("=" * 78)
    print("SCALING ANALYSIS")
    print("=" * 78)

    completed = [(b, N, per, ops, a, t, ok)
                 for b, N, per, ops, a, t, ok in results if t > 0.000001]

    if len(completed) >= 2:
        print(f"\n{'Bits':>5} | {'Time (s)':>12} | {'Ops':>12} | "
              f"{'Time Ratio':>12} | {'Ops Ratio':>12}")
        print("-" * 60)

        for i, (bits, N, per, ops, a, t, ok) in enumerate(completed):
            if i == 0:
                print(f"{bits:>5} | {t:>12.6f} | {ops:>12,} | "
                      f"{'(base)':>12} | {'(base)':>12}")
            else:
                prev_t = completed[i - 1][5]
                prev_ops = completed[i - 1][3]
                t_ratio = t / prev_t if prev_t > 0 else float('inf')
                ops_ratio = ops / prev_ops if prev_ops > 0 else float('inf')
                print(f"{bits:>5} | {t:>12.6f} | {ops:>12,} | "
                      f"{t_ratio:>11.2f}x | {ops_ratio:>11.2f}x")

    print()
    print("=" * 78)
    print("INTEGRITY VERIFICATION")
    print("=" * 78)
    print(f"  Tape SHA-256 (initial):  {initial_hash[:32]}...")
    print(f"  Tape SHA-256 (final):    {final_hash[:32]}...")
    print(f"  Full tape restored:      {tape_fully_restored}")
    print(f"  Clean RAM peak:          {tracker.max_observed} bytes / {tracker.limit_bytes} byte limit")
    print(f"  Total tape reads:        {tape.read_count:,}")
    print(f"  Total tape writes:       {tape.write_count:,}")
    print(f"  Total semiprimes tested: {len(results)}")
    print(f"  Successfully factored:   {sum(1 for r in results if r[6])}")
    print()

    if tape_fully_restored:
        print("  [PASS] Catalytic tape integrity verified (zero bits erased)")
    else:
        print("  [FAIL] Catalytic tape corruption detected!")

    print()
    print("=" * 78)
    print("RAW DATA -- INTERPRET AS YOU SEE FIT")
    print("=" * 78)


if __name__ == "__main__":
    main()

