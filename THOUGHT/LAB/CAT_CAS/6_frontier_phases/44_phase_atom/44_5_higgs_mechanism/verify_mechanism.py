"""
Independent verification of Exp 47.5 mechanism claim:
'The Higgs Boson is a hardware cache miss when a fragment crosses a memory-page boundary.'
Tests whether the 320-bit latency spike is from Python bigint limb boundary
or actual CPU cache/memory-page effects.
"""
import mpmath, time, numpy as np, random, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from catalytic_tape import BennettHistoryTape

def time_bigint_addition(bits, iterations=10000):
    """Time raw Python bigint addition (no mpmath overhead)."""
    a = random.getrandbits(bits)
    b = random.getrandbits(bits)
    for _ in range(500): _ = a + b  # warmup
    times = np.zeros(iterations)
    for i in range(iterations):
        start = time.perf_counter_ns()
        _ = a + b
        times[i] = time.perf_counter_ns() - start
    p99 = np.percentile(times, 99)
    return np.mean(times[times < p99])

def time_mpf_addition(bits, iterations=10000):
    """Time mpmath mpf addition (includes normalization overhead)."""
    mpmath.mp.dps = 10000
    a = mpmath.mpf(random.getrandbits(bits)) if bits > 0 else mpmath.mpf(0)
    b = mpmath.mpf(1.0)
    for _ in range(500): _ = a + b
    times = np.zeros(iterations)
    for i in range(iterations):
        start = time.perf_counter_ns()
        _ = a + b
        times[i] = time.perf_counter_ns() - start
    p99 = np.percentile(times, 99)
    return np.mean(times[times < p99])

def time_mpf_construction(bits, iterations=5000):
    """Time just constructing mpf from bigint (no addition)."""
    val = random.getrandbits(bits) if bits > 0 else 0
    times = np.zeros(iterations)
    for i in range(iterations):
        start = time.perf_counter_ns()
        _ = mpmath.mpf(val)
        times[i] = time.perf_counter_ns() - start
    p99 = np.percentile(times, 99)
    return np.mean(times[times < p99])

def run():
    lines = []
    def log(msg):
        print(msg); lines.append(msg)

    log("=" * 70)
    log("47.5 MECHANISM VERIFICATION: WHAT CAUSES THE 320-BIT SPIKE?")
    log("=" * 70)
    log("")

    tape = BennettHistoryTape()
    mpmath.mp.dps = 10000
    random.seed(137)

    log("Hypothesis: latency spike is from Python bigint limb boundary,")
    log("NOT from CPU cache lines or memory pages.")
    log("")

    # Test raw bigint addition (pure Python, no mpmath)
    log("--- RAW BIGINT ADDITION (no mpmath overhead) ---")
    log(f"{'Bits':>6} {'Limbs':>6} {'BigInt ns':>12} {'mpf+1.0 ns':>12} {'mpf constr':>12}")
    log("-" * 55)
    tape.record_operation("bigint_baseline")
    for bits in [64, 128, 256, 300, 320, 340, 384, 448, 512, 1024, 2048]:
        limbs = (bits + 29) // 30
        bigint_ns = time_bigint_addition(bits)
        mpf_ns = time_mpf_addition(bits)
        constr_ns = time_mpf_construction(bits)
        log(f"{bits:>6} {limbs:>6} {bigint_ns:>12.0f} {mpf_ns:>12.0f} {constr_ns:>12.0f}")

    log("")
    log("--- ANALYSIS ---")

    # Find the biggest jump in raw bigint addition
    bigint_times = {}
    for bits in [64, 128, 256, 300, 320, 340, 384, 448, 512, 1024]:
        bigint_times[bits] = time_bigint_addition(bits, 2000)

    log("Bigint addition cost per limb:")
    for b1, b2 in [(64, 128), (256, 300), (300, 320), (320, 384), (384, 512), (512, 1024)]:
        t1 = bigint_times[b1]
        t2 = bigint_times[b2]
        l1 = (b1 + 29) // 30
        l2 = (b2 + 29) // 30
        cost_per_limb = (t2 - t1) / (l2 - l1) if l2 > l1 else 0
        log(f"  {b1}->{b2} bits ({l1}->{l2} limbs): {t2-t1:+.0f}ns, {cost_per_limb:.0f}ns/limb")

    log("")
    log("--- VERDICT ---")
    log("If the bigint addition cost jumps at 256->300 (9->10 limbs), the spike is from")
    log("Python's internal bigint digit array crossing an allocator boundary, NOT from")
    log("CPU cache lines (64 bytes = 512 bits) or memory pages (4096 bytes).")
    log("")
    log("Cache line = 64 bytes = 512 bits. If the spike were from cache, it would be at 512.")
    log("Python limb = 30 bits. 256 bits = 9 limbs. 300 bits = 10 limbs.")
    log("If the spike is at 300 bits, the mechanism is Python bigint, not CPU hardware.")

    tape.uncompute()
    tape.verify()
    log("\n[TAPE] Verified. 0 bits erased.")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "TELEMETRY_44_5_MECHANISM.txt")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"\nSaved: {path}")

if __name__ == "__main__":
    run()
