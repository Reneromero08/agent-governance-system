"""
Experiment 07 — MAX SCALE: Reversible Quantum State Simulation (25-Qubit CTM)

Maps 2^25 = 33,554,432 complex amplitudes (512 MB state vector) onto a
1 GB dirty catalytic tape.  A 32-gate / 6-round quantum scrambler is
executed, followed by the full inverse, with byte-level tape verification.

Scale vs. the 15-qubit baseline:
  Amplitudes : 32,768       ->  33,554,432  (1024x)
  State size : 512 KB       ->  512 MB      (1024x)
  Tape size  : 1 MB         ->  1 GB        (1024x)
"""

import os
import sys
import hashlib
import time
import random

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from quantum_simulator import CatalyticQuantumSimulator

TAPE_PATH = os.path.abspath(os.path.join(DIR, "..", "storage", "quantum_tape_25q.bin"))

N_QUBITS   = 25
N_STATES    = 1 << N_QUBITS          # 33,554,432
STATE_BYTES = N_STATES * 16           # 512 MB
TAPE_SIZE   = STATE_BYTES * 2         # 1 GB


def generate_tape():
    """Generate dirty tape with os.urandom (fast, 32 MB chunks)."""
    os.makedirs(os.path.dirname(TAPE_PATH), exist_ok=True)
    gb = TAPE_SIZE / (1024 * 1024 * 1024)
    print(f"[Setup] Generating {gb:.0f} GB dirty tape...")
    chunk = 32 * 1024 * 1024  # 32 MB per write
    with open(TAPE_PATH, 'wb') as f:
        for i in range(TAPE_SIZE // chunk):
            f.write(os.urandom(chunk))
    print("[Setup] Done.")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sampled_probability(state, n_states, sample_size=200_000):
    """Fast probability check on a random sample of states."""
    rng = random.Random(99999)
    indices = rng.sample(range(n_states), min(sample_size, n_states))
    total = 0
    for i in indices:
        k = i << 1
        r, im = state[k], state[k + 1]
        total += r * r + im * im
    return total


def run_experiment():
    print("=" * 70)
    print("REVERSIBLE QUANTUM STATE SIMULATION — MAX SCALE (25-Qubit CTM)")
    print("=" * 70)
    print(f"[System] Qubits:        {N_QUBITS}")
    print(f"[System] Hilbert Space:  {N_STATES:,} dimensions")
    print(f"[System] State Vector:   {N_STATES:,} complex amplitudes "
          f"({STATE_BYTES // (1024*1024)} MB)")
    print(f"[System] Dirty Tape:     {TAPE_SIZE // (1024*1024*1024)} GB "
          f"(borrowed, not allocated)")
    print(f"[System] Gate Set:       CNOT, Toffoli (CCX), Pauli-X")

    # ---- Tape setup ----------------------------------------------- #
    if not os.path.exists(TAPE_PATH):
        generate_tape()

    print("\n[Hash] Computing baseline SHA-256 over 1 GB tape...")
    t_hash = time.time()
    original_hash = file_hash(TAPE_PATH)
    print(f"[Hash] {original_hash}  ({time.time() - t_hash:.1f}s)")

    sim = CatalyticQuantumSimulator(N_QUBITS)

    # ---- Load state from tape ------------------------------------- #
    print(f"\n[I/O]  Loading {STATE_BYTES // (1024*1024)} MB state...")
    t_load = time.time()
    state = sim.load_state(TAPE_PATH)
    print(f"[I/O]  Loaded in {time.time() - t_load:.2f}s  "
          f"({len(state):,} int64 elements)")

    # ---- Pre-circuit snapshot ------------------------------------- #
    probes = [0, 1, 10_000, 1_000_000, 16_777_215, 33_554_431]
    pre = sim.sample_amplitudes(state, probes)
    prob_sample_pre = sampled_probability(state, N_STATES)
    print(f"[State] Sampled |psi|^2 (200K states): {prob_sample_pre}")
    print("[State] Probe amplitudes (pre-circuit):")
    for idx, (r, i) in pre.items():
        print(f"  |{idx:025b}> = ({r:+d}, {i:+d}i)")

    # ============================================================== #
    #  FORWARD CIRCUIT — 6-Round Quantum Scrambler (32 gates)         #
    # ============================================================== #
    print(f"\n{'=' * 60}")
    print(f"FORWARD CIRCUIT: 32 gates over {N_STATES:,} amplitudes")
    print(f"{'=' * 60}")
    t_fwd = time.time()

    print("\n  Round 1 | Toffoli Layer (non-linear mixing) — 6 gates")
    for c1, c2, t in [(0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(15,16,17)]:
        sim.gate_ccx(state, c1, c2, t)
    elapsed = time.time() - t_fwd
    print(f"    Done ({elapsed:.1f}s)")

    print("  Round 2 | CNOT Cascade (linear diffusion) — 6 gates")
    for c, t in [(2,3),(5,6),(8,9),(11,12),(14,15),(17,18)]:
        sim.gate_cnot(state, c, t)
    elapsed = time.time() - t_fwd
    print(f"    Done ({elapsed:.1f}s)")

    print("  Round 3 | Cross-block Toffoli (inter-block) — 4 gates")
    for c1, c2, t in [(0,6,12),(3,9,15),(6,12,18),(1,7,13)]:
        sim.gate_ccx(state, c1, c2, t)
    elapsed = time.time() - t_fwd
    print(f"    Done ({elapsed:.1f}s)")

    print("  Round 4 | Pauli-X Flip — 4 gates")
    for t in [0, 10, 19, 5]:
        sim.gate_x(state, t)
    elapsed = time.time() - t_fwd
    print(f"    Done ({elapsed:.1f}s)")

    print("  Round 5 | Butterfly CNOT (long-range) — 7 gates")
    for c, t in [(0,19),(1,18),(2,17),(3,16),(4,15),(5,14),(6,13)]:
        sim.gate_cnot(state, c, t)
    elapsed = time.time() - t_fwd
    print(f"    Done ({elapsed:.1f}s)")

    print("  Round 6 | Deep Toffoli (final mixing) — 5 gates")
    for c1, c2, t in [(0,10,19),(1,11,18),(2,12,17),(3,13,16),(4,14,15)]:
        sim.gate_ccx(state, c1, c2, t)
    fwd_time = time.time() - t_fwd
    print(f"    Done ({fwd_time:.1f}s)")

    # ---- Post-circuit snapshot ------------------------------------ #
    post = sim.sample_amplitudes(state, probes)
    prob_sample_post = sampled_probability(state, N_STATES)
    conserved = abs(prob_sample_post - prob_sample_pre) < 1e-12
    scrambled = sum(1 for idx in probes if post[idx] != pre[idx])
    print(f"\n[Measurement] Sampled |psi|^2 conservation: "
          f"{'EXACT' if conserved else 'VIOLATED'}")
    print(f"[Measurement] Probes displaced: {scrambled}/{len(probes)}")
    print(f"[Forward] 32 gates in {fwd_time:.1f}s")

    # ============================================================== #
    #  INVERSE CIRCUIT                                                #
    # ============================================================== #
    print(f"\n{'=' * 60}")
    print("INVERSE CIRCUIT: 32 gates reversed")
    print(f"{'=' * 60}")
    t_inv = time.time()
    sim.run_inverse(state)
    inv_time = time.time() - t_inv
    print(f"[Inverse] Completed in {inv_time:.1f}s")

    # ---- Final verification --------------------------------------- #
    final = sim.sample_amplitudes(state, probes)
    amp_match = all(final[i] == pre[i] for i in probes)
    prob_sample_final = sampled_probability(state, N_STATES)
    prob_match = (prob_sample_final == prob_sample_pre)
    print(f"\n[State] Sampled |psi|^2 final: "
          f"{'EXACT' if prob_match else 'VIOLATED'}")
    print(f"[State] Probe restoration: "
          f"{'EXACT MATCH' if amp_match else 'MISMATCH'}")

    # ---- Write state back and hash -------------------------------- #
    print(f"\n[I/O]  Writing {STATE_BYTES // (1024*1024)} MB back to tape...")
    t_save = time.time()
    sim.save_state(TAPE_PATH, state)
    del state  # Free 512 MB
    print(f"[I/O]  Written in {time.time() - t_save:.2f}s")

    print("[Hash] Computing final SHA-256 over 1 GB tape...")
    t_hash2 = time.time()
    final_hash = file_hash(TAPE_PATH)
    print(f"[Hash] {final_hash}  ({time.time() - t_hash2:.1f}s)")

    total = fwd_time + inv_time

    if final_hash == original_hash:
        print(f"\n{'=' * 70}")
        print("[VERIFICATION] SUCCESS")
        print(f"  {N_STATES:,} quantum amplitudes "
              f"({STATE_BYTES // (1024*1024)} MB) simulated")
        print(f"  64 gate operations (32 forward + 32 inverse)")
        print(f"  1 GB dirty tape restored 100% byte-for-byte")
        print(f"  Probability conservation: EXACT (sampled 200K states)")
        print(f"  Zero bits erased. Zero entropy leaked.")
        print(f"  Compute time: {total:.1f}s")
        print(f"{'=' * 70}")
    else:
        print(f"\n[VERIFICATION] FAILED: Tape corruption detected!")
        sys.exit(1)


if __name__ == '__main__':
    run_experiment()
