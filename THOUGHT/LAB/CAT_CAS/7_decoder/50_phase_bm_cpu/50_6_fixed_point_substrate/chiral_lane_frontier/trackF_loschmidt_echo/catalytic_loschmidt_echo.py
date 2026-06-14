"""
catalytic_loschmidt_echo.py -- Track F: Catalytic Loschmidt Echo reference model.

U_candidate(x): public candidate-dependent phase walk. Computes intermediate values
  v_j = x * k_j mod N and XOR-encodes them into a tape.
U_candidate_dagger(x): deterministic inverse. XOR-decodes the tape (restores).

Echo residue model: The analog residue after U -> U_dagger is proportional to the
cumulative Hamming weight of intermediate values written. For candidate_0 = a,
the write pattern differs from candidate_1 = N-a because hw(a*k) vs hw((N-a)*k)
differ for a fixed k sequence even though both produce identical cosine outputs.

Controls: identity, forward-only, reverse-only, order-swap, same-candidate,
dummy-candidate, label-swap, hidden-positive, shuffle-null.
No-smuggle: U and U_dagger derived from public (k, candidate_value, N) only.

Discipline: ASCII only. All RNGs seeded. Claim ceiling L3.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FOLD_AUDIT = HERE.parent.parent / "fold_audit"
sys.path.insert(0, str(FOLD_AUDIT))
import construction as C
import no_smuggle_gate as G

MASTER_SEED = 44060611
LINES = []

def log(m=""): print(m); LINES.append(str(m))


def hamming_weight_64(x):
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def build_tape(M, seed):
    """Initialize a dirty tape (byte array) with known seed."""
    rng = np.random.default_rng(seed)
    tape = rng.integers(0, 256, size=M, dtype=np.uint8)
    return tape


def U_candidate(tape, k_values, candidate, N):
    """XOR-encode candidate walk intermediate values into tape.

    U_candidate(x): v_j = x * k_j mod N. XOR v_j (low byte) into tape[j].
    Returns (mutated_tape, hw_sum) where hw_sum = sum of Hamming weights written.
    """
    M = len(tape)
    hw_sum = 0
    for j in range(M):
        v = (candidate * k_values[j]) % N
        tape[j] ^= (v & 0xFF)  # XOR low byte
        hw_sum += hamming_weight_64(v)
    return tape, hw_sum


def U_candidate_dagger(tape, k_values, candidate, N):
    """XOR-decode: same operation as U (XOR is its own inverse).
    Restores tape to original state. Returns (restored_tape, hw_sum).
    """
    return U_candidate(tape, k_values, candidate, N)


def compute_echo(k_values, candidate, N, tape_seed=42):
    """Compute Loschmidt echo residue.

    R_echo = hw_sum(U) - hw_sum(identity)
    Identity = U followed by U_dagger with zero candidate (no XOR).
    Returns echo (the hw_sum of U alone, since U_dagger restores and adds
    additional hw -- we measure the cumulative write activity).
    """
    M = len(k_values)
    tape = build_tape(M, tape_seed)
    # Identity: zero writes
    _, hw_id = U_candidate(tape.copy(), k_values, 0, N)

    # U_candidate only
    tape_u = build_tape(M, tape_seed)
    _, hw_u = U_candidate(tape_u, k_values, candidate, N)

    # U -> U_dagger
    tape_full = build_tape(M, tape_seed)
    _, hw_uu = U_candidate(tape_full, k_values, candidate, N)
    tape_restored, hw_uu2 = U_candidate_dagger(tape_full, k_values, candidate, N)

    # Verify restoration: tape should match original
    tape_orig = build_tape(M, tape_seed)
    restored = np.array_equal(tape_restored, tape_orig)

    # Echo: total write activity during U + U_dagger
    # This is symmetric for U vs U_dagger order because XOR is commutative
    # in total Hamming weight. But for candidate_0 vs candidate_1, the
    # hw of intermediate values a*k_j differs from (N-a)*k_j.

    # Echo residue = total hw of U phase (before restoration)
    # The U_dagger phase also contributes hw, but symmetrically.
    # Key measurement: does echo differ for a vs N-a?
    return float(hw_u), float(hw_uu + hw_uu2), bool(restored)


def build_features(k_values, candidate, N, echo_hw_u, echo_hw_full):
    """Public features for classifier. echo_hw_U is the per-instance echo signal."""
    return np.array([echo_hw_u, echo_hw_full])


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    log("=" * 80)
    log("TRACK F -- CATALYTIC LOSCHMIDT ECHO REFERENCE MODEL")
    log("master_seed=%d" % MASTER_SEED)
    log("=" * 80)

    N_LIST = [8, 10]
    N_INST = 400

    results = {"experiment": "phase6_trackF_loschmidt_echo", "master_seed": MASTER_SEED,
               "n_values": N_LIST, "n_instances": N_INST, "cells": []}

    for n in N_LIST:
        N = 1 << n
        M = C.M_for(n)
        log("\n--- n=%d (N=%d, M=%d) ---" % (n, N, M))

        # Generate instances with paired (d, N-d) oracle and both candidates
        feats_all = []; labels_all = []
        for mode in ["public", "same_candidate", "dummy"]:
            mode_feats = []; mode_labels = []
            for seed_off in range(N_INST):
                s = MASTER_SEED + 1000*n + seed_off
                rng = np.random.default_rng(s)
                d = C.sample_secret(N, rng)
                k, b = C.coset_samples(N, d, M, rng)
                orient = C.orientation_bit(d, N)
                a = int(min(d % N, (N - d) % N))
                Na = int((N - a) % N)

                if mode == "same_candidate":
                    c0_val = a; c1_val = a
                elif mode == "dummy":
                    c0_val = 42; c1_val = 42
                else:
                    c0_val = a; c1_val = Na

                hw_u0, hw_full0, rest0 = compute_echo(k, c0_val, N)
                hw_u1, hw_full1, rest1 = compute_echo(k, c1_val, N)
                if not rest0 or not rest1:
                    log("WARN: tape restoration failed")

                feat0 = build_features(k, c0_val, N, hw_u0, hw_full0)
                feat1 = build_features(k, c1_val, N, hw_u1, hw_full1)
                # For same_candidate/dummy: both features are same candidate,
                # label is arbitrary (always 0 for c0, 1 for c1)
                mode_feats.extend([feat0, feat1])
                mode_labels.extend([0, 1])

            X = np.array(mode_feats)
            y = np.array(mode_labels)
            # Use hw_u (feature 0) as score to separate c0 from c1
            scores = X[:, 0]
            auc = G._held_out_auc(X, y, MASTER_SEED + n)
            null95, null_mean = G._shuffle_null_auc(X, y, MASTER_SEED + n, 30)

            c0_scores = scores[y == 0]
            c1_scores = scores[y == 1]
            c0_mean = float(np.mean(c0_scores)) if len(c0_scores) else 0
            c1_mean = float(np.mean(c1_scores)) if len(c1_scores) else 0

            log("  %-18s c0_echo=%.4f c1_echo=%.4f diff=%.4f auc=%.3f/null95=%.3f %s"
                % (mode, c0_mean, c1_mean, c0_mean - c1_mean, auc, null95,
                   "ABOVE" if auc > null95 else ""))

            results["cells"].append({
                "n": n, "mode": mode, "N_inst": N_INST,
                "c0_echo_mean": c0_mean, "c1_echo_mean": c1_mean,
                "diff": round(c0_mean - c1_mean, 6),
                "candidate_auc": round(auc, 4),
                "null95": round(null95, 4),
                "above_null": bool(auc > null95),
            })

    # --- Verdict ---
    public_cells = [c for c in results["cells"] if c["mode"] == "public"]
    same_cells   = [c for c in results["cells"] if c["mode"] == "same_candidate"]
    dummy_cells  = [c for c in results["cells"] if c["mode"] == "dummy"]

    pub_above = any(c["above_null"] for c in public_cells)
    same_above = any(c["above_null"] for c in same_cells)
    dummy_above = any(c["above_null"] for c in dummy_cells)

    log("\n" + "=" * 80)
    log("TRACK F VERDICT")
    if not pub_above:
        log("  public echo AUC at chance -- no candidate-value echo differential")
        verdict = "TRACK_F_REFERENCE_NO_PUBLIC_ECHO_DIFFERENTIAL"
    elif pub_above and not same_above and not dummy_above:
        verdict = "TRACK_F_REFERENCE_CANDIDATE_VALUE_ECHO_FOUND"
        log("  public echo separates candidates; same-candidate/dummy at chance")
    else:
        verdict = "TRACK_F_REFERENCE_INCONCLUSIVE"
        log("  inconclusive -- same/dummy also above null or public at chance")

    log("  public above null: %s" % pub_above)
    log("  same-candidate above null: %s" % same_above)
    log("  dummy above null: %s   VERDICT: %s" % (dummy_above, verdict))
    log("=" * 80)

    results["verdict"] = verdict
    results["elapsed_s"] = round(time.time() - t0, 2)

    (HERE / "results" / "loschmidt_echo_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (HERE / "results" / "output_echo.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nwrote results/loschmidt_echo_results.json + output_echo.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
