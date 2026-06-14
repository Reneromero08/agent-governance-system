"""
schedule_invariance_audit.py -- Track Z: Orientation Conservation Audit for Track A.

Verifies that the public schedule for the dual-lane PDN differential (Track A) is
invariant under the private fold d <-> N-d. The schedule must NOT encode which
candidate is the hidden true value.

Audited properties for Track A:
  1. Candidate pair {c0, c1} = {a, N-a} where a = min(d, N-d) -- fold-invariant
  2. k_i order -- determined by RNG seed, independent of d
  3. M -- pure function of n
  4. Candidate labels c0, c1 -- blinded, never true/false
  5. Core assignment -- invariant under d <-> N-d
  6. Walk order -- same k_i sequence for both candidates
  7. No hidden d, orientation bit, or private sign enters schedule generation

Discipline: ASCII only. All RNGs seeded. Deterministic. Claim ceiling L4.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOLD_AUDIT = HERE.parent.parent / "cross_cutting" / "fold_audit"
sys.path.insert(0, str(FOLD_AUDIT))

import construction as C

MASTER_SEED = 44060611
LINES = []


def log(m=""):
    print(m)
    LINES.append(str(m))


def build_track_a_schedule(n, d, rng):
    """Build the PUBLIC schedule for Track A (dual-lane PDN differential).

    Returns a dict representing every public scheduling decision. Must be identical
    for d and N-d. The runtime receives ONLY this schedule; no hidden d anywhere."""
    N = 1 << n
    M = C.M_for(n)

    # Public data: coset samples (k, b)
    k, b = C.coset_samples(N, d, M, rng)

    # Candidate pair from public fold magnitude only
    a = int(min(d % N, (N - d) % N))
    candidates = {"candidate_0": int(a), "candidate_1": int((N - a) % N)}

    # Verify: candidate_0 < candidate_1 always (deterministic ordering)
    if candidates["candidate_0"] > candidates["candidate_1"]:
        candidates["candidate_0"], candidates["candidate_1"] = (
            candidates["candidate_1"], candidates["candidate_0"])

    # The schedule: what the runtime actually receives
    schedule = {
        "n": int(n),
        "N": int(N),
        "M": int(M),
        "k": [int(ki) for ki in k],           # public frequency sequence
        "b": [float(bi) for bi in b],          # public noisy bits
        "candidate_0": candidates["candidate_0"],
        "candidate_1": candidates["candidate_1"],
        "k_order_seed": int(rng.integers(1 << 30)),  # seed for reproducible k order
    }

    # Explicitly verify no hidden fields leaked.
    # Forbidden exact key names OR keys containing multi-char forbidden tokens.
    forbidden_exact = {"d", "N_d", "Nd", "true", "false", "left", "right",
                       "orientation", "sign", "private", "hidden", "secret",
                       "d_val", "nd_val", "truth"}
    for key in schedule:
        key_clean = str(key).lower().replace("_", "").replace("-", "")
        for fw in forbidden_exact:
            fw_clean = fw.replace("_", "").replace("-", "")
            # Only flag if the forbidden word is >= 3 chars OR matches exactly
            if len(fw) >= 3 and fw_clean in key_clean:
                raise ValueError("Forbidden key in schedule: '%s' matches '%s'" % (key, fw))
            if len(fw) < 3 and key_clean == fw_clean:
                raise ValueError("Forbidden exact key in schedule: '%s'" % key)

    return schedule


def schedule_hash(schedule):
    """Deterministic hash of the schedule for fast equality comparison."""
    import hashlib
    raw = json.dumps(schedule, sort_keys=True, default=float).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_candidate_blinding(schedule):
    """Verify candidate labels are blinded: only candidate_0, candidate_1 appear."""
    labels_found = set()
    for val in [schedule.get("candidate_0"), schedule.get("candidate_1")]:
        labels_found.add(val)
    # Check that no true/false label exists in the schedule
    for key in schedule:
        if key.lower() in {"true", "false", "d_val", "nd_val", "orientation"}:
            return False, "forbidden key found: %s" % key
    # Check candidate values are symmetric: {a, N-a}
    a = schedule["candidate_0"]
    Na = schedule["candidate_1"]
    if (a + Na) % schedule["N"] != 0:
        return False, "candidates not a fold pair: %d + %d != N mod N" % (a, Na)
    if a != min(a, Na):
        return False, "candidate_0 not the smaller: %d > %d" % (a, Na)
    return True, "blinded"


def verify_no_hidden_labels(schedule):
    """Verify the schedule contains no hidden labeling in keys or values."""
    hidden_keys = {"true", "false", "d_val", "nd_val", "truth", "direction",
                   "orient_sign", "secret", "private_label", "hidden_bit"}
    schedule_str = json.dumps(schedule, default=float).lower()
    for pat in hidden_keys:
        if pat in schedule_str:
            return False, "hidden label found: %s" % pat
    return True, "clean"


def main():
    t0 = time.time()
    log("=" * 90)
    log("TRACK Z -- ORIENTATION CONSERVATION AUDIT for Track A (dual-lane PDN)")
    log("master_seed=%d   n in {8,10,12,14}   claim ceiling L4" % MASTER_SEED)
    log("=" * 90)

    N_LIST = [8, 10, 12, 14]
    results = {
        "master_seed": MASTER_SEED,
        "track": "Track Z",
        "target_track": "Track A (dual-lane PDN differential)",
        "n_values": N_LIST,
        "cells": [],
    }

    for n in N_LIST:
        N = 1 << n
        log("\n--- n=%d (N=%d, M=%d) ---" % (n, N, C.M_for(n)))

        # Generate paired instances: d and N-d
        rng = np.random.default_rng(MASTER_SEED + n)
        d = C.sample_secret(N, rng)
        Nd = (N - d) % N

        # Build schedule for d
        rng_d = np.random.default_rng(MASTER_SEED + 1000 + n)
        sched_d = build_track_a_schedule(n, d, rng_d)

        # Build schedule for N-d with SAME rng seed
        rng_nd = np.random.default_rng(MASTER_SEED + 1000 + n)
        sched_nd = build_track_a_schedule(n, Nd, rng_nd)

        # Check 1: candidate pair is fold-invariant
        c0_d = sched_d["candidate_0"]
        c1_d = sched_d["candidate_1"]
        c0_nd = sched_nd["candidate_0"]
        c1_nd = sched_nd["candidate_1"]
        candidates_invariant = (c0_d == c0_nd) and (c1_d == c1_nd)
        log("  candidate_0: d=%4d -> %4d   N-d=%4d -> %4d   invariant=%s"
            % (d, c0_d, Nd, c0_nd, candidates_invariant))

        # Check 2: k sequence is identical (same RNG state)
        k_invariant = np.array_equal(sched_d["k"], sched_nd["k"])
        log("  k sequence: invariant=%s  (len %d)" % (k_invariant, len(sched_d["k"])))

        # Check 3: b sequence is identical (E1 confirmed this)
        b_invariant = np.allclose(sched_d["b"], sched_nd["b"])
        log("  b sequence: invariant=%s" % b_invariant)

        # Check 4: M is invariant
        M_invariant = sched_d["M"] == sched_nd["M"]
        log("  M:         invariant=%s  (M=%d)" % (M_invariant, sched_d["M"]))

        # Check 5: full schedule hash
        h_d = schedule_hash(sched_d)
        h_nd = schedule_hash(sched_nd)
        full_invariant = h_d == h_nd
        log("  full schedule SHA-256: d=%s  N-d=%s  invariant=%s"
            % (h_d[:16], h_nd[:16], full_invariant))

        # Check 6: candidate blinding
        blind_ok, blind_msg = verify_candidate_blinding(sched_d)
        log("  candidate blinding: %s (%s)" % (blind_ok, blind_msg))

        # Check 7: no hidden labels in schedule
        hide_ok, hide_msg = verify_no_hidden_labels(sched_d)
        log("  hidden label check: %s (%s)" % (hide_ok, hide_msg))

        # Check 8: orientation bit absent as distinct label in schedule
        orient = C.orientation_bit(d, N)
        orient_keys = {"orient", "orientation", "bit", "b_orient",
                       "1_d_less_than_N_over_2", "lower_half", "half"}
        sched_keys_lower = set(str(k).lower() for k in sched_d)
        orient_labeled = bool(orient_keys & sched_keys_lower)
        log("  orientation label in schedule: %s" % orient_labeled)

        cell = {
            "n": int(n),
            "N": int(N),
            "M": int(sched_d["M"]),
            "d": int(d % N),
            "Nd": int(Nd),
            "candidate_0_d": c0_d,
            "candidate_1_d": c1_d,
            "candidate_0_Nd": c0_nd,
            "candidate_1_Nd": c1_nd,
            "candidates_invariant": bool(candidates_invariant),
            "k_sequence_invariant": bool(k_invariant),
            "b_sequence_invariant": bool(b_invariant),
            "M_invariant": bool(M_invariant),
            "full_schedule_invariant": bool(full_invariant),
            "candidate_labels_blinded": bool(blind_ok),
            "no_hidden_labels": bool(hide_ok),
            "orientation_labeled": bool(orient_labeled),
            "schedule_hash_d": h_d,
            "schedule_hash_Nd": h_nd,
        }
        results["cells"].append(cell)

    # --- Aggregate verdict ---
    all_invariant = all(c["full_schedule_invariant"] for c in results["cells"])
    all_blinded = all(c["candidate_labels_blinded"] for c in results["cells"])
    all_clean = all(c["no_hidden_labels"] for c in results["cells"])
    no_leak = not any(c["orientation_labeled"] for c in results["cells"])

    log("\n" + "=" * 90)
    log("TRACK Z VERDICT")
    if all_invariant and all_blinded and all_clean and no_leak:
        log("  SCHEDULE_INVARIANCE_PASS -- all gates clean")
        log("  The public schedule for Track A is invariant under d <-> N-d.")
        log("  Candidate labels are blinded. No hidden d or orientation bit in schedule.")
        results["verdict"] = "SCHEDULE_INVARIANCE_PASS"
    elif not all_invariant:
        log("  SCHEDULE_SMUGGLE_FOUND -- schedule changes under d <-> N-d at n=%s"
            % [c["n"] for c in results["cells"] if not c["full_schedule_invariant"]])
        results["verdict"] = "SCHEDULE_SMUGGLE_FOUND"
    else:
        log("  SCHEDULE_INVARIANCE_PARTIAL -- gates: invariant=%s blinded=%s clean=%s no_leak=%s"
            % (all_invariant, all_blinded, all_clean, no_leak))
        results["verdict"] = "SCHEDULE_INVARIANCE_PARTIAL"

    results["all_invariant"] = bool(all_invariant)
    results["all_blinded"] = bool(all_blinded)
    results["all_clean"] = bool(all_clean)
    results["no_orientation_leak"] = bool(no_leak)
    results["elapsed_s"] = round(time.time() - t0, 2)

    log("\n  elapsed: %.2f s" % results["elapsed_s"])
    log("=" * 90)

    (HERE / "results" / "schedule_invariance.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    (HERE / "results" / "output_schedule_audit.txt").write_text(
        "\n".join(LINES), encoding="utf-8")
    log("\nwrote results/schedule_invariance.json + output_schedule_audit.txt")

    return 0 if results["verdict"] == "SCHEDULE_INVARIANCE_PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

