"""
Prime+Probe CONFLICT-MISS cross-core .holo carrier - REALISTIC-noise sim.
See header docs inline. ASCII only. All seeds recorded.
"""

import json
import numpy as np

ASSOC = 48
N_SETS_TOTAL = 2048
PARK_WAYS = 12
N_SETS_PER_BANK = 64
MODES = [0, 1, 2, 3]

L3_HIT = 50.0
DRAM_MISS = 240.0
RDTSC_JITTER = 38.0

P_EVICT_SIGNAL = 0.78
P_EVICT_BASE = 0.12


def p_signal_set(a):
    return a * P_EVICT_SIGNAL + (1.0 - a) * P_EVICT_BASE


def p_null_set(spill):
    return min(0.95, P_EVICT_BASE + spill)


def probe_set_latency(p_evict, rng):
    ev = rng.random(PARK_WAYS) < p_evict
    lat = np.where(ev, DRAM_MISS, L3_HIT) + rng.normal(0.0, RDTSC_JITTER, PARK_WAYS)
    return float(lat.mean())


def trial_vector(target_idx, probed_banks, a, spill, rng, writer_active=True):
    """Observer probes its PRIMED banks (probed_banks). The writer floods either
    one of those banks (target_idx in 0..3) or disjoint decoy sets (target_idx=None).
    writer_active=True applies the homogenization spill to every primed bank."""
    vec = {}
    for m, sets in enumerate(probed_banks):
        targeted = (target_idx is not None and m == target_idx)
        for s in sets:
            if targeted:
                p = p_signal_set(a)
            else:
                p = p_null_set(spill if writer_active else 0.0)
            vec[s] = probe_set_latency(p, rng)
    return vec


def bank_contrast(vec, banks):
    allv = np.array(list(vec.values()))
    gmean = allv.mean()
    feats = []
    for sets in banks:
        inside = np.mean([vec[s] for s in sets])
        feats.append(inside - gmean)
    return np.array(feats)


def classify(vec, banks):
    return int(np.argmax(bank_contrast(vec, banks)))


def run_condition(a, spill, seed, n_trials=240):
    rng = np.random.default_rng(seed)
    all_sets = rng.permutation(N_SETS_TOTAL)
    real_banks = [list(all_sets[m * N_SETS_PER_BANK:(m + 1) * N_SETS_PER_BANK]) for m in MODES]
    off = 4 * N_SETS_PER_BANK
    pseudo_banks = [list(all_sets[off + m * N_SETS_PER_BANK: off + (m + 1) * N_SETS_PER_BANK]) for m in MODES]

    real_correct = real_total = 0
    pseudo_decl = pseudo_total = 0
    wrong_actual = wrong_decl = wrong_total = 0
    nullB_correct = nullB_total = 0

    for t in range(n_trials):
        for declared in MODES:
            # REAL: writer floods the declared real bank (observer primes real banks).
            v = trial_vector(declared, real_banks, a, spill, rng)
            if classify(v, real_banks) == declared:
                real_correct += 1
            real_total += 1

            # PSEUDO: writer floods disjoint non-.holo (pseudo) sets the observer did
            # NOT prime -> observer's real-bank readout sees only homogenization spill.
            # Should NOT classify as declared.
            vp = trial_vector(None, real_banks, a, spill, rng, writer_active=True)
            if classify(vp, real_banks) == declared:
                pseudo_decl += 1
            pseudo_total += 1

            # WRONG: declared label false; writer actually floods the next real bank.
            actual = (declared + 1) & 3
            vw = trial_vector(actual, real_banks, a, spill, rng)
            cw = classify(vw, real_banks)
            if cw == actual:
                wrong_actual += 1
            if cw == declared:
                wrong_decl += 1
            wrong_total += 1

            # MATCHED NULL B: writer active at full intensity but flooding fully
            # outside the primed map (pure global pressure). Identical mechanism to
            # pseudo here; kept explicit as the homogenization stress gate.
            vn2 = trial_vector(None, real_banks, a, spill, rng, writer_active=True)
            if classify(vn2, real_banks) == declared:
                nullB_correct += 1
            nullB_total += 1

    return {
        "alignment": a, "spill": spill, "seed": seed,
        "real_accuracy": real_correct / real_total,
        "pseudo_declared_match": pseudo_decl / pseudo_total,
        "wrong_actual_match": wrong_actual / wrong_total,
        "wrong_declared_match": wrong_decl / wrong_total,
        "nullB_spurious_match": nullB_correct / nullB_total,
    }


def gates(r):
    return {
        "real_accuracy>=0.60": r["real_accuracy"] >= 0.60,
        "pseudo_declared<=0.35": r["pseudo_declared_match"] <= 0.35,
        "wrong_actual>=0.60": r["wrong_actual_match"] >= 0.60,
        "wrong_declared<=0.20": r["wrong_declared_match"] <= 0.20,
        "nullB_spurious<=0.35": r["nullB_spurious_match"] <= 0.35,
    }


def scenario(name, a_draw, spill_draw, n_seeds=6):
    print("=" * 78)
    print("SCENARIO:", name)
    rows = []
    rng = np.random.default_rng(abs(hash(name)) & 0x7fffffff)
    for k in range(n_seeds):
        a = float(np.clip(a_draw(rng), 0.0, 1.0))
        spill = float(max(0.0, spill_draw(rng)))
        r = run_condition(a, spill, seed=1000 + k)
        g = gates(r)
        r["all_gates_pass"] = all(g.values())
        rows.append(r)
        print("  seed %d  a=%.2f spill=%.2f | real=%.3f pseudo=%.3f wrongA=%.3f nullB=%.3f | %s"
              % (k, a, spill, r["real_accuracy"], r["pseudo_declared_match"],
                 r["wrong_actual_match"], r["nullB_spurious_match"],
                 "PASS" if r["all_gates_pass"] else "FAIL"))
    accs = [r["real_accuracy"] for r in rows]
    npass = sum(r["all_gates_pass"] for r in rows)
    print("  -> real_acc mean=%.3f  min=%.3f  max=%.3f | seeds passing all gates: %d/%d"
          % (np.mean(accs), np.min(accs), np.max(accs), npass, n_seeds))
    return {"name": name, "rows": rows, "acc_mean": float(np.mean(accs)),
            "acc_min": float(np.min(accs)), "seeds_pass": npass, "n_seeds": n_seeds}


def main():
    out = []
    out.append(scenario("THP_aligned_lowspill",
                         a_draw=lambda r: r.normal(0.92, 0.03),
                         spill_draw=lambda r: r.normal(0.06, 0.02)))
    out.append(scenario("THP_aligned_modspill",
                         a_draw=lambda r: r.normal(0.85, 0.05),
                         spill_draw=lambda r: r.normal(0.18, 0.04)))
    out.append(scenario("homogenized_highspill",
                         a_draw=lambda r: r.normal(0.80, 0.06),
                         spill_draw=lambda r: r.normal(0.40, 0.06)))
    out.append(scenario("misaligned_no_thp",
                         a_draw=lambda r: r.uniform(0.15, 0.55),
                         spill_draw=lambda r: r.normal(0.20, 0.05)))
    out.append(scenario("partial_align_homogenized",
                         a_draw=lambda r: r.normal(0.55, 0.12),
                         spill_draw=lambda r: r.normal(0.35, 0.06)))

    print("=" * 78)
    print("SUMMARY (reproducibility gate = all 6 seeds pass):")
    for s in out:
        print("  %-28s acc_mean=%.3f acc_min=%.3f  seeds_pass=%d/%d"
              % (s["name"], s["acc_mean"], s["acc_min"], s["seeds_pass"], s["n_seeds"]))


if __name__ == "__main__":
    main()
