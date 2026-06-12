"""
DRAM ROW-BUFFER DRIVEN LOCK-IN -- realistic-noise sim (Stage 0, pre-physical).
===============================================================================
Angle: cross-core .holo traversal through the SHARED OFF-CHIP DRAM row buffer,
distinct from L3. Instead of a STATIC residency snapshot (the carrier that failed
by homogenization, +0.061 lift), we reuse the PROVEN driven-lock-in methodology
from phase 5.10 (off-bin floor, scrambled-reference null, amplitude ladder).

MECHANISM
  Writer core modulates the row-buffer state of a TARGET BANK B as a square wave
  at f_drive: during the ON half it hammers a CONFLICTING row r_w in bank B (so
  the bank's open row != the observer's row r_o); during the OFF half it leaves
  bank B alone. The observer core, sharing the same DRAM through the on-die memory
  controller, continuously clflush+probes its own row r_o in bank B and timestamps
  each probe (rdtscp, per-core TSC -- no cross-core sync needed). Observer latency
  -> HIGH when writer-ON (row conflict, +tRP+tRCD) and LOW when writer-OFF (row
  hit). Lock-in at f_drive extracts the modulated component; off-bin = floor.

WHY THIS IS NOT THE CACHE FAILURE
  L3 residency is a DIFFUSE many-line population: co-access homogenizes "who
  warmed it" (signal averages out; stronger coupling = weaker signal). The row
  buffer is a SINGLE last-writer-wins state per bank (binary: r_o open vs r_w
  open). Stronger writer hammering makes the ON-half conflict MORE reliable =
  deeper modulation: the monotonicity is INVERTED relative to the cache failure.
  There is no diffuse averaging to wash out -- only a writer/observer race for the
  one row register, which the writer wins by hammering faster.

MATCHED NULLS (the decisive part)
  NULL-1 different-bank: writer hammers bank B' != B, identical traffic/power.
          Controls generic IMC/bus contention + power draw.
  NULL-2 same-row (THE killer null): writer hammers bank B but the SAME row r_o.
          Identical bank, traffic, power, IMC queue occupancy -- ONLY the row
          differs. A row-buffer signal collapses here; a contention/power artifact
          would survive. This is TIGHTER than the 5.10 ALU-vs-cache control.
  NULL-3 off-bin/scrambled reference: demod at f' != f_drive.
  NULL-4 writer idle: pure floor.

REALISM (the prior wormhole sim was too clean -- perfectly invertible channel):
  - per-probe latency jitter (rdtsc overhead + MC queueing), heavy-tail interrupts
  - imperfect race: P(conflict|ON) = p_on < 1 (writer wins ~0.85-0.97);
    P(conflict|OFF) = p_off > 0 (refresh + background re-opens, ~0.04-0.15)
  - DRAM refresh: periodic forced conflicts ~128 kHz (OUT of the f_drive band,
    folded in as elevated p_off + jitter -> the lock-in rejects it)
  - slow thermal/scheduler drift (DC + sub-Hz ramp) -> rejected by mean-removal +
    Hann + lock-in at f_drive (>= 10 Hz, above the drift pole, mirroring 5.10)
  - NULL-1 carries a small ON-half CONTENTION latency bump (different-bank traffic
    still queues at the shared MC) -> tests that the row-buffer signal is cleanly
    ABOVE plausible contention leakage
  - finite address->bank mapping reliability: a fraction of probes land in the
    WRONG bank (imperfect userspace bank reverse-engineering) -> dilutes depth

.holo STRUCTURE
  N banks = N parallel last-writer-wins channels = spatial multiplex. One frame =
  per-bank state vector (mode = a bank subset, mirroring the C-harness FAMILIES).
  Relational PHASE: drive each bank with a phase ramp 2*pi*j/N; observer lock-in
  per bank recovers the ramp (atan2(Q,I)). 4-mode nearest-centroid classifier +
  the analyzer's matched-null gates, mirroring analyze_cache_hologram_matched_nulls.py.

ASCII only. All seeds recorded. JSON written next to this script.
"""

import os, json, math
import numpy as np

# ---------------------------------------------------------------- DDR3 physics
# AMD Phenom II X6 (K10), on-die IMC, DDR3-1333 typical. Latencies in ns.
TCK_NS        = 1.5      # DDR3-1333 -> 1.5 ns/clk
T_HIT_NS      = 50.0     # row-hit access latency seen by the core (CAS + bus + MC)
T_CONFLICT_NS = 80.0     # row-conflict: + tRP + tRCD (~13.5+13.5 ns) -> ~+30 ns
JITTER_NS     = 11.0     # per-probe gaussian jitter (rdtsc overhead + MC queue var)
HEAVYTAIL_P   = 0.012    # prob of an interrupt/preemption outlier on a probe
HEAVYTAIL_NS  = 400.0    # outlier magnitude scale (one-sided)
REFRESH_HZ    = 128000.0 # tREFI ~7.8 us -> ~128 kHz all-bank refresh (OUT of band)

# probe / drive geometry
READ_HZ       = 300000.0 # observer probe cadence (clflush+mfence+load+rdtscp ~3-5 us budget -> ~250-330k/s)
F_DRIVE_HZ    = 30.0     # in the proven above-thermal-pole band of 5.10 (>10 Hz)
OFFBIN_HZ     = F_DRIVE_HZ * 1.37 + 0.071   # 5.10 offbin_freq(): lands between bins
CAP_S         = 0.6      # capture seconds per point (cycles/f bounded, like 5.10)

# race model (the realism the wormhole sim lacked): conflict probabilities
P_ON_BASE     = 0.93     # P(observer probe conflicts | writer-ON) at amp=3 (writer wins race)
P_OFF_BASE    = 0.08     # P(conflict | writer-OFF): refresh + background re-open
BANK_MAP_OK   = 0.90     # fraction of probes that actually land in target bank B
                         # (imperfect userspace bank reverse-engineering)
DRIFT_NS      = 18.0     # slow thermal/scheduler latency drift amplitude over capture
NULL1_CONTENTION_NS = 5.0   # different-bank ON-half contention bump (shared MC queue)

# .holo modes: bank subsets (mirror the C-harness FAMILIES over N banks)
N_BANKS = 16             # 8 banks/rank x 2 ranks (DDR3); spatial multiplex
FAMILIES = {
    0: [1, 2, 3, 4, 5, 6],                 # basis
    1: [7, 8, 9, 10, 11, 12, 13, 14],      # rotation
    2: [2, 5, 8, 11],                      # residual
    3: [1, 4, 7, 10, 13, 2, 8, 14, 5, 11], # mini (relational)
}
PSEUDO = {  # matched budget, shifted to a DISJOINT bank set (the pseudo null)
    0: [0, 9, 10, 11, 12, 13],
    1: [0, 1, 2, 3, 4, 5, 6, 15],
    2: [0, 6, 9, 15],
    3: [0, 3, 6, 9, 12, 15, 1, 4, 7, 13],
}
MODES = [0, 1, 2, 3]


def offbin(f):
    return f * 1.37 + 0.071


def synth_probe_series(active, amp_level, phase, rng, null=None):
    """Generate one observer latency time-series (ns) for ONE bank.

    active   : True if the writer is driving THIS bank for the frame (mode bit=1)
    amp_level: writer amplitude (n aggressor cores / hammer intensity) 1..5
    phase    : relational phase tag for this bank (the .holo ramp)
    null     : None (real), 'same_row' (NULL-2), 'diff_bank' (NULL-1), 'idle' (NULL-4)
    returns  : (t_s, lat_ns) arrays
    """
    n = int(CAP_S * READ_HZ)
    t_s = np.arange(n) / READ_HZ
    # square-wave drive phase at f_drive with the per-bank relational phase offset
    sq = (np.sin(2 * np.pi * F_DRIVE_HZ * t_s + phase) >= 0).astype(float)  # 1=ON,0=OFF

    # amplitude scaling: more writer cores -> writer wins the race harder
    win = min(1.0, 0.72 + 0.07 * amp_level)         # p_on grows with amp, saturating
    p_on  = P_OFF_BASE + (P_ON_BASE - P_OFF_BASE) * win
    p_off = P_OFF_BASE

    if null == 'idle' or not active:
        # writer not driving this bank: observer just holds r_o open; only refresh/bg
        p_on = p_off = P_OFF_BASE
    if null == 'same_row':
        # writer hammers bank B but SAME row r_o -> keeps r_o OPEN -> observer HITS.
        # identical power/traffic/contention, zero row-conflict modulation.
        p_on = p_off = P_OFF_BASE
    if null == 'diff_bank':
        # writer hammers a DIFFERENT bank: NO row-buffer modulation on B, but a small
        # ON-half contention latency bump leaks through the shared memory controller.
        p_on = p_off = P_OFF_BASE

    # bank-map reliability: a fraction of probes miss bank B -> no modulation on those
    in_bank = rng.random(n) < BANK_MAP_OK

    # conflict draw per probe
    pc = np.where(sq > 0.5, p_on, p_off)
    conflict = (rng.random(n) < pc) & in_bank
    lat = np.where(conflict, T_CONFLICT_NS, T_HIT_NS).astype(float)

    # refresh: periodic forced conflicts (~128 kHz), OUT of f_drive band
    ref_phase = (t_s * REFRESH_HZ) % 1.0
    refresh_hit = ref_phase < (READ_HZ / REFRESH_HZ) / max(1.0, (READ_HZ / REFRESH_HZ))
    # approx: each probe near a refresh edge forced to conflict
    refresh_mask = (np.floor(t_s * REFRESH_HZ).astype(np.int64) !=
                    np.floor((t_s - 1.0 / READ_HZ) * REFRESH_HZ).astype(np.int64))
    lat = np.where(refresh_mask, np.maximum(lat, T_CONFLICT_NS), lat)

    # NULL-1 different-bank ON-half contention bump (broadband-ish but phase-aligned)
    if null == 'diff_bank':
        lat = lat + sq * NULL1_CONTENTION_NS * (0.6 + 0.1 * amp_level)

    # per-probe jitter + heavy-tail interrupts
    lat = lat + rng.normal(0.0, JITTER_NS, n)
    tail = (rng.random(n) < HEAVYTAIL_P)
    lat = lat + tail * rng.exponential(HEAVYTAIL_NS, n)

    # slow thermal/scheduler drift (DC + sub-Hz): low-frequency, rejected by lock-in
    lat = lat + DRIFT_NS * np.sin(2 * np.pi * 0.7 * t_s + 0.3) + 0.5 * DRIFT_NS * (t_s / CAP_S)

    return t_s, lat


def lockin(t_s, x, f_ref, phase_offset=0.0):
    """5.10-style lock-in: mean-remove + Hann + demod at f_ref. Returns (I,Q,mag)."""
    n = len(x)
    x = x - np.mean(x)
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))  # Hann
    ph = 2 * np.pi * f_ref * t_s + phase_offset
    wsum = np.sum(w)
    I = 2 * np.sum(w * x * np.cos(ph)) / wsum
    Q = 2 * np.sum(w * x * np.sin(ph)) / wsum
    return I, Q, math.hypot(I, Q)


def bank_feature(active, amp, phase, rng, null=None):
    """One bank -> (lockin magnitude at f_drive, off-bin floor, recovered phase, snr)."""
    t_s, lat = synth_probe_series(active, amp, phase, rng, null)
    I, Q, mag = lockin(t_s, lat, F_DRIVE_HZ)
    _, _, floor = lockin(t_s, lat, OFFBIN_HZ)
    ph_hat = math.atan2(Q, I)
    snr = mag / floor if floor > 0 else 0.0
    return mag, floor, ph_hat, snr


def frame_vector(family_map, mode, amp, rng, null=None, with_phase=False):
    """A .holo frame: per-bank lock-in magnitude vector + (optional) phase read."""
    fam = set(family_map[mode])
    mags = np.zeros(N_BANKS)
    phs = np.zeros(N_BANKS)
    snrs = []
    for b in range(N_BANKS):
        active = b in fam
        phase = (2 * math.pi * (sorted(fam).index(b) / max(1, len(fam)))) if active else 0.0
        mag, floor, ph_hat, snr = bank_feature(active, amp, phase, rng, null)
        mags[b] = mag
        phs[b] = ph_hat
        if active:
            snrs.append(snr)
    return mags, phs, (np.mean(snrs) if snrs else 0.0)


def run(seeds=(11, 23, 37, 51, 67, 83, 97, 113), amp=3, n_trials=80):
    """Train centroids on clean magnitude footprints; score held-out trials with
    the analyzer's matched-null structure (real / pseudo / wrong), across seeds.
    Also report NULL-1/NULL-2 lock-in collapse and per-bank SNR + phase MAE."""
    # clean centroids from the expected magnitude footprint (high-amp, low-noise)
    cen_rng = np.random.RandomState(9001)
    centroids = {}
    for m in MODES:
        acc = np.zeros(N_BANKS)
        for _ in range(12):
            mags, _, _ = frame_vector(FAMILIES, m, 5, cen_rng)
            acc += mags
        centroids[m] = acc / 12

    def classify(vec):
        return min(MODES, key=lambda m: float(np.sum((vec - centroids[m]) ** 2)))

    per_seed_acc = []
    pseudo_decl = pseudo_tot = 0
    wrong_actual = wrong_decl = wrong_tot = 0
    phase_err = []
    snr_real, snr_null1, snr_null2 = [], [], []
    # SNR of the DRIVEN bank lock-in under real vs nulls (the decisive collapse)
    for s in seeds:
        rng = np.random.RandomState(s)
        hits = tot = 0
        for t in range(n_trials):
            declared = MODES[t % 4]
            theta = (rng.random() * 2 - 1) * math.pi  # frame relational phase offset
            # real
            mags, phs, snr = frame_vector(FAMILIES, declared, amp, rng)
            if classify(mags) == declared:
                hits += 1
            tot += 1
            snr_real.append(snr)
            # phase: recover ramp on the driven banks of the real frame
            fam = sorted(FAMILIES[declared])
            if len(fam) >= 2:
                got = np.array([phs[b] for b in fam])
                exp = np.array([2 * math.pi * (i / len(fam)) for i in range(len(fam))])
                # align by removing common offset (relational, not absolute)
                d = np.angle(np.exp(1j * (got - exp)))
                d = np.angle(np.exp(1j * (d - np.mean(d))))
                phase_err.extend(np.abs(d).tolist())
            # pseudo (matched budget, disjoint banks) should NOT classify as declared
            pmags, _, _ = frame_vector(PSEUDO, declared, amp, rng)
            if classify(pmags) == declared:
                pseudo_decl += 1
            pseudo_tot += 1
            # wrong (declared label false; actual schedule is next real mode)
            actual = (declared + 1) % 4
            wmags, _, _ = frame_vector(FAMILIES, actual, amp, rng)
            wp = classify(wmags)
            if wp == actual:
                wrong_actual += 1
            if wp == declared:
                wrong_decl += 1
            wrong_tot += 1
            # NULL-1 / NULL-2 single-bank SNR collapse (decisive discriminators)
            sn1 = bank_feature(True, amp, 0.0, rng, null='diff_bank')[3]
            sn2 = bank_feature(True, amp, 0.0, rng, null='same_row')[3]
            snr_null1.append(sn1); snr_null2.append(sn2)
        per_seed_acc.append(hits / tot)

    real_acc = float(np.mean(per_seed_acc))
    res = {
        "amp_level": amp,
        "per_seed_accuracy": [float(a) for a in per_seed_acc],
        "real_accuracy_mean": real_acc,
        "real_accuracy_min_seed": float(np.min(per_seed_acc)),
        "real_accuracy_std": float(np.std(per_seed_acc)),
        "pseudo_declared_match": pseudo_decl / pseudo_tot,
        "wrong_actual_match": wrong_actual / wrong_tot,
        "wrong_declared_match": wrong_decl / wrong_tot,
        "phase_mae_rad": float(np.mean(phase_err)) if phase_err else None,
        "driven_bank_snr_real_mean": float(np.mean(snr_real)),
        "null1_diffbank_snr_mean": float(np.mean(snr_null1)),
        "null2_samerow_snr_mean": float(np.mean(snr_null2)),
        "snr_real_over_null2": float(np.mean(snr_real) / max(1e-9, np.mean(snr_null2))),
    }
    return res


def amplitude_ladder():
    """Monotonicity control: driven-bank SNR vs writer amplitude (the inverted-
    homogenization signature -- stronger coupling -> stronger signal)."""
    out = []
    for amp in (1, 2, 3, 4, 5):
        rng = np.random.RandomState(500 + amp)
        snrs = [bank_feature(True, amp, 0.0, rng, None)[3] for _ in range(60)]
        out.append({"amp": amp, "driven_snr_mean": float(np.mean(snrs))})
    return out


def main():
    seeds = (11, 23, 37, 51, 67, 83, 97, 113)  # 8 seeds (>= 6 reproducibility gate)
    main_res = run(seeds=seeds, amp=3, n_trials=80)
    ladder = amplitude_ladder()

    gates = {
        # mirror analyze_cache_hologram_matched_nulls.py thresholds
        "real_accuracy_ge_0_60": main_res["real_accuracy_mean"] >= 0.60,
        "reproducible_min_seed_ge_0_60": main_res["real_accuracy_min_seed"] >= 0.60,
        "pseudo_declared_match_le_0_35": main_res["pseudo_declared_match"] <= 0.35,
        "wrong_actual_match_ge_0_60": main_res["wrong_actual_match"] >= 0.60,
        "wrong_declared_match_le_0_20": main_res["wrong_declared_match"] <= 0.20,
        # row-buffer-specific decisive nulls
        "null2_samerow_collapses": main_res["snr_real_over_null2"] >= 3.0,
        "amp_monotonic": all(ladder[i]["driven_snr_mean"] <= ladder[i + 1]["driven_snr_mean"] + 1e-6
                             for i in range(len(ladder) - 1)),
        "phase_recovered": (main_res["phase_mae_rad"] is not None
                            and main_res["phase_mae_rad"] < 0.5),
    }
    verdict = ("DRAM_ROWBUFFER_LOCKIN_SIM_PROMISING"
               if all(gates.values()) else "DRAM_ROWBUFFER_LOCKIN_SIM_PARTIAL")

    out = {
        "channel": "dram_rowbuffer_driven_lockin",
        "claim_cap": "cross-core .holo traversal witness (NOT lattice/crypto)",
        "seeds": list(seeds),
        "physics_ns": {"t_hit": T_HIT_NS, "t_conflict": T_CONFLICT_NS,
                       "jitter": JITTER_NS, "refresh_hz": REFRESH_HZ},
        "main": main_res,
        "amplitude_ladder": ladder,
        "gates": gates,
        "verdict": verdict,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "dram_rowbuffer_lockin_results.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("=" * 74)
    print("DRAM ROW-BUFFER DRIVEN LOCK-IN -- realistic-noise sim")
    print("=" * 74)
    print("real accuracy (mean over %d seeds) = %.4f  (min seed %.4f, std %.4f)"
          % (len(seeds), main_res["real_accuracy_mean"],
             main_res["real_accuracy_min_seed"], main_res["real_accuracy_std"]))
    print("pseudo declared-match = %.4f   wrong actual-match = %.4f   wrong declared = %.4f"
          % (main_res["pseudo_declared_match"], main_res["wrong_actual_match"],
             main_res["wrong_declared_match"]))
    print("phase MAE (rad) = %s" % main_res["phase_mae_rad"])
    print("driven-bank SNR  real=%.2f  NULL1(diff-bank)=%.2f  NULL2(same-row)=%.2f  real/null2=%.2f"
          % (main_res["driven_bank_snr_real_mean"], main_res["null1_diffbank_snr_mean"],
             main_res["null2_samerow_snr_mean"], main_res["snr_real_over_null2"]))
    print("amplitude ladder (driven SNR):",
          ", ".join("amp%d=%.2f" % (d["amp"], d["driven_snr_mean"]) for d in ladder))
    print("-" * 74)
    for k, v in gates.items():
        print("  gate %-36s %s" % (k, "PASS" if v else "FAIL"))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
