#!/usr/bin/env python3
"""
finalize.py -- build the SLOT 2 result JSON from the aggregate + preflight numbers.
Reads the aggregate.py output JSON and the preflight numbers, emits result_slot2_pdn.json.
Usage: finalize.py <aggregate.json> <out_result.json>
ASCII only.
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print("usage: finalize.py <aggregate.json> <out.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as fh:
        agg = json.load(fh)

    overall = agg["overall"]
    witness = overall["witness_holds_reproducibly"]

    result = {
        "experiment": "EXP50_SLOT2_PDN_DRIVEN_LOCKIN_CROSS_CORE_HOLO_TRAVERSAL",
        "date": "2026-06-12",
        "box": {
            "host": "192.168.137.100",
            "cpu": "AMD Phenom II X6 1090T (K10, 45nm)",
            "os": "Debian 13 (Linux 6.12.86)",
            "gcc": "14.2.0",
            "isolcpus": "2,3,4,5",
            "tsc": "constant_tsc + nonstop_tsc, measured 3214.823 MHz",
            "k10temp_path": "/sys/class/hwmon/hwmon0/temp1_input",
            "k10temp_veto_threshold_C": 68.0,
            "userspace_only": True,
        },
        "channel": "shared package power-delivery network (PDN) / IR-drop, "
                   "register/L1-only alu_burst drive -> victim ring-osc timing lock-in",
        "claim_cap": "cross-core .holo traversal readout (MODE + relational phase) above "
                     "matched nulls on real silicon. NOT a wall crossing, NOT d-recovery "
                     "(sender owns the drive phase), NOT a lattice/crypto/quadrature claim.",
        "preflight": {
            "gate": "single-bin cross-process compute-only SNR_eff >= ~3",
            "snr_eff_v2s3_200hz": 25.3,
            "snr_eff_v4s5_200hz": 212.6,
            "snr_eff_v2s3_80hz": 16.4,
            "snr_eff_v2s3_1200hz": 37.6,
            "verdict": "PASS (16-213, far above the ~3 cliff; consistent with the 5.10 "
                       "isolation SNR 50-86)",
        },
        "method": {
            "architecture": "two userspace PROCESSES (sender, receiver), forked by an "
                            "orchestrator, pinned to isolated cores; shared absolute-TSC "
                            "origin t0 = rdtsc+100ms via handshake file; per-slot phase "
                            "reference + deadline-bounded capture keep every slot aligned.",
            "drive": "register/L1-only alu_burst (5.10 power virus, lifted verbatim) gated "
                     "as a 50%-duty square wave at the bin tone; per-bin +/-1 sign carried "
                     "as a pi drive-phase flip; global theta as a shared drive-phase offset.",
            "readout": "victim software ring-osc timing lock-in at f_b + off-bin floor "
                       "(5.10 lock-in math, verbatim).",
            "codebook": "4 codewords / 12 bins, distinct flip-weights {4,5,6,7}, min "
                        "pairwise Hamming 7 (verified permutation-inequivalent fix).",
            "tones": "12 log-spaced non-harmonic tones in [20,1500] Hz, time-multiplexed "
                     "one bin per slot (sequential single-tone drive; no intermod).",
            "decode": "de-rotate zhat=(z*exp(-i*g))/||z|| (g=carrier phase), L2-normalize "
                      "(kills 1/f amp drift), nearest-codeword for MODE, differential theta "
                      "for the relational tag; rho energy-concentration matched null.",
            "reversible": "XOR-borrow + byte-hash restore, bit-exact (hash_restored==1 all rows).",
            "safety": "userspace only; cpufreq P-state pin (1600 MHz, boost off) restored on "
                      "clean exit AND on SIGTERM/SIGINT (signal safety net); k10temp veto "
                      "before every drive slot; -lm last at link.",
        },
        "sweep": {
            "trials_per_family": 48,
            "slot_s": 0.5,
            "read_hz": 4000,
            "seeds": [0, 1, 2, 3, 4, 5],
            "core_pairs": ["2:3", "4:5"],
            "symbols_per_run": 148,
        },
        "success_criteria": "6/6 seeds: all_rows_restore AND real_accuracy>=0.60 AND "
                            "real_vs_pseudo floor>=0.95 AND pseudo_reject>=0.95 AND "
                            "wrong_actual_match>=0.60 AND wrong_declared_match<=0.20 AND "
                            "(phase_corr_true - phase_corr_null) > 0.30",
        "aggregate": agg,
        "verdict": overall["verdict"],
        "witness_label": "PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS",
        "witness_holds_reproducibly": witness,
        "next_slot": None if witness else "SLOT 3 (memctl Manchester)",
    }

    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
