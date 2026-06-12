"""
Exp 50.1 - The Extractive Proof.

Claim: the holographic/spectral readout recovers an answer (a global frequency /
the Riemann zeros) that NO lookup-class decoder can, on the SAME encoding -
because the answer is a global coherent-integration property, not a local or
low-order-statistical one. The lookup-nulls get the full data and unbounded
compute; only their FUNCTIONAL FORM is constrained (bounded receptive field OR
bounded statistical order). The wrong-answer control certifies the extractive
decoder is not keying on a stored statistic.

Two testbeds:
  - synth : controllable weak tone in noise (provable locality barrier).
  - zeta  : the real lab decoder, primes -> Riemann zeros (global resonance).

Run:  python 49_1_extractive_proof.py
Exit 0 iff all gates pass AND catalytic verify() is True.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import decoder_lib as dl
from catalytic_tape import CatalyticTape
import testbed_synth as ts
import testbed_zeta as tz
import wrong_answer_control as wac

LINES = []
def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def call_decoder(name, fn, E, meta, train=None):
    if name == "null_histogram_regressor":
        return fn(E, meta, train=train)
    if name.startswith("null_windowed"):
        return fn(E, meta, window_frac=meta["window_frac"])
    return fn(E, meta)


def main():
    log("=" * 78)
    log("EXP 50.1  -  THE EXTRACTIVE PROOF")
    log("  global coherent spectral readout  vs  lookup-class (local/statistical) nulls")
    log("=" * 78)

    gates = {}

    # ---------------------------------------------------------------
    # SYNTHETIC TESTBED: weak tone in noise
    # ---------------------------------------------------------------
    M, N_SIG, SNR = 4096, 60, 64.0
    bank = ts.signal_bank(M=M, n_signals=N_SIG, snr_global=SNR, seed=1)
    train = [(E, k) for (E, k, _m) in ts.signal_bank(M=M, n_signals=40, snr_global=SNR, seed=999)]
    meta0 = bank[0][2]
    chance = (2 * meta0["tol_bins"] + 1) / (M // 2)

    log("\n[SYNTH] M=%d  signals=%d  global SNR=%.1f  per-window SNR=%.2f (w=M/%d)"
        % (M, N_SIG, SNR, meta0["snr_per_window"], int(1 / meta0["window_frac"])))
    log("        success tol = +/-%d bins ; random-guess chance ~= %.3f" % (meta0["tol_bins"], chance))

    decoders = {**dl.EXTRACTIVE_DECODERS, **dl.LOOKUP_NULLS}
    succ = {name: [] for name in decoders}
    for (E, k_true, meta) in bank:
        for name, fn in decoders.items():
            est = call_decoder(name, fn, E, meta, train=train)
            succ[name].append(1 if ts.success(est, k_true, meta) else 0)

    log("\n  decoder                     class       success   95%% CI            vs-extractive perm p / Cohen h")
    ext_succ = succ["fft_peak"]
    rate = {}
    for name in decoders:
        s = np.array(succ[name]); r = s.mean(); rate[name] = r
        lo, hi = dl.wilson_ci(int(s.sum()), len(s))
        cls = "EXTRACTIVE" if name in dl.EXTRACTIVE_DECODERS else "lookup-null"
        if name in dl.LOOKUP_NULLS:
            p = dl.permutation_p(ext_succ, s, n_perm=5000, seed=3)
            h = dl.cohen_h(np.mean(ext_succ), r)
            extra = "p=%.4g  Cohen h=%.2f" % (p, h)
        else:
            extra = "-"
        log("  %-26s %-11s %6.3f    CI[%.3f, %.3f]   %s" % (name, cls, r, lo, hi, extra))

    # G1: extractive recovers the tone
    g1 = rate["fft_peak"] >= 0.8
    gates["G1 extractive(synth) >=0.8"] = (g1, "fft_peak rate=%.3f (eigenmode=%.3f)" % (rate["fft_peak"], rate["eigenmode"]))

    # G3: extractive DEFEATS every lookup null with large effect
    #     (the claim is separation, not that a null sits exactly at chance).
    null_ok, null_detail = True, []
    ext_rate = rate["fft_peak"]
    for name in dl.LOOKUP_NULLS:
        p = dl.permutation_p(ext_succ, np.array(succ[name]), n_perm=5000, seed=5)
        h = dl.cohen_h(ext_rate, rate[name])
        defeated = (p < 1e-3) and (rate[name] < 0.5 * ext_rate) and (h > 0.8)
        null_ok = null_ok and defeated
        null_detail.append("%s rate=%.3f p=%.4g h=%.2f %s" % (name, rate[name], p, h, "OK" if defeated else "FAIL"))
    gates["G3 null separation"] = (null_ok, " | ".join(null_detail))

    # ---------------------------------------------------------------
    # WRONG-ANSWER CONTROL (anti-circularity)
    # ---------------------------------------------------------------
    log("\n[WRONG-ANSWER CONTROL]  matched value-statistics, different true answer")
    (Ea, ka), (Eb, kb) = wac.matched_pair_synth(M=M, k_a=137, k_b=613, snr_global=SNR, seed=7)
    ident = wac.statistical_identity(Ea, Eb)
    log("  statistical identity: min KS p=%.3f  max |moment diff|=%.2e  (want p>0.5, diff small)"
        % (ident["min_ks_p"], ident["max_abs_moment_diff"]))
    ext_a = dl.extract_fft_peak(Ea); ext_b = dl.extract_fft_peak(Eb)
    hist_a = dl.null_histogram_regressor(Ea, {}, train=train)
    hist_b = dl.null_histogram_regressor(Eb, {}, train=train)
    ext_ok = abs(ext_a - ka) <= 40 and abs(ext_b - kb) <= 40 and ext_a != ext_b
    null_tracks = abs(hist_a - ka) <= 40 and abs(hist_b - kb) <= 40
    log("  extractive(fft):  E_A->%d (true %d)   E_B->%d (true %d)   TRACKS TRUTH=%s"
        % (ext_a, ka, ext_b, kb, ext_ok))
    log("  lookup(histogram): E_A->%d   E_B->%d   TRACKS TRUTH=%s (identical stats -> returns wrong values)"
        % (hist_a, hist_b, null_tracks))
    g4 = (ident["min_ks_p"] > 0.5) and ext_ok and (not null_tracks)
    gates["G4 wrong-answer control"] = (g4, "extractive tracks true answer on statistics-matched pair; statistics-null cannot")

    # ---------------------------------------------------------------
    # CATALYTIC WRAP (mechanistic: decode reads the grating out of the tape)
    # ---------------------------------------------------------------
    log("\n[CATALYTIC]  borrow tape -> XOR-encode grating -> decode FROM tape -> uncompute -> verify")
    E_demo, k_demo, meta_demo = bank[0]
    payload = E_demo.astype(np.complex64).tobytes()
    tape = CatalyticTape(size_mb=1, seed=42)
    h0 = tape.hash()
    off, length = tape.record_operation(payload)
    region = tape.read_region(off, length)
    dirty = tape.dirty_baseline(off, length, seed=42)
    recovered = bytes(r ^ d for r, d in zip(region, dirty))
    E_from_tape = np.frombuffer(recovered, dtype=np.complex64).astype(np.complex128)
    k_tape = dl.extract_fft_peak(E_from_tape)
    k_direct = dl.extract_fft_peak(E_demo.astype(np.complex64).astype(np.complex128))
    tape.uncompute()
    verified = False
    try:
        verified = tape.verify()
    except Exception as e:
        log("  verify() raised: %s" % e)
    h1 = tape.hash()
    log("  decoded-from-tape k=%d   decoded-direct k=%d   match=%s" % (k_tape, k_direct, k_tape == k_direct))
    log("  SHA-256 initial=%s..  final=%s..  restored=%s  was_modified=%s  verify=%s"
        % (h0[:12], h1[:12], h0 == h1, tape.was_modified, verified))
    g5 = (k_tape == k_direct) and verified and (h0 == h1) and tape.was_modified
    gates["G5 catalytic restoration"] = (g5, "decode reads grating from mutated tape; SHA restored; 0 bits erased")

    # ---------------------------------------------------------------
    # ZETA TESTBED: the real lab decoder
    # ---------------------------------------------------------------
    log("\n[ZETA]  primes -> explicit-formula grating -> global resonance sweep -> Riemann zeros")
    primes = tz.primes_upto_count(6000)
    amp, ln_p = tz.prime_grating(primes)
    found = tz.extract_zeros(amp, ln_p, n_bins=6000)
    true_z = tz.zeta_zeros(10)
    score = tz.score_zeros(found, true_z, tol=0.5)
    # scrambled wrong-answer control
    amp_s, ln_s = wac.scrambled_zeta(amp, ln_p, seed=11)
    found_s = tz.extract_zeros(amp_s, ln_s, n_bins=6000)
    score_s = tz.score_zeros(found_s, true_z, tol=0.5)
    log("  extracted (first 6): %s" % [round(f, 2) for f in found[:6]])
    log("  true zeros (first 6): %s" % [round(z, 2) for z in true_z[:6]])
    log("  zero-recovery score (real primes)      = %.2f of first %d" % (score, len(true_z)))
    log("  zero-recovery score (phase-scrambled)  = %.2f  (control: coherence destroyed, stats identical)" % score_s)
    g2 = score >= 0.5 and score > score_s + 0.2
    gates["G2 extractive(zeta) recovers zeros"] = (g2, "real score=%.2f vs scrambled-control=%.2f" % (score, score_s))

    # ---------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------
    log("\n" + "=" * 78)
    log("GATES")
    all_pass = True
    for name, (ok, detail) in gates.items():
        all_pass = all_pass and ok
        log("  [%s] %-34s  %s" % ("PASS" if ok else "FAIL", name, detail))
    log("=" * 78)
    verdict = ("EXTRACTIVE_CONFIRMED" if all_pass else "EXTRACTIVE_NOT_CONFIRMED")
    log("VERDICT: %s   (claim level 4-5: extraction beats lookup nulls + survives wrong-answer control)" % verdict)
    log("Effect: extractive success ~%.2f vs lookup-null success ~%.2f ; barrier = integration length (locality), not compute."
        % (rate["fft_peak"], np.mean([rate[n] for n in dl.LOOKUP_NULLS])))

    (HERE / "output.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
