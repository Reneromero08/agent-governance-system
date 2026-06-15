"""
run_pt_sensor.py - Phase 6 nonhermitian_sensor / pt_symmetry runner.

Stages:
  [V] validate the O(N) corner-expansion determinant against dense determinants
  [T] confirm structure facts: T1 (public operator byte-identical under the fold at
      fixed public data), T2 (smuggle spectrum fold-blind: similarity), T3 (point-gap
      winding = constant public direction; flips only with the walk direction)
  [G] hardened no-smuggle gate (fold_audit/stage3/hardened_gate.py, reused verbatim)
      on the public PT readouts at n = 8, 10, 12, 14, plus two smuggle controls at n=8
  [C] cost scaling of the invariant readout across n = 8..14 (the make-or-break)

Writes pt_sensor_result.json and output_pt_sensor.txt into this directory.
Seeded (MASTER_SEED below). ASCII only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE6 = HERE.parent.parent
FOLD = PHASE6 / "02_fold_audit"
STAGE3 = FOLD / "stage3"
for p in (str(HERE), str(FOLD), str(STAGE3)):
    if p not in sys.path:
        sys.path.insert(0, p)

import construction as C            # noqa: E402  (fold_audit, verbatim source of truth)
import no_smuggle_gate as G         # noqa: E402
import hardened_gate as HG          # noqa: E402
import pt_operator as P             # noqa: E402

MASTER_SEED = 44061101
LINES = []


def log(m=""):
    print(m, flush=True)
    LINES.append(str(m))


def stage_validate(out):
    worst = P.validate_dets(seed=MASTER_SEED + 1)
    ok = worst < 1e-8
    out["det_validation"] = {"worst_rel_err": worst, "ok": bool(ok)}
    log("[V] corner-expansion determinant vs dense: worst rel err = %.3e  ok = %s" % (worst, ok))
    if not ok:
        raise SystemExit("determinant formula failed validation - aborting")


def stage_structure(out):
    rng = np.random.default_rng(MASTER_SEED + 2)
    n = 8
    N = 1 << n
    d = C.sample_secret(N, rng)
    inst = G.make_instance(n, d, rng)
    finst = G.folded_instance(inst)

    g1 = P.g_even_profile(inst["k"], inst["b"], inst["N"])
    g2 = P.g_even_profile(finst["k"], finst["b"], finst["N"])
    t1 = bool(np.array_equal(g1, g2))
    log("[T1] H_public byte-identical under fold at fixed public data: %s" % t1)

    gd = P.g_odd_profile(inst["k"], d, N)
    gfd = P.g_odd_profile(inst["k"], (N - d) % N, N)
    odd_flip = float(np.max(np.abs(gd + gfd)))
    w1 = np.linalg.eigvals(P.build_dense(N, gd, 0.0, P.GAMMA_SMUG))
    w2 = np.linalg.eigvals(P.build_dense(N, gfd, 0.0, P.GAMMA_SMUG))
    DM = np.abs(w1[:, None] - w2[None, :])
    spec_diff = float(max(DM.min(axis=0).max(), DM.min(axis=1).max()))  # Hausdorff SET distance
    log("[T2] g_odd(N-d) == -g_odd(d): max|g(d)+g(N-d)| = %.3e ; smuggle spectrum Hausdorff "
        "distance under fold = %.3e  (similarity P H(g) P = H(-g): spectrum is fold-blind)" % (odd_flip, spec_diff))

    Ws = []
    for _ in range(20):
        dd = C.sample_secret(N, rng)
        ii = G.make_instance(n, dd, rng)
        gg = P.g_even_profile(ii["k"], ii["b"], N)
        W, _ = P.winding(P.E0_IN, 1j * P.GAMMA_PUB * gg, P.A_HOP)
        Ws.append(int(W))
    W_rev, _ = P.winding(P.E0_IN, 1j * P.GAMMA_PUB * g1, -P.A_HOP)
    W_out, _ = P.winding(P.E0_OUT, 1j * P.GAMMA_PUB * g1, P.A_HOP)
    log("[T3] point-gap winding W(E0_IN) across 20 instances: %s ; W outside spectrum: %d ; "
        "W with the walk direction reversed: %d" % (sorted(set(Ws)), int(W_out), int(W_rev)))
    log("     -> the +1 directionality of f DOES survive into a genuine non-Hermitian "
        "invariant, but it is a public constant: it carries the walk direction, not the half.")
    out["structure"] = {"t1_byte_identical": t1, "t2_odd_flip_max": odd_flip,
                        "t2_spec_diff_under_fold": spec_diff,
                        "t3_windings_inside": Ws, "t3_w_outside": int(W_out),
                        "t3_w_reversed": int(W_rev)}


def stage_gates(out):
    cells = []
    cases = [
        ("public_full_n8", P.O_public_full, 8, 150, "FAIL_CHANCE"),
        ("public_evals_n10", P.O_public_evals, 10, 90, "FAIL_CHANCE"),
        ("public_winding_n12", P.O_public_winding, 12, 80, "FAIL_CHANCE"),
        ("public_winding_n14", P.O_public_winding, 14, 60, "FAIL_CHANCE"),
        ("smuggle_evals_n8", P.O_smuggle_evals, 8, 120, "FAIL_CHANCE"),
        ("smuggle_evecs_n8", P.O_smuggle_evecs, 8, 120, "FAIL_SMUGGLE"),
    ]
    for ci, (name, O, n, ninst, expected) in enumerate(cases):
        seed = (MASTER_SEED + 7919 * n + 101 * ci) & 0x7FFFFFFF
        tic = time.time()
        res = HG.hardened_gate(O, n, n_instances=ninst, seed=seed, n_shuffles=20)
        dt = time.time() - tic
        ok = res["verdict"] == expected
        log("  [%s] %-20s n=%-2d verdict=%-13s (exp %-13s)" %
            ("OK " if ok else "!! ", name, n, res["verdict"], expected))
        log("        orient_auc=%.3f (null95=%.3f)  rf_auc=%.3f (null95=%.3f)  "
            "delta=%.3g  reason=%s  [%.1fs]" %
            (res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
             res["random_fold_null_95"], res["max_fold_delta"], res["smuggle_reason"], dt))
        cells.append({
            "name": name, "n": n, "seed": int(seed), "n_instances": ninst,
            "expected": expected, "verdict": res["verdict"], "matches": bool(ok),
            "orientation_auc": res["auc"], "orientation_null95": res["shuffle_null_95"],
            "random_fold_auc": res["random_fold_auc"],
            "random_fold_null95": res["random_fold_null_95"],
            "invariance_delta": res["max_fold_delta"],
            "smuggle_reason": res["smuggle_reason"], "elapsed_s": dt,
        })
    out["gates"] = cells


def stage_cost(out):
    rng = np.random.default_rng(MASTER_SEED + 3)
    rows = []
    log("[C] invariant readout cost (point-gap winding, O(N) exact): per-instance seconds")
    for n in (8, 10, 12, 14):
        N = 1 << n
        ts = []
        for _ in range(5):
            d = C.sample_secret(N, rng)
            inst = G.make_instance(n, d, rng)
            tic = time.perf_counter()
            P.O_public_winding(inst)
            ts.append(time.perf_counter() - tic)
        med = float(np.median(ts))
        rows.append({"n": n, "dim": N, "median_s": med})
        log("    n=%-3d dim=%-6d M=%-4d median=%.4f s" % (n, N, C.M_for(n), med))
    g814 = rows[-1]["median_s"] / max(rows[0]["median_s"], 1e-12)
    g1014 = rows[-1]["median_s"] / max(rows[1]["median_s"], 1e-12)
    log("    growth n=8 -> n=14 (dim x64): %.1fx ; n=10 -> n=14 (dim x16): %.1fx" % (g814, g1014))

    dts = []
    log("[C] full-spectrum (dense eigvals, O(dim^3)) cost:")
    for n, reps in ((8, 3), (10, 3), (12, 1)):
        N = 1 << n
        ts = []
        for _ in range(reps):
            d = C.sample_secret(N, rng)
            inst = G.make_instance(n, d, rng)
            g = P.g_even_profile(inst["k"], inst["b"], N)
            H = P.build_dense(N, g, P.A_HOP, P.GAMMA_PUB)
            tic = time.perf_counter()
            np.linalg.eigvals(H)
            ts.append(time.perf_counter() - tic)
        med = float(np.median(ts))
        dts.append({"n": n, "dim": N, "median_s": med})
        log("    n=%-3d dim=%-6d median=%.3f s" % (n, N, med))
    out["cost"] = {
        "winding_O_N": rows, "dense_eig_O_N3": dts,
        "h_dimension": "2^n (site space of Z_N)",
        "growth_winding_n8_to_n14": g814, "growth_winding_n10_to_n14": g1014,
        "note": "winding readout is Theta(N) = Theta(2^n) exact per instance even with "
                "the O(N) transfer/corner method; dense spectra are O(2^{3n}). The "
                "invariant computation IS the forward-walk cost relocated.",
    }


def main():
    t0 = time.time()
    out = {"master_seed": MASTER_SEED, "approach": "pt_symmetry_biorthogonal",
           "operator": "H = e^{+a} S + e^{-a} S^T + i*gamma*diag(score(x)/M) on Z_N"}
    log("=" * 100)
    log("PHASE 6 NONHERMITIAN SENSOR - PT-SYMMETRIC / BIORTHOGONAL CANDIDATE  (seed %d)" % MASTER_SEED)
    log("=" * 100)
    stage_validate(out)
    stage_structure(out)
    log("[G] hardened no-smuggle gate (fold_audit/stage3, reused) on PT readouts:")
    stage_gates(out)
    stage_cost(out)

    pub = [c for c in out["gates"] if c["name"].startswith("public")]
    ctrl_evals = [c for c in out["gates"] if c["name"] == "smuggle_evals_n8"]
    ctrl_evecs = [c for c in out["gates"] if c["name"] == "smuggle_evecs_n8"]
    pub_chance = all(c["verdict"] == "FAIL_CHANCE" for c in pub)
    pub_delta0 = all(c["invariance_delta"] == 0.0 for c in pub)
    evals_blind = all(c["verdict"] == "FAIL_CHANCE" for c in ctrl_evals)
    evecs_caught = all(c["verdict"] == "FAIL_SMUGGLE" for c in ctrl_evecs)

    log("")
    log("=" * 100)
    log("WHAT THE DATA SAYS")
    log("  public PT/topological readouts at chance for n=8,10,12,14: %s (deltas all 0: %s)"
        % (pub_chance, pub_delta0))
    log("  smuggle control, EIGENVALUE-only readout at chance (spectrum fold-blind even "
        "with quadrature): %s" % evals_blind)
    log("  smuggle control, EIGENVECTOR readout caught by the hardened gate: %s" % evecs_caught)
    if pub_chance and pub_delta0:
        verdict = "FOLD_INVARIANT_OPERATOR_CONFIRMS_WALL"
        log("VERDICT: %s  (honest outcome class iv)" % verdict)
        log("  Any H built from public data is byte-identical under the fold, so every")
        log("  invariant (winding, PT phase, EP, biorthogonal IPR) is fold-blind - measured.")
        log("  The directionality of f DOES survive as a nonzero point-gap winding, but it")
        log("  is the same public constant for every instance: direction of the walk, not")
        log("  the half. A PT-symmetric operator whose P is the fold needs a parity-ODD")
        log("  gain/loss profile = the quadrature channel = the smuggle. And even then the")
        log("  PT ORDER PARAMETER (spectrum) stays fold-blind by similarity; orientation")
        log("  sits only in WHICH site hosts the gain mode (eigenvector data). The wall")
        log("  stands; the invariant computation costs Theta(2^n) on top.")
    else:
        verdict = "UNEXPECTED_REAUDIT_REQUIRED"
        log("VERDICT: %s  - a public PT readout moved off chance or a control misfired;" % verdict)
        log("  treat as suspect, re-audit before any claim.")
    out["verdict"] = verdict
    out["elapsed_s"] = time.time() - t0
    log("total elapsed: %.1f s" % out["elapsed_s"])
    log("=" * 100)

    (HERE / "pt_sensor_result.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    (HERE / "output_pt_sensor.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()

