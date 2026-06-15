"""
run_godel_phi.py - build + run + smuggle-gate + price the Godel-edge-phi sensor (6th
non-Hermitian sensor) on the REAL Exp 50.14 construction, across n in {8,10,12,14}.

Pipeline (all instruments REUSED verbatim from fold_audit / stage3):
  (a) hardened_gate (stage3) on the PUBLIC sensor O_public_godel_phi at every n:
      random-private-fold + exact d<->N-d invariance + orientation AUC vs shuffle null.
  (b) the construction-native SMUGGLE control (self-loop placed at the hidden d) -> must
      be FAIL_SMUGGLE.
  (c) the three standard sensitivity controls (gate's reads_d, reads_sin, useless_even).
  (d) PRICE the phi-swept winding: time the rank-1 / log-space readout (O(N) per
      instance, O(1) per phi after one chain pass) against a DIRECT dense det-per-phi
      winding (O(n_phi N^3)); confirm the lemma is cheap and the speedup grows in N.

ASCII only. All seeds recorded.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE6 = os.path.dirname(os.path.dirname(HERE))                     # .../phase6
FOLD = os.path.join(PHASE6, "fold_audit")
STAGE3 = os.path.join(FOLD, "stage3")
for p in (HERE, FOLD, STAGE3):
    if p not in sys.path:
        sys.path.insert(0, p)

import construction as C
import no_smuggle_gate as G
import hardened_gate as H
import godel_operator as O

MASTER_SEED = 44060611
N_BUDGET = {8: (400, 25), 10: (350, 20), 12: (220, 15), 14: (140, 12)}


def _row(tag, res):
    return ("  [%-4s] %-30s verdict=%-13s orient_auc=%.3f(null95=%.3f) "
            "rf_auc=%.3f(null95=%.3f) delta=%.3g reason=%s" % (
                "OK" if res.get("_ok", True) else "!!", tag, res["verdict"],
                res["auc"], res["shuffle_null_95"], res["random_fold_auc"],
                res["random_fold_null_95"], res["max_fold_delta"], res["smuggle_reason"]))


# ---------------------------------------------------------------------------
# (d) PRICING: rank-1/log-space winding vs direct dense det-per-phi winding
# ---------------------------------------------------------------------------
def direct_dense_winding(diag, E_ref, a, R, n_phi=64):
    """Honest O(n_phi * N^3) baseline: build the dense N x N (E I - H(phi)) for each phi
    and take slogdet, then count the winding of det around 0. No lemma, no corner trick."""
    N = len(diag)
    tR = np.exp(a); tL = np.exp(-a)
    lam = R * np.exp(-a * (N - 1))
    T = np.zeros((N, N), dtype=complex)
    idx = np.arange(N)
    T[idx[1:], idx[1:] - 1] = tR
    T[idx[:-1], idx[:-1] + 1] = tL
    np.fill_diagonal(T, diag)
    EI = E_ref * np.eye(N)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=True)
    ang = np.zeros(n_phi)
    for j, phi in enumerate(phis):
        Hm = T.copy()
        Hm[0, N - 1] = lam * np.exp(1j * phi)
        s, ld = np.linalg.slogdet(EI - Hm)
        ang[j] = np.angle(s)
    ang = np.unwrap(ang)
    return int(round((ang[-1] - ang[0]) / (2.0 * np.pi)))


def price(n, seed, n_phi=64, reps=3, do_dense=True):
    rng = np.random.default_rng(seed)
    N = 1 << n
    # rank-1/log-space readout timing
    insts = []
    for _ in range(reps):
        d = C.sample_secret(N, rng)
        insts.append(G.make_instance(n, d, rng))
    t0 = time.time()
    Wc = None
    for inst in insts:
        acc = O.accept_profile(inst["k"], inst["b"], N)
        diag = (-1j * O.ELL) + O.S_LOOP * acc
        feat, Wnum, Wclosed = O.phi_features(diag, O.E_REFS, n_phi=n_phi, want_numeric=True)
        Wc = (Wnum.tolist(), Wclosed.tolist())
    rank1_per = (time.time() - t0) / reps
    out = {"n": n, "N": N, "rank1_s_per_inst": rank1_per,
           "W_numeric": Wc[0], "W_closed": Wc[1],
           "W_match": bool(Wc[0] == Wc[1])}
    if do_dense:
        inst = insts[0]
        acc = O.accept_profile(inst["k"], inst["b"], N)
        diag = (-1j * O.ELL) + O.S_LOOP * acc
        t1 = time.time()
        Wd = direct_dense_winding(diag, O.E_REFS[3], O.A_HOP, O.LOOP_RADIUS, n_phi=n_phi)
        dense_per = time.time() - t1
        # compare to closed-form winding at the SAME E_ref (index 3)
        feat, Wnum, Wclosed = O.phi_features(diag, O.E_REFS, n_phi=n_phi, want_numeric=True)
        out.update({"dense_s_per_inst": dense_per,
                    "speedup_dense_over_rank1": dense_per / max(rank1_per, 1e-9),
                    "dense_W_Eref3": int(Wd), "closed_W_Eref3": int(Wclosed[3]),
                    "dense_matches_closed": bool(Wd == int(Wclosed[3]))})
    else:
        out.update({"dense_s_per_inst": None, "speedup_dense_over_rank1": None,
                    "dense_W_Eref3": None, "closed_W_Eref3": None,
                    "dense_matches_closed": None})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    report = {"master_seed": MASTER_SEED, "operator": {
        "a_hop": O.A_HOP, "ell": O.ELL, "s_loop": O.S_LOOP,
        "loop_radius_R": O.LOOP_RADIUS, "E_refs": [str(z) for z in O.E_REFS],
        "note": "det(EI-H(phi)) = D_open(E) - R e^{i phi};  R = lambda t_R^{N-1} is a "
                "PUBLIC CONSTANT; D_open is fold-even -> phi handle is fold-blind."},
        "primary": [], "smuggle_native": [], "controls": [], "pricing": []}

    print("=" * 92)
    print(" GODEL-EDGE-PHI : 6th non-Hermitian sensor for the Exp 50.14 dihedral fold")
    print("=" * 92)

    # ---- (a) PRIMARY public sensor at every n ----
    print("\n--- PRIMARY: O_public_godel_phi (phi-swept winding from PUBLIC accept) ---")
    for n in (8, 10, 12, 14):
        ninst, nshuf = N_BUDGET[n]
        seed = (MASTER_SEED + 911 * n) & 0x7FFFFFFF
        t = time.time()
        res = H.hardened_gate(O.O_public_godel_phi, n, n_instances=ninst, seed=seed,
                              n_shuffles=nshuf)
        res["_ok"] = (res["verdict"] == "FAIL_CHANCE")
        res["seed"] = seed; res["n_instances"] = ninst; res["elapsed_s"] = time.time() - t
        print("  n=%-2d  %s  [%.1fs]" % (n, _row("public", res), res["elapsed_s"]))
        report["primary"].append({k: res[k] for k in (
            "n", "verdict", "auc", "shuffle_null_95", "random_fold_auc",
            "random_fold_null_95", "max_fold_delta", "smuggle_reason", "seed",
            "n_instances", "elapsed_s")})

    # ---- (b) construction-native SMUGGLE control ----
    print("\n--- SMUGGLE CONTROL: O_smuggle_godel_phi (self-loop placed at hidden d) ---")
    for n in (8, 10):
        ninst, nshuf = N_BUDGET[n]
        seed = (MASTER_SEED + 333 * n + 7) & 0x7FFFFFFF
        res = H.hardened_gate(O.O_smuggle_godel_phi, n, n_instances=ninst, seed=seed,
                              n_shuffles=nshuf)
        res["_ok"] = (res["verdict"] == "FAIL_SMUGGLE")
        print("  n=%-2d  %s" % (n, _row("smuggle", res)))
        report["smuggle_native"].append({k: res[k] for k in (
            "n", "verdict", "auc", "shuffle_null_95", "random_fold_auc",
            "random_fold_null_95", "max_fold_delta", "smuggle_reason")})

    # ---- (c) standard sensitivity controls ----
    print("\n--- STANDARD SENSITIVITY CONTROLS (gate battery) ---")
    ctrls = [("reads_d", G.O_cheat_reads_d, "FAIL_SMUGGLE"),
             ("reads_sin", G.O_cheat_reads_sin, "FAIL_SMUGGLE"),
             ("useless_even", G.O_useless_even, "FAIL_CHANCE")]
    for n in (8, 10):
        ninst, nshuf = N_BUDGET[n]
        for ci, (name, Ofn, expected) in enumerate(ctrls):
            seed = (MASTER_SEED + 55 * n + ci) & 0x7FFFFFFF
            res = H.hardened_gate(Ofn, n, n_instances=ninst, seed=seed, n_shuffles=nshuf)
            res["_ok"] = (res["verdict"] == expected)
            print("  n=%-2d  %s  (expect %s)" % (n, _row(name, res), expected))
            report["controls"].append({"name": name, "n": n, "expected": expected,
                "verdict": res["verdict"], "auc": res["auc"],
                "random_fold_auc": res["random_fold_auc"],
                "max_fold_delta": res["max_fold_delta"],
                "matches": res["_ok"]})

    # ---- (d) PRICING ----
    print("\n--- PRICING: rank-1/log-space winding vs direct dense det-per-phi ---")
    print("  %-3s %-8s %-16s %-16s %-10s %-22s" % (
        "n", "N", "rank1_s/inst", "dense_s/inst", "speedup", "winding(num=closed=dense?)"))
    for n in (8, 10, 12, 14):
        seed = (MASTER_SEED + 17 * n) & 0x7FFFFFFF
        do_dense = (n <= 10)
        pr = price(n, seed, n_phi=64, reps=3, do_dense=do_dense)
        sp = pr["speedup_dense_over_rank1"]
        match = "num=closed:%s" % pr["W_match"]
        if do_dense:
            match += " dense=closed:%s" % pr["dense_matches_closed"]
        print("  %-3d %-8d %-16.4g %-16s %-10s %s" % (
            n, pr["N"], pr["rank1_s_per_inst"],
            ("%.4g" % pr["dense_s_per_inst"]) if do_dense else "n/a (2^n N^3)",
            ("%.1fx" % sp) if sp else "n/a", match))
        report["pricing"].append(pr)

    # ---- verdict roll-up ----
    prim_all_chance = all(p["verdict"] == "FAIL_CHANCE" for p in report["primary"])
    smug_all_caught = all(s["verdict"] == "FAIL_SMUGGLE" for s in report["smuggle_native"])
    ctrl_all_ok = all(c["matches"] for c in report["controls"])
    report["rollup"] = {"primary_all_FAIL_CHANCE": prim_all_chance,
                        "native_smuggle_all_caught": smug_all_caught,
                        "standard_controls_all_ok": ctrl_all_ok,
                        "outcome_class": ("iii_inherits_the_fold__FAIL_CHANCE"
                                          if prim_all_chance else "REAUDIT_REQUIRED")}
    with open(os.path.join(HERE, "godel_edge_phi_result.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("\n" + "=" * 92)
    print(" ROLLUP  primary_all_FAIL_CHANCE=%s  native_smuggle_caught=%s  controls_ok=%s"
          % (prim_all_chance, smug_all_caught, ctrl_all_ok))
    print(" OUTCOME CLASS: %s" % report["rollup"]["outcome_class"])
    print(" wrote godel_edge_phi_result.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
