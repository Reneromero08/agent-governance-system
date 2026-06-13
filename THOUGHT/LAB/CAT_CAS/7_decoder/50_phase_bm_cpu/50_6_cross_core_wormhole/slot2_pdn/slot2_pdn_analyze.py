#!/usr/bin/env python3
"""
slot2_pdn_analyze.py -- matched-null scorer for the SLOT 2 PDN driven-lock-in CSV.

Reuses the VERIFIED pdn_catalytic_tape_fix_probe.py discriminator on the LIVE rail
lock-in vectors:
  - distinct-weight codebook {4,5,6,7}, min pairwise Hamming 7 (read from the CSV
    header comments so analyzer and harness agree by construction);
  - per-symbol the 12-bin complex lock-in vector z = b##_I + i*b##_Q;
  - de-rotate z by the recovered global (carrier) phase, L2-normalize (kills the 1/f
    amplitude drift), classify on the FULL normalized vector via mode centroids
    (trained on EVEN real trials), AND compute the rho energy-concentration matched
    null (rho = max_k |<zhat, h_k>|^2);
  - real/pseudo/wrong matched-null families scored exactly as
    analyze_cache_hologram_matched_nulls.py conventions:
      real_accuracy, real_mode_floor, real_vs_pseudo accuracy/reject floors,
      pseudo_declared_match, wrong_actual_match, wrong_declared_match;
  - relational PHASE: differential theta (consecutive real symbols) correlation,
    true vs shuffled null.

CSV schema (matrix): family,declared_mode,actual_mode,trial,hash_restored,theta_idx,
  b00_I,b00_Q,...,b11_I,b11_Q,fl00..fl11
Header comments carry tones + codeword_0..3 + seed.

Usage: slot2_pdn_analyze.py input.csv output.json
ASCII only.
"""
import csv
import json
import sys
import numpy as np

MODE_NAMES = ["basis", "rotation", "residual", "mini"]
MODE_IDX = {n: i for i, n in enumerate(MODE_NAMES)}
PHASE_LEVELS = 8


def parse_header(path):
    nbin = None
    codewords = {}
    seed = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "codeword_" in line:
                # "# codeword_0=+1,-1,..."
                lhs, rhs = line[1:].split("=", 1)
                m = int(lhs.strip().split("_")[1])
                codewords[m] = [int(x) for x in rhs.strip().split(",")]
            if "nbin=" in line:
                for tok in line.replace("#", " ").split():
                    if tok.startswith("nbin="):
                        nbin = int(tok.split("=")[1])
                    if tok.startswith("seed="):
                        seed = int(tok.split("=")[1])
    if nbin is None:
        nbin = len(next(iter(codewords.values())))
    code = np.array([codewords[m] for m in range(4)], dtype=float)
    return nbin, code, seed


def load_rows(path, nbin):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = None
        for raw in reader:
            if not raw:
                continue
            if raw[0].startswith("#"):
                continue
            if header is None:
                header = raw
                col = {name: k for k, name in enumerate(header)}
                continue
            d = {}
            d["family"] = raw[col["family"]]
            d["declared"] = MODE_IDX[raw[col["declared_mode"]]]
            d["actual"] = MODE_IDX[raw[col["actual_mode"]]]
            d["trial"] = int(raw[col["trial"]])
            d["hash_restored"] = int(raw[col["hash_restored"]])
            d["theta_idx"] = int(raw[col["theta_idx"]])
            z = np.zeros(nbin, dtype=complex)
            for b in range(nbin):
                z[b] = float(raw[col[f"b{b:02d}_I"]]) + 1j * float(raw[col[f"b{b:02d}_Q"]])
            d["z"] = z
            rows.append(d)
    return rows


def feats(z, code, HN):
    g = np.angle(np.sum(z))
    zr = z * np.exp(-1j * g)
    n = np.linalg.norm(zr) + 1e-12
    zhat = zr / n
    corr = np.abs(HN @ zhat)
    rho = float(np.max(corr) ** 2)
    fvec = np.concatenate([zhat.real, zhat.imag])
    mhat = int(np.argmax(corr))
    theta_hat = float(np.angle(code[mhat] @ z))
    return fvec, rho, mhat, theta_hat


def circ_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0:
        return 0.0
    ca, cb = np.exp(1j * a), np.exp(1j * b)
    return float(np.real(np.sum(ca * np.conj(cb))) / len(a))


def analyze(path):
    nbin, code, seed = parse_header(path)
    HN = code / np.sqrt(nbin)
    rows = load_rows(path, nbin)
    for r in rows:
        fvec, rho, mhat, theta_hat = feats(r["z"], code, HN)
        r["fvec"] = fvec; r["rho"] = rho; r["mhat"] = mhat; r["theta_hat"] = theta_hat

    # preamble (trial<0) + even real trials train centroids
    train_real = [r for r in rows if r["family"] == "real" and (r["trial"] < 0 or r["trial"] % 2 == 0)]
    cent = {}
    for m in range(4):
        sub = [r["fvec"] for r in train_real if r["declared"] == m]
        cent[m] = np.mean(sub, axis=0) if sub else np.zeros(2 * nbin)

    def predict(fv):
        return int(min(range(4), key=lambda m: np.linalg.norm(fv - cent[m])))

    test = [r for r in rows if r["trial"] >= 0 and r["trial"] % 2 == 1]

    # real accuracy + per-mode floor
    rc = rt = 0
    bym = {m: [0, 0] for m in range(4)}
    for r in test:
        if r["family"] != "real":
            continue
        rt += 1
        ok = predict(r["fvec"]) == r["declared"]
        rc += ok
        bym[r["declared"]][1] += 1
        bym[r["declared"]][0] += ok
    real_acc = rc / max(rt, 1)
    real_floor = min((c / t if t else 0.0) for c, t in bym.values())

    # wrong: read ACTUAL, not declared
    wa = wd = wt = 0
    for r in test:
        if r["family"] != "wrong":
            continue
        wt += 1
        pr = predict(r["fvec"])
        wa += (pr == r["actual"])
        wd += (pr == r["declared"])
    wrong_actual = wa / max(wt, 1)
    wrong_declared = wd / max(wt, 1)

    # pseudo declared match
    pdm = pt = 0
    for r in test:
        if r["family"] != "pseudo":
            continue
        pt += 1
        pdm += (predict(r["fvec"]) == r["declared"])
    pseudo_declared = pdm / max(pt, 1)

    # real-vs-pseudo via rho threshold (5th percentile of train real rho)
    rho_tr = np.array([r["rho"] for r in train_real]) if train_real else np.array([0.0])
    thr = float(np.percentile(rho_tr, 5))
    rvp_acc, rvp_rej = [], []
    for m in range(4):
        te_real = [r["rho"] for r in test if r["family"] == "real" and r["declared"] == m]
        te_ps = [r["rho"] for r in test if r["family"] == "pseudo" and r["mhat"] == m]
        if not te_real or not te_ps:
            continue
        acc = (np.sum([x >= thr for x in te_real]) + np.sum([x < thr for x in te_ps])) \
            / (len(te_real) + len(te_ps))
        rvp_acc.append(float(acc))
        rvp_rej.append(float(np.mean([x < thr for x in te_ps])))
    rvp_floor = min(rvp_acc) if rvp_acc else 0.0
    rej_floor = min(rvp_rej) if rvp_rej else 0.0

    # relational phase: differential theta over consecutive real symbols
    real_seq = [r for r in rows if r["family"] == "real"]
    real_seq.sort(key=lambda r: (r["trial"] if r["trial"] >= 0 else -1000 + r["trial"]))
    th_true = np.array([2 * np.pi * r["theta_idx"] / PHASE_LEVELS for r in real_seq])
    th_hat = np.array([r["theta_hat"] for r in real_seq])
    d_true, d_hat = np.diff(th_true), np.diff(th_hat)
    corr_true = circ_corr(d_hat, d_true)
    rng = np.random.default_rng((seed or 0) + 99)
    sh = rng.permutation(len(d_true)) if len(d_true) else np.array([], dtype=int)
    corr_null = circ_corr(d_hat, d_true[sh]) if len(d_true) else 0.0

    all_restore = all(r["hash_restored"] == 1 for r in rows)

    gates = {
        "all_rows_restore": bool(all_restore),
        "real_accuracy_ge_0_60": real_acc >= 0.60,
        "real_mode_floor_ge_0_45": real_floor >= 0.45,
        "real_vs_pseudo_floor_ge_0_95": rvp_floor >= 0.95,
        "pseudo_reject_floor_ge_0_95": rej_floor >= 0.95,
        "pseudo_declared_match_le_0_35": pseudo_declared <= 0.35,
        "wrong_actual_match_ge_0_60": wrong_actual >= 0.60,
        "wrong_declared_match_le_0_20": wrong_declared <= 0.20,
        "phase_recovered_gt_0_30": (corr_true - corr_null) > 0.30,
    }
    verdict = "PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS" if all(gates.values()) else "PHASE4B_PDN_PARTIAL"

    return {
        "input": path,
        "nbin": nbin,
        "seed": seed,
        "rows": len(rows),
        "test_rows": len(test),
        "minham": int(min(int(np.sum(code[i] != code[j]))
                          for i in range(4) for j in range(i + 1, 4))),
        "real_accuracy": real_acc,
        "real_mode_floor": real_floor,
        "real_vs_pseudo_floor": rvp_floor,
        "pseudo_reject_floor": rej_floor,
        "pseudo_declared_match": pseudo_declared,
        "wrong_actual_match": wrong_actual,
        "wrong_declared_match": wrong_declared,
        "rho_threshold": thr,
        "phase_corr_true": corr_true,
        "phase_corr_null": corr_null,
        "phase_delta": corr_true - corr_null,
        "all_rows_restore": bool(all_restore),
        "gates": gates,
        "verdict": verdict,
    }


def main():
    if len(sys.argv) != 3:
        print("usage: slot2_pdn_analyze.py input.csv output.json", file=sys.stderr)
        return 2
    out = analyze(sys.argv[1])
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(out["verdict"])
    print("rows=%d real_acc=%.4f real_floor=%.4f rvp=%.4f rej=%.4f ps_decl=%.4f "
          "wr_act=%.4f wr_decl=%.4f ph[T=%.3f N=%.3f d=%.3f] restore=%s"
          % (out["rows"], out["real_accuracy"], out["real_mode_floor"],
             out["real_vs_pseudo_floor"], out["pseudo_reject_floor"],
             out["pseudo_declared_match"], out["wrong_actual_match"],
             out["wrong_declared_match"], out["phase_corr_true"],
             out["phase_corr_null"], out["phase_delta"], out["all_rows_restore"]))
    return 0 if all(out["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
