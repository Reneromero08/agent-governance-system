"""
Exp 50.2 - The Decodability Gradient (frontier piece).

Slides a problem family from abelian (Shor/Fourier-decodable) through dihedral
(the known hard wall, lattice-linked) to symmetric (deeply non-abelian), and
measures where the holographic/scalar readout's ability to recover the hidden
subgroup COLLAPSES.  Order parameter: normalized decodability

    D = (Phi_signal - Phi_null) / (1 - Phi_null)

from the 1-D-character channel (Phi_character), confirmed by the literal FFT
readout channel.  D -> 1 for abelian (decodable), D -> 0 for non-abelian.

The collapse is a QUESTION for Brick 3 / Mythos ("known non-abelian-HSP barrier,
or a frontier our scalar readout simply has not crossed?"), never a verdict.

Run:  python 50_2_decodability_gradient.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import hsp_family as hf
import decoder_lib as dl

LINES = []
def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def measure(inst, n_inst=40, seed=0):
    """Return dict of channel decodabilities + raw signal/null samples."""
    rng = np.random.default_rng(seed)
    sig_c, null_c, sig_f, null_f = [], [], [], []
    for _ in range(n_inst):
        v = inst.coset_grating(rng)
        r = inst.random_grating(rng)
        sig_c.append(inst.phi_character(v))
        null_c.append(inst.phi_character(r))
        sig_f.append(inst.phi_fft(v))
        null_f.append(inst.phi_fft(r))
    # label-shuffle null: destroy coset structure by permuting v
    shuf_c = []
    for _ in range(n_inst):
        v = inst.coset_grating(rng)
        rng.shuffle(v)
        shuf_c.append(inst.phi_character(v))

    def norm_D(sig, null):
        ms, mn = np.mean(sig), np.mean(null)
        return (ms - mn) / (1 - mn + 1e-12), ms, mn

    Dc, sc, nc = norm_D(sig_c, null_c)
    Df, sf, nf = norm_D(sig_f, null_f)
    Dshuf, _, _ = norm_D(shuf_c, null_c)
    return {
        "D_char": Dc, "D_fft": Df, "D_shuffle_null": Dshuf,
        "sig_c": np.array(sig_c), "null_c": np.array(null_c),
        "phi_sig": sc, "phi_null": nc,
    }


def boot_ci(sig, null, n_boot=2000, seed=1):
    rng = np.random.default_rng(seed)
    mn = np.mean(null)
    ds = []
    for _ in range(n_boot):
        s = rng.choice(sig, size=len(sig), replace=True)
        ds.append((s.mean() - mn) / (1 - mn + 1e-12))
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def main():
    log("=" * 84)
    log("EXP 50.2  -  THE DECODABILITY GRADIENT")
    log("  abelian (Shor-decodable)  ->  dihedral (hard wall)  ->  symmetric (deep non-abelian)")
    log("  order parameter D = (Phi_sig - Phi_null)/(1 - Phi_null) ; 1=decodable, 0=collapsed")
    log("=" * 84)

    ladder = hf.build_ladder()
    rows = []
    log("\n  group   |G|   |[G,G]|  d_max  abelianness   D_char [95%% CI]        D_fft   shuffle-null")
    for (name, gens, pts, d_max) in ladder:
        inst = hf.GroupInstance(name, gens, pts, d_max)
        m = measure(inst, n_inst=40, seed=7)
        lo, hi = boot_ci(m["sig_c"], m["null_c"])
        rows.append({
            "name": name, "order": inst.order, "comm": len(inst.comm), "d_max": d_max,
            "ab": inst.abelianness, "D_char": m["D_char"], "ci": (lo, hi),
            "D_fft": m["D_fft"], "D_shuf": m["D_shuffle_null"],
            "sig_c": m["sig_c"], "null_c": m["null_c"], "abelian": d_max == 1,
        })
        log("  %-6s %4d   %5d    %3d     %.3f        %.3f [%.3f, %.3f]    %.3f    %.3f"
            % (name, inst.order, len(inst.comm), d_max, inst.abelianness,
               m["D_char"], lo, hi, m["D_fft"], m["D_shuffle_null"]))

    abelian = [r for r in rows if r["abelian"]]
    nonab = [r for r in rows if not r["abelian"]]

    # ---- locate the collapse (first group, abelian-ordered, with D below midpoint)
    shelf_mean = np.mean([r["D_char"] for r in abelian])
    pole_mean = np.mean([r["D_char"] for r in nonab])
    thresh = 0.5 * (shelf_mean + pole_mean)
    wall = None
    for r in rows:  # ladder is already abelian-first
        if r["D_char"] < thresh:
            wall = r
            break
    log("\n  shelf mean D_char=%.3f   pole mean D_char=%.3f   collapse threshold=%.3f" % (shelf_mean, pole_mean, thresh))
    if wall:
        log("  >> COLLAPSE located at: %s  (|G|=%d, d_max %d, abelianness %.3f)  D_char=%.3f"
            % (wall["name"], wall["order"], wall["d_max"], wall["ab"], wall["D_char"]))

    # ---- statistics: shelf vs pole
    shelf_D = [r["D_char"] for r in abelian]
    pole_D = [r["D_char"] for r in nonab]
    d_eff = dl.cohen_d(shelf_D, pole_D)
    # permutation p that shelf D > pole D
    pooled = np.array(shelf_D + pole_D); na = len(shelf_D)
    rng = np.random.default_rng(2)
    obs = np.mean(shelf_D) - np.mean(pole_D)
    cnt = sum(1 for _ in range(10000) if (rng.shuffle(pooled) or (pooled[:na].mean() - pooled[na:].mean()) >= obs))
    p_sep = (cnt + 1) / 10001

    # ===================== GATES =====================
    gates = {}
    # G1 discrimination: abelian shelf high, non-abelian pole low, separated.
    # (permutation p has a ~8e-4 resolution floor at n=13 groups, so the decisive
    #  statistic is the effect size; p is reported at the 0.01 level.)
    g1 = (shelf_mean > 0.8) and (pole_mean < 0.4) and (abs(d_eff) > 2.0) and (p_sep < 0.01)
    gates["G1 shelf vs pole discrimination"] = (g1, "shelf=%.3f pole=%.3f p=%.4g Cohen d=%.2f" % (shelf_mean, pole_mean, p_sep, d_eff))

    # G2 knobs agree: wall is first non-abelian under BOTH abelianness and d_max ordering
    by_ab = sorted(rows, key=lambda r: (-r["ab"], r["d_max"]))
    by_dmax = sorted(rows, key=lambda r: (r["d_max"], -r["ab"]))
    first_nonab_ab = next(r for r in by_ab if not r["abelian"])
    first_nonab_dm = next(r for r in by_dmax if not r["abelian"])
    g2 = (wall is not None) and (first_nonab_ab["d_max"] == first_nonab_dm["d_max"] == 2) and (wall["d_max"] == 2)
    gates["G2 knobs agree (abelianness & d_max)"] = (g2, "collapse at d_max 1->2 under both orderings")

    # G3 scale-independence: at each shared |G|, cyclic decodable, dihedral collapsed
    g3 = True; g3_detail = []
    for order in (16, 24, 32, 48, 64):
        zc = next((r for r in abelian if r["order"] == order), None)
        dd = next((r for r in nonab if r["order"] == order and r["d_max"] == 2), None)
        if zc and dd:
            ok = zc["D_char"] > 0.8 and dd["D_char"] < 0.5
            g3 = g3 and ok
            g3_detail.append("|G|=%d Z=%.2f D=%.2f %s" % (order, zc["D_char"], dd["D_char"], "OK" if ok else "FAIL"))
    gates["G3 scale-independence (wall != |G| artifact)"] = (g3, " | ".join(g3_detail))

    # G4 robustness: re-measure shelf/pole with a different seed; collapse persists
    r2 = [measure(hf.GroupInstance(*[*g[:3], g[3]]), n_inst=40, seed=99) for g in ladder]
    shelf2 = np.mean([r2[i]["D_char"] for i, r in enumerate(rows) if r["abelian"]])
    pole2 = np.mean([r2[i]["D_char"] for i, r in enumerate(rows) if not r["abelian"]])
    g4 = (shelf2 > 0.8) and (pole2 < 0.4)
    gates["G4 robustness (seed)"] = (g4, "seed2 shelf=%.3f pole=%.3f" % (shelf2, pole2))

    # G5 null floor: label-shuffle destroys decodability everywhere
    max_shuf = max(abs(r["D_shuf"]) for r in rows)
    g5 = max_shuf < 0.3
    gates["G5 label-shuffle null floor"] = (g5, "max |shuffle-null D|=%.3f (want <0.3)" % max_shuf)

    log("\n" + "=" * 84)
    log("GATES")
    all_pass = True
    for nm, (ok, det) in gates.items():
        all_pass = all_pass and ok
        log("  [%s] %-44s  %s" % ("PASS" if ok else "FAIL", nm, det))
    log("=" * 84)

    verdict = "BOUNDED_AT_ABELIAN_HSP_WALL" if (all_pass and wall and wall["d_max"] == 2) else \
              ("NO_MEASUREMENT_GRADE_COLLAPSE" if not (wall and shelf_mean - pole_mean > 0.3) else "GRADIENT_PARTIAL")
    log("VERDICT: %s" % verdict)
    log("  (provisional, given weak-model arsenal) decodability collapses at the abelian->non-abelian")
    log("  boundary; whether this is the KNOWN non-abelian-HSP barrier or an uncrossed frontier of the")
    log("  scalar/holographic readout is the question handed to Brick 3 / Mythos.")

    # write located-wall summary for Brick 3
    if wall:
        import json
        (HERE / "located_wall.json").write_text(json.dumps({
            "wall_group": wall["name"], "order": wall["order"], "d_max": wall["d_max"],
            "abelianness": wall["ab"], "D_char_at_wall": wall["D_char"],
            "shelf_mean": shelf_mean, "pole_mean": pole_mean, "cohen_d": d_eff, "p_sep": p_sep,
            "verdict": verdict,
            "rows": [{"name": r["name"], "order": r["order"], "d_max": r["d_max"],
                      "abelianness": r["ab"], "D_char": r["D_char"], "ci": r["ci"], "D_fft": r["D_fft"]}
                     for r in rows],
        }, indent=2), encoding="utf-8")

    (HERE / "output.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
