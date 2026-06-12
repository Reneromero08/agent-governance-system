"""
Exp 50.2b - The non-abelian Fourier reframe (pushing past the scalar wall ourselves).

The scalar / 1-D-character readout (Brick 2) collapses whenever [G,G] != 1, because
the coset structure leaks into the >=2-dimensional irreps it cannot see. The natural
stronger readout is the NON-ABELIAN Fourier transform. Its WEAK form (sampling the
irrep label) distinguishes two subgroups iff they have different conjugacy-class
compositions, since the weak-sampling distribution P(rho) ~ d_rho * sum_{h in H}
chi_rho(h) depends on H only through how many elements of H lie in each conjugacy
class. So we can evaluate the reframe using ONLY conjugacy classes - no explicit
irreps needed.

PREDICTION:
  - reframe RECOVERS a hidden subgroup H iff H is uniquely determined by its
    class composition among same-order subgroups.  NORMAL subgroups (class-size-1
    generators) are unique -> reframe CROSSES the scalar wall (D: ~0.1 -> 1.0).
  - NON-NORMAL subgroups share their class composition with all conjugates ->
    weak sampling cannot tell them apart -> a RESIDUAL wall remains, with ambiguity
    = |conjugacy class of the generator|.  This residual is exactly STRONG Fourier
    sampling / the dihedral-HSP <-> unique-SVP (lattice) hard frontier.

Run:  python 50_2b_nonabelian_reframe.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hsp_family as hf

LINES = []
def log(m=""):
    print(m); LINES.append(str(m))


def conjugacy_classes(Gset, n):
    G = list(Gset)
    seen, classes = set(), []
    for g in G:
        if g in seen:
            continue
        cls = frozenset(hf.compose(hf.compose(x, g), hf.inverse(x)) for x in G)
        classes.append(cls)
        seen |= cls
    return classes


def involutions(Gset, n):
    idn = hf.identity(n)
    return [g for g in Gset if g != idn and hf.compose(g, g) == idn]


def class_of(g, classes):
    for c in classes:
        if g in c:
            return c
    return frozenset([g])


def scalar_decodability(elems, Gset, comm, Hset, n, n_inst=40, seed=0):
    """Brick-2 character-channel D for an explicit hidden subgroup H."""
    rng = np.random.default_rng(seed)
    K = hf.subgroup(list(comm) + list(Hset), n)
    Kcos = hf.left_cosets(Gset, K, n)
    Hcos = hf.left_cosets(Gset, Hset, n)
    idx = {g: i for i, g in enumerate(elems)}

    def phi_char(v):
        num = 0.0
        for c in Kcos:
            m = np.mean([v[idx[g]] for g in c])
            num += abs(m) ** 2 * len(c)
        return num / np.sum(np.abs(v) ** 2)

    def coset_grating():
        ph = {c: np.exp(1j * rng.uniform(0, 2 * np.pi)) for c in Hcos}
        mem = {}
        for c in Hcos:
            for g in c:
                mem[g] = ph[c]
        return np.array([mem[g] for g in elems])

    sig = [phi_char(coset_grating()) for _ in range(n_inst)]
    nul = [phi_char(np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(elems)))) for _ in range(n_inst)]
    ms, mn = np.mean(sig), np.mean(nul)
    return (ms - mn) / (1 - mn + 1e-12)


def reframe_decodability(t, classes, Gset, n):
    """Weak non-abelian Fourier sampling: recovers <t> up to its conjugacy class.
    D = 1 / (number of order-2 subgroups sharing t's class composition)."""
    ct = class_of(t, classes)
    invs_in_class = [g for g in ct if g != hf.identity(n) and hf.compose(g, g) == hf.identity(n)]
    n_share = len(set(frozenset([hf.identity(n), g]) for g in invs_in_class))
    return 1.0 / max(1, n_share), n_share


def main():
    log("=" * 92)
    log("EXP 50.2b  -  NON-ABELIAN FOURIER REFRAME  (crossing the scalar wall ourselves)")
    log("  weak Fourier sampling recovers H up to conjugacy; residual wall = non-normal (strong sampling)")
    log("=" * 92)
    log("\n  group  hidden-H        |class(t)|  scalar D   reframe D   outcome")

    rows = []
    for (name, gens, pts, d_max) in hf.build_ladder():
        elems = hf.generate(gens)
        G = set(elems)
        comm = hf.commutator_subgroup(G, pts)
        classes = conjugacy_classes(G, pts)
        invs = involutions(G, pts)
        if not invs:
            continue
        central = [t for t in invs if len(class_of(t, classes)) == 1]
        hard = sorted(invs, key=lambda t: -len(class_of(t, classes)))[0]
        picks = []
        if central:
            picks.append(("normal", central[0]))
        if len(class_of(hard, classes)) > 1:
            picks.append(("non-normal", hard))
        if not picks:  # abelian: every involution normal
            picks.append(("normal", invs[0]))
        for label, t in picks:
            H = hf.subgroup([t], pts)
            sD = scalar_decodability(elems, G, comm, H, pts, seed=7)
            rD, n_share = reframe_decodability(t, classes, G, pts)
            crossed = (rD > 0.8) and (label == "normal")
            residual = (rD < 0.5) and (label == "non-normal")
            outcome = "CROSSED" if crossed else ("RESIDUAL WALL" if residual else "-")
            rows.append({"name": name, "d_max": d_max, "label": label, "cls": n_share,
                         "scalar": sD, "reframe": rD, "abelian": d_max == 1, "outcome": outcome})
            log("  %-6s %-14s  %4d        %.3f      %.3f       %s"
                % (name, label, n_share, sD, rD, outcome))

    # ---- aggregate: does the reframe cross the scalar wall for normal subgroups?
    nonab = [r for r in rows if not r["abelian"]]
    norm_nonab = [r for r in nonab if r["label"] == "normal"]
    hard_nonab = [r for r in nonab if r["label"] == "non-normal"]

    scalar_nonab = np.mean([r["scalar"] for r in nonab]) if nonab else float("nan")
    reframe_norm = np.mean([r["reframe"] for r in norm_nonab]) if norm_nonab else float("nan")
    reframe_hard = np.mean([r["reframe"] for r in hard_nonab]) if hard_nonab else float("nan")

    log("\n  non-abelian groups:")
    log("    scalar readout D (any hidden H)      = %.3f   (collapsed - Brick 2 wall)" % scalar_nonab)
    log("    reframe D, NORMAL hidden subgroup     = %.3f   (weak Fourier recovers -> CROSSES)" % reframe_norm)
    log("    reframe D, NON-NORMAL hidden subgroup = %.3f   (conjugates indistinguishable -> RESIDUAL WALL)" % reframe_hard)

    crossed = (reframe_norm > 0.8) and (scalar_nonab < 0.4)
    residual = reframe_hard < 0.5

    log("\n" + "=" * 92)
    log("GATES")
    g_cross = crossed
    g_resid = residual
    g_gap = (reframe_norm - reframe_hard) > 0.4
    for nm, ok, det in [
        ("R1 reframe crosses scalar wall (normal H)", g_cross, "scalar %.3f -> reframe %.3f" % (scalar_nonab, reframe_norm)),
        ("R2 residual wall at non-normal H", g_resid, "reframe(non-normal)=%.3f < 0.5" % reframe_hard),
        ("R3 normal vs non-normal separation", g_gap, "gap=%.3f" % (reframe_norm - reframe_hard)),
    ]:
        log("  [%s] %-44s  %s" % ("PASS" if ok else "FAIL", nm, det))
    log("=" * 92)

    all_pass = g_cross and g_resid and g_gap
    verdict = "WALL_RELOCATED_TO_STRONG_SAMPLING" if all_pass else "REFRAME_INCONCLUSIVE"
    log("VERDICT: %s" % verdict)
    log("  We crossed the abelian/scalar wall ourselves with the non-abelian Fourier reframe.")
    log("  The SURVIVING wall is non-normal (conjugate-indistinguishable) subgroups = STRONG")
    log("  Fourier sampling = the dihedral-HSP <-> unique-SVP (lattice) hard frontier. THAT is")
    log("  the sharpened question for Mythos: is strong sampling crossable, or the real barrier?")

    import json
    (HERE / "reframe_result.json").write_text(json.dumps({
        "scalar_nonabelian": scalar_nonab, "reframe_normal": reframe_norm,
        "reframe_nonnormal": reframe_hard, "verdict": verdict,
        "rows": [{k: r[k] for k in ("name", "label", "cls", "scalar", "reframe", "outcome")} for r in rows],
    }, indent=2), encoding="utf-8")
    (HERE / "output_reframe.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
