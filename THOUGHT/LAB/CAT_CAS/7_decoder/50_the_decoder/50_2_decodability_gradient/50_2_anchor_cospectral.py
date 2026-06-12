"""
Exp 50.2 anchor - the cospectral hard case (the binary probe Exp 31 left unreported).

Shrikhande and Rook(4x4) are both SRG(16,6,2,2): they have IDENTICAL eigenvalue
spectra (cospectral) but are NON-isomorphic. A purely spectral / holographic
signature therefore CANNOT distinguish them - which pins the deep-non-abelian pole
of the gradient: the holographic readout is spectrum-bounded, exactly as it is
1-D-character-bounded in the HSP family. A non-spectral invariant (4-clique count)
separates them, proving they really are different.

Run:  python 50_2_anchor_cospectral.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import decoder_lib as dl

LINES = []
def log(m=""):
    print(m); LINES.append(str(m))


def rook_4x4():
    A = np.zeros((16, 16), int)
    for i in range(16):
        ri, ci = divmod(i, 4)
        for j in range(16):
            if i == j:
                continue
            rj, cj = divmod(j, 4)
            if ri == rj or ci == cj:
                A[i, j] = 1
    return A


def shrikhande():
    conn = [(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)]
    A = np.zeros((16, 16), int)
    idx = lambda a, b: (a % 4) * 4 + (b % 4)
    for a in range(4):
        for b in range(4):
            for da, db in conn:
                A[idx(a, b), idx(a + da, b + db)] = 1
    return A


def holo_signature(A):
    """Spectral / holographic signature: sorted eigenvalues + participation dim."""
    ev = np.sort(np.linalg.eigvalsh(A.astype(float)))[::-1]
    return ev, dl.participation_dimension(np.abs(ev))


def count_4cliques(A):
    n = len(A)
    c = 0
    for q in combinations(range(n), 4):
        if all(A[i, j] for i, j in combinations(q, 2)):
            c += 1
    return c


def main():
    log("=" * 78)
    log("EXP 50.2 ANCHOR  -  cospectral hard case (Shrikhande vs Rook 4x4)")
    log("=" * 78)
    R, S = rook_4x4(), shrikhande()
    log("  both 6-regular on 16 vertices: Rook deg=%d  Shrikhande deg=%d" % (R.sum(1)[0], S.sum(1)[0]))

    evR, pR = holo_signature(R)
    evS, pS = holo_signature(S)
    spec_dist = float(np.linalg.norm(evR - evS))
    log("\n  eigenvalues (rounded):")
    log("    Rook       : %s" % np.round(evR, 3).tolist())
    log("    Shrikhande : %s" % np.round(evS, 3).tolist())
    log("  holographic signature distance (spectral)         = %.2e   participation R=%.3f S=%.3f"
        % (spec_dist, pR, pS))

    cR, cS = count_4cliques(R), count_4cliques(S)
    log("\n  non-spectral witness (4-clique count): Rook=%d  Shrikhande=%d  -> NON-ISOMORPHIC=%s"
        % (cR, cS, cR != cS))

    cospectral = spec_dist < 1e-6
    nonisomorphic = cR != cS
    holo_fails = cospectral and nonisomorphic
    log("\n" + "=" * 78)
    log("  cospectral (spectral signature identical)         : %s" % cospectral)
    log("  genuinely non-isomorphic (clique witness)         : %s" % nonisomorphic)
    log("  => holographic/spectral readout CANNOT distinguish: %s" % holo_fails)
    verdict = "SPECTRUM_BOUNDED_CONFIRMED" if holo_fails else "ANCHOR_INCONCLUSIVE"
    log("VERDICT: %s" % verdict)
    log("  Anchors the deep-non-abelian pole: the holographic readout fails the cospectral")
    log("  graph-iso case exactly as it fails non-abelian HSP - it is spectrum-bounded. A")
    log("  reframe that crosses this is the question for Brick 3 / Mythos.")

    (HERE / "cospectral_anchor.json").write_text(
        '{"spec_dist": %.3e, "cliques_rook": %d, "cliques_shrik": %d, "holo_fails": %s, "verdict": "%s"}'
        % (spec_dist, cR, cS, str(holo_fails).lower(), verdict), encoding="utf-8")
    (HERE / "output_cospectral.txt").write_text("\n".join(LINES), encoding="utf-8")
    sys.exit(0 if holo_fails else 1)


if __name__ == "__main__":
    main()
