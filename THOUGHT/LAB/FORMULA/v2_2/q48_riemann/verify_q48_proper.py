"""Q48: Proper eigenvalue spacing test with 777 words + Gram matrix."""
import sys; sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sentence_transformers import SentenceTransformer
from large_anchor_generator import ANCHOR_1024

A1024 = ANCHOR_1024  # type: ignore
WORDS = list(A1024)
n_words = len(WORDS)
print(f"Words: {n_words}")

def proper_unfold(eigenvalues):
    """Staircase unfolding with local spline fit."""
    ev = np.sort(eigenvalues)
    n = len(ev)
    idx = np.arange(1, n + 1, dtype=float)
    # Fit log(N) vs log(E) — power-law spectrum
    ok = ev > 1e-15
    log_e = np.log(ev[ok])
    log_N = np.log(idx[ok])
    try:
        spline = UnivariateSpline(log_e, log_N, s=n*0.001, k=3)
        N_smooth = np.exp(spline(log_e))
    except:
        slope, intercept = np.polyfit(log_e, log_N, 1)
        N_smooth = np.exp(intercept) * ev[ok] ** slope
    N_smooth = np.maximum(N_smooth, 0.1)
    return N_smooth

def test_model(model_id, name):
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)

    # Gram matrix: NxN, all eigenvalues available
    gram = embs @ embs.T
    ev_gram = np.sort(np.linalg.eigvalsh(gram))

    # Only use positive eigenvalues (rank = min(N-1, D))
    ev_pos = ev_gram[ev_gram > 1e-10]

    # Unfold
    unfolded = proper_unfold(ev_pos)
    spacings = np.diff(unfolded)
    s_norm = spacings / spacings.mean()

    # KS tests
    ks_p = stats.kstest(s_norm, "expon")
    # GUE: compare to Wigner surmise via sampling
    np.random.seed(42)
    gue_sample = np.sort(np.random.rand(len(s_norm) * 20))
    gue_sample = np.sqrt(-np.pi / 4 * np.log(1 - gue_sample + 1e-15))
    gue_sample = gue_sample / gue_sample.mean()
    ks_g = stats.ks_2samp(s_norm, gue_sample)

    # Level repulsion
    frac_small = (s_norm < 0.3).mean()
    frac_med = (s_norm < 0.5).mean()

    # Poisson expected: 1-exp(-0.3)=0.259, 1-exp(-0.5)=0.393
    # GUE expected (Wigner): P(s<0.3)~0.012, P(s<0.5)~0.083

    # Pair correlation function test (Montgomery conjecture)
    # g(r) = 1 - (sin(pi*r)/(pi*r))^2 for GUE
    # Test: does g(r) follow GUE or Poisson (g(r)=1)?

    best = "GUE" if ks_g.pvalue > ks_p.pvalue else "POISSON"
    if ks_p.pvalue < 0.01 and ks_g.pvalue < 0.01:
        best = "NEITHER"

    print(f"\n{name} ({model.get_sentence_embedding_dimension()}d):")
    print(f"  N words: {n_words}, N eigenvalues: {len(ev_pos)}, N spacings: {len(s_norm)}")
    print(f"  Mean spacing: {spacings.mean():.4f}, std: {spacings.std():.4f}")
    print(f"  KS Poisson: D={ks_p.statistic:.4f} p={ks_p.pvalue:.4e}")
    print(f"  KS GUE:     D={ks_g.statistic:.4f} p={ks_g.pvalue:.4e}")
    print(f"  P(s<0.3):  {frac_small:.4f} (Poisson=0.259, GUE=0.012)")
    print(f"  P(s<0.5):  {frac_med:.4f} (Poisson=0.393, GUE=0.083)")
    print(f"  Best match: {best}")

    return {
        "name": name,
        "n_spacings": len(s_norm),
        "ks_poisson_D": float(ks_p.statistic),
        "ks_poisson_p": float(ks_p.pvalue),
        "ks_gue_D": float(ks_g.statistic),
        "ks_gue_p": float(ks_g.pvalue),
        "frac_small": float(frac_small),
        "frac_med": float(frac_med),
        "best": best,
    }

results = []
for mid, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    results.append(test_model(mid, name))

print(f"\n{'='*60}")
print("VERDICT")
for r in results:
    if r["best"] == "GUE":
        print(f"  {r['name']}: GUE -> Riemann zeta connection POSSIBLE")
    elif r["best"] == "POISSON":
        print(f"  {r['name']}: Poisson -> NO Riemann connection (v1 was right)")
    else:
        print(f"  {r['name']}: NEITHER -> NO Riemann connection")
        print(f"    P(s<0.3)={r['frac_small']:.3f} is not GUE ({0.012}) and not Poisson ({0.259})")
        print(f"    The spacing distribution is specific to the rank-deficient Gram matrix")
