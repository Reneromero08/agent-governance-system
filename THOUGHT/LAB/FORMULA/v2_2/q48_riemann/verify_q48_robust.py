"""Q48: Robust test — multiple unfolding methods, correct eigenvalue count."""
import sys; sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sentence_transformers import SentenceTransformer
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)  # type: ignore
n = len(WORDS)

def unfold_polynomial(ev, deg=4):
    """Polynomial fit to staircase in log-log."""
    ev_pos = np.sort(ev[ev > 1e-15])
    N_stair = np.arange(1, len(ev_pos) + 1, dtype=float)
    log_e, log_N = np.log(ev_pos), np.log(N_stair)
    coeffs = np.polyfit(log_e, log_N, deg)
    N_smooth = np.exp(np.polyval(coeffs, log_e))
    return np.maximum(N_smooth, 0.1)

def unfold_spline(ev, s_frac=0.01):
    """Spline fit to staircase in log-log."""
    ev_pos = np.sort(ev[ev > 1e-15])
    N_stair = np.arange(1, len(ev_pos) + 1, dtype=float)
    log_e, log_N = np.log(ev_pos), np.log(N_stair)
    n_ev = len(ev_pos)
    spline = UnivariateSpline(log_e, log_N, s=s_frac * n_ev, k=3)
    N_smooth = np.exp(spline(log_e))
    return np.maximum(N_smooth, 0.1)

def test_spacings(ev_unfolded, label):
    """KS tests vs Poisson and GUE."""
    s = np.diff(ev_unfolded)
    s_norm = s / s.mean()

    ks_p = stats.kstest(s_norm, "expon")
    np.random.seed(42)
    gue = np.sqrt(-np.pi / 4 * np.log(1 - np.random.rand(len(s_norm) * 50) + 1e-15))
    gue = gue / gue.mean()
    ks_g = stats.ks_2samp(s_norm, gue)

    frac_03 = (s_norm < 0.3).mean()
    return {
        "n": len(s_norm), "ks_p_D": ks_p.statistic, "ks_p_p": ks_p.pvalue,
        "ks_g_D": ks_g.statistic, "ks_g_p": ks_g.pvalue, "frac_small": frac_03,
    }


for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()

    # Gram matrix eigenvalues — USE ONLY TOP D (rank-limited by embedding dim)
    gram = embs @ embs.T
    ev_all = np.sort(np.linalg.eigvalsh(gram))
    max_rank = min(n - 1, D)
    ev_signal = ev_all[-max_rank:]  # top max_rank eigenvalues (highest)
    ev_signal = ev_signal[ev_signal > 1e-10]

    print(f"\n{'='*60}")
    print(f"{name} (D={D}, N={n}, max_rank={max_rank})")
    print(f"Signal eigenvalues: {len(ev_signal)}")
    print(f"{'Method':<20s} {'n_sp':>6s} {'KS Pois D':>10s} {'KS Pois p':>10s} {'KS GUE D':>10s} {'KS GUE p':>10s} {'P(s<0.3)':>10s}")

    methods = {
        "spline (s=0.1%)": lambda e: unfold_spline(e, 0.001),
        "spline (s=1%)":   lambda e: unfold_spline(e, 0.01),
        "spline (s=10%)":  lambda e: unfold_spline(e, 0.1),
        "poly (deg=2)":    lambda e: unfold_polynomial(e, 2),
        "poly (deg=4)":    lambda e: unfold_polynomial(e, 4),
        "poly (deg=6)":    lambda e: unfold_polynomial(e, 6),
    }

    for method_name, fn in methods.items():
        try:
            unfolded = fn(ev_signal)
            r = test_spacings(unfolded, method_name)
            gue_rej = "GUE REJ" if r["ks_g_p"] < 0.05 else "gue ok"
            print(f"{method_name:<20s} {r['n']:>6d} {r['ks_p_D']:>10.4f} {r['ks_p_p']:>10.2e} {r['ks_g_D']:>10.4f} {r['ks_g_p']:>10.2e} {r['frac_small']:>10.4f}  {gue_rej}")
        except Exception as e:
            print(f"{method_name:<20s} FAILED: {e}")

print(f"\n{'='*60}")
print("EXPECTED VALUES:")
print(f"  Poisson: P(s<0.3) = {1 - np.exp(-0.3):.4f}")
print(f"  GUE:     P(s<0.3) ~ 0.012")
print(f"  GUE p>0.05 means 'cannot reject GUE'")
print(f"  Consistent across all 6 unfolding methods = robust")
print(f"  Inconsistent across methods = unfolding-sensitive (not robust)")
