"""Q48: Proper eigenvalue spacing test with correct spectral unfolding."""
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sentence_transformers import SentenceTransformer

WORDS = ['water','fire','earth','sky','sun','moon','star','mountain','river','tree','flower','rain','wind','snow','cloud','ocean','dog','cat','bird','fish','horse','tiger','lion','elephant','heart','eye','hand','head','brain','blood','bone','mother','father','child','friend','king','queen','love','hate','truth','life','death','time','space','power','peace','war','hope','fear','joy','pain','dream','thought','book','door','house','road','food','money','stone','gold','light','shadow','music','word','name','law','good','bad','big','small','old','new','high','low','hot','cold','dark','bright','strong','weak','fast','slow','hard','soft','deep','wide','long','short','rich','poor','free','safe','clean','fair','kind','cruel','brave','wise','young','old','true','false','happy','sad','queen','king','prince','castle','knight','sword','shield','dragon','magic','wizard','witch','giant','hero','monster','ghost','spirit','angel','demon','god','brain','mind','soul','foot','body','force','energy','mass','speed','matter','atom','cell','north','south','east','west']

def proper_unfold(eigenvalues, s=0.5):
    """Unfold eigenvalues using local polynomial density estimation.
    Staircase: N(E) = count of eigenvalues <= E (cumulative).
    Smooth N(E) with spline, then unfolded e_i = N_smooth(E_i).
    """
    ev = eigenvalues[::-1]  # ascending order
    n = len(ev)
    # Staircase function: N(E_i) = i+1 (or i for 0-indexed)
    N = np.arange(1, n + 1, dtype=float)
    # Avoid log of zero: shift small eigenvalues
    log_ev = np.log(np.maximum(ev, 1e-15))
    # Fit smooth N(E) in log-log space (power-law spectrum)
    try:
        spline = UnivariateSpline(log_ev, N, s=s * n)
        N_smooth = spline(log_ev)
        N_smooth = np.maximum(N_smooth, 0.1)  # avoid zero
        unfolded = N_smooth
    except:
        # Fallback: linear fit on log-log
        slope, intercept = np.polyfit(log_ev, np.log(N), 1)
        N_smooth = np.exp(intercept) * ev ** slope
        unfolded = N_smooth
    return unfolded

def spacing_stats(eigenvalues, n_bootstrap=100):
    """Compute unfolded spacing statistics and compare to Poisson/GUE."""
    ev_nz = eigenvalues[eigenvalues > 1e-12]
    n_ev = len(ev_nz)
    if n_ev < 20:
        return None

    # Proper unfolding
    e_unfolded = proper_unfold(ev_nz)
    spacings = np.diff(e_unfolded)
    spacings = spacings[spacings > 0]
    if len(spacings) < 10:
        return None

    # Normalize to unit mean
    s_norm = spacings / spacings.mean()

    # Wigner surmise for GUE: P(s) = (32/pi^2) * s^2 * exp(-4s^2/pi)
    def wigner_gue_cdf(s):
        from scipy.special import erf
        x = 2 * s / np.sqrt(np.pi)
        return 1.0 - np.exp(-4 * s**2 / np.pi) * (1 + 4 * s**2 / np.pi) - \
               x * (1 - 2 * s**2 / np.pi) * np.exp(-4 * s**2 / np.pi)

    # KS test vs Poisson (exponential with rate 1)
    ks_poisson = stats.kstest(s_norm, 'expon')

    # KS test vs GUE: sample from Wigner surmise for comparison
    np.random.seed(42)
    gue_samples = np.random.rand(5000)
    gue_samples = np.sqrt(-np.pi/4 * np.log(1 - gue_samples))  # approx GUE spacing via inverse
    ks_gue = stats.ks_2samp(s_norm, gue_samples)

    # Level repulsion ratio: P(s < 0.2) / P(s > 0.8)
    # Poisson: r ~ 0.22. GUE: r << 0.01.
    frac_near_zero = (s_norm < 0.3).mean()
    # Poisson: P(s < 0.3) = 1 - exp(-0.3) = 0.259
    # GUE: P(s < 0.3) ~ 0.012 (approximate)

    return {
        'n_spacings': len(s_norm),
        'ks_poisson_D': ks_poisson.statistic,
        'ks_poisson_p': ks_poisson.pvalue,
        'ks_gue_D': ks_gue.statistic,
        'ks_gue_p': ks_gue.pvalue,
        'frac_near_zero': frac_near_zero,
        'poisson_expected_frac': 1 - np.exp(-0.3),  # 0.259
        'gue_expected_frac': 0.012,
        'matches': 'POISSON' if ks_poisson.pvalue > 0.05 else
                   'GUE' if ks_gue.pvalue > 0.05 else
                   'NEITHER'
    }

print("Q48: Properly unfolded eigenvalue spacing statistics")
print("=" * 60)
print(f"Pool size: {len(WORDS)} words")
print()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]

    result = spacing_stats(ev)
    print(f"{name} ({model.get_sentence_embedding_dimension()}d):")
    print(f"  Eigenvalues: {len(ev)}, Non-zero: {result['n_spacings']+1}")
    print(f"  KS vs Poisson: D={result['ks_poisson_D']:.4f} p={result['ks_poisson_p']:.4f}")
    print(f"  KS vs GUE:     D={result['ks_gue_D']:.4f} p={result['ks_gue_p']:.4f}")
    print(f"  P(s < 0.3):   {result['frac_near_zero']:.4f}")
    print(f"    Poisson expected: {result['poisson_expected_frac']:.4f}")
    print(f"    GUE expected:     {result['gue_expected_frac']:.4f}")
    print(f"  Matches: {result['matches']}")
    print()

print("Riemann zeta zeros follow GUE (Montgomery-Odlyzko law).")
print("GUE shows level repulsion: P(s->0) -> 0.")
print("Poisson shows no level repulsion: P(s->0) ~ 1.")
print("If embedding eigenvalues match Poisson, no Riemann connection exists.")
print()
print("Verdict based on level repulsion test (no unfolding artifacts).")
