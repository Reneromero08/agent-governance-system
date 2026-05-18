"""Q32 integrity: proper causal test with 100+ clusters."""
import sys
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS = list(ANCHOR_1024)

def compute_c_sem(emb_set, indices):
    x = emb_set[indices]
    n = len(indices)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(x[i]).dot(x[j])
            H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H)
    ev = np.maximum(ev, 1e-15); ev = ev / ev.sum()
    sigma = 1.0 / max(ev.sum()**2 / (ev**2).sum(), 1e-10)
    nabla = -np.sum(ev * np.log(ev + 1e-15))
    c_sem = np.sqrt(max(sigma, 1e-10) / max(nabla, 1e-10))
    # Also compute mean pairwise cosine (real, for comparison)
    x_real = np.real(x)
    cos_mat = x_real @ x_real.T
    mean_cos = cos_mat[np.tril_indices(n, k=-1)].mean()
    return c_sem, sigma, nabla, mean_cos

for mid, name in [("all-MiniLM-L6-v2","MiniLM")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    proj = (embs - embs.mean(axis=0)) @ evecs[:, :96]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1; proj = proj / norms
    z = hilbert(proj, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

    # Permuted complex
    z_perm = z.copy()
    for d in range(z_perm.shape[1]):
        np.random.shuffle(z_perm[:, d])

    # Generate 200 random clusters of size 8
    np.random.seed(42)
    real_cs, real_mcs = [], []
    perm_cs, perm_mcs = [], []

    for trial in range(200):
        idx = np.random.choice(len(WORDS), 8, replace=False)
        rcs, _, _, rmc = compute_c_sem(z, idx)
        pcs, _, _, _ = compute_c_sem(z_perm, idx)
        real_cs.append(rcs); real_mcs.append(rmc)
        perm_cs.append(pcs)

    real_cs = np.array(real_cs); perm_cs = np.array(perm_cs); real_mcs = np.array(real_mcs)

    # Test 1: does permuting reduce c_sem?
    delta = real_cs - perm_cs
    t, p = stats.ttest_1samp(delta, 0)
    effect = delta.mean() / real_cs.mean() * 100
    print(f"{name} K=96 — 200 random clusters:")
    print(f"  Real c_sem: {real_cs.mean():.4f}+/-{real_cs.std():.4f}")
    print(f"  Perm c_sem: {perm_cs.mean():.4f}+/-{perm_cs.std():.4f}")
    print(f"  Delta: {delta.mean():+.4f} ({effect:+.1f}%)")
    print(f"  Paired t-test: t={t:.2f} p={p:.6f}")
    print(f"  Verdict: {'SEMANTIC (permuting destroys it)' if p<0.01 and effect>5 else 'GEOMETRIC (permuting has no effect)'}")

    # Test 2: does c_sem correlate with mean cosine similarity?
    r, p2 = stats.pearsonr(real_cs, real_mcs)
    print(f"  Corr(c_sem, mean_cos_sim): r={r:.4f} p={p2:.6f}")
    print(f"  Verdict: {'c_sem IS mean_cos_sim (transformed)' if r>0.9 else 'c_sem ADDS information beyond mean_cos_sim' if r<0.5 else 'c_sem PARTIALLY reflects mean_cos_sim'}")

    # Test 3: residual — does c_sem add anything beyond mean_cos_sim?
    from scipy import stats
    slope, intercept, r_val, _, _ = stats.linregress(real_mcs, real_cs)
    residual = real_cs - (slope * real_mcs + intercept)
    # Does the residual have any structure? Test: can residual separate tight from loose?
    # (Not enough data for that, but we can check residual variance)
    print(f"  c_sem ~ {slope:.4f} * mean_cos + {intercept:.4f}  (r^2={r_val**2:.4f})")
    print(f"  Residual std: {residual.std():.4f} (c_sem std: {real_cs.std():.4f})")
    print(f"  Fraction unexplained: {residual.var()/real_cs.var():.4f}")
