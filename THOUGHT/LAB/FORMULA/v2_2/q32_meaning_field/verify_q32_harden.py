"""Q32 hardening: PCA sweep, causality, seed stability, cross-model, predictive."""
import sys, time
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS = list(ANCHOR_1024)

CLUSTERS = {
    "royalty_tight": ["king","queen","prince","castle","crown","knight","throne","royal"],
    "animals_tight": ["dog","cat","bird","fish","horse","lion","tiger","elephant"],
    "nature_tight": ["water","fire","earth","sky","sun","moon","star","mountain"],
    "emotions_tight": ["love","hate","joy","fear","pain","hope","dream","peace"],
    "mixed_loose": ["king","water","book","love","time","gold","music","science"],
    "random_loose": ["fire","dream","money","horse","peace","truth","light","dark"],
}

def compute_c_sem(words, emb_set):
    idx = [WORDS.index(w) for w in words if w in WORDS]
    if len(idx) < 2: return np.nan, np.nan, np.nan, np.nan
    x = emb_set[idx]
    n = len(idx)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(x[i]).dot(x[j])
            H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H)
    ev = np.maximum(ev, 1e-15); ev = ev / ev.sum()
    Df = ev.sum()**2 / (ev**2).sum()
    sigma_c = 1.0 / Df
    nabla_S = -np.sum(ev * np.log(ev + 1e-15))
    c_sem = np.sqrt(sigma_c / max(nabla_S, 1e-10))
    return c_sem, sigma_c, nabla_S, Df

print("Q32 HARDENING BATTERY")
t0 = time.time()

all_data = {}
for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    all_data[name] = {"embs": embs, "D": D, "proj_full": (embs - embs.mean(axis=0)) @ evecs}

# ===========================================================================
# ANGLE 1: PCA sweep — optimal K for tight/loose separation?
# ===========================================================================
print(f"\n{'='*64}")
print("ANGLE 1: PCA sweep — c_sem tight vs loose separation")
print(f"{'K':>6s} {'tight_csem':>10s} {'loose_csem':>10s} {'ratio':>8s} {'sep?':>6s}")

for model_name, data in all_data.items():
    proj_full = data["proj_full"]
    D = data["D"]
    print(f"\n{model_name}:")
    for K in [8, 16, 32, 64, 96, 128, 192, 256, min(D, 384)]:
        proj = proj_full[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

        tight_cs = [compute_c_sem(w, z)[0] for cname, w in CLUSTERS.items() if "tight" in cname]
        loose_cs = [compute_c_sem(w, z)[0] for cname, w in CLUSTERS.items() if "loose" in cname]
        tight_cs = [c for c in tight_cs if not np.isnan(c)]
        loose_cs = [c for c in loose_cs if not np.isnan(c)]

        if tight_cs and loose_cs:
            mt, ml = np.mean(tight_cs), np.mean(loose_cs)
            ratio = mt / ml if ml > 0 else np.inf
            t, p = stats.ttest_ind(tight_cs, loose_cs)
            sep = "YES" if p < 0.1 and ratio > 1.3 else "no"
            print(f"{K:6d} {mt:10.4f} {ml:10.4f} {ratio:8.2f}x {sep:>6s}")

# ===========================================================================
# ANGLE 2: Causality — permuted embeddings
# ===========================================================================
print(f"\n{'='*64}")
print("ANGLE 2: Causal control — permuted vs real (K=96)")

for model_name, data in all_data.items():
    K = 96
    proj = data["proj_full"][:, :K]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
    proj = proj / norms
    z = hilbert(proj, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

    # Permuted complex
    z_perm = z.copy()
    for d in range(z_perm.shape[1]):
        np.random.shuffle(z_perm[:, d])

    real_cs = []; perm_cs = []
    for cname, words in CLUSTERS.items():
        rc, _, _, _ = compute_c_sem(words, z)
        pc, _, _, _ = compute_c_sem(words, z_perm)
        if not np.isnan(rc): real_cs.append(rc)
        if not np.isnan(pc): perm_cs.append(pc)

    mr, mp = np.mean(real_cs), np.mean(perm_cs)
    t, p = stats.ttest_ind(real_cs, perm_cs)
    print(f"  {model_name}: real_c_sem={mr:.4f} perm_c_sem={mp:.4f} t={t:.1f} p={p:.4f}")
    print(f"    {'SEMANTIC (real > perm)' if p < 0.05 and mr > mp else 'NOISE (real ~ perm)'}")

# ===========================================================================
# ANGLE 3: Seed stability (10 seeds, same words)
# ===========================================================================
print(f"\n{'='*64}")
print("ANGLE 3: Seed stability (K=96, 10 perturbation seeds)")

for model_name, data in all_data.items():
    K = 96
    proj = data["proj_full"][:, :K]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
    proj = proj / norms

    csem_seeds = []
    for seed in range(10):
        np.random.seed(seed)
        z = hilbert(proj, axis=0).astype(np.complex128)
        z = z + 1e-8 * np.random.randn(*z.shape) * np.exp(1j * np.random.randn(*z.shape))
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
        c_vals = []
        for cname, words in CLUSTERS.items():
            cs, _, _, _ = compute_c_sem(words, z)
            if not np.isnan(cs): c_vals.append(cs)
        csem_seeds.append(np.mean(c_vals))
    csem_arr = np.array(csem_seeds)
    print(f"  {model_name}: c_sem={csem_arr.mean():.4f}+/-{csem_arr.std():.4f}  CV={csem_arr.std()/csem_arr.mean()*100:.2f}%")

# ===========================================================================
# ANGLE 4: Cross-model agreement
# ===========================================================================
print(f"\n{'='*64}")
print("ANGLE 4: Cross-model agreement (K=96)")

K = 96
model_cs_all = {}
for model_name, data in all_data.items():
    proj = data["proj_full"][:, :K]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
    proj = proj / norms
    z = hilbert(proj, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    model_cs_all[model_name] = {}
    for cname, words in CLUSTERS.items():
        cs, _, _, _ = compute_c_sem(words, z)
        if not np.isnan(cs): model_cs_all[model_name][cname] = cs

cs1 = [model_cs_all["MiniLM"].get(c, np.nan) for c in CLUSTERS.keys()]
cs2 = [model_cs_all["MPNet"].get(c, np.nan) for c in CLUSTERS.keys()]
valid = ~np.isnan(cs1) & ~np.isnan(cs2)
r, p = stats.pearsonr(np.array(cs1)[valid], np.array(cs2)[valid])
print(f"  Cross-model corr(c_sem): r={r:.4f} p={p:.4f}")
print(f"  {'SHARED FIELD' if r > 0.8 and p < 0.01 else 'MODEL-SPECIFIC'}")

# ===========================================================================
# ANGLE 5: Predictive — AUROC for tight vs loose classification
# ===========================================================================
print(f"\n{'='*64}")
print("ANGLE 5: Predictive — c_sem classifies tight vs loose (K=96)")

for model_name, data in all_data.items():
    proj = data["proj_full"][:, :K]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
    proj = proj / norms
    z = hilbert(proj, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

    scores, labels = [], []
    for cname, words in CLUSTERS.items():
        cs, _, _, _ = compute_c_sem(words, z)
        if not np.isnan(cs):
            scores.append(cs)
            labels.append(1 if "tight" in cname else 0)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    print(f"  {model_name}: AUROC = {auc:.4f}")

print(f"\nTime: {time.time()-t0:.1f}s")
