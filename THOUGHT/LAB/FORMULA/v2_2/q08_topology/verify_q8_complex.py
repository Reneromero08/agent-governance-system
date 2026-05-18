"""Q8 complexify: does Hilbert transform tighten topological structure?"""
import sys, time
import numpy as np
from scipy.signal import hilbert
from ripser import ripser
from persim import bottleneck
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

N = 300

def compute_dgms(x):
    cos_sim = np.real(x @ np.conj(x).T)  # Hermitian real part
    dists = np.sqrt(np.maximum(2 - 2*cos_sim, 0))
    return ripser(dists, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]

def h1_stats(dgms):
    h1 = dgms[1]; h1p = h1[h1[:,1] < np.inf]
    return len(h1p), np.mean(h1p[:,1] - h1p[:,0]) if len(h1p) > 0 else 0

all_data = {}
for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    proj = (embs - embs.mean(axis=0)) @ evecs
    all_data[name] = {"proj": proj, "D": D}

print(f"N={N}, 5 trials per condition")
print(f"{'='*80}")
print(f"{'K':>5s} {'Real H1':>8s} {'Real_life':>10s} {'Cmplx H1':>8s} {'Cmplx_life':>10s} {'Ratio':>6s} {'Cmplx/Perm':>10s}")

for model_name, data in all_data.items():
    D = data["D"]
    print(f"\n{model_name}:")
    for K in [64, 96, 128, 192, 256, min(384, D)]:
        if K > D: continue
        proj = data["proj"][:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms

        # Complexify
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

        real_h1s, real_lives = [], []
        cmplx_h1s, cmplx_lives = [], []

        for seed in range(5):
            np.random.seed(seed)
            idx = np.random.choice(len(WORDS), N, replace=False)
            x_real = proj[idx]
            x_cmplx = z[idx]

            dgms_r = compute_dgms(x_real.astype(np.complex128))
            dgms_c = compute_dgms(x_cmplx)
            rh, rl = h1_stats(dgms_r)
            ch, cl = h1_stats(dgms_c)
            real_h1s.append(rh); real_lives.append(rl)
            cmplx_h1s.append(ch); cmplx_lives.append(cl)

        rh1 = np.mean(real_h1s); rl1 = np.mean(real_lives)
        ch1 = np.mean(cmplx_h1s); cl1 = np.mean(cmplx_lives)
        ratio = rh1 / ch1 if ch1 > 0 else 0
        print(f"{K:5d} {rh1:8.0f} {rl1:10.4f} {ch1:8.0f} {cl1:10.4f} {ratio:6.2f}x")

# Cross-model bottleneck with complexification
print(f"\n{'='*80}")
print("CROSS-MODEL BOTTLENECK: Real vs Complex")
np.random.seed(0)
idx = np.random.choice(len(WORDS), N, replace=False)

for K in [96, 192]:
    results = {}
    for model_name in ["MiniLM", "MPNet"]:
        proj = all_data[model_name]["proj"][:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
        results[model_name] = {
            "real": proj[idx],
            "complex": z[idx]
        }

    for mode1, mode2 in [("real","real"), ("complex","complex"), ("real","complex")]:
        dgms_ml = compute_dgms(results["MiniLM"][mode1])
        dgms_mp = compute_dgms(results["MPNet"][mode2])

        h1_ml = dgms_ml[1][dgms_ml[1][:,1] < np.inf]
        h1_mp = dgms_mp[1][dgms_mp[1][:,1] < np.inf]
        h0_ml = dgms_ml[0][dgms_ml[0][:,1] < np.inf]
        h0_mp = dgms_mp[0][dgms_mp[0][:,1] < np.inf]

        b_h1 = bottleneck(h1_ml, h1_mp) if len(h1_ml)>0 and len(h1_mp)>0 else -1
        b_h0 = bottleneck(h0_ml, h0_mp) if len(h0_ml)>0 and len(h0_mp)>0 else -1
        print(f"  K={K} {mode1:>8s}-{mode2:<8s}: H0={b_h0:.4f} H1={b_h1:.4f}")
