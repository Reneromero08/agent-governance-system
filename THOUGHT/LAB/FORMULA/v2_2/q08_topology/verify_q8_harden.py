"""Q8 hardening: multi-model, PCA sweep, causal control, cross-model agreement."""
import sys, time
import numpy as np
from ripser import ripser
from persim import bottleneck
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

N = 300

all_models = {}
for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    all_models[name] = {"embeddings": embs, "D": D, "projected": (embs - embs.mean(axis=0)) @ evecs}

def compute_dgms(x, label=""):
    cos_sim = x @ x.T
    dists = np.sqrt(np.maximum(2 - 2*cos_sim, 0))
    return ripser(dists, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]

def h1_stats(dgms):
    h1 = dgms[1]
    h1p = h1[h1[:,1] < np.inf]
    return len(h1p), np.mean(h1p[:,1] - h1p[:,0]) if len(h1p) > 0 else 0

print(f"N={N}, 5 trials per condition")
print(f"{'='*72}")

results = {}
for model_name, data in all_models.items():
    D = data["D"]
    print(f"\n{model_name}:")
    print(f"{'K':>5s} {'Real H1':>8s} {'Real_life':>10s} {'Permuted H1':>12s} {'Perm_life':>12s} {'H1_ratio':>8s}")
    print(f"{'='*60}")

    for K in [64, 96, 128, 192, 256, D]:
        if K > D: continue
        proj = data["projected"][:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms

        real_h1s, real_lives = [], []
        perm_h1s, perm_lives = [], []

        for seed in range(5):
            np.random.seed(seed)
            idx = np.random.choice(len(WORDS), N, replace=False)
            x_real = proj[idx]

            # Permuted: shuffle each dimension independently → destroy semantics
            x_perm = x_real.copy()
            for d in range(x_perm.shape[1]):
                np.random.shuffle(x_perm[:, d])

            dgms_r = compute_dgms(x_real)
            dgms_p = compute_dgms(x_perm)
            rh, rl = h1_stats(dgms_r)
            ph, pl = h1_stats(dgms_p)
            real_h1s.append(rh); real_lives.append(rl)
            perm_h1s.append(ph); perm_lives.append(pl)

        rh1 = np.mean(real_h1s); rl1 = np.mean(real_lives)
        ph1 = np.mean(perm_h1s); pl1 = np.mean(perm_lives)
        ratio = ph1 / rh1 if rh1 > 0 else 0
        print(f"{K:5d} {rh1:8.0f} {rl1:10.4f} {ph1:12.0f} {pl1:12.4f} {ratio:8.2f}x")

        results[f"{model_name}_K{K}"] = {
            "real_h1": float(rh1), "real_life": float(rl1),
            "perm_h1": float(ph1), "perm_life": float(pl1),
            "ratio": float(ratio)
        }

# Cross-model bottleneck: MiniLM vs MPNet (same K, same words)
print(f"\n{'='*72}")
print("CROSS-MODEL BOTTLENECK (N=300)")
np.random.seed(0)
idx = np.random.choice(len(WORDS), N, replace=False)

for K in [96, 192]:
    # MiniLM PCA-K
    ml_embs = all_models["MiniLM"]["projected"][:, :K]
    norms = np.linalg.norm(ml_embs, axis=1, keepdims=True); norms[norms==0]=1
    ml_proj = ml_embs / norms
    x_ml = ml_proj[idx]
    dgms_ml = compute_dgms(x_ml)

    # MPNet PCA-K
    mp_embs = all_models["MPNet"]["projected"][:, :K]
    norms = np.linalg.norm(mp_embs, axis=1, keepdims=True); norms[norms==0]=1
    mp_proj = mp_embs / norms
    x_mp = mp_proj[idx]
    dgms_mp = compute_dgms(x_mp)

    # Random
    x_rd = np.random.randn(N, K); x_rd = x_rd / np.linalg.norm(x_rd, axis=1, keepdims=True)
    dgms_rd = compute_dgms(x_rd)

    # Bottleneck distances
    h1_ml = dgms_ml[1][dgms_ml[1][:,1] < np.inf]
    h1_mp = dgms_mp[1][dgms_mp[1][:,1] < np.inf]
    h1_rd = dgms_rd[1][dgms_rd[1][:,1] < np.inf]
    h0_ml = dgms_ml[0][dgms_ml[0][:,1] < np.inf]
    h0_mp = dgms_mp[0][dgms_mp[0][:,1] < np.inf]
    h0_rd = dgms_rd[0][dgms_rd[0][:,1] < np.inf]

    b_ml_mp_h1 = bottleneck(h1_ml, h1_mp) if len(h1_ml) > 0 and len(h1_mp) > 0 else -1
    b_ml_rd_h1 = bottleneck(h1_ml, h1_rd) if len(h1_ml) > 0 and len(h1_rd) > 0 else -1
    b_mp_rd_h1 = bottleneck(h1_mp, h1_rd) if len(h1_mp) > 0 and len(h1_rd) > 0 else -1
    b_ml_mp_h0 = bottleneck(h0_ml, h0_mp) if len(h0_ml) > 0 and len(h0_mp) > 0 else -1
    b_ml_rd_h0 = bottleneck(h0_ml, h0_rd) if len(h0_ml) > 0 and len(h0_rd) > 0 else -1

    print(f"\n  K={K}:")
    print(f"    H1 bottleneck: MiniLM-MPNet={b_ml_mp_h1:.4f}  MiniLM-Random={b_ml_rd_h1:.4f}  MPNet-Random={b_mp_rd_h1:.4f}")
    print(f"    H0 bottleneck: MiniLM-MPNet={b_ml_mp_h0:.4f}  MiniLM-Random={b_ml_rd_h0:.4f}")
    if b_ml_mp_h1 < b_ml_rd_h1:
        print(f"    -> Cross-model MORE similar than either to random (shared topology)")
    else:
        print(f"    -> Cross-model similarity DOES NOT exceed similarity to random")
