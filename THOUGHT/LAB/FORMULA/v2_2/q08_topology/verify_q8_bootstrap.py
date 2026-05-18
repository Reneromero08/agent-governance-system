"""Q8 deepen: bootstrap H1 cycle counts, bottleneck distance, Wasserstein."""
import sys, time
import numpy as np
from ripser import ripser
from persim import bottleneck
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

N = 300  # subsample size for bootstrap

for model_id, name in [("all-MiniLM-L6-v2","MiniLM")]:
    m = SentenceTransformer(model_id, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()

    # PCA-96
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    proj_all = (embs - embs.mean(axis=0)) @ evecs[:, :96]
    norms = np.linalg.norm(proj_all, axis=1, keepdims=True); norms[norms==0]=1
    proj_all = proj_all / norms

    print(f"\n{name} PCA-96  (N={N})")
    print(f"{'Trial':>6s} {'Real H1':>10s} {'Rand H1':>10s} {'Real H1_life':>14s} {'Rand H1_life':>14s}")
    print(f"{'='*56}")

    real_h1_counts = []; rand_h1_counts = []
    real_h1_lives = []; rand_h1_lives = []

    for trial in range(20):
        np.random.seed(trial)
        idx = np.random.choice(len(WORDS), N, replace=False)
        x_real = proj_all[idx]

        x_rand = np.random.randn(N, 96)
        x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)

        for label, x in [("Real", x_real), ("Rand", x_rand)]:
            cos_sim = x @ x.T
            dists = np.sqrt(np.maximum(2 - 2*cos_sim, 0))
            dgms = ripser(dists, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]
            h1 = dgms[1]
            h1_persistent = h1[h1[:,1] < np.inf]
            n_h1 = len(h1_persistent)
            life = np.mean(h1_persistent[:,1] - h1_persistent[:,0]) if n_h1 > 0 else 0

            if label == "Real":
                real_h1_counts.append(n_h1); real_h1_lives.append(life)
            else:
                rand_h1_counts.append(n_h1); rand_h1_lives.append(life)

        if trial < 5 or trial % 5 == 0:
            print(f"{trial:6d} {real_h1_counts[-1]:10d} {rand_h1_counts[-1]:10d} {real_h1_lives[-1]:14.6f} {rand_h1_lives[-1]:14.6f}")

    rc = np.array(real_h1_counts); rrc = np.array(rand_h1_counts)
    rl = np.array(real_h1_lives); rrl = np.array(rand_h1_lives)

    print(f"\n  H1 count:  Real={rc.mean():.0f}+/-{rc.std():.0f}  Rand={rrc.mean():.0f}+/-{rrc.std():.0f}  ratio={rrc.mean()/rc.mean():.1f}x")
    from scipy import stats
    t_c, p_c = stats.ttest_ind(rc, rrc)
    print(f"  H1 count t-test: t={t_c:.1f} p={p_c:.2e}")
    t_l, p_l = stats.ttest_ind(rl, rrl)
    print(f"  H1 life t-test:  t={t_l:.1f} p={p_l:.2e}")

    # Bottleneck distance: how different are the H1 persistence diagrams?
    print(f"\n  Bottleneck distance (H1 diagrams, trial 0 vs trial 0):")
    idx0 = np.random.choice(len(WORDS), N, replace=False)
    np.random.seed(0)
    x_r = proj_all[idx0]
    x_rd = np.random.randn(N, 96); x_rd = x_rd / np.linalg.norm(x_rd, axis=1, keepdims=True)
    cos_r = x_r @ x_r.T; dists_r = np.sqrt(np.maximum(2 - 2*cos_r, 0))
    cos_rd = x_rd @ x_rd.T; dists_rd = np.sqrt(np.maximum(2 - 2*cos_rd, 0))
    dgm_r = ripser(dists_r, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]
    dgm_rd = ripser(dists_rd, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]
    bdist_h1 = bottleneck(dgm_r[1][dgm_r[1][:,1] < np.inf], dgm_rd[1][dgm_rd[1][:,1] < np.inf])
    bdist_h0 = bottleneck(dgm_r[0][dgm_r[0][:,1] < np.inf], dgm_rd[0][dgm_rd[0][:,1] < np.inf])
    print(f"  H1 bottleneck = {bdist_h1:.6f}")
    print(f"  H0 bottleneck = {bdist_h0:.6f}")
