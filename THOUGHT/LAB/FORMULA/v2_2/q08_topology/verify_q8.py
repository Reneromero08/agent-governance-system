"""Q8: Persistent homology — do embeddings have topological structure beyond random?"""
import sys, json, time
import numpy as np
from ripser import ripser
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

N = 400  # number of words to sample

for model_id, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(model_id, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()

    # Sample N words
    np.random.seed(42)
    idx = np.random.choice(len(WORDS), N, replace=False)
    x_real = embs[idx]

    # Random vectors as null (same D, same N)
    x_rand = np.random.randn(N, D)
    x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)

    # PCA-96 for embeddings
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    proj_all = (embs - embs.mean(axis=0)) @ evecs[:, :96]
    norms = np.linalg.norm(proj_all, axis=1, keepdims=True); norms[norms==0]=1
    proj_all = proj_all / norms
    x_pca = proj_all[idx]

    for label, x in [("Real (D)",x_real),("PCA-96",x_pca),("Random",x_rand)]:
        t0 = time.time()
        # Distance matrix: d = sqrt(2 - 2*cos_sim) = chord distance
        cos_sim = x @ x.T
        dists = np.sqrt(np.maximum(2 - 2*cos_sim, 0))

        # Persistent homology (H_0, H_1)
        dgms = ripser(dists, maxdim=1, distance_matrix=True, thresh=1.5)["dgms"]

        h0 = dgms[0]  # H0: [birth, death] for each connected component
        h1 = dgms[1]  # H1: [birth, death] for each 1-cycle

        # Summary statistics
        h0_lifetimes = h0[:,1] - h0[:,0]
        h0_persistence = h0[h0[:,1] < np.inf]  # exclude the infinite component
        h1_persistence = h1[h1[:,1] < np.inf]

        # Key metric: number of persistent H1 features (holes)
        n_h1 = len(h1_persistence)
        # Mean H1 lifetime
        mean_h1_life = h1_persistence[:,1].mean() - h1_persistence[:,0].mean() if n_h1 > 0 else 0
        # H1 birth-death ratio
        h1_bd_ratio = np.mean(h1_persistence[:,0]) / (np.mean(h1_persistence[:,1]) + 1e-10) if n_h1 > 0 else 0

        print(f"  {name} {label:>10s}: H0={len(h0_persistence)} components, H1={n_h1} cycles, H1_lifetime={mean_h1_life:.4f}, H1_bd={h1_bd_ratio:.4f} ({time.time()-t0:.1f}s)")

print(f"\n{'='*64}")
print("VERDICT:")
print("  If embeddings have MORE persistent H1 features than random,")
print("  there is genuine topological structure (holes/cycles).")
print("  If H1 ~ random, the embedding topology is trivial (spherical).")
