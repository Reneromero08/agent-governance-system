"""Q32: Complex-plane field dynamics — c_sem from Fubini-Study manifold."""
import sys, time
import numpy as np
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS = list(ANCHOR_1024)

CHAINS = {
    "king->queen->prince->castle->crown": ["king","queen","prince","castle","crown"],
    "dog->cat->bird->fish->horse": ["dog","cat","bird","fish","horse"],
    "water->river->ocean->sea->lake": ["water","river","ocean","sea","lake"],
    "love->peace->war->fire->stone": ["love","peace","war","fire","stone"],
    "random_mix": ["king","water","book","love","time"],
}

for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()

    for K in [96, D]:
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        proj = centered @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1; proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

        print(f"\n{name} K={K}")
        print(f"{'Chain':<35s} {'c_sem':>8s} {'sigma':>8s} {'nabla_S':>8s} {'phase_vel':>10s} {'FS_dist':>8s}")

        c_sem_vals, pv_vals = [], []
        for cname, words in CHAINS.items():
            idx = [WORDS.index(w) for w in words if w in WORDS]
            if len(idx) < 2: continue
            zi = z[idx]

            # Complex Hermitian Gram of the chain
            H = np.zeros((len(idx), len(idx)), dtype=np.complex128)
            for i in range(len(idx)):
                for j in range(i, len(idx)):
                    v = np.conj(zi[i]).dot(zi[j])
                    H[i,j] = v; H[j,i] = np.conj(v)
            ev_H = np.linalg.eigvalsh(H)
            ev_H = np.maximum(ev_H, 1e-15)
            ev_H = ev_H / ev_H.sum()

            # sigma = 1/Df (inverse participation ratio)
            Df = ev_H.sum()**2 / (ev_H**2).sum()
            sigma_c = 1.0 / Df

            # nabla_S = von Neumann entropy
            nabla_S_c = -np.sum(ev_H * np.log(ev_H + 1e-15))

            # c_sem = sqrt(sigma / nabla_S)
            csem = np.sqrt(sigma_c / max(nabla_S_c, 1e-10))

            # Phase velocity along chain
            total_phase = 0.0; total_dist = 0.0
            for i in range(len(zi)-1):
                o = np.conj(zi[i]).dot(zi[i+1])
                total_phase += abs(np.angle(o))
                total_dist += np.arccos(np.clip(np.abs(o), 0, 1))
            pv = total_phase / total_dist if total_dist > 0 else np.nan

            # FS geodesic distance (sum)
            fs_dist = np.sum([np.arccos(np.clip(np.abs(np.conj(zi[i]).dot(zi[i+1])), 0, 1))
                            for i in range(len(zi)-1)])

            if not np.isnan(csem) and not np.isnan(pv):
                c_sem_vals.append(csem); pv_vals.append(pv)

            print(f"{cname:<35s} {csem:8.4f} {sigma_c:8.4f} {nabla_S_c:8.4f} {pv:10.4f} {fs_dist:8.4f}")

        if len(c_sem_vals) >= 5:
            from scipy import stats
            r, p = stats.pearsonr(c_sem_vals, pv_vals)
            print(f"  Corr(c_sem_complex, phase_vel): r={r:.4f} p={p:.4f}")
            print(f"  {'FIELD PREDICTS' if r > 0.7 and p < 0.1 else 'DOES NOT PREDICT'} wave speed")
