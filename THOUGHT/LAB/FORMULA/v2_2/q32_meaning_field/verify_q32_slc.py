"""Q32: SLC Kuramoto threshold — does sigma > 2*nabla_S for semantic clusters?"""
import numpy as np
from sentence_transformers import SentenceTransformer

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]

CLUSTERS = {
    "royalty": ["king","queen","prince","castle","crown","knight"],
    "animals": ["dog","cat","bird","fish","horse","lion"],
    "nature": ["water","fire","earth","sky","sun","moon"],
    "emotions": ["love","hate","joy","fear","pain","hope"],
    "mixed": ["king","water","book","love","stone","time"],
    "random": ["fire","dream","money","horse","peace","truth"],
}

for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS_POOL, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    proj = centered @ evecs[:, :96]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1; proj = proj / norms

    print(f"\n{name} K=96 — SLC Kuramoto: sigma > 2*nabla_S ?")
    print(f"{'Cluster':<12s} {'sigma':>8s} {'nabla_S':>8s} {'2*nabla_S':>10s} {'Sync?':>8s}")

    for cname, words in CLUSTERS.items():
        indices = [WORDS_POOL.index(w) for w in words if w in WORDS_POOL]
        if len(indices) < 2: continue
        x = proj[indices]
        n = len(indices)

        # sigma = 1/Df = inverse participation ratio of density matrix
        rho = x.T @ x / n
        rho_ev = np.linalg.eigvalsh(rho)
        rho_ev = np.maximum(rho_ev, 1e-15)
        rho_ev = rho_ev / rho_ev.sum()
        Df = rho_ev.sum()**2 / (rho_ev**2).sum()
        sigma = 1.0 / Df

        # nabla_S = von Neumann entropy
        nabla_S = -np.sum(rho_ev * np.log(rho_ev))

        sync = "YES" if sigma > 2 * nabla_S else "no"
        print(f"{cname:<12s} {sigma:8.4f} {nabla_S:8.4f} {2*nabla_S:10.4f} {sync:>8s}")
