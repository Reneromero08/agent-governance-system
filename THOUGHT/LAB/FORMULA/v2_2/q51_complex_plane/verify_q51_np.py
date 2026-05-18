"""Q51: Berry phase via numpy overlap product — reliable, no ctypes issues."""
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river",
    "tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish",
    "horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood",
    "mother","father","child","friend","king","queen","love","hate","truth","life",
    "death","time","space","power","peace","war","hope","fear","joy","pain","dream",
    "thought","book","door","house","road","food","money","stone","gold"]

LOOPS = {
    "king-man-woman-queen": ["king","man","woman","queen","king"],
    "paris-france-berlin-germany": ["paris","france","berlin","germany","paris"],
    "walk-walking-run-running": ["walk","walking","run","running","walk"],
    "good-better-bad-worse": ["good","better","bad","worse","good"],
    "cat-kitten-dog-puppy": ["cat","kitten","dog","puppy","cat"],
    "doctor-hospital-teacher-school": ["doctor","hospital","teacher","school","doctor"],
    "hot-cold-fast-slow": ["hot","cold","fast","slow","hot"],
    "buy-sell-give-take": ["buy","sell","give","take","buy"],
}

def berry_phase_np(states):
    """Berry phase = -Im[sum_i log(conj(psi_i) @ psi_{i+1})]"""
    total = np.complex128(0.0)
    for i in range(len(states)-1):
        overlap = np.conj(states[i]).dot(states[i+1])
        if abs(overlap) < 1e-15:
            return 0.0, 0.0
        total += np.log(overlap)
    phi = (-np.imag(total)) % (2*np.pi)
    mag = abs(np.exp(1j * total))
    return phi, mag

for mid, name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    all_embs = m.encode(WORDS_POOL, normalize_embeddings=True)
    centered = all_embs - all_embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
    D = m.get_sentence_embedding_dimension()

    print(f"\n{'='*64}")
    print(f"{name}")
    print(f"{'='*64}")

    for K in [96, D]:
        print(f"\n  K={K}:")
        print(f"  {'Loop':<35s} {'phi':>8s} {'|prod|':>8s}")
        print(f"  {'-'*51}")

        analogy_phis = []
        for loop_name, words in LOOPS.items():
            embs = m.encode(words, normalize_embeddings=True)
            centered_w = embs - all_embs.mean(axis=0)
            proj = centered_w @ evecs[:, :K]
            norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
            proj = proj / norms
            z = hilbert(proj, axis=0).astype(np.complex128)
            zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
            z = z / (zn + 1e-12)
            phi, mag = berry_phase_np([z[i] for i in range(len(words))])
            analogy_phis.append(phi)
            print(f"  {loop_name:<35s} {phi:8.4f} {mag:8.4f}")

        # Random loops
        np.random.seed(42)
        rand_phis = []
        for _ in range(50):
            words = list(np.random.choice(WORDS_POOL, 5, replace=False))
            words.append(words[0])  # close the loop
            embs = m.encode(words, normalize_embeddings=True)
            centered_w = embs - all_embs.mean(axis=0)
            proj = centered_w @ evecs[:, :K]
            norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
            proj = proj / norms
            z = hilbert(proj, axis=0).astype(np.complex128)
            zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
            z = z / (zn + 1e-12)
            phi, _ = berry_phase_np([z[i] for i in range(len(words))])
            rand_phis.append(phi)

        rand_phis = np.array(rand_phis)
        analogy_phis = np.array(analogy_phis)

        # Circular statistics
        C = np.mean(np.cos(rand_phis)); S = np.mean(np.sin(rand_phis))
        R = np.sqrt(C**2 + S**2)
        circ_mean = np.arctan2(S, C) % (2*np.pi)
        circ_std = np.sqrt(-2 * np.log(max(R, 1e-15)))

        # KS test: are analogy loop phases uniform?
        # If analogy loops cluster, they're NOT uniform → structure exists
        ks_uniform = stats.kstest(analogy_phis % (2*np.pi), 'uniform', args=(0, 2*np.pi))

        # Two-sample: do analogy and random phases differ?
        ks_2samp = stats.ks_2samp(analogy_phis % (2*np.pi), rand_phis % (2*np.pi))

        print(f"  Random loops (n=50): circ_mean={circ_mean:.4f} circ_std={circ_std:.4f}")
        print(f"  Analogy vs Uniform KS p: {ks_uniform.pvalue:.4f}")
        print(f"  Analogy vs Random KS p:  {ks_2samp.pvalue:.4f}")

        # Real (no complexification) control
        words_ctrl = ["king","man","woman","queen","king"]
        embs_ctrl = m.encode(words_ctrl, normalize_embeddings=True)
        centered_c = embs_ctrl - all_embs.mean(axis=0)
        proj_c = centered_c @ evecs[:, :K]
        norms_c = np.linalg.norm(proj_c, axis=1, keepdims=True); norms_c[norms_c==0]=1
        proj_c = proj_c / norms_c
        z_real = proj_c.astype(np.complex128)  # no Hilbert
        phi_real, mag_real = berry_phase_np([z_real[i] for i in range(len(words_ctrl))])
        print(f"  Real (no Hilbert) control: phi={phi_real:.4f} mag={mag_real:.4f}")
