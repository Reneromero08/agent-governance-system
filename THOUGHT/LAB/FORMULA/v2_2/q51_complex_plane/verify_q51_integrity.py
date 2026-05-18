"""Q51 integrity: Hilbert vs random phases, phase-size dependence."""
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]

m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
all_embs = m.encode(WORDS_POOL, normalize_embeddings=True)
centered = all_embs - all_embs.mean(axis=0)
cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov)
idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]

for K in [96, 384]:
    print(f"\nK={K}:")
    h_phis = []; r_phis = []
    np.random.seed(42)
    for _ in range(50):
        words = list(np.random.choice(WORDS_POOL, 5, replace=False))
        words.append(words[0])
        embs = m.encode(words, normalize_embeddings=True)
        cw = embs - all_embs.mean(axis=0)
        proj = cw @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms

        # Hilbert
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
        total = 0j
        for i in range(len(z)-1):
            o = np.conj(z[i]).dot(z[i+1])
            if abs(o) > 1e-15: total += np.log(o)
        h_phis.append((-np.imag(total)) % (2*np.pi))

        # Random complex phases
        zr = proj.astype(np.complex128) * np.exp(1j * 2 * np.pi * np.random.rand(*proj.shape))
        total = 0j
        for i in range(len(zr)-1):
            o = np.conj(zr[i]).dot(zr[i+1])
            if abs(o) > 1e-15: total += np.log(o)
        r_phis.append((-np.imag(total)) % (2*np.pi))

    h = np.array(h_phis); r = np.array(r_phis)
    Ch, Sh = np.mean(np.cos(h)), np.mean(np.sin(h))
    Rh = np.sqrt(Ch**2 + Sh**2)
    Cr, Sr = np.mean(np.cos(r)), np.mean(np.sin(r))
    Rr = np.sqrt(Cr**2 + Sr**2)
    print(f"  Hilbert: mean={np.arctan2(Sh,Ch)%(2*np.pi):.4f} std={np.sqrt(-2*np.log(max(Rh,1e-15))):.4f}  min={h.min():.4f} max={h.max():.4f}")
    print(f"  Random:  mean={np.arctan2(Sr,Cr)%(2*np.pi):.4f} std={np.sqrt(-2*np.log(max(Rr,1e-15))):.4f}  min={r.min():.4f} max={r.max():.4f}")
    ks = stats.ks_2samp(h, r)
    print(f"  KS Hilbert vs Random: p={ks.pvalue:.4f}")

# Also: does Berry phase scale with loop size?
print(f"\nLoop size test (MiniLM K=96):")
for n_words in [3, 5, 7, 10]:
    np.random.seed(42)
    phis = []
    for _ in range(30):
        words = list(np.random.choice(WORDS_POOL, n_words, replace=False))
        words.append(words[0])
        embs = m.encode(words, normalize_embeddings=True)
        cw = embs - all_embs.mean(axis=0)
        proj = cw @ evecs[:, :96]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
        total = 0j
        for i in range(len(z)-1):
            o = np.conj(z[i]).dot(z[i+1])
            if abs(o) > 1e-15: total += np.log(o)
        phis.append((-np.imag(total)) % (2*np.pi))
    p = np.array(phis)
    C, S = np.mean(np.cos(p)), np.mean(np.sin(p))
    R = np.sqrt(C**2 + S**2)
    print(f"  N={n_words+1}: circ_mean={np.arctan2(S,C)%(2*np.pi):.4f} circ_std={np.sqrt(-2*np.log(max(R,1e-15))):.4f}")
