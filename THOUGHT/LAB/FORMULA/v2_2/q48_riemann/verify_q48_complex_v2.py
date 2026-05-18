"""Q48: Verify dimwise complexification result across seeds, models, and controls."""
import sys, json, time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)
rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi); rz_sp /= rz_sp.mean()

def unfold(eigenvalues):
    ev = np.sort(eigenvalues); ev = ev[ev > 1e-15]
    N = np.arange(1, len(ev)+1, dtype=float)
    s = UnivariateSpline(np.log(ev), np.log(N), s=len(ev)*0.001, k=3)
    ns = np.exp(s(np.log(ev))); ns = np.maximum(ns, 0.1)
    sp = np.diff(ns); return sp/sp.mean()

def compute_spacings(embs, method):
    """Compute unfolded spacings from (possibly complexified) embeddings."""
    if method == "real":
        gram = embs @ embs.T
        D = embs.shape[1]; max_r = min(len(embs)-1, D)
        ev = np.sort(np.linalg.eigvalsh(gram))[-max_r:]
        return unfold(ev)
    elif method == "hilbert_dimwise":
        z = hilbert(embs, axis=0)
        n = len(z)
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(i, n):
                v = np.conj(z[i]) @ z[j]
                H[i,j] = v; H[j,i] = np.conj(v)
        ev = np.linalg.eigvalsh(H)
        ev = ev[ev > 1e-12]
        return unfold(ev)
    elif method == "hilbert_eigenspace":
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        _, evecs = np.linalg.eigh(cov)
        projected = centered @ evecs
        z = hilbert(projected, axis=-1) @ evecs.T + embs.mean(axis=0)
        n = len(z)
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(i, n):
                v = np.conj(z[i]) @ z[j]
                H[i,j] = v; H[j,i] = np.conj(v)
        ev = np.linalg.eigvalsh(H)
        ev = ev[ev > 1e-12]
        return unfold(ev)
    elif method == "random_complex":
        # Control: add random complex noise
        z = embs.astype(np.complex128) + 0.1j * np.random.randn(*embs.shape)
        n = len(z)
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(i, n):
                v = np.conj(z[i]) @ z[j]
                H[i,j] = v; H[j,i] = np.conj(v)
        ev = np.linalg.eigvalsh(H)
        ev = ev[ev > 1e-12]
        return unfold(ev)

print(f"Words: {len(WORDS)}, Riemann zeros: {len(rz)}")
print(f"Riemann P(s<0.3) = {(rz_sp<0.3).mean():.4f}")
print()

t0 = time.time()
for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'Method':<25s} {'N_sp':>6s} {'KS D':>8s} {'KS p':>10s} {'P(s<0.3)':>10s} {'Verdict':>15s}")
    print("-" * 75)

    for method, label in [
        ("real", "Real Gram"),
        ("hilbert_dimwise", "Hilbert (dimwise)"),
        ("hilbert_eigenspace", "Hilbert (eigenspace)"),
        ("random_complex", "Random complex noise"),
    ]:
        sp = compute_spacings(embs, method)
        ks = stats.ks_2samp(sp, rz_sp)
        ps3 = (sp < 0.3).mean()
        if ks.pvalue > 0.05:
            v = "MATCH (p>0.05) ***"
        elif ks.pvalue > 0.01:
            v = "marginal"
        else:
            v = "NO MATCH"
        print(f"{label:<25s} {len(sp):>6d} {ks.statistic:>8.4f} {ks.pvalue:>10.2e} {ps3:>10.4f}  {v}")

    # Multiple seeds for dimwise (robustness check)
    print(f"\n  Seed stability (dimwise, 5 seeds):")
    seeds_p, seeds_ps3 = [], []
    for seed in [0, 1, 2, 3, 4]:
        np.random.seed(seed)
        z = hilbert(embs, axis=0)
        # Add small seed-dependent perturbation to break any symmetries
        z = z + 1e-4 * np.random.randn(*z.shape) * np.exp(1j * np.random.randn(*z.shape))
        n = len(z)
        H = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(i, n):
                v = np.conj(z[i]) @ z[j]
                H[i,j] = v; H[j,i] = np.conj(v)
        ev = np.linalg.eigvalsh(H)
        ev = ev[ev > 1e-12]
        if len(ev) > 10:
            sp = unfold(ev)
            ks = stats.ks_2samp(sp, rz_sp)
            seeds_p.append(ks.pvalue)
            seeds_ps3.append((sp < 0.3).mean())
    if seeds_p:
        print(f"    KS p range: [{min(seeds_p):.4f}, {max(seeds_p):.4f}], mean={np.mean(seeds_p):.4f}")
        print(f"    P(s<0.3) range: [{min(seeds_ps3):.4f}, {max(seeds_ps3):.4f}], mean={np.mean(seeds_ps3):.4f}")
        print(f"    p>0.05 in {sum(1 for p in seeds_p if p>0.05)}/5 seeds")

print(f"\nTime: {time.time() - t0:.1f}s")
