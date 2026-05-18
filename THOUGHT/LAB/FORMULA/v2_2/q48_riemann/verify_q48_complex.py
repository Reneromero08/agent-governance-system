"""Q48 complex-plane test: Hilbert-complexified embeddings, Hermitian Gram, eigenvalue analysis."""
import sys, json, math, time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

# Path setup
sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)
print(f"Words: {len(WORDS)}")

def complexify_eigenspace(embeddings):
    """Hilbert-transform each PC dimension to complexify embeddings.
    Returns complex-valued embeddings z_i with z_i† z_j as Hermitian inner product.
    """
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    projected = centered @ evecs  # shape (N, D) in eigenspace
    analytic = hilbert(projected, axis=-1)  # complex-valued
    z = analytic @ evecs.T + embeddings.mean(axis=0)  # back to original space
    return z

def complexify_dimwise(embeddings):
    """Hilbert-transform each dimension independently.
    z_{i,d} = emb_{i,d} + i * H[emb_{:,d}]
    """
    analytic = hilbert(embeddings, axis=0)  # complexify along sample axis per dim
    return analytic

def unfold_eigenvalues(eigenvalues):
    """Unfold real eigenvalues to unit mean spacing."""
    ev = np.sort(eigenvalues)
    ev_pos = ev[ev > 1e-15]
    N_stair = np.arange(1, len(ev_pos) + 1, dtype=float)
    log_e, log_N = np.log(ev_pos), np.log(N_stair)
    s = UnivariateSpline(log_e, log_N, s=len(ev_pos) * 0.001, k=3)
    N_smooth = np.exp(s(log_e))
    N_smooth = np.maximum(N_smooth, 0.1)
    sp = np.diff(N_smooth)
    return sp / sp.mean()

def complex_eigenvalue_stats(z_embs):
    """Compute complex-plane eigenvalue statistics from Hermitian Gram.
    H_ij = conj(z_i) @ z_j  (Hermitian, real eigenvalues).
    Also: complex symmetric G_ij = z_i^T z_j (complex eigenvalues).
    """
    n = len(z_embs)
    # Hermitian Gram
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            H[i, j] = np.conj(z_embs[i]) @ z_embs[j]
            H[j, i] = np.conj(H[i, j])  # Hermitian

    # Eigenvalues of H (real, since Hermitian)
    ev_H = np.linalg.eigvalsh(H)

    # Complex symmetric Gram: G_ij = z_i^T z_j (no conjugate)
    G = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            G[i, j] = z_embs[i] @ z_embs[j]  # complex symmetric
            G[j, i] = G[i, j]

    ev_G = np.linalg.eigvals(G)  # complex eigenvalues

    return ev_H, ev_G


# Load Riemann reference
rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi)
rz_sp = rz_sp / rz_sp.mean()

t0 = time.time()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    max_r = min(len(WORDS) - 1, D)

    print(f"\n{'='*64}")
    print(f"{name} ({D}d)")
    print("=" * 64)

    # --- BASELINE: real Gram ---
    gram = embs @ embs.T
    ev_real = np.sort(np.linalg.eigvalsh(gram))[-max_r:]
    ev_real = ev_real[ev_real > 1e-12]
    sp_real_unfolded = unfold_eigenvalues(ev_real)
    ks_real = stats.ks_2samp(sp_real_unfolded, rz_sp)
    print(f"Real Gram:     {len(sp_real_unfolded)} spacings  KS vs Riemann: D={ks_real.statistic:.4f} p={ks_real.pvalue:.2e}  P(s<0.3)={(sp_real_unfolded<0.3).mean():.4f}")

    # --- COMPLEXIFY: eigenspace mode ---
    z_eig = complexify_eigenspace(embs)
    ev_H_eig, ev_G_eig = complex_eigenvalue_stats(z_eig)
    ev_H_eig_sig = ev_H_eig[ev_H_eig > 1e-12]

    if len(ev_H_eig_sig) > 10:
        sp_H = unfold_eigenvalues(ev_H_eig_sig)
        ks_H = stats.ks_2samp(sp_H, rz_sp)
        print(f"Complex (eigenspace): {len(sp_H)} spacings  KS vs Riemann: D={ks_H.statistic:.4f} p={ks_H.pvalue:.2e}  P(s<0.3)={(sp_H<0.3).mean():.4f}")

    # --- COMPLEXIFY: dimwise mode ---
    z_dim = complexify_dimwise(embs)
    ev_H_dim, ev_G_dim = complex_eigenvalue_stats(z_dim)
    ev_H_dim_sig = ev_H_dim[ev_H_dim > 1e-12]

    if len(ev_H_dim_sig) > 10:
        sp_Hd = unfold_eigenvalues(ev_H_dim_sig)
        ks_Hd = stats.ks_2samp(sp_Hd, rz_sp)
        print(f"Complex (dimwise):  {len(sp_Hd)} spacings  KS vs Riemann: D={ks_Hd.statistic:.4f} p={ks_Hd.pvalue:.2e}  P(s<0.3)={(sp_Hd<0.3).mean():.4f}")

    # --- Complex plane: eigenvalues of G (complex symmetric) ---
    # Check critical line alignment: fraction of eigenvalues where |Re(ev_G)-1/2| < threshold
    ev_G_norm = ev_G_eig / np.max(np.abs(ev_G_eig))  # normalize to [-1, 1] range
    near_critical = np.abs(np.real(ev_G_norm) - 0.5) < 0.05
    near_real_line = np.abs(np.imag(ev_G_norm)) < 0.05
    print(f"Complex-plane G eigenvalues: {len(ev_G_eig)}")
    print(f"  Near Re(lambda)=1/2:  {near_critical.sum()}/{len(ev_G_norm)} ({near_critical.mean()*100:.1f}%)")
    print(f"  Near Im(lambda)=0:    {near_real_line.sum()}/{len(ev_G_norm)} ({near_real_line.mean()*100:.1f}%)")
    print(f"  Mean Re: {np.real(ev_G_norm).mean():.4f},  Mean Im: {np.imag(ev_G_norm).mean():.4f}")
    print(f"  Std Re:  {np.real(ev_G_norm).std():.4f},  Std Im:  {np.imag(ev_G_norm).std():.4f}")

    # Save complex-plane eigenvalues for visualization
    out = {
        "model": name, "D": D, "n_words": len(WORDS),
        "ev_G_norm_real": np.real(ev_G_norm).tolist(),
        "ev_G_norm_imag": np.imag(ev_G_norm).tolist(),
        "near_critical_frac": float(near_critical.mean()),
        "near_real_line_frac": float(near_real_line.mean()),
    }
    json.dump(out, open(f"THOUGHT/LAB/FORMULA/v2_2/q48_riemann/complex_plane_{name}.json", "w"), indent=2)

print(f"\nTime: {time.time() - t0:.1f}s")
print("\nRiemann reference: P(s<0.3) = 0.018, 500 zeros")
print("If complexification reveals Riemann structure:")
print("  - KS p > 0.05 vs Riemann zero spacings")
print("  - P(s<0.3) ~ 0.02 (strong level repulsion)")
print("  - Complex G eigenvalues cluster near Re(lambda)=1/2")
print("Without Riemann structure:")
print("  - KS p << 0.05")
print("  - P(s<0.3) ~ 0.07-0.12 (weak repulsion, same as real)")
print("  - Complex eigenvalues uniformly distributed")
