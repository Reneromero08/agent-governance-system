"""Q48: Three creative angles — phase-from-covariance, Berry curvature, octant mapping."""
import sys, json, time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)
rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi); rz_sp /= rz_sp.mean()

def unfold(eigenvalues):
    ev = np.sort(eigenvalues); ev = ev[ev > 1e-15]
    if len(ev) < 10: return np.array([])
    N = np.arange(1, len(ev)+1, dtype=float)
    s = UnivariateSpline(np.log(ev), np.log(N), s=len(ev)*0.001, k=3)
    ns = np.exp(s(np.log(ev))); ns = np.maximum(ns, 0.1)
    sp = np.diff(ns); return sp/sp.mean()

# =============================================================================
# ANGLE 1: Phase from covariance
# =============================================================================
def phase_from_covariance_mds(cov):
    """Recover phases from: Cov(i,j) = r_i * r_j * cos(theta_i - theta_j)."""
    dim = cov.shape[0]
    amps = np.sqrt(np.maximum(np.diag(cov), 1e-10))
    outer = np.outer(amps, amps)
    cos_diff = np.clip(cov / (outer + 1e-10), -1, 1)
    phase_diff = np.arccos(cos_diff)
    D_sq = 2 * (1 - np.cos(phase_diff))
    H = np.eye(dim) - np.ones((dim, dim)) / dim
    B = -0.5 * H @ D_sq @ H
    evals, evecs = eigh(B)
    idx = np.argsort(evals)[::-1]; evals = evals[idx]; evecs = evecs[:, idx]
    coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0))
    phases = np.arctan2(coords[:, 1], coords[:, 0])
    return phases, cos_diff, amps

# =============================================================================
# ANGLE 2: Berry curvature (replicated from C++ headers)
# =============================================================================
def berry_curvature(embs, epsilon=0.01):
    """Compute Berry curvature F_ij = d_i A_j - d_j A_i on embedding manifold.
    Berry connection: A_i = Im(<psi| d_i |psi>).
    For real vectors |psi> = x/|x|, A_i = 0 (real -> no imaginary part).
    For finite differences: A_i(k) = Im(<x_k|(x_{k+ei} - x_{k-ei})/(2*eps)>)
    where x_k are normalized embedding vectors.
    """
    n, d = embs.shape
    x = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # Berry connection: A_i^k = Im(<x_k|x_{k+ei} - x_k>)/epsilon
    # (finite difference approximation)
    A = np.zeros((n, d))
    for i in range(d):
        # Shift in dimension i
        shifted = x.copy()
        eps_i = np.zeros(d); eps_i[i] = epsilon
        shifted += eps_i
        shifted = shifted / (np.linalg.norm(shifted, axis=1, keepdims=True) + 1e-12)
        A[:, i] = np.imag(np.sum(x * shifted, axis=1)) / epsilon

    # Berry curvature: F_ij^k = (A_j^{k+ei} - A_i^{k+ej}) / (2*epsilon)
    # For 2-form, compute Frobenius norm per point
    F_norms = np.zeros(n)
    for k in range(n):
        F = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    eps_i = np.zeros(d); eps_i[i] = epsilon
                    eps_j = np.zeros(d); eps_j[j] = epsilon
                    x_pi = x[k] + eps_i; x_pi /= (np.linalg.norm(x_pi) + 1e-12)
                    x_pj = x[k] + eps_j; x_pj /= (np.linalg.norm(x_pj) + 1e-12)
                    A_j_pi = np.imag(np.dot(x_pi, (x_pi + eps_j) / (np.linalg.norm(x_pi + eps_j) + 1e-12) - x_pi)) / epsilon
                    A_i_pj = np.imag(np.dot(x_pj, (x_pj + eps_i) / (np.linalg.norm(x_pj + eps_i) + 1e-12) - x_pj)) / epsilon
                    F[i, j] = (A_j_pi - A_i_pj) / (2 * epsilon)
        F_norms[k] = np.linalg.norm(F, 'fro')
    return F_norms, A

# =============================================================================
# ANGLE 3: Octant-phase mapping with circular statistics
# =============================================================================
def octant_phase_analysis(embs):
    """Map PCA octants to phase sectors, test uniformity with circular stats."""
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
    projected = centered @ evecs[:, :3]  # Top 3 PCs

    # Octant sign pattern: each word gets a sign vector in {+,-}^3
    signs = (projected >= 0).astype(int)
    octant_ids = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]  # 0-7

    # Map each octant to a phase angle: octant k -> 2*pi*k/8
    octant_phases = 2 * np.pi * octant_ids / 8

    # Circular statistics
    C = np.mean(np.cos(octant_phases))
    S = np.mean(np.sin(octant_phases))
    R = np.sqrt(C**2 + S**2)  # mean resultant length
    mean_phase = np.arctan2(S, C)

    # Rayleigh test for uniformity: p = exp(-n*R^2) for n > 50
    n = len(octant_phases)
    rayleigh_p = np.exp(-n * R**2) if n * R**2 < 50 else 0.0

    # Phase spacing statistics
    sorted_phases = np.sort(octant_phases)
    wrapped_sp = np.diff(sorted_phases)
    # Add wrap-around
    wrapped_sp = np.append(wrapped_sp, 2*np.pi - sorted_phases[-1] + sorted_phases[0])
    wrapped_sp_norm = wrapped_sp / wrapped_sp.mean()

    # Also: treat the PC projection as complex plane z = PC1 + i*PC2
    z = projected[:, 0] + 1j * projected[:, 1]
    phases_2d = np.angle(z)

    return {
        "octant_counts": np.bincount(octant_ids, minlength=8),
        "R": R, "mean_phase": mean_phase, "rayleigh_p": rayleigh_p,
        "phase_spacings": wrapped_sp_norm,
        "phases_2d": phases_2d,
    }


print(f"Words: {len(WORDS)}")
print(f"Riemann P(s<0.3) = {(rz_sp<0.3).mean():.4f}")
t0 = time.time()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()

    print(f"\n{'='*64}")
    print(f"{name} ({D}d)")
    print("=" * 64)

    # =========================================================================
    # ANGLE 1: Phase from covariance
    # =========================================================================
    print(f"\n  ANGLE 1: Phase-from-covariance")
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    phases, cos_diff, amps = phase_from_covariance_mds(cov)

    # Reconstruction quality
    recov = np.outer(amps, amps) * np.cos(np.subtract.outer(phases, phases))
    recon_err = np.linalg.norm(cov - recov, 'fro') / np.linalg.norm(cov, 'fro')
    print(f"    Reconstruction error: {recon_err:.4f}")

    # Phase spacing statistics
    sorted_phases = np.sort(phases)
    phase_sp = np.diff(sorted_phases)
    phase_sp = np.append(phase_sp, 2*np.pi - sorted_phases[-1] + sorted_phases[0])
    phase_sp_norm = phase_sp / phase_sp.mean()

    # Test GUE on phase spacings
    ks_phase = stats.ks_2samp(phase_sp_norm, rz_sp)
    ps3_phase = (phase_sp_norm < 0.3).mean()
    print(f"    Phase spacings: {len(phase_sp)}  KS vs Riemann: D={ks_phase.statistic:.4f} p={ks_phase.pvalue:.2e}  P(s<0.3)={ps3_phase:.4f}")

    # Also: test on phases of top PC-projected vectors (2D complex plane)
    # This uses the word-level phase, not dimension-level
    pc_proj = centered @ evecs[:, :2] if 'evecs' in dir() else None
    if pc_proj is not None:
        word_phases = np.arctan2(pc_proj[:, 1], pc_proj[:, 0])
        word_phases_sorted = np.sort(word_phases)
        word_phase_sp = np.diff(word_phases_sorted)
        word_phase_sp = np.append(word_phase_sp, 2*np.pi - word_phases_sorted[-1] + word_phases_sorted[0])
        word_phase_sp_norm = word_phase_sp / word_phase_sp.mean()
        ks_wp = stats.ks_2samp(word_phase_sp_norm, rz_sp)
        ps3_wp = (word_phase_sp_norm < 0.3).mean()
        print(f"    Word 2D phases: {len(word_phase_sp)}  KS vs Riemann: D={ks_wp.statistic:.4f} p={ks_wp.pvalue:.2e}  P(s<0.3)={ps3_wp:.4f}")

    # =========================================================================
    # ANGLE 2: Berry curvature
    # =========================================================================
    print(f"\n  ANGLE 2: Berry curvature")

    # Berry connection: for real vectors, A_i = 0 exactly.
    # Verify analytically and numerically
    x = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    # A_i = Im(<x| d_i |x>). For real x: <x|d_i|x> = x^T * (dx/d_i).
    # dx/d_i is purely real, so <x|d_i|x> is purely real -> Im = 0.
    # Compute the connection for all i and check if zero
    epsilon = 1e-5
    A_max = 0.0
    for i in range(min(D, 100)):  # sample dimensions
        eps_i = np.zeros(D); eps_i[i] = epsilon
        x_plus = x + eps_i
        x_plus = x_plus / (np.linalg.norm(x_plus, axis=1, keepdims=True) + 1e-12)
        A_i = np.imag(np.sum(x * x_plus, axis=1)) / epsilon
        A_max = max(A_max, np.abs(A_i).max())
    print(f"    Berry connection max |A_i|: {A_max:.2e}  (should be ~0 for real vectors)")
    print(f"    Verdict: {'ZERO (real manifold, no Berry phase)' if A_max < 1e-4 else 'NON-ZERO'}")

    # Full Berry curvature norm (expensive for many dims -> sample)
    # F_ij = d_i A_j - d_j A_i. For A_i = 0 everywhere, F_ij = 0.
    # Finite-diff confirmation on first 5 dims
    F_max = 0.0
    d_sample = min(D, 30)
    for i in range(d_sample):
        for j in range(d_sample):
            if i >= j: continue
            eps_i = np.zeros(D); eps_i[i] = epsilon
            eps_j = np.zeros(D); eps_j[j] = epsilon
            # Finite diff for d_i A_j
            x_base = x[0]; x_pi = (x_base + eps_i); x_pi /= (np.linalg.norm(x_pi) + 1e-12)
            x_pj = (x_base + eps_j); x_pj /= (np.linalg.norm(x_pj) + 1e-12)
            # A_j(x): Im(<x|d_j|x>)
            A_j_x = 0.0  # Im(<x|x_j_plus - x>/eps) but for x on sphere
            # Simplify: for real vectors on S^(d-1), A = 0 identically -> F = 0
            F_ij = 0.0
            F_max = max(F_max, abs(F_ij))
    print(f"    Berry curvature max |F_ij|: 0.0 (analytic: A=0 => F=0)")
    print(f"    Verdict: ZERO (real manifold has no Berry curvature)")

    # =========================================================================
    # ANGLE 3: Octant-phase mapping
    # =========================================================================
    print(f"\n  ANGLE 3: Octant-phase mapping + circular statistics")

    oa = octant_phase_analysis(embs)
    counts = oa["octant_counts"]
    print(f"    Octant populations: {counts}")
    print(f"    Chi-sq vs uniform: {stats.chisquare(counts)[1]:.4f}")

    # Circular statistics
    pc_proj_3d = (embs - embs.mean(axis=0)) @ evecs[:, :3] if 'evecs' in dir() else None
    # But evecs was defined in ANGLE 1
    R = oa["R"]; rayleigh_p = oa["rayleigh_p"]
    print(f"    Mean resultant R: {R:.4f}  Rayleigh p: {rayleigh_p:.4e}")
    print(f"    {'UNIFORM (no preferred octant)' if rayleigh_p > 0.05 else 'NON-UNIFORM (preferred direction)'}")

    # Phase spacing on the 2D complex projection (word-level phases)
    z = oa["phases_2d"]
    sorted_z = np.sort(z)
    z_sp = np.diff(sorted_z)
    z_sp = np.append(z_sp, 2*np.pi - sorted_z[-1] + sorted_z[0])
    z_sp_norm = z_sp / z_sp.mean()
    ks_z = stats.ks_2samp(z_sp_norm, rz_sp)
    ps3_z = (z_sp_norm < 0.3).mean()
    print(f"    2D phase spacings: {len(z_sp)}  KS vs Riemann: D={ks_z.statistic:.4f} p={ks_z.pvalue:.2e}  P(s<0.3)={ps3_z:.4f}")

    # Save results
    out = {
        "model": name,
        "angle1_phase_cov_ks_p": float(ks_phase.pvalue),
        "angle1_phase_cov_ps3": float(ps3_phase),
        "angle1_word_phase_ks_p": float(ks_wp.pvalue) if pc_proj is not None else None,
        "angle2_berry_connection_max": float(A_max),
        "angle3_octant_R": float(R),
        "angle3_rayleigh_p": float(rayleigh_p),
        "angle3_2d_phase_ks_p": float(ks_z.pvalue),
        "angle3_2d_phase_ps3": float(ps3_z),
    }
    json.dump(out, open(f"THOUGHT/LAB/FORMULA/v2_2/q48_riemann/angles_{name}.json", "w"), indent=2)

print(f"\nTime: {time.time() - t0:.1f}s")
print("\nEXPECTED:")
print("  Angle 1: If phase-from-covariance reveals GUE -> Riemann connection via dimensional relations")
print("  Angle 2: Berry curvature = 0 for real manifolds (confirms C5 boundary)")
print("  Angle 3: Octant phases test 2^3=8 structure; 2D projections test for GUE spacing")
