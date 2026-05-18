"""Q48: Comprehensive battery using compiled QGT library + complex geometry."""
import sys, json, time, ctypes
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)

# Load compiled library
so = "THOUGHT/LAB/EIGEN_ALIGNMENT/qgt_lib/build_minimal/lib/libdiffgeo.so"
lib = ctypes.CDLL(so)

class ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

class DiffgeoEngine(ctypes.Structure):
    pass

lib.bridge_engine_create.restype = ctypes.POINTER(DiffgeoEngine)
lib.bridge_compute_berry_curvature.argtypes = [
    ctypes.POINTER(DiffgeoEngine),
    ctypes.POINTER(ComplexDouble), ctypes.c_size_t,
    ctypes.POINTER(ComplexDouble), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
]
lib.bridge_compute_berry_curvature.restype = ctypes.c_bool

def berry_curvature_c(state, param_derivs):
    """Call C library for Berry curvature F_ij."""
    engine = lib.bridge_engine_create()
    dim, np_params = state.shape[0], param_derivs.shape[1]
    state_c = (ComplexDouble * dim)()
    for i in range(dim):
        state_c[i].real = state[i].real; state_c[i].imag = state[i].imag
    n_total = dim * np_params
    derivs_c = (ComplexDouble * n_total)()
    for j in range(np_params):
        for i in range(dim):
            idx = j * dim + i
            derivs_c[idx].real = param_derivs[i, j].real
            derivs_c[idx].imag = param_derivs[i, j].imag
    curv_out = (ctypes.c_double * (np_params * np_params))()
    lib.bridge_compute_berry_curvature(engine, state_c, dim, derivs_c, np_params, curv_out)
    return np.array(curv_out).reshape(np_params, np_params)

def unfold(eigenvalues):
    ev = np.sort(eigenvalues); ev = ev[ev > 1e-15]
    if len(ev) < 10: return np.array([])
    N = np.arange(1, len(ev)+1, dtype=float)
    s = UnivariateSpline(np.log(ev), np.log(N), s=len(ev)*0.001, k=3)
    ns = np.exp(s(np.log(ev))); ns = np.maximum(ns, 0.1)
    sp = np.diff(ns); return sp/sp.mean()

rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi); rz_sp /= rz_sp.mean()

print(f"Words: {len(WORDS)}, Riemann zeros: {len(rz)}")
print(f"Riemann P(s<0.3) = {(rz_sp<0.3).mean():.4f}")
t0 = time.time()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    n = len(WORDS)
    max_r = min(n - 1, D)

    print(f"\n{'='*64}")
    print(f"{name} ({D}d)")
    print("=" * 64)

    # Complexify
    z = hilbert(embs, axis=0).astype(np.complex128)
    z_norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    z = z / (z_norms + 1e-12)

    # ---- TEST A: Complex Hermitian Gram eigenvalues ----
    print(f"\n  TEST A: Complex Hermitian Gram eigenvalues")
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(z[i]) @ z[j]
            H[i, j] = v; H[j, i] = np.conj(v)
    ev_H = np.linalg.eigvalsh(H)
    ev_H_sig = ev_H[ev_H > 1e-12]
    if len(ev_H_sig) > 10:
        sp_H = unfold(ev_H_sig)
        ks_H = stats.ks_2samp(sp_H, rz_sp)
        print(f"  N spacings: {len(sp_H)}  KS vs Riemann: D={ks_H.statistic:.4f} p={ks_H.pvalue:.2e}  P(s<0.3)={(sp_H<0.3).mean():.4f}")
        print(f"  Verdict: {'MATCH' if ks_H.pvalue > 0.05 else 'REJECTED'}")

    # ---- TEST B: Berry curvature per word (via C library) ----
    print(f"\n  TEST B: Berry curvature eigenvalues (per-word, C library)")
    n_sample = min(50, n)
    np.random.seed(42); indices = np.random.choice(n, n_sample, replace=False)
    d_berry = min(D, 60)
    all_berry_evals = []
    for idx in indices:
        psi = z[idx]
        derivs = np.zeros((D, d_berry), dtype=np.complex128)
        param_idx = np.sort(np.random.choice(D, d_berry, replace=False))
        for k, j in enumerate(param_idx):
            derivs[j, k] = 1.0
        F = berry_curvature_c(psi, derivs)
        if F is not None:
            # F is antisymmetric: eigenvalues are imaginary pairs
            ev_F = np.linalg.eigvals(F)
            mags = np.abs(ev_F)
            mags = mags[mags > 1e-12]
            all_berry_evals.extend(mags.tolist())
    all_berry_evals = np.array(all_berry_evals)
    all_berry_evals = all_berry_evals[all_berry_evals > 1e-12]
    if len(all_berry_evals) > 10:
        sp_B = unfold(all_berry_evals)
        ks_B = stats.ks_2samp(sp_B, rz_sp)
        print(f"  N eigenvalues: {len(all_berry_evals)}  spacings: {len(sp_B)}  KS vs Riemann: D={ks_B.statistic:.4f} p={ks_B.pvalue:.2e}  P(s<0.3)={(sp_B<0.3).mean():.4f}")
        print(f"  Verdict: {'MATCH' if ks_B.pvalue > 0.05 else 'REJECTED'}")

    # ---- TEST C: Multi-word Berry curvature (library: combined state) ----
    print(f"\n  TEST C: Multi-word Berry curvature (combined state, C library)")
    n_joint = min(20, n)
    np.random.seed(99); joint_idx = np.random.choice(n, n_joint, replace=False)
    psi_joint = z[joint_idx].flatten()  # (n_joint * D,) stacked
    d_joint = min(D * n_joint, 500)
    derivs_joint = np.zeros((D * n_joint, d_joint), dtype=np.complex128)
    param_idx = np.sort(np.random.choice(D * n_joint, d_joint, replace=False))
    for k, j in enumerate(param_idx):
        derivs_joint[j, k] = 1.0
    F_joint = berry_curvature_c(psi_joint, derivs_joint)
    if F_joint is not None:
        ev_joint = np.abs(np.linalg.eigvals(F_joint))
        ev_joint = ev_joint[ev_joint > 1e-12]
        if len(ev_joint) > 10:
            sp_j = unfold(ev_joint)
            ks_j = stats.ks_2samp(sp_j, rz_sp)
            print(f"  N eigenvalues: {len(ev_joint)}  spacings: {len(sp_j)}  KS vs Riemann: D={ks_j.statistic:.4f} p={ks_j.pvalue:.2e}  P(s<0.3)={(sp_j<0.3).mean():.4f}")
            print(f"  Verdict: {'MATCH' if ks_j.pvalue > 0.05 else 'REJECTED'}")

    # ---- TEST D: Phase coherence of complex Gram ----
    print(f"\n  TEST D: Phase coherence from complex Gram")
    phases_H = np.angle(H[np.triu_indices(n, k=1)])
    sorted_phases = np.sort(phases_H)
    phase_sp = np.diff(sorted_phases)
    phase_sp = np.append(phase_sp, 2*np.pi - sorted_phases[-1] + sorted_phases[0])
    phase_sp_norm = phase_sp / phase_sp.mean()
    ks_PH = stats.ks_2samp(phase_sp_norm, rz_sp)
    print(f"  Phase spacings: {len(phase_sp)}  KS vs Riemann: D={ks_PH.statistic:.4f} p={ks_PH.pvalue:.2e}  P(s<0.3)={(phase_sp_norm<0.3).mean():.4f}")

print(f"\n{'='*64}")
print(f"SUMMARY: Complex-plane Q48 battery")
print(f"Time: {time.time() - t0:.1f}s")
