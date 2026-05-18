"""Q48 push: Fubini-Study metric eigenvalues via compiled C library."""
import sys, json, time, ctypes
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)
so = "THOUGHT/LAB/EIGEN_ALIGNMENT/qgt_lib/build_minimal/lib/libdiffgeo.so"
lib = ctypes.CDLL(so)

class ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

class DiffgeoEngine(ctypes.Structure):
    pass

lib.bridge_engine_create.restype = ctypes.POINTER(DiffgeoEngine)
lib.bridge_compute_fubini_study.argtypes = [
    ctypes.POINTER(DiffgeoEngine),
    ctypes.POINTER(ComplexDouble), ctypes.c_size_t,
    ctypes.POINTER(ComplexDouble), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
]
lib.bridge_compute_fubini_study.restype = ctypes.c_bool

def fs_metric_c(state, param_derivs):
    """Compute Fubini-Study metric via C library."""
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
    metric_out = (ctypes.c_double * (np_params * np_params))()
    lib.bridge_compute_fubini_study(engine, state_c, dim, derivs_c, np_params, metric_out)
    return np.array(metric_out).reshape(np_params, np_params)

def unfold(eigenvalues):
    ev = np.sort(eigenvalues); ev = ev[ev > 1e-15]
    if len(ev) < 10: return np.array([])
    N = np.arange(1, len(ev)+1, dtype=float)
    s = UnivariateSpline(np.log(ev), np.log(N), s=len(ev)*0.001, k=3)
    ns = np.exp(s(np.log(ev))); ns = np.maximum(ns, 0.1)
    sp = np.diff(ns); return sp/sp.mean()

rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi); rz_sp /= rz_sp.mean()

print(f"Words: {len(WORDS)}")
print(f"Riemann P(s<0.3) = {(rz_sp<0.3).mean():.4f}")
t0 = time.time()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    n = len(WORDS)
    print(f"\n{'='*64}")
    print(f"{name} ({D}d)")

    # Complexify
    z = hilbert(embs, axis=0).astype(np.complex128)
    z_norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    z = z / (z_norms + 1e-12)

    # ---- Complex Hermitian Gram (baseline) ----
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(z[i]) @ z[j]
            H[i, j] = v; H[j, i] = np.conj(v)
    ev_H = np.linalg.eigvalsh(H)
    ev_H = ev_H[ev_H > 1e-12]
    sp_H = unfold(ev_H)
    ks_H = stats.ks_2samp(sp_H, rz_sp)
    print(f"  Gram baseline: {len(sp_H)} spacings  KS p={ks_H.pvalue:.2e}  P(s<0.3)={(sp_H<0.3).mean():.4f}")

    # ---- Fubini-Study metric per word (C library) ----
    n_sample = min(40, n)
    np.random.seed(42); indices = np.random.choice(n, n_sample, replace=False)
    d_fs = min(D, 80)
    all_fs_evals = []

    for idx in indices:
        psi = z[idx]
        param_idx = np.sort(np.random.choice(D, d_fs, replace=False))
        derivs = np.zeros((D, d_fs), dtype=np.complex128)
        for k, j in enumerate(param_idx):
            derivs[j, k] = 1.0

        G = fs_metric_c(psi, derivs)
        if G is not None:
            ev_G = np.linalg.eigvalsh(G)
            ev_G = ev_G[ev_G > 1e-12]
            all_fs_evals.extend(ev_G.tolist())

    all_fs_evals = np.array(all_fs_evals)
    all_fs_evals = all_fs_evals[all_fs_evals > 1e-12]
    if len(all_fs_evals) > 10:
        sp_FS = unfold(all_fs_evals)
        ks_FS = stats.ks_2samp(sp_FS, rz_sp)
        print(f"  FS metric (lib): {len(all_fs_evals)} evals {len(sp_FS)} spacings  KS p={ks_FS.pvalue:.2e}  P(s<0.3)={(sp_FS<0.3).mean():.4f}")

    # ---- Combined FS metric across words (treat as block-diagonal state) ----
    n_joint = min(10, n)
    np.random.seed(99); joint_idx = np.random.choice(n, n_joint, replace=False)
    psi_joint = z[joint_idx].flatten()
    dim_joint = D * n_joint
    d_joint = min(dim_joint, 200)
    param_idx = np.sort(np.random.choice(dim_joint, d_joint, replace=False))
    derivs_joint = np.zeros((dim_joint, d_joint), dtype=np.complex128)
    for k, j in enumerate(param_idx):
        derivs_joint[j, k] = 1.0

    G_joint = fs_metric_c(psi_joint, derivs_joint)
    if G_joint is not None:
        ev_Gj = np.linalg.eigvalsh(G_joint)
        ev_Gj = ev_Gj[ev_Gj > 1e-12]
        if len(ev_Gj) > 10:
            sp_Gj = unfold(ev_Gj)
            ks_Gj = stats.ks_2samp(sp_Gj, rz_sp)
            print(f"  FS joint ({n_joint} words): {len(ev_Gj)} evals {len(sp_Gj)} spacings  KS p={ks_Gj.pvalue:.2e}  P(s<0.3)={(sp_Gj<0.3).mean():.4f}")

print(f"\nTime: {time.time() - t0:.1f}s")
