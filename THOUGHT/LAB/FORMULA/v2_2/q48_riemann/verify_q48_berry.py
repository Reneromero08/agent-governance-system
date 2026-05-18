"""Q48: Berry curvature via compiled QGT library on complexified embeddings."""
import sys, json, time, ctypes
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)

# Load shared library
so_path = "THOUGHT/LAB/EIGEN_ALIGNMENT/qgt_lib/build_minimal/lib/libdiffgeo.so"
lib = ctypes.CDLL(so_path)

# Struct types
class ComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

class DiffgeoEngine(ctypes.Structure):
    pass  # opaque

# Function signatures
lib.bridge_engine_create.restype = ctypes.POINTER(DiffgeoEngine)
lib.bridge_compute_berry_curvature.argtypes = [
    ctypes.POINTER(DiffgeoEngine),
    ctypes.POINTER(ComplexDouble),  # state
    ctypes.c_size_t,                # dim
    ctypes.POINTER(ComplexDouble),  # param_derivatives (dim x num_params)
    ctypes.c_size_t,                # num_params
    ctypes.POINTER(ctypes.c_double),# curvature_out (num_params x num_params)
]
lib.bridge_compute_berry_curvature.restype = ctypes.c_bool

def compute_berry_curvature_c(state_complex, param_derivs_complex):
    """Call C library to compute Berry curvature F_ij.
    state: (dim,) complex
    param_derivs: (dim, num_params) complex — d|psi>/d(param_j) for dim i
    Returns: (num_params, num_params) real Berry curvature tensor
    """
    engine = lib.bridge_engine_create()
    dim = state_complex.shape[0]
    num_params = param_derivs_complex.shape[1]

    # Pack state as ComplexDouble array
    state_c = (ComplexDouble * dim)()
    for i in range(dim):
        state_c[i].real = state_complex[i].real
        state_c[i].imag = state_complex[i].imag

    # Pack derivatives: column-major parameters
    # param_derivs[i, j] = d(psi_i)/d(param_j)
    n_total = dim * num_params
    derivs_c = (ComplexDouble * n_total)()
    for j in range(num_params):
        for i in range(dim):
            idx = j * dim + i  # column-major: parameter j first
            derivs_c[idx].real = param_derivs_complex[i, j].real
            derivs_c[idx].imag = param_derivs_complex[i, j].imag

    # Output: (num_params, num_params) flat
    curvature_out = (ctypes.c_double * (num_params * num_params))()

    success = lib.bridge_compute_berry_curvature(
        engine, state_c, dim, derivs_c, num_params, curvature_out
    )

    if not success:
        return None

    F = np.array(curvature_out).reshape(num_params, num_params)
    return F


def unfold(eigenvalues):
    ev = np.sort(eigenvalues); ev = ev[ev > 1e-15]
    if len(ev) < 10: return np.array([])
    N = np.arange(1, len(ev)+1, dtype=float)
    s = UnivariateSpline(np.log(ev), np.log(N), s=len(ev)*0.001, k=3)
    ns = np.exp(s(np.log(ev))); ns = np.maximum(ns, 0.1)
    sp = np.diff(ns); return sp/sp.mean()


# Load Riemann reference
rz = np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp = np.diff(rz) / (2 * np.pi); rz_sp /= rz_sp.mean()

t0 = time.time()

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    n = len(WORDS)

    print(f"\n{'='*64}")
    print(f"{name} ({D}d, {n} words)")
    print("=" * 64)

    # Complexify: Hilbert transform per dimension
    z = hilbert(embs, axis=0).astype(np.complex128)
    # Normalize complex vectors
    z_norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    z = z / (z_norms + 1e-12)

    # Sample embeddings (Berry curvature is O(D^3), expensive for all 777 words)
    n_sample = min(50, n)
    np.random.seed(42)
    indices = np.random.choice(n, n_sample, replace=False)

    # For each sampled word, compute Berry curvature
    F_norms = []
    for idx in indices:
        psi = z[idx]  # (D,) complex state

        # Parameter derivatives: d|psi>/d(x_j) = e_j (unit vector in dim j)
        # Sample parameters to keep computation tractable
        d_sample = min(D, 100)
        param_indices = np.sort(np.random.choice(D, d_sample, replace=False))
        derivs = np.zeros((D, d_sample), dtype=np.complex128)
        for k, j in enumerate(param_indices):
            derivs[j, k] = 1.0  # unit vector in dim j

        F = compute_berry_curvature_c(psi, derivs)
        if F is not None:
            F_norm = np.linalg.norm(F, 'fro')
            F_norms.append(F_norm)

    F_norms = np.array(F_norms)
    print(f"  Berry curvature ||F||: mean={F_norms.mean():.6e}  max={F_norms.max():.6e}  non_zero={(F_norms > 1e-10).mean()*100:.0f}%")

    # Compute eigenvalues from the Hermitian QGT (real part = metric, imag part = curvature)
    # For each word, QGT = g + i*Omega is a Hermitian matrix
    # Its eigenvalues are real. Test GUE spacing.
    all_qgt_evals = []
    for idx in indices[:10]:  # 10 words, each producing a d_sample x d_sample QGT
        psi = z[idx]
        d_sample = min(D, 60)
        param_indices = np.random.choice(D, d_sample, replace=False)
        derivs = np.zeros((D, d_sample), dtype=np.complex128)
        for k, j in enumerate(param_indices):
            derivs[j, k] = 1.0

        # Compute QGT: Q_ij = <d_i|d_j> - <d_i|psi><psi|d_j>
        # d_i = e_i = unit vector
        # <e_i|e_j> = delta_ij
        # <e_i|psi> = psi_i
        # Q_ij = delta_ij - conj(psi_i) * psi_j
        # Q = I - |psi><psi| (projector)
        psi_col = psi.reshape(-1, 1)
        psi_row = psi.conj().reshape(1, -1)
        Q = np.eye(D, dtype=np.complex128) - psi_col @ psi_row
        # Subset to sampled parameters
        Q_sub = Q[param_indices][:, param_indices]
        # Ensure Hermitian
        Q_sub = (Q_sub + Q_sub.conj().T) / 2
        evals = np.linalg.eigvalsh(Q_sub)
        all_qgt_evals.extend(evals.tolist())

    all_qgt_evals = np.array(all_qgt_evals)
    all_qgt_evals = all_qgt_evals[all_qgt_evals > 1e-12]
    if len(all_qgt_evals) > 10:
        sp_qgt = unfold(all_qgt_evals)
        ks_qgt = stats.ks_2samp(sp_qgt, rz_sp)
        ps3_qgt = (sp_qgt < 0.3).mean()
        print(f"  QGT eigenvalues: {len(sp_qgt)} spacings  KS vs Riemann: D={ks_qgt.statistic:.4f} p={ks_qgt.pvalue:.2e}  P(s<0.3)={ps3_qgt:.4f}")

print(f"\nTime: {time.time() - t0:.1f}s")
