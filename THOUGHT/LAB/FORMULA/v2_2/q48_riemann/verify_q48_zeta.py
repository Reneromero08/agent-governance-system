"""Q48 angle 2: Spectral zeta function — zeros on the critical line?"""
import sys; sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
import numpy as np
from sentence_transformers import SentenceTransformer
from large_anchor_generator import ANCHOR_1024

WORDS = list(ANCHOR_1024)  # type: ignore

def zeta_sem(eigenvalues, s):
    """zeta_sem(s) = sum lambda_k^(-s) for complex s."""
    ev = eigenvalues[eigenvalues > 1e-12]
    return np.sum(ev ** (-s))

def find_zeros(ev, sigma=0.5, t_range=(0, 100), n_points=2000):
    """Search for sign changes (potential zeros) on line Re(s)=sigma."""
    ts = np.linspace(t_range[0], t_range[1], n_points)
    real_vals = np.zeros(n_points, dtype=np.float64)
    imag_vals = np.zeros(n_points, dtype=np.float64)

    for i, t in enumerate(ts):
        z = zeta_sem(ev, complex(sigma, t))
        real_vals[i] = z.real
        imag_vals[i] = z.imag

    # Find sign changes in real part
    sign_real = np.sign(real_vals)
    sign_imag = np.sign(imag_vals)
    zero_crossings_real = np.where(np.diff(sign_real) != 0)[0]
    zero_crossings_imag = np.where(np.diff(sign_imag) != 0)[0]

    # Potential zeros: where both real and imag cross zero near each other
    potential_zeros = []
    for r_idx in zero_crossings_real:
        t_r = ts[r_idx]
        # Check if imag crosses near same t
        nearby = zero_crossings_imag[np.abs(ts[zero_crossings_imag] - t_r) < 0.5]
        if len(nearby) > 0:
            potential_zeros.append(t_r)

    return ts, real_vals, imag_vals, potential_zeros


for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    max_r = min(len(WORDS)-1, D)

    # Gram matrix eigenvalues (top max_r only — signal)
    gram = embs @ embs.T
    ev_all = np.sort(np.linalg.eigvalsh(gram))
    ev = ev_all[-max_r:]
    ev = ev[ev > 1e-12]

    print(f"\n{'='*60}")
    print(f"{name} (D={D}, signal evals: {len(ev)})")

    # Scan critical line for zeros
    ts, real_vals, imag_vals, pz = find_zeros(ev, sigma=0.5, t_range=(0, 50), n_points=2000)
    print(f"  Potential zeros on Re(s)=0.5, t in [0,50]: {len(pz)}")
    if len(pz) > 0:
        for t0 in pz[:10]:
            z0 = zeta_sem(ev, complex(0.5, t0))
            print(f"    t={t0:.4f}  zeta={z0.real:.6f}+{z0.imag:.6f}i  |z|={abs(z0):.6f}")

    # Also test: does zeta_sem(1) converge? (Riemann: zeta(1) diverges)
    z1 = zeta_sem(ev, 1.0)
    print(f"  zeta_sem(1) = {z1:.6f}  (Riemann: diverges)")

    # Test functional equation: zeta_sem(s) vs zeta_sem(1-s)?
    # Riemann: pi^(-s/2) Gamma(s/2) zeta(s) = pi^(-(1-s)/2) Gamma((1-s)/2) zeta(1-s)
    # For spectral: any symmetry?
    for s_test in [0.25, 0.5, 0.75]:
        zs = zeta_sem(ev, complex(s_test, 0))
        z1s = zeta_sem(ev, complex(1 - s_test, 0))
        ratio = zs / z1s if abs(z1s) > 1e-10 else float('inf')
        print(f"  zeta({s_test})={zs:.4f}  zeta({1-s_test})={z1s:.4f}  ratio={ratio:.4f}")

# Compare: Riemann zeta has simple pole at s=1 (diverges)
# Spectral zeta: finite at s=1 since sum lambda_k^(-1) converges for finite k
print(f"\n{'='*60}")
print("VERDICT:")
print("  Riemann zeta: pole at s=1 (harmonic series diverges)")
print("  Spectral zeta: finite at s=1 (finite sum of eigenvalues)")
print("  Riemann zeta: infinite zeros on critical line Re(s)=1/2")
print("  Spectral zeta: finite zeros (or none) — finite polynomial-like sum")
print("  The spectral zeta is a finite Dirichlet polynomial, not a true zeta function.")
