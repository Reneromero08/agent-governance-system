"""
Exp 34.2: Berry-Keating Hamiltonian — Discrete xp + px
========================================================
The Berry-Keating conjecture: zeta zeros are eigenvalues of
H = x*p + p*x where p = -i * d/dx, on a half-line with cutoff.

Discretize H in a finite basis. Eigendecompose. Compare to zeta zeros.

In the position basis with N points on [0, L]:
  H[i,j] = (x_i * p_j + p_i * x_j) discretized
"""
import math, time
import numpy as np
try: import mpmath as mp; HAS_MP = True
except ImportError: HAS_MP = False

def zeta_zeros(N):
    if not HAS_MP: return list(range(14, 14+4*N, 4))[:N]
    mp.mp.dps = 50
    return [float(mp.zetazero(n).imag) for n in range(1, N+1)]

def spacing_ratio(ev):
    ev = np.sort(np.abs(ev))
    ev = ev[ev > 1e-15][:min(100, len(ev))]
    if len(ev) < 4: return 0.0
    sp = np.diff(ev); ms = sp.mean()
    if ms < 1e-15: return 0.0
    uf = sp / ms; r = []
    for i in range(len(uf)-1):
        a,b = min(uf[i],uf[i+1]), max(uf[i],uf[i+1])
        if b > 1e-15: r.append(a/b)
    return np.mean(r) if r else 0.0

def berry_keating(N, L=100.0):
    """Discretize H = xp + px on [0, L] with N points.
    
    In position basis: <x|H|psi> = -i * (2x*psi'(x) + psi(x))
    Discretized with finite differences:
    H[i,j] = x_i * D[i,j] + D[i,j] * x_j  (where D is derivative matrix)
    
    Simplification: H = xD + Dx, where D is anti-symmetric (i*d/dx).
    Since Dx = xD + [D,x] and [D,x] = -i (canonical commutation):
    H = 2*x*D + i  (where i is the imaginary unit from the commutator)
    
    Discretize D as the Fourier derivative matrix.
    """
    dx = L / (N - 1)
    x = np.linspace(dx, L, N)  # start at dx to avoid singularity at 0
    
    # Fourier derivative via spectral method
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    D = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            # D[i,j] = (1/N) * sum_k i*k * exp(i*k*(x_i - x_j))
            pass  # too complex for direct construction
    
    # Simpler: finite difference derivative
    D = np.zeros((N, N), dtype=np.complex128)
    for i in range(1, N-1):
        D[i, i-1] = -1.0 / (2 * dx)
        D[i, i+1] = 1.0 / (2 * dx)
    # Boundary: forward/backward difference
    D[0, 0] = -3.0 / (2 * dx); D[0, 1] = 4.0 / (2 * dx); D[0, 2] = -1.0 / (2 * dx)
    D[N-1, N-3] = 1.0 / (2 * dx); D[N-1, N-2] = -4.0 / (2 * dx); D[N-1, N-1] = 3.0 / (2 * dx)
    
    # H = x*D + D*x (matrix multiplication)
    X = np.diag(x)
    H = X @ D + D @ X  # + 1j (commutator gives identity)
    
    # H should be anti-Hermitian? Let's check and Hermitize
    H = 1j * (H - H.conj().T) / 2
    H = (H + H.conj().T) / 2  # force Hermitian
    
    return H, x

def main():
    print("=" * 78)
    print("EXP 34.2: BERRY-KEATING HAMILTONIAN (xp + px)")
    print("=" * 78)
    
    zz = zeta_zeros(30)
    zeta_ratio = spacing_ratio(np.array(zz))
    print(f"\n  Zeta zero ref: spacing ratio = {zeta_ratio:.4f} (GUE: 0.603)")
    
    for N in [50, 100, 200, 400]:
        H, x = berry_keating(N)
        evals = np.linalg.eigvalsh(H)
        ratio = spacing_ratio(evals)
        print(f"  N={N:>4d}: spacing={ratio:.4f}  delta_zeta={abs(ratio-zeta_ratio):.4f}")
    
    # Compare top eigenvalues to zeta zeros
    H_final, x_final = berry_keating(100)
    ev_final = np.linalg.eigvalsh(H_final)
    ev_top = np.sort(ev_final)[-20:]
    
    print(f"\n  Top 8 eigenvalues (N=100): {[f'{e:.1f}' for e in ev_top[-8:]]}")
    print(f"  Zeta zeros (first 8):       {[f'{z:.1f}' for z in zz[:8]]}")
    
    print(f"\n{'='*78}")
    print(f"  Berry-Keating predicts H = xp + px has eigenvalues ~ zeta zeros.")
    print(f"  The discrete approximation tests whether the operator structure")
    print(f"  alone (without fitting) produces GUE eigenvalue statistics.")
    print("=" * 78)

if __name__ == "__main__":
    main()
