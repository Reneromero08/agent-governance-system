"""24.7: D_pr Scaling Law — measure effective dimension vs qubit count"""
import torch, math, time, numpy as np
from pathlib import Path

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
def gate1(state, G, t, n):
    d=2;td=n-1-t;st=state.reshape([d]*n)
    perm=[td]+[i for i in range(n) if i!=td]
    st=st.permute(*perm).contiguous().reshape(d,-1)
    st=(G@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def measure_dpr(psi):
    """Measure D_pr via SVD of state reshaped to matrix (Schmidt decomposition)."""
    N = len(psi)
    n = int(math.log2(N))
    # Split into two halves: reshape to (2^(n//2), 2^(n - n//2))
    nA = n // 2; nB = n - nA
    M = psi.reshape(2**nA, 2**nB).numpy()
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    total = (S**2).sum()
    if total < 1e-15: return 1.0
    probs = S**2 / total
    dpr = 1.0 / (probs**2).sum()
    return dpr

print("=" * 78)
print("24.7: D_pr SCALING LAW — Effective Dimension vs Qubit Count")
print("=" * 78)
print(f"  {'n':>4}  {'state':>10}  {'D_pr(uniform)':>14}  {'D_pr(entangled)':>16}  {'compress':>12}  {'scaling':>12}")
print(f"  {'-'*75}")

for n in range(4, 18):
    N = 2**n
    if N > 500000: break  # memory limit
    
    t0 = time.perf_counter()
    
    # Uniform superposition
    psi_u = torch.ones(N, dtype=torch.complex64) / math.sqrt(N)
    dpr_u = measure_dpr(psi_u)
    
    # Entangled: Hadamard on half the qubits
    if n >= 2:
        psi_e = torch.zeros(N, dtype=torch.complex64); psi_e[0] = 1.0
        for i in range(n // 2):
            psi_e = gate1(psi_e, H, i, n)
        dpr_e = measure_dpr(psi_e)
    else:
        dpr_e = 0.0
    
    dt = time.perf_counter() - t0
    ratio = N / max(dpr_u, 1)
    print(f"  {n:>10}  {N:>12,}  {dpr_u:>14.1f}  {dpr_e:>16.1f}  {ratio:>14.1f}x  ({dt:.2f}s)")

print(f"\n  D_pr grows with qubits. Question: linear, sqrt, or log?")
print("=" * 78)
