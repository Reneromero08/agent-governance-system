"""24.3: Catalytic Massive Scale — fast init, direct gates"""
import math, torch, time

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
X = torch.tensor([[0,1],[1,0]], dtype=torch.complex64)
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64)
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=torch.complex64)
CZ = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=torch.complex64)

def gate1(state, G, t, n):
    d = 2; st = state.reshape([d]*n)
    perm = [t] + [i for i in range(n) if i != t]
    st = st.permute(perm).reshape(d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(inv).reshape(-1)

def gate2(state, G, c, t, n):
    d = 2; st = state.reshape([d]*n)
    perm = [c, t] + [i for i in range(n) if i not in (c, t)]
    st = st.permute(perm).reshape(d*d, -1)
    st = (G @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(inv).reshape(-1)

def overlap(a, b):
    return torch.abs(torch.dot(a.conj(), b)).item()

def test(n, entangled, cycles, depth):
    """GHZ on `entangled` qubits, borrow to work qubits, restore."""
    N = 2**n
    # Fast init: state[0] = 1.0
    psi = torch.zeros(N, dtype=torch.complex64)
    psi[0] = 1.0+0j
    
    # GHZ on first `entangled` qubits
    psi = gate1(psi, H, 0, n)
    for i in range(entangled - 1):
        psi = gate2(psi, CNOT, i, i+1, n)
    
    initial = psi.clone()
    work = list(range(entangled, n))
    
    for cycle in range(cycles):
        for wi, wq in enumerate(work):
            eq = wi % entangled
            psi = gate2(psi, CNOT, eq, wq, n)
            for _ in range(depth):
                psi = gate1(psi, H if (wi+cycle)%2==0 else Z, wq, n)
                psi = gate2(psi, CZ, eq, wq, n)
                psi = gate1(psi, X if cycle%3==0 else Z, eq, n)
        
        for wi, wq in reversed(list(enumerate(work))):
            eq = wi % entangled
            for _ in range(depth):
                psi = gate1(psi, X if cycle%3==0 else Z, eq, n)
                psi = gate2(psi, CZ, eq, wq, n)
                psi = gate1(psi, H if (wi+cycle)%2==0 else Z, wq, n)
            psi = gate2(psi, CNOT, eq, wq, n)
    
    return initial, psi

print("=" * 78)
print("24.3: CATALYTIC MASSIVE SCALE")
print("=" * 78)
print(f"  {'n':>6}  {'ent':>5}  {'cyc':>5}  {'dep':>5}  {'state':>10}  {'overlap':>10}  {'time':>8}")
print(f"  {'-'*65}")

import numpy as np
overlaps = []
for n, ent, cyc, dep in [(10,5,2,2),(12,6,3,2),(14,7,2,3),(16,8,3,3),(18,9,4,3)]:
    t0 = time.perf_counter()
    try:
        initial, final = test(n, ent, cyc, dep)
        ov = overlap(initial, final)
        dt = time.perf_counter() - t0
        overlaps.append(ov)
        print(f"  {n:>6}  {ent:>5}  {cyc:>5}  {dep:>5}  {2**n:>10,}  {ov:>10.6f}  {dt:>7.2f}s")
    except Exception as e:
        print(f"  {n:>6}  {ent:>5}  {cyc:>5}  {dep:>5}  {2**n:>10,}  {'ERR':>10}  {str(e)[:30]}")

print(f"\n  Overlap stats: mean={np.mean(overlaps):.6f}  std={np.std(overlaps):.6f}  "
      f"min={np.min(overlaps):.6f}  max={np.max(overlaps):.6f}")
print("=" * 78)
