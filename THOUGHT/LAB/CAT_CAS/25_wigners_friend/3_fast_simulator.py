"""25.3: Fast Quantum Simulator — Hadamard transform + push qubits"""
import torch, math, time, numpy as np

H = torch.tensor([[1,1],[1,-1]], dtype=torch.float32)/math.sqrt(2)
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=torch.complex64)
SWAP = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.complex64)

def hadamard_all(state, n_qubits, start=0):
    """Fast Hadamard on n_qubits starting at 'start'. Tensor product approach.
    Much faster than per-qubit gate1 loop. O(N log N) total."""
    N = len(state)
    # Reshape to tensor: first 'start' qubits, then 'n_qubits', then rest
    n_before = start; n_after = int(math.log2(N)) - start - n_qubits
    st = state.reshape([2]*n_before + [2]*n_qubits + [2]*n_after)
    
    # Apply H to each of the n_qubits dimensions
    for d in range(n_before, n_before + n_qubits):
        # Bring dimension d to front
        ndims = st.ndim
        perm = [d] + [i for i in range(ndims) if i != d]
        st = st.permute(*perm).contiguous().reshape(2, -1)
        # H gate
        st = H.to(st.dtype) @ st
        # Restore shape
        st = st.reshape([2]*ndims)
        inv = [0]*ndims
        for i, p in enumerate(perm): inv[p] = i
        st = st.permute(*inv).contiguous()
    
    return st.reshape(-1)

def gate1(state, G, t, n):
    d = 2; td = n - 1 - t; st = state.reshape([d]*n)
    perm = [td] + [i for i in range(n) if i != td]
    st = st.permute(*perm).contiguous().reshape(d, -1)
    st = (G.to(st.dtype) @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def gate2(state, G, c, t, n):
    d = 2; cd = n - 1 - c; td = n - 1 - t; st = state.reshape([d]*n)
    perm = [cd, td] + [i for i in range(n) if i not in (cd, td)]
    st = st.permute(*perm).contiguous().reshape(d*d, -1)
    st = (G.to(st.dtype) @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def build_mod_mult(a, N, n_num):
    d = 2**n_num; M = torch.zeros(d, d, dtype=torch.complex64)
    for x in range(d): y = (a * x) % N if x < N else x; M[y, x] = 1.0
    return M

def controlled_mod_mult(state, a, power, N, ctrl, num_start, n_num, n):
    U = build_mod_mult(pow(a, power, N), N, n_num)
    d_num = 2**n_num
    CU = torch.zeros(d_num*2, d_num*2, dtype=torch.complex64)
    CU[:d_num, :d_num] = torch.eye(d_num, dtype=torch.complex64)
    CU[d_num:, d_num:] = U
    d = 2; st = state.reshape([d]*n)
    cd = n - 1 - ctrl
    num_dims = [n - 1 - (num_start + n_num - 1 - i) for i in range(n_num)]
    perm = [cd] + num_dims + [i for i in range(n) if i != cd and i not in num_dims]
    st = st.permute(*perm).contiguous().reshape(d * d_num, -1)
    st = (CU @ st).reshape([d]*n)
    inv = [0]*n
    for i, p in enumerate(perm): inv[p] = i
    return st.permute(*inv).contiguous().reshape(-1)

def qft_inverse(state, reg_start, n_reg, n):
    for i in range(n_reg):
        qi = reg_start + i; state = gate1(state, H.to(torch.complex64), qi, n)
        for j in range(i+1, n_reg):
            qj = reg_start + j; angle = -math.pi / (2**(j-i))
            Rk = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                [0,0,0,complex(math.cos(angle), math.sin(angle))]], dtype=torch.complex64)
            state = gate2(state, Rk, qi, qj, n)
    for i in range(n_reg // 2):
        a = reg_start + i; b = reg_start + n_reg - 1 - i
        state = gate2(state, SWAP, a, b, n)
    return state

def gcd(a, b):
    while b: a, b = b, a % b
    return a

from fractions import Fraction

print("=" * 78)
print("25.3: FAST QUANTUM SIMULATOR + PUSH QUBITS")
print("=" * 78)

for N, a, n_p, n_n in [(15,2,4,4),(21,2,5,5),(35,2,6,6),(15,2,8,4),(21,2,6,5)]:
    n = n_p + n_n
    Nstate = 2**n
    if Nstate > 500000: continue
    
    t0 = time.perf_counter()
    psi = torch.zeros(Nstate, dtype=torch.complex64)
    psi[1 << n_p] = 1.0
    
    # FAST Hadamard on all period qubits
    psi = hadamard_all(psi, n_p, start=0)
    
    for i in range(n_p):
        psi = controlled_mod_mult(psi, a, 2**i, N, i, n_p, n_n, n)
    
    psi = qft_inverse(psi, 0, n_p, n)
    
    from fractions import Fraction
    probs = torch.zeros(2**n_p)
    for k in range(2**n_p):
        p = 0.0
        for x in range(2**n_n): p += (psi[k + x*(2**n_p)] * psi[k + x*(2**n_p)].conj()).real.item()
        probs[k] = p
    
    top_vals, top_idxs = torch.topk(probs, min(8, len(probs)))
    dt = time.perf_counter() - t0
    
    factored = False
    for i in range(len(top_idxs)):
        k = top_idxs[i].item()
        if k > 0:
            r = Fraction(k, 2**n_p).limit_denominator(N).denominator
        else: r = 0
        if r > 0 and pow(a,r,N)==1 and r%2==0:
            v = pow(a, r//2, N)
            g1 = gcd(v-1,N); g2 = gcd(v+1,N)
            if g1*g2==N and g1>1 and g2>1 and not factored:
                print(f"  N={N}: {g1}x{g2} (r={r}, {n}q, state={Nstate}, {dt:.3f}s)")
                factored = True
    if not factored:
        print(f"  N={N}: FAILED ({n}q, state={Nstate}, {dt:.3f}s)")

# Push qubit count — just Hadamard (no Shor circuit, too expensive)
print(f"\n  Pushing raw qubit count with Hadamard transform:")
for n_q in [18, 20, 22, 24, 26]:
    N = 2**n_q
    if N * 8 > 4e9: print(f"  {n_q} qubits: state={N:,} SKIP (>4GB)"); continue
    try:
        t0 = time.perf_counter()
        psi = torch.ones(N, dtype=torch.complex64) / math.sqrt(N)
        psi = hadamard_all(psi, n_q, start=0)
        dt = time.perf_counter() - t0
        norm = psi.norm().item()
        print(f"  {n_q} qubits: state={N:,} Hadamard time={dt:.3f}s norm={norm:.4f}")
    except Exception as e:
        print(f"  {n_q} qubits: state={N:,} ERROR: {e}")
print("=" * 78)
