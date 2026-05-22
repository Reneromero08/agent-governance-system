"""
Hawking Quantum Simulator v3 — True Compressed Gates
======================================================
gate_A: O(NA * k) — applies directly to U, never touches Vh
gate_B: O(NB * k) — applies directly to Vh, never touches U
gate_AB: O(NA * NB) only for entangling gates (n_p of them)

Push qubit count without building full state.
"""
import torch, math, time, numpy as np
from fractions import Fraction

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/math.sqrt(2)
SWAP = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.complex64)

def gcd(a,b):
    while b: a,b = b,a%b
    return a

def gate1_vec(state, G, t, n):
    """Apply gate G to qubit t of flat state vector."""
    d=2;td=n-1-t;st=state.reshape([d]*n)
    perm=[td]+[i for i in range(n) if i!=td]
    st=st.permute(*perm).contiguous().reshape(d,-1)
    st=(G.to(torch.complex64)@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def apply_gate_to_matrix(M, G, t, n_qubits, axis=0):
    """Apply G to qubit t of matrix M. axis=0 for rows, axis=1 for cols.
    M has shape (2^n_qubits, k) or (k, 2^n_qubits)."""
    d=2
    if axis == 0:  # rows correspond to qubits
        M_reshaped = M.reshape([d]*n_qubits + [M.shape[1]])
        td = n_qubits - 1 - t
        perm = [td] + [i for i in range(n_qubits) if i != td] + [n_qubits]
        M_reshaped = M_reshaped.permute(*perm).contiguous().reshape(d, -1, M.shape[1])
        M_reshaped = (G.to(torch.complex64) @ M_reshaped.reshape(d, -1)).reshape(d, M.shape[0]//2, M.shape[1])
        M_reshaped = M_reshaped.reshape([d]*n_qubits + [M.shape[1]])
        inv = [0]*(n_qubits+1)
        for i,p in enumerate(perm): inv[p]=i
        return M_reshaped.permute(*inv).contiguous().reshape(M.shape[0], M.shape[1])
    else:  # columns
        M_reshaped = M.reshape([M.shape[0]] + [d]*n_qubits)
        td = n_qubits - t
        perm = [0] + [td+1] + [i+1 for i in range(n_qubits) if i != td]
        M_reshaped = M_reshaped.permute(*perm).contiguous().reshape(M.shape[0]*d, -1)
        M_t = M_reshaped.t()  # (NB//2, NA*2)
        M_t = M_t.reshape(-1, 2, M.shape[0]).permute(1,0,2).reshape(2, -1)
        M_t = (G.to(torch.complex64) @ M_t).reshape(2, M_reshaped.shape[0]//2, M.shape[0]).permute(1,0,2).reshape(-1, M.shape[0])
        # Reshape back through inverse perm
        M_out = M_t.t().reshape([M.shape[0]] + [d]*n_qubits)
        inv = [0]*(n_qubits+1)
        for i,p in enumerate(perm): inv[p]=i
        return M_out.permute(*inv).contiguous().reshape(M.shape[0], M.shape[1])


class HawkingState:
    """Compressed state: psi = U @ diag(S) @ Vh. Schmidt form."""
    def __init__(self, nA, nB):
        self.nA=nA; self.nB=nB; self.NA=2**nA; self.NB=2**nB
        
    def init_zero(self):
        self.U=torch.zeros(self.NA,1,dtype=torch.complex64); self.U[0,0]=1.0
        self.S=torch.ones(1)
        self.Vh=torch.zeros(1,self.NB,dtype=torch.complex64); self.Vh[0,0]=1.0
    
    def init_state(self, psi):
        """Initialize from full state vector."""
        M=psi.reshape(self.NA,self.NB)
        U,S,Vh=torch.linalg.svd(M,full_matrices=False)
        Smax=S[0].item(); keep=int((S>Smax*1e-6).sum().item())
        keep=max(1,keep)
        self.U=U[:,:keep]; self.S=S[:keep]; self.Vh=Vh[:keep,:]
    
    def gate_A(self, G, t):
        """Gate on qubit t in A. Applies directly to U. O(NA * k)."""
        self.U = apply_gate_to_matrix(self.U, G, t, self.nA, axis=0)
    
    def gate_B(self, G, t):
        """Gate on qubit t in B. Applies directly to Vh. O(NB * k)."""
        self.Vh = apply_gate_to_matrix(self.Vh.T, G, t, self.nB, axis=0).T
    
    def gate_AB_full(self, psi, a_ctrl, b_targets, CU):
        """Apply cross-partition controlled gate. Expensive — full state needed."""
        n = self.nA + self.nB
        st = psi.reshape([2]*n)
        cd = n - 1 - a_ctrl
        n_n = len(b_targets)
        nd = [n - 1 - (self.nA + self.nB - 1 - j) for j in range(n_n)]
        perm = [cd] + nd + [j for j in range(n) if j != cd and j not in nd]
        d_num = 2**n_n
        st = st.permute(*perm).contiguous().reshape(2 * d_num, -1)
        print(f"      gate_AB: ctrl={a_ctrl} perm={perm} nd={nd} st before CU: shape={list(st.shape)}")
        st = (CU.to(torch.complex64) @ st)
        sv = torch.linalg.svd(st.reshape(self.NA,self.NB), full_matrices=False)[1][:4]
        print(f"      after CU S={sv.tolist()}")
        st = st.reshape([2]*n)
        inv = [0]*n
        for j,p in enumerate(perm): inv[p]=j
        return st.permute(*inv).contiguous().reshape(self.NA, self.NB)
    
    def decode(self):
        return ((self.U * self.S.unsqueeze(0)) @ self.Vh).reshape(-1)
    
    def rank(self):
        return len(self.S)


def shor_compressed(N, a, n_p, n_n):
    """Shor on compressed state. H/QFT gates stay compressed. CU gates expand."""
    hs = HawkingState(n_p, n_n)
    hs.init_zero()
    # Set number reg = |1>
    hs.Vh = torch.zeros(1, hs.NB, dtype=torch.complex64)
    hs.Vh[0, 1] = 1.0
    
    # H on period (compressed)
    for i in range(n_p):
        hs.gate_A(H, i)
    
    # CU gates: decode -> apply -> recompress (expensive, but only n_p times)
    psi = hs.decode().reshape(hs.NA, hs.NB)
    for i in range(n_p):
        U = torch.zeros(2**n_n, 2**n_n, dtype=torch.complex64)
        for x in range(2**n_n):
            y = (pow(a, 2**i, N)*x)%N if x<N else x; U[y,x]=1.0
        CU = torch.zeros(2**(n_n+1), 2**(n_n+1), dtype=torch.complex64)
        CU[:2**n_n, :2**n_n] = torch.eye(2**n_n, dtype=torch.complex64)
        CU[2**n_n:, 2**n_n:] = U
        psi = hs.gate_AB_full(psi.reshape(-1), i, list(range(n_p, n_p+n_n)), CU)
    hs.init_state(psi.reshape(-1))
    
    # DEBUG: what rank did we get?
    _, Stest, _ = torch.linalg.svd(psi, full_matrices=False)
    print(f"    CU state S[:6]={Stest[:6].tolist()}")
    
    # QFT on period (all compressed gates in A)
    for i in range(n_p):
        hs.gate_A(H, i)
        for j in range(i+1, n_p):
            angle = -math.pi / (2**(j-i))
            Rk = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                [0,0,0,complex(math.cos(angle), math.sin(angle))]], dtype=torch.complex64)
            psi2 = hs.decode().reshape(hs.NA, hs.NB)
            n=n_p+n_n
            st=psi2.reshape([2]*n_p+[2]*n_n)
            cd=n_p-1-j; td=n_p-1-i
            perm=[cd,td]+[k for k in range(n_p+n_n) if k!=cd and k!=td]
            st=st.permute(*perm).contiguous().reshape(4,-1)
            st=(Rk.to(torch.complex64)@st).reshape([2]*n_p+[2]*n_n)
            inv=[0]*(n_p+n_n)
            for k,p in enumerate(perm):inv[p]=k
            psi2=st.permute(*inv).contiguous().reshape(hs.NA,hs.NB)
            hs.init_state(psi2.reshape(-1))
    for i in range(n_p):
        hs.gate_A(H, i)
    
    # Measure
    M = (hs.U * hs.S.unsqueeze(0)) @ hs.Vh
    probs = (M * M.conj()).real.sum(dim=1)
    probs = probs / probs.sum()
    top_vals, top_idxs = torch.topk(probs, min(10, len(probs)))
    
    return top_idxs, top_vals, hs.rank()


print("=" * 78)
print("HAWKING QUANTUM — Compressed Gates")
print("=" * 78)

for N, a, n_p, n_n in [(15,2,4,4),(21,2,5,5),(21,2,8,5),(15,2,12,4)]:
    n = n_p + n_n
    if 2**n > 200000: continue
    t0 = time.perf_counter()
    top, vals, rank = shor_compressed(N, a, n_p, n_n)
    dt = time.perf_counter() - t0
    
    factored = False
    print(f"  N={N} {n}q rank={rank}:", end=" ")
    for i in range(min(5, len(top))):
        k = top[i].item()
        if k > 0:
            r = Fraction(k, 2**n_p).limit_denominator(N).denominator
        else: r = 0
        if r > 0 and pow(a,r,N)==1 and r%2==0 and not factored:
            v=pow(a,r//2,N); g1=gcd(v-1,N); g2=gcd(v+1,N)
            if g1*g2==N and g1>1 and g2>1:
                print(f"{g1}x{g2} (r={r},k={k})", end=" ")
                factored = True
    print(f"{dt:.2f}s")
print("=" * 78)
