"""25.2: Deep Wigner's Friend — non-trivial reversible thought"""
import torch, math

H=torch.tensor([[1,1],[1,-1]],dtype=torch.complex64)/math.sqrt(2)
X=torch.tensor([[0,1],[1,0]],dtype=torch.complex64)
Z=torch.tensor([[1,0],[0,-1]],dtype=torch.complex64)
CNOT=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=torch.complex64)
CZ=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]],dtype=torch.complex64)

def gate1(s,G,t,n):
    d=2;td=n-1-t;st=s.reshape([d]*n)
    perm=[td]+[i for i in range(n) if i!=td]
    st=st.permute(*perm).contiguous().reshape(d,-1);st=(G@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def gate2(s,G,c,t,n):
    d=2;cd=n-1-c;td=n-1-t;st=s.reshape([d]*n)
    perm=[cd,td]+[i for i in range(n) if i not in (cd,td)]
    st=st.permute(*perm).contiguous().reshape(d*d,-1);st=(G@st).reshape([d]*n)
    inv=[0]*n
    for i,p in enumerate(perm):inv[p]=i
    return st.permute(*inv).contiguous().reshape(-1)

def overlap(a,b):
    return torch.abs(torch.dot(a.conj(),b)).item()

def think(state, mem_start, n_mem, n):
    """Non-trivial reversible thought process on memory qubits."""
    # Layer 1: entangle memory qubits
    for i in range(n_mem-1):
        state = gate2(state, CZ, mem_start+i, mem_start+i+1, n)
    # Layer 2: phase rotations
    for i in range(n_mem):
        state = gate1(state, H if i%2==0 else Z, mem_start+i, n)
    # Layer 3: cross-entangle
    for i in range(n_mem//2):
        state = gate2(state, CNOT, mem_start+i*2, mem_start+i*2+1, n)
    # Layer 4: more phase
    for i in range(n_mem):
        state = gate1(state, X if i%3==0 else H, mem_start+i, n)
    return state

def unthink(state, mem_start, n_mem, n):
    """Precise inverse of think()."""
    for i in range(n_mem-1,-1,-1):
        state = gate1(state, X if i%3==0 else H, mem_start+i, n)  # X,H self-inv
    for i in range(n_mem//2-1,-1,-1):
        state = gate2(state, CNOT, mem_start+i*2, mem_start+i*2+1, n)  # CNOT self-inv
    for i in range(n_mem-1,-1,-1):
        state = gate1(state, H if i%2==0 else Z, mem_start+i, n)  # H,Z self-inv
    for i in range(n_mem-2,-1,-1):
        state = gate2(state, CZ, mem_start+i, mem_start+i+1, n)  # CZ self-inv
    return state

def observe(state, target, mem_start, n_mem, n):
    for i in range(n_mem):
        state = gate2(state, CNOT, target, mem_start+i, n)
    return state

def unobserve(state, target, mem_start, n_mem, n):
    for i in range(n_mem-1,-1,-1):
        state = gate2(state, CNOT, target, mem_start+i, n)
    return state

def main():
    print("=" * 78)
    print("25.2: DEEP WIGNER'S FRIEND — Non-trivial Reversible Thought")
    print("=" * 78)
    
    for n_mem in [3, 5, 7, 9, 11]:
        n = 1 + n_mem
        if 2**n > 10000: break
        
        psi = torch.zeros(2**n, dtype=torch.complex64); psi[0] = 1.0
        psi = gate1(psi, H, 0, n)  # target = |+⟩
        initial = psi.clone()
        
        # Friend measures
        psi = observe(psi, 0, 1, n_mem, n)
        after_measure = psi.clone()
        
        # Entropy check: is target entangled with friend?
        # If friend qubits are all |0⟩ when target=|0⟩ and all |1⟩ when target=|1⟩,
        # that's maximum entanglement.
        
        # Friend processes the outcome
        psi = think(psi, 1, n_mem, n)
        after_think = psi.clone()
        
        # Wigner erases: unthink + unobserve
        psi = unthink(psi, 1, n_mem, n)
        psi = unobserve(psi, 0, 1, n_mem, n)
        
        ov = overlap(initial, psi)
        status = "RESTORED" if ov > 0.9999 else f"FAIL({ov:.4f})"
        
        # Measure entanglement entropy of the "measurement" state
        # Schmidt decomposition across target vs friend
        M = after_measure.reshape(2, 2**n_mem).numpy()
        import numpy as np
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        S2 = S**2; S2 = S2/S2.sum()
        entropy = -sum(p*np.log2(p) for p in S2 if p>1e-15)
        
        print(f"  n_mem={n_mem:>2} qubits={n} state={2**n} overlap={ov:.6f} {status}  S_entropy={entropy:.2f}")
    
    print(f"\n  Non-trivial 4-layer thought process on Friend's memory.")
    print(f"  Measurement creates entanglement (S_entropy > 0).")
    print(f"  Catalytic uncomputation perfectly erases it.")
    print(f"  Wigner sees only the original superposition.")
    print("=" * 78)

if __name__ == "__main__":
    main()
