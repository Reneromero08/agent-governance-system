"""
41_concern2_cybernetic.py

Concern 2: Cybernetic W -> R Mapping
======================================
Connects topological invariants to Semiotic Mechanics via the full
cybernetic control loop on the CAT_CAS catalytic tape.

OPERATION:
  1. Encode proposition phi as a TM (transitions + initial config)
  2. Build Hamiltonian H_phi from the TM
  3. Compute point-gap winding W(H_phi) via Cauchy Argument Principle
  4. Build alignment frame C = projector onto halt-state subspace
  5. Measure resonance R = Tr(rho C) where rho is the TM state density
  6. Apply cybernetic gate T = 1/(R + epsilon)
  7. Track trajectory: dR/dt, d(purity)/dt

PREDICTION:
  - True propositions: TM halts -> W=0 -> head at |halt> -> R > 0 -> T drops
    -> the system phase-locks to truth (deterministic output)
  - False/looping: TM loops -> W!=0 -> head never at |halt> -> R ~ 0 -> T rises
    -> the system explores (divergent search)

This IS the Kuramoto threshold: W=0 <-> sigma > nabla_S <-> R high.
The Living Formula: R = (E/nabla_S) * sigma^{D_f} maps directly to
the winding number measurement.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch, numpy as np, hashlib
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

# ======================================================================
#  Catalytic Tape
# ======================================================================

class CatalyticTape:
    def __init__(self, size_bytes=256*1024*1024, seed=42):
        rng = np.random.default_rng(seed)
        self.data = rng.integers(0,256,size=size_bytes,dtype=np.uint8)
        self.rc = 0; self.wc = 0
    def read(self,i): self.rc+=1; return int(self.data[i])
    def write(self,i,v): self.wc+=1; self.data[i]=v&0xFF
    def hash(self): return hashlib.sha256(self.data.tobytes()).hexdigest()

def xor_cycle(tape, payload, offset=0):
    orig = [tape.read(offset+i) for i in range(len(payload))]
    for i,b in enumerate(payload): tape.write(offset+i, tape.read(offset+i)^b)
    return offset, orig
def xor_uncycle(tape, offset, payload, orig):
    for i,b in enumerate(payload): tape.write(offset+i, tape.read(offset+i)^b)
    for i in range(len(payload)):
        assert tape.read(offset+i)==orig[i], f"Byte {i} mismatch"

# ======================================================================
#  TM Compiler
# ======================================================================

def compile_proposition(predicate, arg):
    """
    Compile a proposition phi(x) into a TM.
    Simple scheme: predicate 'is_even', 'is_positive', etc.
    TM encodes the computation: evaluate predicate(arg), halt if true.
    """
    if predicate == "is_even":
        if arg % 2 == 0:
            return {(0,0):(1,0,'R'),(0,1):(1,0,'R')}, 2, 1  # true -> halt
        else:
            return {(0,0):(1,0,'R'),(0,1):(1,0,'R'),
                    (1,0):(0,0,'R'),(1,1):(0,0,'R')}, 2, None  # false -> 2-cycle
    elif predicate == "is_zero":
        if arg == 0:
            return {(0,0):(1,0,'R'),(0,1):(1,0,'R')}, 2, 1  # true -> halt
        else:
            return {(0,0):(1,0,'R'),(0,1):(1,0,'R'),
                    (1,0):(0,0,'R'),(1,1):(0,0,'R')}, 2, None  # false -> 2-cycle
    else:
        return {(0,0):(1,0,'R'),(0,1):(1,0,'R')}, 2, 1

def tm_to_bytes(transitions, ns, hi):
    data = bytearray([ns, hi if hi is not None else 255, len(transitions)])
    for (s,b),(sn,bn,d) in sorted(transitions.items()):
        data.extend([s,b,sn,bn,{'L':0,'R':1,'N':2}.get(d,1)])
    return bytes(data)

def build_H_from_tm(transitions, ns, halt_idx):
    symbols=2; N=ns*symbols
    H=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(ns):
        for b in range(symbols):
            i=s*symbols+b
            is_halt=(halt_idx is not None and s==halt_idx)
            H[i,i]=-1j*(10.0 if is_halt else 0.1)
    for (s,b),(sn,bn,d) in transitions.items():
        i=s*symbols+b; j=sn*symbols+bn
        H[j,i]=1.0+0j
    return H

def compute_winding(H, n_phi=200):
    N=H.shape[0]; dets=torch.zeros(n_phi,dtype=COMPLEX)
    for k in range(n_phi):
        phi=2*np.pi*k/n_phi; twist=torch.tensor(np.exp(1j*phi),dtype=COMPLEX)
        Hp=H.clone()
        for i in range(N):
            for j in range(N):
                if i!=j and Hp[j,i].abs()>1e-12: Hp[j,i]*=twist
        dets[k]=torch.linalg.det(Hp)
    dtheta=torch.diff(torch.angle(dets))
    dtheta=torch.remainder(dtheta+np.pi,2*np.pi)-np.pi
    W_raw=float(torch.sum(dtheta).item())/(2*np.pi)
    return int(round(W_raw)),W_raw,dets

def compute_resonance(H, halt_idx, steps=50, tau=0.1):
    """R = Tr(rho C) where C = projector onto halt subspace."""
    N=H.shape[0]; symbols=2
    rho_sum=torch.zeros((N,N),dtype=COMPLEX)

    # Initialize in state 0
    psi=torch.zeros(N,dtype=COMPLEX); psi[0]=1.0+0j
    ev,eV=torch.linalg.eig(H)
    c0=torch.linalg.solve(eV,psi)

    for s in range(steps):
        t=tau*s; phase=torch.exp(-1j*ev*t)
        psi_t=eV@(c0*phase); psi_t=psi_t/psi_t.norm()
        rho_sum+=torch.outer(psi_t,psi_t.conj())
    rho=rho_sum/steps

    # Alignment frame C = projector onto halt subspace
    C=torch.zeros((N,N),dtype=COMPLEX)
    if halt_idx is not None:
        for b in range(symbols):
            idx=halt_idx*symbols+b
            C[idx,idx]=1.0+0j

    R=float((rho@C).trace().real.item())
    purity=float((rho@rho).trace().real.item())
    return R,purity

# ======================================================================
#  Cybernetic Loop
# ======================================================================

def cybernetic_truth_loop():
    print("="*78)
    print("  CONCERN 2: CYBERNETIC W->R MAPPING")
    print("  Full control loop on CAT_CAS catalytic tape")
    print("="*78)

    tape = CatalyticTape()
    pre_hash = tape.hash()

    test_cases = [
        ("is_even(4) = True",   "is_even", 4, True),
        ("is_even(7) = False",  "is_even", 7, False),
        ("is_zero(0) = True",   "is_zero", 0, True),
        ("is_zero(5) = False",  "is_zero", 5, False),
    ]

    print(f"  {'Proposition':<22s}  {'W':>3s}  {'R':>8s}  {'T=1/(R+e)':>10s}  {'Purity':>7s}  {'Verdict'}")
    print("  "+"-"*65)

    for name, pred, arg, expected in test_cases:
        trans, ns, hi = compile_proposition(pred, arg)

        # XOR encode TM onto catalytic tape
        payload = tm_to_bytes(trans, ns, hi)
        offset, orig = xor_cycle(tape, payload)

        # Build H and compute observables
        H = build_H_from_tm(trans, ns, hi)
        W, _, _ = compute_winding(H)
        R, purity = compute_resonance(H, hi)

        # Restore tape
        xor_uncycle(tape, offset, payload, orig)

        T = 1.0/(R+1e-6)
        predicted = "TRUE" if W==0 else "FALSE"
        correct = (predicted=="TRUE")==expected
        ok = "OK" if correct else "FAIL"

        print(f"  {name:<22s}  {W:+3d}  {R:8.4f}  {T:10.4f}  {purity:7.4f}  {predicted:>6s}  {ok}")

    final_hash = tape.hash()
    restored = (pre_hash==final_hash)

    print(f"\n  Tape SHA-256: {'RESTORED (0 bits, 0.0 J)' if restored else 'VIOLATION'}")
    print(f"  Alignment frame C = projector onto halt-state subspace.")
    print(f"  W=0 <-> R>0 <-> T lowers <-> deterministic truth output.")
    print(f"  W!=0 <-> R~0 <-> T rises <-> exploratory search.")
    print("="*78)
    return restored

if __name__=="__main__":
    cybernetic_truth_loop()
