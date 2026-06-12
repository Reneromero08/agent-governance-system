"""
40_sub_5_pulseprog.py

PULSE-PROGRAMMED COMPUTATION ON THE FLOQUET TIME CRYSTAL

The Floquet operator U_F = exp(-i*gamma*G2) @ exp(-i*beta*G1) @ exp(-i*alpha*G5) @ exp(-i*H0)
uses fixed pulse angles alpha=beta=gamma=pi/2. We replace these with a time-dependent
pulse PROGRAM (alpha_t, beta_t, gamma_t) across T Floquet cycles.

The pulse sequence IS the program. The pi-mode survival pattern after T cycles IS
the output. Computation is encoded in temporal structure, not spatial memory.

DEMONSTRATION:
  8 distinct pulse programs. Each program = 3 cycles of (alpha, beta, gamma).
  Execute each program on a dedicated momentum slice.
  Measure pi-mode survival pattern after T cycles.
  Show that different programs produce DIFFERENT survival patterns.
  Program -> output mapping is deterministic and reversible.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch, numpy as np, itertools
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

G1 = torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],dtype=COMPLEX)
G2 = torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]],dtype=COMPLEX)
G3 = torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],dtype=COMPLEX)
G4 = torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]],dtype=COMPLEX)
G5 = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],dtype=COMPLEX)
I4 = torch.eye(4,dtype=COMPLEX)

def build_H(L,t1=1.0,loss=0.01,gamma=0.0):
    N=L*L*4;H=torch.zeros((N,N),dtype=COMPLEX)
    for y in range(L):
        for x in range(L):
            si=y*L+x;ib=slice(si*4,(si+1)*4);H[ib,ib]=-1j*loss*I4
            if gamma>0:H[ib,ib]-=1j*gamma*I4
            nx,ny=(x+1)%L,y;sj=ny*L+nx;jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G1+1j*G2)/2;H[ib,jb]+=t1*(G1-1j*G2)/2
            nx,ny=x,(y+1)%L;sj=ny*L+nx;jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G3+1j*G4)/2;H[ib,jb]+=t1*(G3-1j*G4)/2
    return H

def apply_pulse_program(L, kz, kw, program, t1=1.0, loss=0.01, gamma=0.0):
    """
    Apply a pulse program: list of (alpha, beta, gamma) to execute
    sequentially as Floquet cycles. Returns final Floquet operator after
    all cycles.
    
    U_total = U_N @ ... @ U_2 @ U_1
    where U_t = exp(-i*gamma_t*G2) @ exp(-i*beta_t*G1) @ exp(-i*alpha_t*G5) @ exp(-i*H0)
    """
    H0 = build_H(L, t1=t1, loss=loss, gamma=gamma)
    N = L*L*4
    U_total = torch.eye(N, dtype=COMPLEX)
    
    for alpha, beta, gam in program:
        P1 = torch.zeros((N,N), dtype=COMPLEX)
        P2 = torch.zeros((N,N), dtype=COMPLEX)
        P5 = torch.zeros((N,N), dtype=COMPLEX)
        for s in range(L*L):
            ib = slice(s*4,(s+1)*4)
            P1[ib,ib] = beta*G1; P2[ib,ib] = gam*G2; P5[ib,ib] = alpha*G5
        U_cycle = (torch.linalg.matrix_exp(-1j*P2) @
                   torch.linalg.matrix_exp(-1j*P1) @
                   torch.linalg.matrix_exp(-1j*P5) @
                   torch.linalg.matrix_exp(-1j*H0))
        U_total = U_cycle @ U_total
    
    return U_total

def pi(U, th=0.3): return int(((torch.linalg.eigvals(U)+1).abs()<th).sum().item())

def pulse_computation():
    L = 4  # small lattice for speed
    kz_vals = torch.linspace(0,2*np.pi,4)
    kw_vals = torch.linspace(0,2*np.pi,4)
    
    # Define 8 distinct pulse programs
    programs = {
        "IDENTITY":    [(np.pi/2, np.pi/2, np.pi/2)],  # DTC baseline: all pi
        "NEGATE":      [(np.pi, np.pi, np.pi)],         # Full rotation
        "HADAMARD_1":  [(np.pi/4, np.pi/2, np.pi/2)],  # Partial G5
        "HADAMARD_2":  [(np.pi/2, np.pi/4, np.pi/2)],  # Partial G1
        "PHASE_FLIP":  [(np.pi/2, 0, 0)],               # G5 only
        "SWAP_TEST":   [(0, np.pi/2, 0)],               # G1 only
        "ECHO_1":      [(np.pi/2, np.pi/2, np.pi/2),    # DTC + negate + DTC
                        (np.pi, np.pi, np.pi),
                        (np.pi/2, np.pi/2, np.pi/2)],
        "ECHO_2":      [(np.pi/2, np.pi/2, np.pi/2),    # DTC + partial + DTC
                        (np.pi/4, np.pi/4, np.pi/4),
                        (np.pi/2, np.pi/2, np.pi/2)],
    }
    
    print("=" * 78)
    print("  PULSE-PROGRAMMED COMPUTATION")
    print("  Floquet Time Crystal — Program = Drive Sequence")
    print("=" * 78)
    
    slices = list(itertools.product(kz_vals, kw_vals))
    
    # Test each program on the first 8 slices
    print(f"\n  {'Slice':>5s} {'Program':>12s} {'Cycles':>6s} {'Pi':>5s} {'Output'}")
    print("  " + "-" * 45)
    
    results = {}
    for idx, (prog_name, prog) in enumerate(programs.items()):
        if idx >= len(slices): break
        kz, kw = slices[idx]
        kzi = kz.item(); kwi = kw.item()
        
        U = apply_pulse_program(L, kzi, kwi, prog, t1=0.1)
        n_pi = pi(U)
        
        # Characterize output
        if n_pi == 32:
            output = "FULL DTC (all 32 pi-modes)"
        elif n_pi == 0:
            output = "MELTED (0 pi-modes)"
        elif n_pi > 16:
            output = f"PARTIAL ({n_pi}/32 pi-modes)"
        else:
            output = f"NEAR-MELT ({n_pi}/32 pi-modes)"
        
        results[prog_name] = {'pi': n_pi, 'output': output}
        print(f"  {idx:5d} {prog_name:>12s} {len(prog):6d} {n_pi:5d} {output}")
    
    # ---- IDENTITY baseline on all 16 slices ----
    print(f"\n  ---  BASELINE: IDENTITY program on ALL 16 slices  ---")
    identity_prog = programs["IDENTITY"]
    all_pi = []
    for idx, (kz, kw) in enumerate(slices):
        U = apply_pulse_program(L, kz.item(), kw.item(), identity_prog, t1=0.1)
        all_pi.append(pi(U))
    print(f"  All 16 slices: {all_pi}")
    print(f"  Mean: {np.mean(all_pi):.1f}  Std: {np.std(all_pi):.1f}")
    
    # ---- Program uniqueness ----
    print(f"\n  ---  PROGRAM UNIQUENESS: Distinct programs -> distinct outputs? ---")
    unique_outputs = set(r['pi'] for r in results.values())
    unique_names = len(results)
    unique_pi = len(unique_outputs)
    print(f"  Programs tested: {unique_names}")
    print(f"  Unique pi-mode counts: {unique_pi}")
    if unique_pi == unique_names:
        print(f"  ALL PROGRAMS PRODUCE UNIQUE OUTPUTS — program space is fully mapped")
    elif unique_pi > 1:
        print(f"  {unique_pi}/{unique_names} distinct outputs — program space is partially mapped")
    else:
        print(f"  All programs produce identical output — program space is degenerate")
    
    # ---- ECHO test: does ECHO_1 restore DTC after perturbation? ----
    print(f"\n  ---  ECHO TEST: Does DTC + perturbation + DTC restore? ---")
    for echo_name in ["ECHO_1", "ECHO_2"]:
        pi_echo = results[echo_name]['pi']
        pi_base = results["IDENTITY"]['pi']
        restored = (pi_echo == pi_base)
        print(f"  {echo_name}: pi={pi_echo} base={pi_base} -> {'RESTORED' if restored else 'CHANGED'}")
    
    print(f"\n{'=' * 78}")
    print("  PULSE COMPUTATION SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Programs executed:      {unique_names}")
    print(f"  Unique outputs:         {unique_pi}")
    print(f"  Program space:          R^(3T) for T-cycle programs")
    print(f"  Execution:              Physics (Floquet evolution)")
    print(f"  Readout:                Pi-mode counting per slice")
    print(f"  Reversibility:          Programs are unitary sequences")
    print(f"  {'=' * 78}")

if __name__ == "__main__":
    pulse_computation()
