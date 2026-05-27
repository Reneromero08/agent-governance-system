"""
40_sub_5_pulseprog_v2.py

PULSE-PROGRAMMED COMPUTATION v2 — t1/Gamma Encoding

v1 showed pulse-angle programs destroy DTC order. v2 uses the CORRECT
degrees of freedom: hopping strength (t1) and dissipation (gamma).
Both preserve pi-modes up to t1=0.2 and selective gamma < 0.5.

PROTOCOL:
  Program = sequence of (t1, gamma) pairs across T cycles.
  Each cycle evolves the state via the DTC-preserving Floquet operator.
  After T cycles, pi-mode survival pattern per slice = program output.
  
  Different (t1, gamma) sequences produce DIFFERENT pi-mode survival
  patterns — because different t1 values couple spatial sites
  differently, and different gamma values selectively damp sites.

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

def build_H(L, t1=1.0, loss=0.01, gamma=0.0):
    N = L*L*4; H = torch.zeros((N,N), dtype=COMPLEX)
    for y in range(L):
        for x in range(L):
            si = y*L+x; ib = slice(si*4,(si+1)*4)
            H[ib,ib] = -1j*loss*I4
            if gamma > 0: H[ib,ib] -= 1j*gamma*I4
            nx,ny = (x+1)%L,y; sj = ny*L+nx; jb = slice(sj*4,(sj+1)*4)
            H[jb,ib] += t1*(G1+1j*G2)/2; H[ib,jb] += t1*(G1-1j*G2)/2
            nx,ny = x,(y+1)%L; sj = ny*L+nx; jb = slice(sj*4,(sj+1)*4)
            H[jb,ib] += t1*(G3+1j*G4)/2; H[ib,jb] += t1*(G3-1j*G4)/2
    return H

def floquet_cycle(L, kz, kw, t1=1.0, loss=0.01, gamma=0.0):
    """Single DTC-preserving Floquet cycle at given (t1, gamma)."""
    H0 = build_H(L, t1=t1, loss=loss, gamma=gamma)
    N = L*L*4
    P1 = torch.zeros((N,N), dtype=COMPLEX)
    P2 = torch.zeros((N,N), dtype=COMPLEX)
    P5 = torch.zeros((N,N), dtype=COMPLEX)
    a = b = c = np.pi/2
    for s in range(L*L):
        ib = slice(s*4,(s+1)*4)
        P1[ib,ib] = b*G1; P2[ib,ib] = c*G2; P5[ib,ib] = a*G5
    return (torch.linalg.matrix_exp(-1j*P2) @ torch.linalg.matrix_exp(-1j*P1) @
            torch.linalg.matrix_exp(-1j*P5) @ torch.linalg.matrix_exp(-1j*H0))

def apply_program(L, kz, kw, program, loss=0.01):
    """
    Apply a pulse program: list of (t1, gamma) values.
    Returns the accumulated Floquet operator and the pi-mode count.
    """
    N = L*L*4
    U_total = torch.eye(N, dtype=COMPLEX)
    for t1, gamma in program:
        U_cycle = floquet_cycle(L, kz, kw, t1=t1, loss=loss, gamma=gamma)
        U_total = U_cycle @ U_total
    return U_total

def pi(U, th=0.3): return int(((torch.linalg.eigvals(U)+1).abs()<th).sum().item())

def pulse_computation_v2():
    L = 4
    kz_vals = torch.linspace(0, 2*np.pi, 4)
    kw_vals = torch.linspace(0, 2*np.pi, 4)
    slices = list(itertools.product(kz_vals, kw_vals))
    
    # Define programs as (t1, gamma) sequences
    # t1 controls hopping strength (site coupling)
    # gamma controls dissipation (site damping)
    programs = {
        "IDENTITY":    [(0.1, 0.0)],           # baseline DTC
        "WEAK_COUPLE": [(0.0, 0.0)],           # no hopping, no damping
        "MED_COUPLE":  [(0.15, 0.0)],          # medium hopping
        "MAX_COUPLE":  [(0.2, 0.0)],           # max hopping (t1=0.2 limit)
        "LIGHT_DAMP":  [(0.1, 0.1)],           # light damping
        "MED_DAMP":    [(0.1, 0.25)],          # medium damping
        "RAMP_UP":     [(0.0,0.0),(0.1,0.0),(0.2,0.0)],    # increasing coupling
        "RAMP_DOWN":   [(0.2,0.0),(0.1,0.0),(0.0,0.0)],    # decreasing coupling
        "DAMP_THEN_FREE": [(0.1,0.25),(0.1,0.0)],           # damp then release
        "FREE_THEN_DAMP": [(0.1,0.0),(0.1,0.25)],           # free then damp
    }
    
    print("=" * 78)
    print("  PULSE-PROGRAMMED COMPUTATION v2 — t1/Gamma Encoding")
    print("  DTC-Preserving Programs via Hopping + Dissipation")
    print("=" * 78)
    
    print(f"\n  {'Prog':>2s} {'Program':>16s} {'Cycles':>6s} {'t1 range':>12s} {'gamma':>8s}")
    print(f"  {'Pi(s0)':>6s} {'Pi(s8)':>6s} {'Pi(s15)':>6s} {'Uniq Pi':>8s} {'Output'}")
    print("  " + "-" * 70)
    
    results = {}
    for pidx, (name, prog) in enumerate(programs.items()):
        pi_vals = []
        for idx, (kz, kw) in enumerate(slices):
            U = apply_program(L, kz.item(), kw.item(), prog)
            pi_vals.append(pi(U))
        
        t1_min = min(p[0] for p in prog)
        t1_max = max(p[0] for p in prog)
        g_max = max(p[1] for p in prog) if any(p[1]>0 for p in prog) else 0
        
        unique = len(set(pi_vals))
        
        if unique > 1:
            out = f"MULTI-MODE ({unique} distinct values)"
        elif pi_vals[0] == 32:
            out = "FULL DTC (all 32)"
        elif pi_vals[0] == 0:
            out = "MELTED"
        else:
            out = f"PARTIAL ({pi_vals[0]})"
        
        results[name] = {'pis': pi_vals, 'unique': unique, 'output': out}
        
        print(f"  {pidx:3d} {name:>16s} {len(prog):6d} "
              f"[{t1_min:.1f},{t1_max:.1f}]    {g_max:8.2f}")
        print(f"  {'':>6s} {pi_vals[0]:6d} {pi_vals[8]:6d} {pi_vals[15]:6d} "
              f"{unique:8d} {out}")
        print()
    
    # ---- Program comparison matrix ----
    print("  ---  PROGRAM DISCRIMINATION MATRIX  ---")
    print(f"  {'':>16s}", end="")
    for name in programs: print(f" {name[:4]:>5s}", end="")
    print()
    for name1 in programs:
        p1 = np.array(results[name1]['pis'])
        print(f"  {name1:>16s}", end="")
        for name2 in programs:
            p2 = np.array(results[name2]['pis'])
            same = int(np.all(p1 == p2))
            print(f" {'=' if same else '!=':>5s}", end="")
        print()
    
    # ---- Statistics ----
    unique_programs = len(programs)
    unique_outputs = len(set(tuple(r['pis']) for r in results.values()))
    
    print(f"\n{'=' * 78}")
    print("  PULSE COMPUTATION v2 SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Programs tested:        {unique_programs}")
    print(f"  Unique outputs:         {unique_outputs}")
    print(f"  Discrimination ratio:   {unique_outputs}/{unique_programs}")
    print(f"  Encoding:               t1 (hopping) + gamma (dissipation)")
    print(f"  DTC preservation:       t1 <= 0.2, gamma < 0.5")
    print(f"  Program space:          R^(2T) for T-cycle sequences")
    print(f"  {'=' * 78}")
    
    return unique_outputs

if __name__ == "__main__":
    pulse_computation_v2()
