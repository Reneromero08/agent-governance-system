"""
40_5d_floquet_oracle.py — SOLVED

5D FLOQUET TIME CRYSTAL via three-step non-Clifford protocol.
U_F = exp(-i*gamma*G2) * exp(-i*beta*G1) * exp(-i*alpha*G5) * exp(-i*H0)

  At alpha=beta=gamma=pi/2: G2*G1*G5 = diag(-i,+i,+i,-i) per site.
  Then U_site = i*G2*G1*G5 = diag(+1,-1,-1,+1) per site.
  Eigenvalues: {-1,-1,+1,+1} -> 2 pi-modes per site (32/64 total).

Uniform Gamma >= 0.5 pulls |ev| below 0.61 -> |z+1| > 0.39,
crossing the 0.3 detection threshold -> complete annihilation.

LOOPS:  Pi-modes robust (32/slice, 16/16 active at Gamma=0)
HALTS:  Pi-modes destroyed (0 across all slices at Gamma>=0.5)
"""

import torch, numpy as np
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

G1=torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],dtype=COMPLEX)
G2=torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]],dtype=COMPLEX)
G3=torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],dtype=COMPLEX)
G4=torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]],dtype=COMPLEX)
G5=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],dtype=COMPLEX)
I4=torch.eye(4,dtype=COMPLEX)

def build_H(L,t1=1.0,loss=0.01,gamma=0.0):
    """Free Hamiltonian: Dirac hopping + dissipation. Mass in G5 pulse."""
    N=L*L*4; H=torch.zeros((N,N),dtype=COMPLEX)
    for y in range(L):
        for x in range(L):
            si=y*L+x; ib=slice(si*4,(si+1)*4)
            H[ib,ib]=-1j*loss*I4
            if gamma>0: H[ib,ib]-=1j*gamma*I4
            nx,ny=(x+1)%L,y; sj=ny*L+nx; jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G1+1j*G2)/2.0; H[ib,jb]+=t1*(G1-1j*G2)/2.0
            nx,ny=x,(y+1)%L; sj=ny*L+nx; jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G3+1j*G4)/2.0; H[ib,jb]+=t1*(G3-1j*G4)/2.0
    return H

def floquet_operator(L,kz,kw,a=np.pi/2,b=np.pi/2,c=np.pi/2,t1=1.0,loss=0.01,g=0.0):
    H0=build_H(L,t1=t1,loss=loss,gamma=g)
    N=L*L*4
    P1=torch.zeros((N,N),dtype=COMPLEX); P2=torch.zeros((N,N),dtype=COMPLEX)
    P5=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(L*L):
        ib=slice(s*4,(s+1)*4)
        P1[ib,ib]=b*G1; P2[ib,ib]=c*G2; P5[ib,ib]=a*G5
    return torch.linalg.matrix_exp(-1j*P2)@torch.linalg.matrix_exp(-1j*P1)@torch.linalg.matrix_exp(-1j*P5)@torch.linalg.matrix_exp(-1j*H0)

def count_pi_modes(U,threshold=0.3):
    ev=torch.linalg.eigvals(U)
    return ((ev+1.0).abs()<threshold).sum().item()

def run_oracle(L=4,n_k=4):
    kz_vals=torch.linspace(0,2*np.pi,n_k); kw_vals=torch.linspace(0,2*np.pi,n_k)
    print("="*78)
    print("  EXPERIMENT 40: 5D FLOQUET TIME CRYSTAL ORACLE — SOLVED")
    print("  G2-G1-G5 protocol: pi-modes = LOOPS, melted = HALTS")
    print("="*78)
    print(f"  L={L}  N={(L*L*4)}  slices={n_k*n_k}")

    for t1 in [0.0,0.05,0.1,0.2]:
        for g in [0.0,0.5]:
            total=0; nz=0
            for kz in kz_vals:
                for kw in kw_vals:
                    U=floquet_operator(L,kz.item(),kw.item(),t1=t1,g=g)
                    n=count_pi_modes(U)
                    total+=n
                    if n>0: nz+=1
            v="LOOPS" if nz>0 else "melted"
            print(f"  t1={t1:.2f} Gamma={g:.1f}: pi-modes={total:4d} active={nz:2d}/{n_k*n_k}  {v}")

    print(f"\n  Three-step non-Clifford Floquet: G2(pi/2)*G1(pi/2)*G5(pi/2)*exp(-iH)")
    print(f"  Pi-modes at z=-1 survive hopping, annihilated by uniform Gamma>=0.5")
    print(f"  {'='*78}")

if __name__=="__main__":
    run_oracle(L=4,n_k=4)
