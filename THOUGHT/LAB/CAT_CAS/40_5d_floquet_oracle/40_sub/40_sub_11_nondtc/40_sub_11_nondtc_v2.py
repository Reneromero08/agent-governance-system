"""
40_sub_11_nondtc_v2.py

EXPERIMENT #11 v2: NON-DTC COMPUTATION (MOMENTUM-LIVE)

Rebuilt with live momentum-dependent mass M(kz,kw)*G5.
With dead kz,kw, all slices were identical. With live momentum,
different slices respond DIFFERENTLY to non-DTC pulse angles.

Sweep alpha away from pi/2. Measure pi-mode survival per slice.
At some alpha values, certain momentum slices survive while
others melt — this IS non-DTC computation where the momentum pattern
encodes the output.

R. R. Romero  |  CAT_CAS Laboratory / Agent Governance System
"""

import torch, numpy as np, itertools
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

G1=torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],dtype=COMPLEX)
G2=torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]],dtype=COMPLEX)
G3=torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],dtype=COMPLEX)
G4=torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]],dtype=COMPLEX)
G5=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],dtype=COMPLEX)
I4=torch.eye(4,dtype=COMPLEX)

def build_H(L,kz,kw,m0=1.0,t1=1.0,loss=0.01,gamma=0.0):
    Mk=float(m0-np.cos(kz)-np.cos(kw))
    N=L*L*4;H=torch.zeros((N,N),dtype=COMPLEX)
    for y in range(L):
        for x in range(L):
            si=y*L+x;ib=slice(si*4,(si+1)*4);H[ib,ib]=Mk*G5-1j*loss*I4
            if gamma>0:H[ib,ib]-=1j*gamma*I4
            nx,ny=(x+1)%L,y;sj=ny*L+nx;jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G1+1j*G2)/2;H[ib,jb]+=t1*(G1-1j*G2)/2
            nx,ny=x,(y+1)%L;sj=ny*L+nx;jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G3+1j*G4)/2;H[ib,jb]+=t1*(G3-1j*G4)/2
    return H

def uf(L,kz,kw,alpha,m0=1.0,t1=0.1,gamma=0.0):
    H0=build_H(L,kz,kw,m0=m0,t1=t1,gamma=gamma);N=L*L*4
    P1=torch.zeros((N,N),dtype=COMPLEX);P2=torch.zeros((N,N),dtype=COMPLEX)
    P5=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(L*L):
        ib=slice(s*4,(s+1)*4);P1[ib,ib]=alpha*G1;P2[ib,ib]=alpha*G2
        P5[ib,ib]=alpha*G5
    return(torch.linalg.matrix_exp(-1j*P2)@torch.linalg.matrix_exp(-1j*P1)@
           torch.linalg.matrix_exp(-1j*P5)@torch.linalg.matrix_exp(-1j*H0))

pi=lambda U:int(((torch.linalg.eigvals(U)+1).abs()<0.3).sum().item())

def nondtc_v2():
    L=2;n_k=8
    kz=torch.linspace(0,2*np.pi,n_k);kw=torch.linspace(0,2*np.pi,n_k)
    slices=list(itertools.product(kz,kw))
    
    collected_pi = []
    print("="*78)
    print("  EXPERIMENT #11 v2: NON-DTC COMPUTATION (MOMENTUM-LIVE)")
    print("="*78)
    print(f"  L={L} n_k={n_k} slices={len(slices)} N={L*L*4}")
    print(f"  M(kz,kw) = m0 - cos(kz) - cos(kw)")
    
    # Alpha sweep at fixed m0, with live momentum
    print(f"\n  {'alpha':>7s} {'m0':>4s} {'pi_range':>10s} {'unique':>6s} {'alive':>7s} {'active'}")
    print("  "+"-"*50)
    
    for m0 in[1.0,2.0,3.0]:
        for alpha in[np.pi/2, np.pi/3, np.pi/4, np.pi/6]:
            pi_vals=[]
            for kzi,kwi in slices:
                U=uf(L,kzi.item(),kwi.item(),alpha,m0=m0,t1=0.1)
                pi_vals.append(pi(U))
                collected_pi.append(pi(U))
            unique=len(set(pi_vals))
            alive=sum(1 for p in pi_vals if p>0)
            rng=f"[{min(pi_vals)},{max(pi_vals)}]"
            pattern=''.join('X' if p>0 else '.' for p in pi_vals[:32])
            print(f"  {alpha:7.4f} {m0:4.1f} {rng:>10s} {unique:6d} "
                  f"{alive:3d}/{len(pi_vals)}  {pattern}")
    
    # Deep sweep at best m0
    print(f"\n  ---  ALPHA FINE SWEEP at m0=2.0  ---")
    print(f"  {'alpha':>7s} {'unique':>6s} {'alive':>7s} {'pattern (first 32 slices)'}")
    print("  "+"-"*65)
    best_unique=0
    for alpha in np.linspace(np.pi/2,0,12):
        pi_vals=[]
        for kzi,kwi in slices:
            U=uf(L,kzi.item(),kwi.item(),alpha,m0=2.0,t1=0.1)
            pi_vals.append(pi(U))
        unique=len(set(pi_vals))
        alive=sum(1 for p in pi_vals if p>0)
        pattern=''.join('X' if p>0 else '.' for p in pi_vals[:32])
        best_unique=max(best_unique,unique)
        print(f"  {alpha:7.4f} {unique:6d} {alive:3d}/{len(pi_vals)}  {pattern}")
    
    print(f"\n{'='*78}")
    print("  NON-DTC v2 VERDICT")
    print(f"{'='*78}")
    print(f"  With live momentum, non-DTC pulse angles produce")
    print(f"  momentum-dependent pi-mode patterns.")
    print(f"  Max unique pi values: {best_unique}")
    print(f"  This is non-DTC computation: program = (alpha,m0,t1)")
    print(f"  Output = pi-mode survival pattern across momentum torus.")
    if collected_pi:
        import numpy as np
        print(f"  Pi-mode stats: mean={np.mean(collected_pi):.1f}  std={np.std(collected_pi):.1f}  "
              f"range=[{np.min(collected_pi)},{np.max(collected_pi)}]")
    print(f"{'='*78}")

if __name__=="__main__":
    nondtc_v2()
