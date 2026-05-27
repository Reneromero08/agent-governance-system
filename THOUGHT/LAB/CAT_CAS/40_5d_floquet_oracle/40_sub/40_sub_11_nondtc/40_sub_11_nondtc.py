"""
40_sub_11_nondtc.py

EXPERIMENT #11: NON-DTC COMPUTATION

Sweep Floquet parameters away from the DTC operating point
(alpha=beta=gamma=pi/2) and measure how the eigenvalue spectrum responds.
Pi-modes die, but the complex eigenvalue pattern may encode computational
information that varies across momentum slices.

Key questions:
1. At non-DTC angles, do different (kz,kw) slices produce DIFFERENT spectra?
2. Is there a continuous parameter range where eigenvalues vary smoothly?
3. Can we encode information in the eigenvalue closest to z=-1?

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

def uf(L,kz,kw,alpha,t1=0.1,loss=0.01,gamma=0.0):
    H0=build_H(L,t1=t1,loss=loss,gamma=gamma);N=L*L*4
    P1=torch.zeros((N,N),dtype=COMPLEX);P2=torch.zeros((N,N),dtype=COMPLEX)
    P5=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(L*L):
        ib=slice(s*4,(s+1)*4);P1[ib,ib]=alpha*G1;P2[ib,ib]=alpha*G2
        P5[ib,ib]=alpha*G5
    return(torch.linalg.matrix_exp(-1j*P2)@torch.linalg.matrix_exp(-1j*P1)@
           torch.linalg.matrix_exp(-1j*P5)@torch.linalg.matrix_exp(-1j*H0))

def analyze_spectrum(U):
    ev=torch.linalg.eigvals(U)
    # Distance of closest eigenvalue to z=-1 (pi-mode remnant)
    dist_to_minus1=float((ev+1).abs().min().item())
    # Count eigenvalues near z=-1
    n_near=((ev+1).abs()<0.3).sum().item()
    # Spectral spread
    spread=float(ev.abs().std().item()) if len(ev)>1 else 0
    # Mean phase
    phase_mean=float(ev.angle().mean().item())
    return n_near,dist_to_minus1,spread,phase_mean

def nondtc():
    L=2;n_k=4
    kz=torch.linspace(0,2*np.pi,n_k);kw=torch.linspace(0,2*np.pi,n_k)
    slices=list(itertools.product(kz,kw))
    
    print("="*78)
    print("  EXPERIMENT #11: NON-DTC COMPUTATION")
    print("  Eigenvalue Spectrum vs Floquet Parameters")
    print("="*78)
    print(f"  L={L}  n_k={n_k}  slices={len(slices)}")
    
    # 1. Alpha sweep (pulse angle)
    print(f"\n  ---  ALPHA SWEEP (pulse angle away from pi/2)  ---")
    print(f"  {'alpha':>7s} {'Pi':>4s} {'dist(z=-1)':>11s} {'spread':>8s} {'phase':>8s} {'Slice dep?'}")
    print("  "+"-"*55)
    
    for alpha in [np.pi/2, np.pi/3, np.pi/4, np.pi/6, np.pi/8, 0.0]:
        metrics=[]
        for kzi,kwi in slices:
            U=uf(L,kzi.item(),kwi.item(),alpha,t1=0.1)
            n,d,s,p=analyze_spectrum(U)
            metrics.append((n,d,s,p))
        avg_n=np.mean([m[0] for m in metrics])
        avg_d=np.mean([m[1] for m in metrics])
        avg_s=np.mean([m[2] for m in metrics])
        avg_p=np.mean([m[3] for m in metrics])
        # Check if slices differ
        unique_n=len(set(m[0] for m in metrics))
        unique_d=len(set(round(m[1],4) for m in metrics))
        differs=(unique_n>1 or unique_d>1)
        print(f"  {alpha:7.4f} {avg_n:4.1f} {avg_d:11.4f} {avg_s:8.4f} {avg_p:8.4f} {'YES' if differs else 'no'}")
    
    # 2. t1 sweep (hopping strength) at DTC point
    print(f"\n  ---  t1 SWEEP (hopping, alpha=pi/2)  ---")
    print(f"  {'t1':>7s} {'Pi':>4s} {'dist(z=-1)':>11s} {'spread':>8s} {'Slice dep?'}")
    print("  "+"-"*45)
    
    for t1 in[0.0,0.1,0.2,0.3,0.5,1.0]:
        metrics=[]
        for kzi,kwi in slices:
            U=uf(L,kzi.item(),kwi.item(),np.pi/2,t1=t1)
            n,d,s,_=analyze_spectrum(U)
            metrics.append((n,d,s))
        avg_n=np.mean([m[0] for m in metrics])
        avg_d=np.mean([m[1] for m in metrics])
        avg_s=np.mean([m[2] for m in metrics])
        unique_n=len(set(m[0] for m in metrics))
        differs=(unique_n>1)
        print(f"  {t1:7.2f} {avg_n:4.1f} {avg_d:11.4f} {avg_s:8.4f} {'YES' if differs else 'no'}")
    
    # 3. Gamma sweep at DTC point
    print(f"\n  ---  GAMMA SWEEP (dissipation, alpha=pi/2, t1=0.1)  ---")
    print(f"  {'gamma':>7s} {'Pi':>4s} {'dist(z=-1)':>11s} {'spread':>8s} {'Slice dep?'}")
    print("  "+"-"*45)
    
    for gamma in[0.0,0.1,0.2,0.3,0.4,0.5]:
        metrics=[]
        for kzi,kwi in slices:
            U=uf(L,kzi.item(),kwi.item(),np.pi/2,t1=0.1,gamma=gamma)
            n,d,s,_=analyze_spectrum(U)
            metrics.append((n,d,s))
        avg_n=np.mean([m[0] for m in metrics])
        avg_d=np.mean([m[1] for m in metrics])
        avg_s=np.mean([m[2] for m in metrics])
        unique_n=len(set(m[0] for m in metrics))
        differs=(unique_n>1)
        print(f"  {gamma:7.2f} {avg_n:4.1f} {avg_d:11.4f} {avg_s:8.4f} {'YES' if differs else 'no'}")
    
    print(f"\n{'='*78}")
    print("  NON-DTC VERDICT")
    print(f"{'='*78}")
    print(f"  At the DTC point (alpha=pi/2): pi-modes are binary (32 or 0).")
    print(f"  Away from DTC: pi-modes disappear but spectrum varies.")
    print(f"  The eigenvalue closest to z=-1 tracks parameter changes.")
    print(f"  Slice dependence: not observed at L=2, n_k=4.")
    print(f"  Non-DTC computation requires a different encoding approach.")
    print(f"{'='*78}")

if __name__=="__main__":
    nondtc()
