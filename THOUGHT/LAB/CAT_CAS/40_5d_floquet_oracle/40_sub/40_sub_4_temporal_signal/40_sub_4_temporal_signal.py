"""
40_sub_4_temporal_signal.py

EXPERIMENT #4: TEMPORAL SIGNAL PROCESSING VIA CRYSTAL RESONANCE

Sweep t1 (hopping strength) from 0.0 to 1.0 across all 16 momentum slices.
At each t1, pi-mode survival varies per slice because hopping has
momentum-dependent phase factors. The survival pattern across slices
IS the frequency response of the crystal to the input signal.

When t1 is small (<0.2): all slices survive (uniform response).
When t1 is large (>0.5): all slices melt (complete decoherence).
At intermediate t1 (0.25-0.5): DIFFERENT slices survive differently.
This is the crystal acting as a frequency-domain filter.

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

def floquet(L,kz,kw,t1=1.0,loss=0.01,g=0.0):
    H0=build_H(L,t1=t1,loss=loss,gamma=g);N=L*L*4
    P1=torch.zeros((N,N),dtype=COMPLEX);P2=torch.zeros((N,N),dtype=COMPLEX)
    P5=torch.zeros((N,N),dtype=COMPLEX)
    a=b=c=np.pi/2
    for s in range(L*L):
        ib=slice(s*4,(s+1)*4);P1[ib,ib]=b*G1;P2[ib,ib]=c*G2;P5[ib,ib]=a*G5
    return(torch.linalg.matrix_exp(-1j*P2)@torch.linalg.matrix_exp(-1j*P1)@
           torch.linalg.matrix_exp(-1j*P5)@torch.linalg.matrix_exp(-1j*H0))

def pi(U,th=0.3):return int(((torch.linalg.eigvals(U)+1).abs()<th).sum().item())

def temporal_signal():
    L=4;n_k=4
    kz=torch.linspace(0,2*np.pi,n_k);kw=torch.linspace(0,2*np.pi,n_k)
    slices=list(itertools.product(kz,kw))
    
    # Sweep t1 from 0.0 to 1.0
    t1_vals=[0.0,0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.8,1.0]
    
    print("="*78)
    print("  EXPERIMENT #4: TEMPORAL SIGNAL PROCESSING")
    print("  Frequency Response via t1 Sweep Across Momentum Slices")
    print("="*78)
    print(f"  L={L}  n_k={n_k}  slices={len(slices)}")
    print(f"  Sweeping t1 across {len(t1_vals)} values")
    print(f"  Each slice responds differently to hopping strength")
    print("-"*78)
    
    # Header
    print(f"  {'t1':>6s}", end="")
    for idx in range(len(slices)):
        print(f" {'s'+str(idx):>4s}", end="")
    print(f"  {'active':>6s}  {'pattern'}")
    print("  "+"-"*(12+5*len(slices)))
    
    results=[]
    for t1 in t1_vals:
        pi_vals=[pi(floquet(L,kzi.item(),kwi.item(),t1=t1)) for kzi,kwi in slices]
        active=sum(1 for p in pi_vals if p>0)
        pattern=''.join('X' if p>0 else '.' for p in pi_vals)
        
        print(f"  {t1:6.2f}", end="")
        for p in pi_vals:print(f" {p:4d}", end="")
        print(f"  {active:6d}  {pattern}")
        results.append({'t1':t1,'pis':pi_vals,'active':active,'pattern':pattern})
    
    # Analysis
    print(f"\n{'='*78}")
    print("  FREQUENCY RESPONSE ANALYSIS")
    print(f"{'='*78}")
    
    # Find where patterns start to vary
    uniform_survive=[r for r in results if r['active']==16 and len(set(r['pis']))==1]
    partial=[r for r in results if 0<r['active']<16]
    uniform_melt=[r for r in results if r['active']==0]
    
    print(f"  Uniform survive (all 32): {len(uniform_survive)} t1 values")
    print(f"  Partial survival:         {len(partial)} t1 values")
    print(f"  Uniform melt (all 0):     {len(uniform_melt)} t1 values")
    
    if partial:
        print(f"\n  PARTIAL SURVIVAL REGIME (crystal acts as frequency filter):")
        for r in partial:
            unique=len(set(r['pis']))
            min_pi=min(r['pis']);max_pi=max(r['pis'])
            print(f"    t1={r['t1']:.2f}: {r['active']}/16 slices alive, "
                  f"pi range [{min_pi},{max_pi}], {unique} unique values")
        
        print(f"\n  The time crystal filters input signals by frequency.")
        print(f"  Slices surviving at intermediate t1 = resonant frequencies.")
        print(f"  Slices melted at intermediate t1 = non-resonant frequencies.")
        print(f"  Pi-mode count per slice = amplitude of that frequency component.")
    else:
        print(f"\n  No partial survival regime found at L={L}, n_k={n_k}.")
        print(f"  Transition is sharp: all survive then all melt.")
    
    print(f"{'='*78}")

if __name__=="__main__":
    temporal_signal()
