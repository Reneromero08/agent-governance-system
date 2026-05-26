"""
40_5d_floquet_oracle.py — v4: KNOWN FLOQUET TIME CRYSTAL
U_F = exp(-i * alpha * G1) * exp(-i * eps * G5 * dt)
At eps*dt = pi and alpha = pi/2: U_F = -I -> pi-modes at -1.
Spatial hoppings added to test robustness.
"""
import torch, numpy as np, hashlib
torch.manual_seed(42); torch.set_default_dtype(torch.float64)
COMPLEX = torch.complex64

G1=torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],dtype=COMPLEX)
G5=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]],dtype=COMPLEX)
I4=torch.eye(4,dtype=COMPLEX)
G2=torch.tensor([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]],dtype=COMPLEX)
G3=torch.tensor([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]],dtype=COMPLEX)
G4=torch.tensor([[0,0,-1j,0],[0,0,0,-1j],[1j,0,0,0],[0,1j,0,0]],dtype=COMPLEX)

def build_H(L,kz,kw,m0=1.0,tz=1.0,tw=1.0,t1=1.0,loss=0.05,gamma=0.0):
    N_sp,N=L*L,L*L*4; H=torch.zeros((N,N),dtype=COMPLEX)
    M=m0-tz*np.cos(kz)-tw*np.cos(kw)
    for y in range(L):
        for x in range(L):
            si=y*L+x; ib=slice(si*4,(si+1)*4)
            H[ib,ib]=M*G5-1j*loss*I4
            if gamma>0: H[ib,ib]-=1j*gamma*I4
            nx,ny=(x+1)%L,y; sj=ny*L+nx; jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G1+1j*G2)/2.0; H[ib,jb]+=t1*(G1-1j*G2)/2.0
            nx,ny=x,(y+1)%L; sj=ny*L+nx; jb=slice(sj*4,(sj+1)*4)
            H[jb,ib]+=t1*(G3+1j*G4)/2.0; H[ib,jb]+=t1*(G3-1j*G4)/2.0
    return H

def floquet_site(L,kz,kw,eps=1.0,alpha=np.pi/2,dt=np.pi,gamma=0.0,t1=1.0):
    """Known Floquet TC: U_F = exp(-i*alpha*G1) * exp(-i*eps*G5*dt) per site + hopping."""
    H0=build_H(L,kz,kw,t1=t1,gamma=gamma)
    N=L*L*4
    # Kick: alpha * G1 on every site
    P=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(L*L):
        ib=slice(s*4,(s+1)*4); P[ib,ib]=alpha*G1
    U_free=torch.linalg.matrix_exp(-1j*H0*dt)
    U_kick=torch.linalg.matrix_exp(-1j*P)
    return U_kick@U_free

print("="*78)
print("  EXPERIMENT 40 v4: KNOWN FLOQUET TIME CRYSTAL")
print("  U_F = exp(-i*alpha*G1) * exp(-i*eps*G5*dt)")
print("="*78)

for dt in [np.pi, np.pi/2, np.pi/4, np.pi/8, 0.5, 1.0, 2.0]:
    H0=build_H(4,0.0,0.0,t1=0.0,loss=0.0,gamma=0.0)
    P=torch.zeros((64,64),dtype=COMPLEX)
    for s in range(16): P[slice(s*4,(s+1)*4),slice(s*4,(s+1)*4)] = (np.pi/2)*G1
    U=torch.linalg.matrix_exp(-1j*P) @ torch.linalg.matrix_exp(-1j*H0*dt)
    ev=torch.linalg.eigvals(U)
    n=((ev+1.0).abs()<0.1).sum().item()
    print(f"  dt={dt:.4f}: near -1: {n:2d}/64  min|z+1|={(ev+1.0).abs().min():.4f}")

print(f"  {'='*78}")
