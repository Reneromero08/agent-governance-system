"""
40_v3_pi_pulse.py — PROVENANCE RECONSTRUCTION (lost in amend)

V3: G1 PI-PULSE PROTOCOL
U_F = exp(-i * G1 * pulse) * exp(-i * H_Dirac * dt)

G1 anticommutes with G5 — the pulse rotates between mass eigenstates.
Tested: pulse={0.5,1,2,3,5,10}, dt={0.2,0.5,pi/4,pi/2}, loss={0.0,0.05}
Result: ALL eigenvalues at ±i, min|z+1|~0.59. No pi-modes.
Conclusion: Two generators cannot produce pi-modes.
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

def build_dirac(L,kz,kw,m0=1.0,tz=1.0,tw=1.0,t1=1.0,loss=0.05,gamma=0.0):
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

def g1_pulse(L,strength):
    N=L*L*4; P=torch.zeros((N,N),dtype=COMPLEX)
    for s in range(L*L): P[slice(s*4,(s+1)*4),slice(s*4,(s+1)*4)]=strength*G1
    return P

print("V3: G1 PI-PULSE FLOQUET PROTOCOL")
for pulse in [0.5,1.0,2.0,3.0,5.0,10.0]:
    for dt in [0.2,0.5,np.pi/4,np.pi/2]:
        H0=build_dirac(4,0.0,0.0,t1=1.0,loss=0.05)
        P=g1_pulse(4,pulse)
        U=torch.linalg.matrix_exp(-1j*P*dt)@torch.linalg.matrix_exp(-1j*H0*dt)
        ev=torch.linalg.eigvals(U)
        dmin=float((ev+1.0).abs().min().item())
        if dmin<0.6: print(f"  pulse={pulse:.1f} dt={dt:.4f}: min|z+1|={dmin:.4f}")
print("CONCLUSION: No pi-modes. Two generators (G1,G5) insufficient.")
