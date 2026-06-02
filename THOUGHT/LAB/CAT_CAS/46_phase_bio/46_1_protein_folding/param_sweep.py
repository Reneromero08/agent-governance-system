"""Parameter sweep for 46.1: find parameters where winding discriminates foldable vs frustrated."""
import numpy as np, os
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def build_H(seq, gamma=0.3, t_base=0.1, frust_scale=2.0):
    L=len(seq); H=np.zeros((L,L),dtype=np.complex128)
    for i in range(L): H[i,i]=-1j*gamma*KD.get(seq[i],1.8)
    for i in range(L):
        j=(i+1)%L
        di=KD.get(seq[i],1.8); dj=KD.get(seq[j],1.8)
        delta=abs(di-dj); frust=delta*frust_scale
        H[j,i]=t_base+frust; H[i,j]=t_base
    return H

def compute_W(H):
    D=np.diag(np.diag(H)); O=H-D
    phis=np.linspace(0,2*np.pi,200)
    dets=np.array([np.linalg.det(D+np.exp(1j*p)*O) for p in phis])
    return int(round((np.unwrap(np.angle(dets))[-1]-np.unwrap(np.angle(dets))[0])/(2*np.pi)))

rng=np.random.RandomState(42)
aa=list(KD.keys())
r1=''.join(rng.choice(aa,30)); r2=''.join(rng.choice(aa,30))

print("Parameter sweep for W(Poly-A)=0, W(GP)!=0, W(random)!=0 at L=30")
print(f"{'gamma':>6s} {'t_base':>8s} {'fscale':>8s} {'Poly-A':>8s} {'GP':>8s} {'R1':>8s} {'R2':>8s} {'OK'}")
print("-"*70)
for gamma in [0.1, 0.2, 0.3, 0.5]:
    for tb in [0.05, 0.1, 0.2]:
        for fs in [1.0, 2.0, 3.0, 5.0]:
            wA=compute_W(build_H('A'*30,gamma,tb,fs))
            wGP=compute_W(build_H('GP'*15,gamma,tb,fs))
            wR1=compute_W(build_H(r1,gamma,tb,fs))
            wR2=compute_W(build_H(r2,gamma,tb,fs))
            ok=(wA==0 and wGP!=0 and wR1!=0 and wR2!=0)
            mark="***" if ok else ""
            print(f"{gamma:6.1f} {tb:8.2f} {fs:8.1f} {wA:+8d} {wGP:+8d} {wR1:+8d} {wR2:+8d} {mark}")
