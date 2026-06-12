"""Fix EP detection: use eigenvalue separation, not eigenvalue magnitude."""
import numpy as np
KD = {'A':1.8,'G':-0.4,'P':-1.6,'V':4.2}

def build_H(seq, lam=0.0):
    L=len(seq); H=np.zeros((L,L),dtype=np.complex128)
    for i in range(L): H[i,i]=-1j*0.1*KD.get(seq[i],1.8)
    for i in range(L):
        j=(i+1)%L
        di=KD.get(seq[i],1.8); dj=KD.get(seq[j],1.8)
        H[j,i]=0.1+abs(di-dj)*2.0; H[i,j]=0.1*lam
    return H

def eig_separation(H):
    evals=np.sort_complex(np.linalg.eigvals(H))
    diffs=[abs(evals[i+1]-evals[i]) for i in range(len(evals)-1)]
    return float(min(diffs)) if diffs else 0

def cond_number(H):
    _,V=np.linalg.eig(H)
    try:
        Vi=np.linalg.inv(V)
        return float(np.linalg.norm(V)*np.linalg.norm(Vi))
    except np.linalg.LinAlgError: return float('inf')

for lam in [0.0, 0.001, 0.01, 0.1, 1.0]:
    H=build_H('A'*10, lam)
    sep=eig_separation(H)
    cond=cond_number(H)
    is_ep = sep<1e-10 or cond>1e6
    print("Poly-A L=10 lam=%.3f: eig_sep=%.2e cond=%.2e %s" % (lam, sep, cond, "EP" if is_ep else "NOT EP"))

# Compare Poly-A vs GP at lam=0
for name, seq in [("Poly-A","A"*10), ("GP","GP"*5), ("AV","AV"*5)]:
    H=build_H(seq, 0.0)
    sep=eig_separation(H)
    cond=cond_number(H)
    is_ep = sep<1e-10 or cond>1e6
    print("%s lam=0: eig_sep=%.2e cond=%.2e %s" % (name, sep, cond, "EP" if is_ep else "NOT EP"))

# CTC test with correct EP detection
def ctc(seq, lam0=1.0, max_iters=50, lr=0.3):
    lam=lam0
    history=[]
    for step in range(max_iters):
        H=build_H(seq,lam)
        sep=eig_separation(H)
        cond=cond_number(H)
        history.append((step,lam,sep,cond))
        if sep<1e-10 or cond>1e6:
            break
        lam-=lr*sep
        if lam<0: lam=0.0
    return history

for name, seq in [("Poly-A","A"*10), ("GP","GP"*5)]:
    h=ctc(seq)
    s,lam,sep,cond=h[-1]
    print("CTC %s: %d steps lam=%.6f sep=%.2e cond=%.2e %s" % (name, s, lam, sep, cond, "EP" if (sep<1e-10 or cond>1e6) else "NOT"))
