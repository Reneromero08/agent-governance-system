"""Q48 integrity check: verify KS test, unfolding, and perturbation sensitivity."""
import sys,json
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

def unfold(ev):
    ev=np.sort(ev);ev=ev[ev>1e-15]
    if len(ev)<10: return np.array([])
    N=np.arange(1,len(ev)+1,dtype=float)
    s=UnivariateSpline(np.log(ev),np.log(N),s=len(ev)*0.001,k=3)
    ns=np.exp(s(np.log(ev)));ns=np.maximum(ns,0.1)
    return np.diff(ns)/np.mean(np.diff(ns))

rz=np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp=np.diff(rz)/(2*np.pi);rz_sp/=rz_sp.mean()

for mid,name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m=SentenceTransformer(mid,device="cpu")
    embs=m.encode(WORDS,normalize_embeddings=True)
    centered=embs-embs.mean(axis=0)
    cov=np.cov(centered.T)
    evals,evecs=np.linalg.eigh(cov)
    idx=np.argsort(evals)[::-1];evecs=evecs[:,idx]
    proj=centered@evecs
    for k in [96]:
        pk=proj[:,:k]
        norms=np.linalg.norm(pk,axis=1,keepdims=True);norms[norms==0]=1
        pk=pk/norms
        z=hilbert(pk,axis=0).astype(np.complex128)
        zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True))
        z=z/(zn+1e-12)
        n=len(WORDS)
        H=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(z[i])@z[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev=np.linalg.eigvalsh(H);ev=ev[ev>1e-12]
        
        # Check 1: Eigenvalue range
        print(f"{name} K={k}: {len(ev)} evals  range=[{ev[0]:.4f},{ev[-1]:.4f}]  sum={ev.sum():.2f}")
        
        # Check 2: Unfolding quality (N(E) vs E on log-log should be smooth)
        sp=unfold(ev)
        print(f"  spacings: {len(sp)}  mean={sp.mean():.4f}  std={sp.std():.4f}")
        
        # Check 3: KS vs Riemann
        ks=stats.ks_2samp(sp,rz_sp)
        print(f"  KS p vs Riemann: {ks.pvalue:.4f}")
        
        # Check 4: KS vs SELF (bootstrap — should give high p)
        np.random.seed(0)
        sp_half1=sp[np.random.choice(len(sp),len(sp)//2,replace=False)]
        sp_half2=sp[np.random.choice(len(sp),len(sp)//2,replace=False)]
        ks_self=stats.ks_2samp(sp_half1,sp_half2)
        print(f"  KS p vs SELF (bootstrap): {ks_self.pvalue:.4f} (should be >>0.05)")
        
        # Check 5: Large perturbation — does result change?
        z2=z+0.01*np.random.randn(*z.shape)*np.exp(1j*np.random.randn(*z.shape))
        zn2=np.sqrt(np.sum(np.abs(z2)**2,axis=1,keepdims=True))
        z2=z2/(zn2+1e-12)
        H2=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(z2[i])@z2[j];H2[i,j]=v;H2[j,i]=np.conj(v)
        ev2=np.linalg.eigvalsh(H2);ev2=ev2[ev2>1e-12]
        sp2=unfold(ev2)
        ks2=stats.ks_2samp(sp2,rz_sp)
        print(f"  Perturbation 0.01: P(s<0.3)={(sp2<0.3).mean():.4f}  KS p vs Riemann: {ks2.pvalue:.4f}")
        
        # Check 6: Negative control — random phases instead of Hilbert
        zr=pk.astype(np.complex128)*np.exp(1j*2*np.pi*np.random.rand(n,k))
        Hr=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(zr[i])@zr[j];Hr[i,j]=v;Hr[j,i]=np.conj(v)
        evr=np.linalg.eigvalsh(Hr);evr=evr[evr>1e-12]
        spr=unfold(evr)
        ksr=stats.ks_2samp(spr,rz_sp)
        print(f"  Random phases (neg ctrl): P(s<0.3)={(spr<0.3).mean():.4f}  KS p: {ksr.pvalue:.2e} (should fail)")
        print()
