"""Q48: Word-level phase recovery + Hilbert + combined complexification."""
import sys,json,time
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.signal import hilbert
from scipy.linalg import eigh
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

def word_phases_from_gram(G):
    n=G.shape[0]
    cos_diff=np.clip(G,-1,1)
    phase_diff=np.arccos(cos_diff)
    D2=2*(1-np.cos(phase_diff))
    Hmat=np.eye(n)-np.ones((n,n))/n
    B=-0.5*Hmat@D2@Hmat
    evals,evecs=eigh(B)
    idx=np.argsort(evals)[::-1]
    coords=evecs[:,idx[:2]]*np.sqrt(np.maximum(evals[idx[:2]],0))
    return np.arctan2(coords[:,1],coords[:,0])

rz=np.array(json.load(open("THOUGHT/LAB/FORMULA/v2_2/q48_riemann/zeros_500.json")))
rz_sp=np.diff(rz)/(2*np.pi);rz_sp/=rz_sp.mean()

for mid,name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m=SentenceTransformer(mid,device="cpu")
    embs=m.encode(WORDS,normalize_embeddings=True)
    n=len(WORDS)
    
    # Word-level phases from Gram
    G=embs@embs.T
    phases=word_phases_from_gram(G)
    z_wp=embs.astype(np.complex128)*np.exp(1j*phases[:,None])
    
    # Hilbert
    zh=hilbert(embs,axis=0).astype(np.complex128)
    zhn=np.sqrt(np.sum(np.abs(zh)**2,axis=1,keepdims=True));zh=zh/(zhn+1e-12)
    
    # Combined
    zc=zh.copy()
    for i in range(n): zc[i]=zc[i]*np.exp(1j*phases[i])
    
    for method, zz in [("Word-phase",z_wp),("Hilbert",zh),("Combined",zc)]:
        H=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(zz[i])@zz[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev=np.linalg.eigvalsh(H);ev=ev[ev>1e-12]
        sp=unfold(ev)
        if len(sp)>10:
            ks=stats.ks_2samp(sp,rz_sp)
            print(f"{name} {method:>12s}: {len(sp):4d} spacings  KS p={ks.pvalue:.2e}  P(s<0.3)={(sp<0.3).mean():.4f}")
    print()
