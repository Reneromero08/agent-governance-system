"""Q48: Real vs Complex at K=96 vs K=384."""
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
    n=len(WORDS)
    
    print(f"{name}:")
    for k in [96, 384]:
        pk=proj[:,:k]
        norms=np.linalg.norm(pk,axis=1,keepdims=True);norms[norms==0]=1
        pk=pk/norms
        
        # Real Gram
        G=pk@pk.T
        ev_G=np.linalg.eigvalsh(G);ev_G=ev_G[ev_G>1e-12]
        sp_G=unfold(ev_G)
        ks_G=stats.ks_2samp(sp_G,rz_sp)
        p3_G=(sp_G<0.3).mean()
        
        # Hilbert complex
        z=hilbert(pk,axis=0).astype(np.complex128)
        zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True));z=z/(zn+1e-12)
        H=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(z[i])@z[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev_H=np.linalg.eigvalsh(H);ev_H=ev_H[ev_H>1e-12]
        sp_H=unfold(ev_H)
        ks_H=stats.ks_2samp(sp_H,rz_sp)
        p3_H=(sp_H<0.3).mean()
        
        # Random phases
        np.random.seed(42)
        zr=pk.astype(np.complex128)*np.exp(1j*2*np.pi*np.random.rand(n,k))
        Hr=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(zr[i])@zr[j];Hr[i,j]=v;Hr[j,i]=np.conj(v)
        ev_R=np.linalg.eigvalsh(Hr);ev_R=ev_R[ev_R>1e-12]
        sp_R=unfold(ev_R)
        ks_R=stats.ks_2samp(sp_R,rz_sp)
        p3_R=(sp_R<0.3).mean()
        
        print(f"  K={k:3d}:  REAL P3={p3_G:.4f} KS={ks_G.pvalue:.4f}  |  HILBERT P3={p3_H:.4f} KS={ks_H.pvalue:.4f}  |  RANDOM P3={p3_R:.4f} KS={ks_R.pvalue:.4f}")
    print()
