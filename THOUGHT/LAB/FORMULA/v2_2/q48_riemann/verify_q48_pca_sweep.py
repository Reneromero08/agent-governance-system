"""Q48: PCA-dimension sweep before Hilbert complexification."""
import sys,json,time
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
    D=m.get_sentence_embedding_dimension()
    n=len(WORDS)
    
    # PCA
    centered=embs-embs.mean(axis=0)
    cov=np.cov(centered.T)
    evals,evecs=np.linalg.eigh(cov)
    idx=np.argsort(evals)[::-1];evecs=evecs[:,idx]
    projected=centered@evecs
    
    print(f"\n{name} ({D}d):")
    print(f"{'K dims':>8s} {'P(s<0.3)':>10s} {'KS p':>10s} {'spacings':>8s}")
    
    for k in [8,16,32,64,96,128,192,256,320,384,512,768]:
        if k>D: break
        # Use top K PCA dimensions
        proj_k=projected[:,:k]
        # Normalize (important after projection)
        norms=np.linalg.norm(proj_k,axis=1,keepdims=True);norms[norms==0]=1
        proj_k=proj_k/norms
        
        # Hilbert complexify
        z=hilbert(proj_k,axis=0).astype(np.complex128)
        zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True))
        z=z/(zn+1e-12)
        
        # Hermitian Gram
        H=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(z[i])@z[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev=np.linalg.eigvalsh(H);ev=ev[ev>1e-12]
        sp=unfold(ev)
        if len(sp)>10:
            ks=stats.ks_2samp(sp,rz_sp)
            marker=" <-- BEST" if (sp<0.3).mean()>0.01 and (sp<0.3).mean()<0.03 else ""
            print(f"{k:8d} {(sp<0.3).mean():10.4f} {ks.pvalue:10.2e} {len(sp):8d}{marker}")
