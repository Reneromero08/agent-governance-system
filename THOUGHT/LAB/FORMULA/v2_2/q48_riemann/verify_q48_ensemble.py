"""Q48: Ensemble sampling to close KS gap."""
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
    z=hilbert(embs,axis=0).astype(np.complex128)
    zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True))
    z=z/(zn+1e-12)
    n=len(WORDS)
    
    all_sp=[]
    np.random.seed(42)
    for trial in range(20):
        idx=np.random.choice(n,300,replace=False)
        zi=z[idx]
        H=np.zeros((300,300),dtype=np.complex128)
        for i in range(300):
            for j in range(i,300):
                v=np.conj(zi[i])@zi[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev=np.linalg.eigvalsh(H);ev=ev[ev>1e-12]
        sp=unfold(ev)
        if len(sp)>0: all_sp.extend(sp.tolist())
    all_sp=np.array(all_sp)
    ks=stats.ks_2samp(all_sp,rz_sp)
    print(f'{name}: {len(all_sp)} ensemble spacings  KS p={ks.pvalue:.2e}  P(s<0.3)={(all_sp<0.3).mean():.4f}')
