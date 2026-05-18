"""Q48: Verify K=96 result across seeds and models."""
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

print("PCA-96 + Hilbert complexification: seed stability test")
print(f"Riemann P(s<0.3) = {(rz_sp<0.3).mean():.4f}\n")

for mid,name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m=SentenceTransformer(mid,device="cpu")
    embs=m.encode(WORDS,normalize_embeddings=True)
    centered=embs-embs.mean(axis=0)
    cov=np.cov(centered.T)
    evals,evecs=np.linalg.eigh(cov)
    idx=np.argsort(evals)[::-1];evecs=evecs[:,idx]
    projected=centered@evecs
    proj96=projected[:,:96]
    norms=np.linalg.norm(proj96,axis=1,keepdims=True);norms[norms==0]=1
    proj96=proj96/norms
    
    ps3_vals,ksp_vals=[],[]
    for seed in range(10):
        np.random.seed(seed)
        # Add tiny perturbation to break symmetries per seed
        z=hilbert(proj96,axis=0).astype(np.complex128)
        z=z+1e-8*np.random.randn(*z.shape)*np.exp(1j*np.random.randn(*z.shape))
        zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True))
        z=z/(zn+1e-12)
        n=len(WORDS)
        H=np.zeros((n,n),dtype=np.complex128)
        for i in range(n):
            for j in range(i,n):
                v=np.conj(z[i])@z[j];H[i,j]=v;H[j,i]=np.conj(v)
        ev=np.linalg.eigvalsh(H);ev=ev[ev>1e-12]
        sp=unfold(ev)
        if len(sp)>10:
            ks=stats.ks_2samp(sp,rz_sp)
            ps3=(sp<0.3).mean()
            ps3_vals.append(ps3);ksp_vals.append(ks.pvalue)
    
    ps3_arr=np.array(ps3_vals);ksp_arr=np.array(ksp_vals)
    n_pass=sum(1 for p in ksp_arr if p>0.05)
    print(f"{name}: P(s<0.3)={ps3_arr.mean():.4f}+/-{ps3_arr.std():.4f}  KS p mean={ksp_arr.mean():.4f}  pass>0.05: {n_pass}/10")
    print(f"  KS p range: [{ksp_arr.min():.4f}, {ksp_arr.max():.4f}]")
    print(f"  P(s<0.3) range: [{ps3_arr.min():.4f}, {ps3_arr.max():.4f}]")
