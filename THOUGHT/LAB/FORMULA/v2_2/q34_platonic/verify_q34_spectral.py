"""Q34: Embedding models converge to shared geometry.

Tests: do Native Eigen's C^2 complex embeddings converge to the same
spectral structure as MiniLM (384d) and MPNet (768d)?

Key metric: cumulative variance curve correlation (the proven invariant
from Eigen Alignment v3.7.28 — 0.994 across 19 model pairs).

C5 test: complex (Hilbert) vs real convergence to Native Eigen.
"""
import torch, torch.nn as nn, numpy as np, math, json
from collections import Counter
from datasets import load_dataset
from scipy.stats import spearmanr
torch.manual_seed(42)

# ============================================================
# 1. Native Eigen embeddings
# ============================================================
print("=" * 70)
print("Q34: NATIVE EIGEN vs MiniLM/MPNet SPECTRAL CONVERGENCE")
print("=" * 70)

class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d); self.im = nn.Embedding(V, d)
        nn.init.normal_(self.re.weight, std=0.02); nn.init.normal_(self.im.weight, std=0.02)
    def forward(self, x): return torch.complex(self.re(x), self.im(x))

class NativeAttention(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.qr=nn.Linear(d,d,bias=False); self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False); self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False); self.vi=nn.Linear(d,d,bias=False)
        self.sc=1.0/math.sqrt(d)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi]:
            nn.init.normal_(w.weight,std=0.02)
    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag); qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag); ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag); vi=self.vr(x.imag)+self.vi(x.real)
        qr,kr,vr=qr.transpose(1,2),kr.transpose(1,2),vr.transpose(1,2)
        qi,ki,vi=qi.transpose(1,2),ki.transpose(1,2),vi.transpose(1,2)
        sr=(qr.transpose(-2,-1)@kr+qi.transpose(-2,-1)@ki)*self.sc
        si=(qi.transpose(-2,-1)@kr-qr.transpose(-2,-1)@ki)*self.sc
        mask=torch.triu(torch.ones(S,S,device=x.device),diagonal=1).bool()
        sr=sr.masked_fill(mask,float('-inf')); si=si.masked_fill(mask,0.0)
        attn=torch.nn.functional.softmax(sr,dim=-1)
        cp,sp=torch.cos(si),torch.sin(si)
        out_r=(attn*cp)@vr.transpose(-2,-1)-(attn*sp)@vi.transpose(-2,-1)
        out_i=(attn*cp)@vi.transpose(-2,-1)+(attn*sp)@vr.transpose(-2,-1)
        return torch.complex(out_r,out_i)

class PhaseRot(nn.Module):
    def __init__(self,d): super().__init__(); self.ang=nn.Parameter(torch.ones(d)*0.1)
    def forward(self,z): c,s=torch.cos(self.ang),torch.sin(self.ang); return torch.complex(z.real*c-z.imag*s,z.real*s+z.imag*c)

class NativeEigen(nn.Module):
    def __init__(self,V=2000,d=2,L=2):
        super().__init__(); self.emb=ComplexEmbed(V,d)
        self.layers=nn.ModuleList([nn.ModuleDict({'a':NativeAttention(d),'p':PhaseRot(d)}) for _ in range(L)])
        self.out=nn.Linear(d,V); nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,x):
        z=self.emb(x)
        for l in self.layers:
            z=l['p'](l['a'](z))
        return self.out(torch.abs(z))

def load_wikitext(V=2000,seq=32,N=1000):
    ds=load_dataset("wikitext","wikitext-2-raw-v1",split="train")
    c=Counter()
    for ex in ds:
        for w in str(ex["text"]).split(): c[w]+=1
    voc=["<pad>","<unk>","<eos>"]+[w for w,_ in c.most_common(V-3)]
    w2i={w:i for i,w in enumerate(voc)}
    toks=[]
    for ex in ds:
        for w in str(ex["text"]).split(): toks.append(w2i.get(w,1))
        toks.append(2)
    data=[]
    for i in range(0,min(len(toks)-seq,N*seq),seq//2):
        s=toks[i:i+seq+1]
        if len(s)==seq+1: data.append((s[:-1],s[1:]))
    return data[:N],len(voc),voc,w2i

D="cuda" if torch.cuda.is_available() else "cpu"
data,V,vocab,w2i=load_wikitext(N=1000)
model=NativeEigen(V=V).to(D)

import torch.nn.functional as F
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.01)
model.train()
for ep in range(5):
    tl=0
    for i in range(0,len(data),16):
        b=data[i:i+16]
        if not b: continue
        x=torch.tensor([p[0] for p in b],device=D,dtype=torch.long)
        y=torch.tensor([p[1] for p in b],device=D,dtype=torch.long)
        loss=F.cross_entropy(model(x).view(-1,V),y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
    print(f"  E{ep+1}: ppl={math.exp(tl/max(1,len(data)//16)):.0f}",flush=True)

# Extract Native Eigen embeddings for top-N vocabulary words
N_words=200
top_words=vocab[:N_words]
word_ids=torch.tensor([w2i.get(w,1) for w in top_words],device=D,dtype=torch.long)
with torch.no_grad():
    z=model.emb(word_ids.unsqueeze(0))
    ne_re=z.real[0].cpu().numpy(); ne_im=z.imag[0].cpu().numpy()
    ne_embeds=np.hstack([ne_re,ne_im])
    ne_complex=z[0].cpu().numpy()

print(f"  Native Eigen: {ne_embeds.shape[1]} real dims from C^2")

# ============================================================
# 2. MiniLM and MPNet embeddings
# ============================================================
print("\n  Loading sentence transformers...")
from sentence_transformers import SentenceTransformer
model_minilm=SentenceTransformer("all-MiniLM-L6-v2")
model_mpnet=SentenceTransformer("all-mpnet-base-v2")

minilm_embeds=model_minilm.encode(top_words[:N_words],show_progress_bar=False)
mpnet_embeds=model_mpnet.encode(top_words[:N_words],show_progress_bar=False)
print(f"  MiniLM: {minilm_embeds.shape[1]}d  MPNet: {mpnet_embeds.shape[1]}d")

# ============================================================
# 3. Hilbert complexification
# ============================================================
from scipy.signal import hilbert
def hilbert_complexify(embeds):
    N,D=embeds.shape
    complexified=np.zeros((N,D),dtype=np.complex128)
    for d in range(D):
        analytic=hilbert(embeds[:,d])
        if analytic.dtype!=np.complex128: analytic=analytic.astype(np.complex128)
        complexified[:,d]=analytic
    return complexified

minilm_cpx=hilbert_complexify(minilm_embeds)
mpnet_cpx=hilbert_complexify(mpnet_embeds)

# ============================================================
# 4. Spectral comparison
# ============================================================
def gram_eigenspectrum(embeds,complex_input=False,normalize=True):
    if complex_input:
        G=np.zeros((embeds.shape[0],embeds.shape[0]),dtype=np.complex128)
        for i in range(embeds.shape[0]):
            for j in range(embeds.shape[0]):
                G[i,j]=np.vdot(embeds[i],embeds[j])
    else:
        embeds_norm=embeds/(np.linalg.norm(embeds,axis=1,keepdims=True)+1e-8)
        G=embeds_norm@embeds_norm.T
    if normalize: G=G/np.trace(G)
    eigvals=np.linalg.eigvalsh(G) if not complex_input else np.linalg.eigvalsh(G).real
    eigvals=eigvals[::-1]
    eigvals=np.maximum(eigvals,0)
    eigvals=eigvals/np.sum(eigvals)
    cumulative=np.cumsum(eigvals)
    participation_ratio=np.sum(eigvals)**2/np.sum(eigvals**2) if np.sum(eigvals**2)>0 else 0
    return eigvals,cumulative,participation_ratio

spectra={}
for name,emb,is_complex in [
    ("NativeEigen_C2",ne_complex,True),
    ("MiniLM_real",minilm_embeds,False),
    ("MiniLM_cpx",minilm_cpx,True),
    ("MPNet_real",mpnet_embeds,False),
    ("MPNet_cpx",mpnet_cpx,True),
]:
    ev,cum,pr=gram_eigenspectrum(emb,complex_input=is_complex)
    spectra[name]={"eigvals":ev,"cumulative":cum,"pr":pr}
    print(f"  {name:>16}: PR={pr:.1f}  top5={ev[:5].sum():.3f}")

# ============================================================
# 5. Cross-model cumulative variance correlation (THE invariant)
# ============================================================
print(f"\n{'='*70}")
print(f"CROSS-MODEL CUMULATIVE VARIANCE CORRELATIONS")
print(f"{'='*70}")

pairs=[
    ("MiniLM_real","MPNet_real","Real vs Real"),
    ("MiniLM_cpx","MPNet_cpx","Complex vs Complex"),
    ("MiniLM_real","MiniLM_cpx","Real vs Hilbert"),
    ("MPNet_real","MPNet_cpx","MPNet real vs Hilbert"),
    ("NativeEigen_C2","MiniLM_cpx","NativeEigen vs MiniLM-cpx"),
    ("NativeEigen_C2","MPNet_cpx","NativeEigen vs MPNet-cpx"),
    ("NativeEigen_C2","MiniLM_real","NativeEigen vs MiniLM-real"),
    ("NativeEigen_C2","MPNet_real","NativeEigen vs MPNet-real"),
]

results={}
for a,b,label in pairs:
    ca=spectra[a]["cumulative"]; cb=spectra[b]["cumulative"]
    n_pts=min(len(ca),len(cb))
    rho,_=spearmanr(ca[:n_pts],cb[:n_pts])
    results[label]=rho
    print(f"  {label:>35}: r={rho:+.4f}")

# ============================================================
# 6. C5 test: complex vs real convergence to Native Eigen
# ============================================================
print(f"\n{'='*70}")
print(f"C5 BOUNDARY TEST: Convergence to Native Eigen")
print(f"{'='*70}")

ne_r_cpx=results["NativeEigen vs MiniLM-cpx"]
ne_r_real=results["NativeEigen vs MiniLM-real"]
ne_m_cpx=results["NativeEigen vs MPNet-cpx"]
ne_m_real=results["NativeEigen vs MPNet-real"]

print(f"  NativeEigen vs MiniLM-real:   r={ne_r_real:+.4f}")
print(f"  NativeEigen vs MiniLM-complex: r={ne_r_cpx:+.4f}")
print(f"  NativeEigen vs MPNet-real:    r={ne_m_real:+.4f}")
print(f"  NativeEigen vs MPNet-complex:  r={ne_m_cpx:+.4f}")

cpx_avg=(ne_r_cpx+ne_m_cpx)/2
real_avg=(ne_r_real+ne_m_real)/2
print(f"  Complex avg: {cpx_avg:+.4f}  Real avg: {real_avg:+.4f}")

if cpx_avg>real_avg+0.05:
    print(f"  C5 CONFIRMED: complex embeddings converge {cpx_avg-real_avg:+.3f} better to Native Eigen")
elif abs(cpx_avg-real_avg)<0.03:
    print(f"  C5 NOT DETECTED: real and complex converge equally")
else:
    print(f"  REAL converges better — Hilbert transform may degrade alignment")

# ============================================================
# 7. Df comparison: effective dimensionality
# ============================================================
print(f"\n{'='*70}")
print(f"EFFECTIVE DIMENSIONALITY (Df / Participation Ratio)")
print(f"{'='*70}")
for name in spectra:
    pr=spectra[name]["pr"]
    print(f"  {name:>20}: Df={pr:.2f}")

# ============================================================
# 8. KL divergence between eigenspectra
# ============================================================
print(f"\n{'='*70}")
print(f"KL DIVERGENCE between eigenspectra (normalized)")
print(f"{'='*70}")
for a,b,label in pairs:
    ev_a=spectra[a]["eigvals"]; ev_b=spectra[b]["eigvals"]
    n=min(len(ev_a),len(ev_b))
    pa=np.maximum(ev_a[:n],1e-8); pb=np.maximum(ev_b[:n],1e-8)
    kl=np.sum(pa*np.log(pa/pb))
    print(f"  {label:>35}: KL={kl:.4f}")

print(f"\n{'='*70}")
print(f"Q34 VEREDICT")
print(f"{'='*70}")
best_pair=max(results,key=results.get)
print(f"  Strongest convergence: {best_pair} (r={results[best_pair]:+.4f})")
if cpx_avg>0.7:
    print(f"  CROSS-MODEL CONVERGENCE CONFIRMED: complex embeddings converge (r={cpx_avg:+.3f})")
    print(f"  The cumulative variance curve is the spectral invariant.")
elif cpx_avg>0.5:
    print(f"  MODERATE convergence: r={cpx_avg:+.3f}")
else:
    print(f"  No spectral convergence detected at d=2 scale")
