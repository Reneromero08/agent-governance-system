"""Q12 Push: real transition metrics — concentration of change."""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

W = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]
m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ae = m.encode(W, normalize_embeddings=True, show_progress_bar=False)
ctr = ae - ae.mean(axis=0); cov = np.cov(ctr.T)
evals, evecs = np.linalg.eigh(cov); evecs = evecs[:, np.argsort(evals)[::-1]]
S = load_dataset("allenai/scifact","claims",split="train",trust_remote_code=True)
C = load_dataset("allenai/scifact","corpus",split="train",trust_remote_code=True)
ds = {d["doc_id"]: d.get("sentences",d.get("abstract",[])) for d in C}

data = []
for c in S:
    ids = c.get("cited_doc_ids",[])
    if not ids or ids[0] not in ds: continue
    s = ds[ids[0]]; s = [t for t in s if isinstance(t,str)]
    if len(s) < 8: continue
    cl = str(c.get("evidence_label",""))
    data.append({"c": c["claim"], "e": s[:10], "l": 1 if "SUPPORT" in cl else 0})
print(f"N={len(data)}")

np.random.seed(42); samp = np.random.choice(len(data), 200, replace=False)

ratios = []; kuramoto_ratio = []; kuramoto_fires = 0
for si in samp:
    d = data[si]; ce = m.encode(d["c"], normalize_embeddings=True)
    ee = [m.encode(t, normalize_embeddings=True) for t in d["e"]]
    all_e = np.vstack([ce.reshape(1,-1), np.array(ee)])
    x = (all_e - ae.mean(axis=0)) @ evecs[:, :96]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    M, sig, nabla = [], [], []
    for k in range(2, len(ee)+1):
        ev_arr = np.array(ee[:k]); mu = ev_arr.mean(axis=0)
        cs = [np.dot(e, mu) for e in ev_arr]; dS = np.std(cs) + 1e-6
        zv = abs(np.dot(ce, mu) - np.mean(cs)) / dS
        E = math.exp(-zv**2/2); R = E / dS; M.append(math.log(max(R, 1e-6)))
        rho = ev_arr.T @ ev_arr / k; re = np.linalg.eigvalsh(rho)
        re = np.maximum(re, 1e-15); re /= re.sum()
        s1 = 1.0 / max(re.sum()**2 / (re**2).sum(), 1e-10)
        nb = -np.sum(re * np.log(re + 1e-15))
        sig.append(s1); nabla.append(nb)
    M = np.array(M); dM = np.diff(M)
    total_change = abs(M[-1] - M[0])
    max_step = max(abs(dM))
    ratio = max_step / (total_change + 1e-10)
    ratios.append(ratio)
    has_kura = (np.array(sig) > 2 * np.array(nabla)).any()
    if has_kura:
        kuramoto_fires += 1
        kuramoto_ratio.append(ratio)

ratios = np.array(ratios)
print("Fraction of total M change in max single step:")
print(f"  mean={ratios.mean():.3f} median={np.median(ratios):.3f}")
print(f"  q10={np.percentile(ratios,10):.3f} q25={np.percentile(ratios,25):.3f} q75={np.percentile(ratios,75):.3f} q90={np.percentile(ratios,90):.3f}")
print(f"  >30%: {(ratios>0.3).mean()*100:.0f}%  >50%: {(ratios>0.5).mean()*100:.0f}%  >70%: {(ratios>0.7).mean()*100:.0f}%")
print(f"  Kuramoto fires: {kuramoto_fires}/200")
if kuramoto_fires > 0:
    kr = np.array(kuramoto_ratio)
    print(f"  Kuramoto ratio mean: {kr.mean():.3f}")
    t, p = stats.ttest_ind(kr, ratios)
    print(f"  t-test vs baseline: t={t:.1f} p={p:.4f}")
    print(f"  {'KURAMOTO TRANSITIONS ARE SHARPER' if p<0.05 and kr.mean()>ratios.mean() else 'NO DIFFERENCE'}")

# Also: AUROC of ratio
from sklearn.metrics import roc_auc_score
labels = np.array([data[i]["l"] for i in samp])
print(f"\nAUROC(ratio): {roc_auc_score(labels, ratios):.4f}")
