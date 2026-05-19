"""Q12 integrity audit: d2M distribution, better thresholds, Kuramoto validation."""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]
m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ae = m.encode(WORDS_POOL, normalize_embeddings=True, show_progress_bar=False)
centered = ae - ae.mean(axis=0); cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov); evecs = evecs[:, np.argsort(evals)[::-1]]

scifact = load_dataset("allenai/scifact","claims",split="train",trust_remote_code=True)
corpus = load_dataset("allenai/scifact","corpus",split="train",trust_remote_code=True)
doc_to_sents = {d["doc_id"]: d.get("sentences",d.get("abstract",[])) for d in corpus}

sf = []
for claim in scifact:
    ids = claim.get("cited_doc_ids",[])
    if not ids or ids[0] not in doc_to_sents: continue
    s = doc_to_sents[ids[0]]; s = [t for t in s if isinstance(t,str)]
    if len(s) < 8: continue
    cl = str(claim.get("evidence_label",""))
    sf.append({"claim": claim["claim"], "evs": s[:10], "label": 1 if "SUPPORT" in cl else 0})

print(f"SF claims: {len(sf)}")
np.random.seed(42)
samp = np.random.choice(len(sf), 200, replace=False)

K = 96
all_ratios = []; all_maxd2m = []; all_stds = []
kuramoto_fires = 0; kuramoto_jumps = 0

for si in samp:
    item = sf[si]
    ce = m.encode(item["claim"], normalize_embeddings=True)
    ee = [m.encode(t, normalize_embeddings=True) for t in item["evs"]]
    all_e = np.vstack([ce.reshape(1,-1), np.array(ee)])
    x = (all_e - ae.mean(axis=0)) @ evecs[:, :K]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

    M_seq, sigma_seq, nabla_seq = [], [], []
    for k in range(2, len(ee)+1):
        ev_arr = np.array(ee[:k]); mu = ev_arr.mean(axis=0)
        cs = [np.dot(e, mu) for e in ev_arr]; dS = np.std(cs) + 1e-6
        z_val = abs(np.dot(ce, mu) - np.mean(cs)) / dS
        E = math.exp(-z_val**2/2); R = E/dS; M_seq.append(math.log(max(R,1e-6)))
        rho = ev_arr.T@ev_arr/k; r_ev = np.linalg.eigvalsh(rho)
        r_ev = np.maximum(r_ev,1e-15); r_ev /= r_ev.sum()
        s_val = 1.0/max(r_ev.sum()**2/(r_ev**2).sum(),1e-10)
        nb = -np.sum(r_ev*np.log(r_ev+1e-15))
        sigma_seq.append(s_val); nabla_seq.append(nb)

    M_seq = np.array(M_seq)
    dM = np.diff(M_seq); d2M = np.diff(dM)
    ratio = np.max(np.abs(d2M)) / (np.std(dM) + 1e-10)
    all_ratios.append(ratio); all_maxd2m.append(np.max(np.abs(d2M))); all_stds.append(np.std(dM))

    has_kura = (np.array(sigma_seq) > 2*np.array(nabla_seq)).any()
    if has_kura:
        kuramoto_fires += 1
        if np.max(np.abs(d2M)) > 2*np.std(dM):
            kuramoto_jumps += 1

all_ratios = np.array(all_ratios)
print(f"\n=== d2M distribution ===")
print(f"  max|d2M|/std(dM): mean={all_ratios.mean():.2f}, median={np.median(all_ratios):.2f}")
print(f"  q10={np.percentile(all_ratios,10):.2f}, q25={np.percentile(all_ratios,25):.2f}")
print(f"  q75={np.percentile(all_ratios,75):.2f}, q90={np.percentile(all_ratios,90):.2f}")
print(f"  Fraction > 2: {(all_ratios > 2).mean()*100:.0f}%")
print(f"  Fraction > 5: {(all_ratios > 5).mean()*100:.0f}%")
print(f"  Fraction > 10: {(all_ratios > 10).mean()*100:.0f}%")

# Better metric: look at M(t) convexity - does M change SIGN? Or does M cross a stability threshold?
print(f"\n=== Better phase transition metrics ===")
# 1. Sigmoid fit: does M(t) have a clear inflection?
# M(t) should start flat, jump, flatten. Fit: M(t) ~ a/(1+exp(-b*(t-c)))
from scipy.optimize import curve_fit
def sigmoid(t, a, b, c, d): return a/(1+np.exp(-b*(t-c))) + d

sig_r2s = []
for si in samp[:100]:
    item = sf[si]
    ce = m.encode(item["claim"], normalize_embeddings=True)
    ee = [m.encode(t, normalize_embeddings=True) for t in item["evs"]]
    M_seq = []
    for k in range(2, len(ee)+1):
        ev_arr = np.array(ee[:k]); mu = ev_arr.mean(axis=0)
        cs = [np.dot(e, mu) for e in ev_arr]; dS = np.std(cs) + 1e-6
        z_val = abs(np.dot(ce, mu) - np.mean(cs)) / dS
        E = math.exp(-z_val**2/2); R = E/dS; M_seq.append(math.log(max(R,1e-6)))
    M_seq = np.array(M_seq)
    t = np.arange(len(M_seq))
    try:
        popt, _ = curve_fit(sigmoid, t, M_seq, p0=[M_seq[-1]-M_seq[0], 1, len(M_seq)//2, M_seq[0]], maxfev=1000)
        pred = sigmoid(t, *popt)
        r2 = 1 - np.sum((M_seq-pred)**2)/np.sum((M_seq-M_seq.mean())**2)
        sig_r2s.append(r2)
    except: pass

sig_r2s = np.array(sig_r2s)
print(f"  Sigmoid fit R^2: mean={sig_r2s.mean():.3f}, med={np.median(sig_r2s):.3f}")
print(f"  R^2 > 0.5: {(sig_r2s>0.5).mean()*100:.0f}%, R^2 > 0.8: {(sig_r2s>0.8).mean()*100:.0f}%")

# 2. Actual Kuramoto validation
print(f"\n  Kuramoto threshold: fires={kuramoto_fires}/200, jumps when fires={kuramoto_jumps}/{kuramoto_fires} ({kuramoto_jumps/kuramoto_fires*100:.0f}%)")

# 3. Transition magnitude vs truth label
from sklearn.metrics import roc_auc_score
labels = [sf[i]["label"] for i in samp]
auc_ratio = roc_auc_score(labels, all_ratios)
auc_d2m = roc_auc_score(labels, all_maxd2m)
print(f"\n  AUROC(max|d2M|): {auc_d2m:.4f}")
print(f"  AUROC(ratio): {auc_ratio:.4f}")
