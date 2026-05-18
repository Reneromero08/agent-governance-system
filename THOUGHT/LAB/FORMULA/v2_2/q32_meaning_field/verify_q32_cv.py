"""Q32 integrity: 5-fold CV on Climate-FEVER."""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]
st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ae = st.encode(WORDS_POOL, normalize_embeddings=True, show_progress_bar=False)
centered = ae - ae.mean(axis=0); cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov); evecs = evecs[:, np.argsort(evals)[::-1]]

def c_sem(ev_embs):
    if len(ev_embs) < 2: return np.nan
    x = np.array(ev_embs); x = (x - ae.mean(axis=0)) @ evecs[:, :96]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    n = len(z); H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(z[i]).dot(z[j]); H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H); ev = np.maximum(ev, 1e-15); ev = ev / ev.sum()
    s = 1.0 / max(ev.sum()**2 / (ev**2).sum(), 1e-10)
    nb = -np.sum(ev * np.log(ev + 1e-15))
    return np.sqrt(s / max(nb, 1e-10))

climate = load_dataset("climate_fever", split="test", trust_remote_code=True)
results = []
for claim in climate:
    evs = claim.get("evidences", [])
    if len(evs) < 3: continue
    cl = claim.get("claim_label", "")
    if cl not in [0, 2]: continue
    ce = st.encode(claim["claim"], normalize_embeddings=True)
    ee = [st.encode(e["evidence"], normalize_embeddings=True) for e in evs[:10]]
    ea = np.array(ee); mu = ea.mean(axis=0)
    cs_arr = [np.dot(e, mu) for e in ee]
    dS = np.std(cs_arr) + 1e-6
    z_val = abs(np.dot(ce, mu) - np.mean(cs_arr)) / dS
    E = math.exp(-z_val**2 / 2); R = E / dS; M = math.log(max(R, 1e-6))
    csem = c_sem(ee)
    if not np.isnan(csem):
        results.append({"M": M, "c_sem": csem, "label": 1 if cl == 2 else 0})

Ms = np.array([r["M"] for r in results])
cs = np.array([r["c_sem"] for r in results])
labels = np.array([r["label"] for r in results])

X_M = Ms.reshape(-1, 1); X_c = cs.reshape(-1, 1); X_both = np.column_stack([Ms, cs])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_M_cv = cross_val_score(LogisticRegression(), X_M, labels, cv=cv, scoring="roc_auc")
auc_c_cv = cross_val_score(LogisticRegression(), X_c, labels, cv=cv, scoring="roc_auc")
auc_both_cv = cross_val_score(LogisticRegression(), X_both, labels, cv=cv, scoring="roc_auc")

print(f"5-fold CV (n={len(results)}):")
print(f"  M alone:    {auc_M_cv.mean():.4f}+/-{auc_M_cv.std():.4f}")
print(f"  c_sem alone:{auc_c_cv.mean():.4f}+/-{auc_c_cv.std():.4f}")
print(f"  M+c_sem:    {auc_both_cv.mean():.4f}+/-{auc_both_cv.std():.4f}")
print(f"  Gain:       {auc_both_cv.mean() - auc_M_cv.mean():+.4f}")
t, p = stats.ttest_rel(auc_both_cv, auc_M_cv)
print(f"  Paired t-test: t={t:.1f} p={p:.4f}")

# Also: what are the LR coefficients?
from sklearn.linear_model import LogisticRegression as LR
lr = LR().fit(X_both, labels)
print(f"\n  LR coefficients: M={lr.coef_[0][0]:.4f}, c_sem={lr.coef_[0][1]:.4f}")
print(f"  (Positive = higher value -> more likely SUPPORTED)")
