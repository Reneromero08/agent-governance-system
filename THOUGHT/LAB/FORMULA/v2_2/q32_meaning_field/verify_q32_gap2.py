"""Q32 Gap 2: Connect c_sem to M=log(R) using local SciFact cache."""
import os, math, time
import json, numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

# Use local cache
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]

print("Loading models...")
st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
all_embs = st.encode(WORDS_POOL, normalize_embeddings=True, show_progress_bar=False)
centered = all_embs - all_embs.mean(axis=0)
cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov)
evecs = evecs[:, np.argsort(evals)[::-1]]
K = 96

def c_sem_from_embs(embs):
    if len(embs) < 2: return np.nan
    x = np.array(embs)
    x = (x - all_embs.mean(axis=0)) @ evecs[:, :K]
    norms = np.linalg.norm(x, axis=1, keepdims=True); norms[norms==0]=1; x = x / norms
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    n = len(z)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(z[i]).dot(z[j]); H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H); ev = np.maximum(ev, 1e-15); ev = ev / ev.sum()
    sigma = 1.0 / max(ev.sum()**2 / (ev**2).sum(), 1e-10)
    nabla = -np.sum(ev * np.log(ev + 1e-15))
    return np.sqrt(sigma / max(nabla, 1e-10))

print("Loading SciFact from cache...")
from datasets import load_dataset
claims = load_dataset("allenai/scifact", "claims", split="train", trust_remote_code=True)
corpus_raw = load_dataset("allenai/scifact", "corpus", split="train", trust_remote_code=True)

# Build doc_id -> sentences mapping
doc_to_sents = {}
for doc in corpus_raw:
    sents = doc.get("sentences", doc.get("abstract", []))
    if isinstance(sents, str): sents = [sents]
    doc_to_sents[doc["doc_id"]] = sents

def R_grounded(e_claim, e_evidence):
    ev = np.array(e_evidence)
    mu_hat = ev.mean(axis=0)
    cos_sims = [np.dot(e, mu_hat) for e in ev]
    deltaS = np.std(cos_sims) + 1e-6
    claim_sim = np.dot(e_claim, mu_hat)
    z = abs(claim_sim - np.mean(cos_sims)) / deltaS
    E = math.exp(-z**2 / 2)
    return E / deltaS

results = []
for claim in claims:
    # SciFact v2 API: cited_doc_ids has abstract IDs, evidence_doc_id may be empty
    doc_ids = claim.get("cited_doc_ids", [])
    if not doc_ids: continue
    # Use first cited abstract's sentences as evidence
    doc_id = doc_ids[0]
    if doc_id not in doc_to_sents: continue
    sents = doc_to_sents[doc_id]
    if len(sents) < 3: continue
    ev_texts = sents[:8]  # first 8 sentences of abstract

    claim_emb = st.encode(claim["claim"], normalize_embeddings=True)
    ev_embs = [st.encode(t, normalize_embeddings=True) for t in ev_texts]
    R = R_grounded(claim_emb, ev_embs)
    M = math.log(max(R, 1e-6))
    csem = c_sem_from_embs(ev_embs)
    # evidence_label: SUPPORT or CONTRADICT
    label_raw = claim.get("evidence_label", "")
    label = 1 if "SUPPORT" in str(label_raw) else 0
    if not np.isnan(csem):
        results.append({"M": M, "c_sem": csem, "label": label, "n_ev": len(ev_texts)})

Ms = np.array([r["M"] for r in results]); cs = np.array([r["c_sem"] for r in results]); labels = np.array([r["label"] for r in results])

print(f"\n{len(results)} claims ({labels.mean()*100:.0f}% supported):")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

auc_M = roc_auc_score(labels, Ms)
auc_c = roc_auc_score(labels, cs)
print(f"  AUROC(M):      {auc_M:.4f}")
print(f"  AUROC(c_sem):  {auc_c:.4f}")

X_both = np.column_stack([Ms, cs])
lr = LogisticRegression().fit(X_both, labels)
auc_comb = roc_auc_score(labels, lr.predict_proba(X_both)[:,1])
print(f"  AUROC(M+c_sem): {auc_comb:.4f}")
print(f"  {'c_sem ADDS VALUE' if auc_comb > auc_M + 0.03 else 'NO ADDED VALUE'}")

r, p = stats.pearsonr(Ms, cs)
print(f"  Corr(M,c_sem): r={r:.4f} p={p:.4f}")
