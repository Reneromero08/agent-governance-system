"""Q32 Phase Final Push: evidence depth, cross-domain, per-label, FFT."""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]
st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ae = st.encode(WORDS_POOL, normalize_embeddings=True, show_progress_bar=False)
centered = ae - ae.mean(axis=0); cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov); evecs = evecs[:, np.argsort(evals)[::-1]]

def batch_phase(embeddings):
    x = (embeddings - ae.mean(axis=0)) @ evecs[:, :96]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    return z / np.abs(z + 1e-15)

print("Loading data...")
climate = load_dataset("climate_fever", split="test", trust_remote_code=True)
scifact = load_dataset("allenai/scifact","claims",split="train",trust_remote_code=True)
corpus = load_dataset("allenai/scifact","corpus",split="train",trust_remote_code=True)
doc_to_sents = {d["doc_id"]: d.get("sentences",d.get("abstract",[])) for d in corpus}

# ---- COLLECT ALL TEXTS FOR BOTH DOMAINS ----
all_texts = []
cf_info = []
for claim in climate:
    evs = claim.get("evidences",[])
    if len(evs)<4: continue
    cl = claim.get("claim_label","")
    if cl not in [0,2]: continue
    texts = [claim["claim"]] + [e["evidence"] for e in evs[:8]]
    all_texts.extend(texts)
    labels = [1 if e["evidence_label"]==2 else 0 for e in evs[:8]]
    cf_info.append({"n_ev":len(texts)-1, "label":1 if cl==2 else 0, "ev_labels":labels})

sf_info = []
for claim in scifact:
    doc_ids = claim.get("cited_doc_ids",[])
    if not doc_ids or doc_ids[0] not in doc_to_sents: continue
    ev_texts = doc_to_sents[doc_ids[0]][:8]
    if len(ev_texts)<3: continue
    cl = str(claim.get("evidence_label",""))
    all_texts.extend([claim["claim"]] + ev_texts)
    # SF evidence doesn't have per-sentence labels, mark as claim-level
    sf_info.append({"n_ev":len(ev_texts), "label":1 if "SUPPORT" in cl else 0})

print(f"Embedding {len(all_texts)} texts...")
all_embs = st.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)
all_pp = batch_phase(all_embs)

# ---- PARSE RESULTS ----
def extract_dataset(info_list, start_idx):
    data = []
    cursor = start_idx
    for meta in info_list:
        n = 1 + meta["n_ev"]
        pp = all_pp[cursor:cursor+n]; cursor += n
        pp_c, pp_ev = pp[0], pp[1:]

        # Compute per-evidence phase coherences
        all_coh = np.array([np.abs(np.mean(np.exp(1j * np.angle(pp_c * np.conj(pe))))) for pe in pp_ev])

        # Evidence depth analysis
        depth_coh = {}
        for k in [2,3,5,len(pp_ev)]:
            if k <= len(pp_ev):
                depth_coh[k] = np.mean(all_coh[:k])

        # Per-label phase coherence (CF only)
        per_label = None
        if "ev_labels" in meta:
            sup_coh = all_coh[np.array(meta["ev_labels"])==1].mean() if (np.array(meta["ev_labels"])==1).any() else np.nan
            ref_coh = all_coh[np.array(meta["ev_labels"])==0].mean() if (np.array(meta["ev_labels"])==0).any() else np.nan
            per_label = {"sup": sup_coh, "ref": ref_coh}

        # FFT of coherence sequence
        coh_fft = np.abs(np.fft.fft(all_coh))[:len(all_coh)//2]
        ff_power = coh_fft.sum()
        ff_peak = coh_fft[1:].max() if len(coh_fft) > 2 else 0

        data.append({
            "mean_coh": all_coh.mean(), "label": meta["label"],
            "depth_coh": depth_coh, "per_label": per_label,
            "fft_power": ff_power, "fft_peak": ff_peak,
         })
    return data, cursor

cf_data, _ = extract_dataset(cf_info, 0)
sf_data, sf_cursor = extract_dataset(sf_info, len([t for info in cf_info for t in [""]*(1+info["n_ev"])]))
# Actually use cursor from CF
_, sf_cursor2 = extract_dataset(cf_info, 0)
sf_data2, _ = extract_dataset(sf_info, sf_cursor2)

print(f"CF: {len(cf_data)}, SF: {len(sf_data2)}")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# ---- ANGLE 1: Evidence depth (fixed) ----
print("\n=== ANGLE 1: Evidence depth ===")
for k in [2, 3, 5, 8]:
    Xk = np.array([r["depth_coh"].get(k, np.nan) for r in cf_data]).reshape(-1,1)
    yk = np.array([r["label"] for r in cf_data])
    valid = ~np.isnan(Xk).flatten()
    if valid.sum() > 50:
        auc = cross_val_score(LogisticRegression(), Xk[valid], yk[valid], cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="roc_auc")
        sup = Xk[valid][yk[valid]==1].mean(); ref = Xk[valid][yk[valid]==0].mean()
        print(f"  n_ev={k}: N={valid.sum()}, AUROC={auc.mean():.4f}, sup={sup:.4f}, ref={ref:.4f}, gap={abs(sup-ref):.4f}")

# ---- ANGLE 2: Per-label phase coherence (SUPPORT vs REFUTE evidence) ----
print("\n=== ANGLE 2: Per-label phase coherence (CF) ===")
sup_coh_all = []; ref_coh_all = []
for r in cf_data:
    if r["per_label"]:
        if not np.isnan(r["per_label"]["sup"]): sup_coh_all.append(r["per_label"]["sup"])
        if not np.isnan(r["per_label"]["ref"]): ref_coh_all.append(r["per_label"]["ref"])
print(f"  SUPPORT evidence coh: {np.mean(sup_coh_all):.4f}+/-{np.std(sup_coh_all):.4f} (n={len(sup_coh_all)})")
print(f"  REFUTE evidence coh:  {np.mean(ref_coh_all):.4f}+/-{np.std(ref_coh_all):.4f} (n={len(ref_coh_all)})")
t, p = stats.ttest_ind(sup_coh_all, ref_coh_all)
print(f"  t={t:.1f} p={p:.2e}  {'REFUTE > SUPPORT' if np.mean(ref_coh_all) > np.mean(sup_coh_all) else 'SUPPORT > REFUTE'}")

# ---- ANGLE 3: Cross-domain phase ----
print("\n=== ANGLE 3: Cross-domain ===")
for dname, data in [("Climate-FEVER", cf_data), ("SciFact", sf_data2)]:
    y = np.array([r["label"] for r in data])
    X = np.array([r["mean_coh"] for r in data]).reshape(-1,1)
    auc = cross_val_score(LogisticRegression(), X, y, cv=5, scoring="roc_auc")
    sup_c = X[y==1].mean(); ref_c = X[y==0].mean()
    print(f"  {dname:<20s}: AUROC={auc.mean():.4f}, sup={sup_c:.4f}, ref={ref_c:.4f}")

# ---- ANGLE 4: FFT of coherence sequence ----
print("\n=== ANGLE 4: FFT power (phase coherence frequency domain) ===")
y_cf = np.array([r["label"] for r in cf_data])
X_fft = np.column_stack([[r["fft_power"] for r in cf_data], [r["fft_peak"] for r in cf_data]])
auc_fft = cross_val_score(LogisticRegression(), X_fft, y_cf, cv=5, scoring="roc_auc")
print(f"  FFT features AUROC: {auc_fft.mean():.4f}+/-{auc_fft.std():.4f}")

# ---- ANGLE 5: Phase + M ensemble across both domains ----
print("\n=== ANGLE 5: Ensemble CF phase + SF phase (transfer) ===")
X_cf_ph = np.array([r["mean_coh"] for r in cf_data]).reshape(-1,1)
X_sf_ph = np.array([r["mean_coh"] for r in sf_data2]).reshape(-1,1)
y_cf_np = np.array([r["label"] for r in cf_data])
y_sf_np = np.array([r["label"] for r in sf_data2])

# Train on SF, test on CF
lr_sf = LogisticRegression().fit(X_sf_ph, y_sf_np)
auc_train_sf = lr_sf.score(X_sf_ph, y_sf_np)
auc_test_cf = lr_sf.score(X_cf_ph, y_cf_np)
print(f"  Train SF -> Test CF: SF_acc={auc_train_sf:.4f}, CF_acc={auc_test_cf:.4f}")

# Train on CF, test on SF
lr_cf = LogisticRegression().fit(X_cf_ph, y_cf_np)
auc_sf_test = lr_cf.score(X_sf_ph, y_sf_np)
print(f"  Train CF -> Test SF: CF_acc={lr_cf.score(X_cf_ph, y_cf_np):.4f}, SF_acc={auc_sf_test:.4f}")
print(f"  {'PHASE TRANSFER WORKS' if auc_test_cf > 0.55 or auc_sf_test > 0.55 else 'FAILS'}")
