"""Q32: Phase lock test — does phase coherence predict truth?"""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold"]

st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ae = st.encode(WORDS_POOL, normalize_embeddings=True, show_progress_bar=False)
centered = ae - ae.mean(axis=0); cov = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov); evecs = evecs[:, np.argsort(evals)[::-1]]
K = 96

def pure_phase_embs(embeddings):
    """Extract pure phase vectors: |z| = 1, only phase information."""
    x = (embeddings - ae.mean(axis=0)) @ evecs[:, :K]
    norms = np.linalg.norm(x, axis=1, keepdims=True); norms[norms==0]=1; x = x / norms
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    z = z / (zn + 1e-12)
    # Pure phase: strip magnitude, keep only phase angle
    return z / np.abs(z + 1e-15)  # unit complex numbers

def phase_coherence(z_claim, z_evidence):
    """Mean phase coherence between claim and evidence."""
    # P = |(1/D) * sum_d exp(i*(phi_c,d - phi_e,d))|
    # High P -> phase-locked; Low P -> random phase
    phase_diff = np.angle(z_claim * np.conj(z_evidence))
    return np.abs(np.mean(np.exp(1j * phase_diff)))

def phase_kuramoto_r(z_evidence):
    """Kuramoto order parameter across evidence sentences."""
    phases = np.angle(z_evidence)
    return np.abs(np.mean(np.exp(1j * phases)))

# Test on Climate-FEVER
climate = load_dataset("climate_fever", split="test", trust_remote_code=True)

results = []
for claim in climate:
    evs = claim.get("evidences", [])
    if len(evs) < 3: continue
    cl = claim.get("claim_label", "")
    if cl not in [0, 2]: continue

    claim_emb = st.encode(claim["claim"], normalize_embeddings=True)
    ev_texts = [e["evidence"] for e in evs[:10]]
    ev_embs = np.array([st.encode(t, normalize_embeddings=True) for t in ev_texts])

    # Pure phase vectors
    all_embs = np.vstack([claim_emb.reshape(1,-1), ev_embs])
    pp = pure_phase_embs(all_embs)
    pp_claim = pp[0]; pp_ev = pp[1:]

    # Phase coherence between claim and each evidence
    coherences = np.array([phase_coherence(pp_claim, pp_ev[i]) for i in range(len(pp_ev))])
    mean_coh = coherences.mean()
    max_coh = coherences.max()
    std_coh = coherences.std()

    # Evidence-internal phase coherence (Kuramoto r)
    ev_phase_r = phase_kuramoto_r(pp_ev)

    # Also: traditional M=log(R) for comparison
    ev_arr = np.array(ev_embs)
    mu_hat = ev_arr.mean(axis=0)
    cos_sims = [np.dot(e, mu_hat) for e in ev_arr]
    deltaS = np.std(cos_sims) + 1e-6
    claim_sim = np.dot(claim_emb, mu_hat)
    z_val = abs(claim_sim - np.mean(cos_sims)) / deltaS
    E = math.exp(-z_val**2 / 2); R = E / deltaS; M = math.log(max(R, 1e-6))

    results.append({
        "label": 1 if cl == 2 else 0,
        "mean_phase_coh": mean_coh,
        "max_phase_coh": max_coh,
        "std_phase_coh": std_coh,
        "ev_phase_r": ev_phase_r,
        "M": M,
        "n_ev": len(evs)
    })

labels = np.array([r["label"] for r in results])
print(f"\n{len(results)} Climate-FEVER claims ({labels.mean()*100:.0f}% SUPPORTED):")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

X_all = np.column_stack([
    [r["mean_phase_coh"] for r in results],
    [r["max_phase_coh"] for r in results],
    [r["std_phase_coh"] for r in results],
    [r["ev_phase_r"] for r in results],
    [r["M"] for r in results],
])

names = ["mean_phase_coh", "max_phase_coh", "std_phase_coh", "ev_phase_r", "M"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n5-fold CV AUROC:")
for i, name in enumerate(names):
    auc = cross_val_score(LogisticRegression(), X_all[:, i:i+1], labels, cv=cv, scoring="roc_auc")
    print(f"  {name:<20s}: {auc.mean():.4f}+/-{auc.std():.4f}")

# Combined: all phase metrics
X_phase = X_all[:, :4]
auc_phase = cross_val_score(LogisticRegression(), X_phase, labels, cv=cv, scoring="roc_auc")
print(f"  {'all_phase':<20s}: {auc_phase.mean():.4f}+/-{auc_phase.std():.4f}")

# Phase + M
X_phase_M = X_all[:, [0, 1, 2, 3, 4]]
auc_all = cross_val_score(LogisticRegression(), X_phase_M, labels, cv=cv, scoring="roc_auc")
print(f"  {'phase + M':<20s}: {auc_all.mean():.4f}+/-{auc_all.std():.4f}")

# Key test: does supported evidence have HIGHER phase coherence?
from scipy import stats
sup = [r for r in results if r["label"] == 1]
ref = [r for r in results if r["label"] == 0]
t, p = stats.ttest_ind([r["mean_phase_coh"] for r in sup], [r["mean_phase_coh"] for r in ref])
print(f"\n  Supported mean_phase_coh: {np.mean([r['mean_phase_coh'] for r in sup]):.4f}")
print(f"  Refuted mean_phase_coh:   {np.mean([r['mean_phase_coh'] for r in ref]):.4f}")
print(f"  t-test: t={t:.2f} p={p:.4f}")
sup_mean = np.mean([r["mean_phase_coh"] for r in sup])
ref_mean = np.mean([r["mean_phase_coh"] for r in ref])
verdict = "PHASE LOCKS ON TRUTH" if p < 0.05 and sup_mean > ref_mean else "NO PHASE LOCK DIFFERENCE"
print(f"  {verdict}")
