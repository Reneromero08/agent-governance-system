"""Q32 Entropy-as-Mass: does nabla_S explain phase coherence?"""
import os
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

print("Loading + embedding...")
climate = load_dataset("climate_fever", split="test", trust_remote_code=True)

all_texts = []
cf_info = []
for claim in climate:
    evs = claim.get("evidences",[])
    if len(evs)<4: continue
    cl = claim.get("claim_label","")
    if cl not in [0,2]: continue
    texts = [claim["claim"]] + [e["evidence"] for e in evs[:8]]
    all_texts.extend(texts)
    ev_labels = [e["evidence_label"] for e in evs[:8]]
    cf_info.append({"n_ev":len(texts)-1, "label":1 if cl==2 else 0, "ev_labels":ev_labels, "ev_texts":[e["evidence"] for e in evs[:8]]})

all_embs = st.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)
all_pp = batch_phase(all_embs)

# Extract per-claim metrics
data = []
cursor = 0
for meta in cf_info:
    n = 1 + meta["n_ev"]
    pp = all_pp[cursor:cursor+n]; cursor += n
    pp_c, pp_ev = pp[0], pp[1:]
    real_ev = all_embs[cursor-n+1:cursor]  # evidence embeddings (real)

    # Phase coherence per evidence
    all_coh = np.array([np.abs(np.mean(np.exp(1j * np.angle(pp_c * np.conj(pe))))) for pe in pp_ev])

    # SUPPORT vs REFUTE per evidence
    sup_coh = all_coh[np.array(meta["ev_labels"])==2].mean() if 2 in meta["ev_labels"] else np.nan
    ref_coh = all_coh[np.array(meta["ev_labels"])==0].mean() if 0 in meta["ev_labels"] else np.nan

    # ---- ENTROPY AS MASS metrics ----
    # nabla_S = von Neumann entropy of evidence density matrix
    x = real_ev; n_ev = len(x)
    rho = x.T @ x / n_ev
    rho_ev = np.linalg.eigvalsh(rho)
    rho_ev = np.maximum(rho_ev, 1e-15); rho_ev = rho_ev / rho_ev.sum()
    nabla_S = -np.sum(rho_ev * np.log(rho_ev + 1e-15))

    # sigma = 1/Df (inverse participation ratio)
    Df = rho_ev.sum()**2 / (rho_ev**2).sum()
    sigma_val = 1.0 / Df

    # c_sem = sqrt(sigma/nabla_S) — wave speed
    c_sem = np.sqrt(sigma_val / max(nabla_S, 1e-10))

    # GR prediction: curvature = nabla_S * density
    # "density" = semantic mass = mean cosine similarity
    cos_sim = x @ x.T
    density = cos_sim[np.tril_indices(n_ev, k=-1)].mean()

    # Semiotic mass: M_sem = nabla_S * density
    semiotic_mass = nabla_S * density

    data.append({
        "label": meta["label"],
        "mean_coh": all_coh.mean(),
        "sup_coh": sup_coh, "ref_coh": ref_coh,
        "nabla_S": nabla_S, "sigma": sigma_val, "c_sem": c_sem,
        "density": density, "sem_mass": semiotic_mass,
        "n_ev": meta["n_ev"],
    })

print(f"\n{len(data)} claims ({np.mean([r['label'] for r in data])*100:.0f}% SUPPORTED)")

# ---- KEY TEST: Does nabla_S (entropy) differ between supported and refuted? ----
sup = [r for r in data if r["label"]==1]
ref = [r for r in data if r["label"]==0]

print("\n=== ENTROPY AS MASS ===")
metrics = ["nabla_S","sigma","c_sem","density","sem_mass","mean_coh"]
for m in metrics:
    sv = np.mean([r[m] for r in sup if not np.isnan(r[m])])
    rv = np.mean([r[m] for r in ref if not np.isnan(r[m])])
    t, p = stats.ttest_ind([r[m] for r in sup if not np.isnan(r[m])],
                           [r[m] for r in ref if not np.isnan(r[m])])
    higher = "SUPPORT" if sv > rv else "REFUTE"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {m:<12s}: sup={sv:.4f}  ref={rv:.4f}  {higher:>8s} t={t:+.1f} p={p:.4f} {sig}")

# ---- Does nabla_S predict phase coherence? ----
print("\n=== Does entropy predict phase coherence? ===")
from scipy import stats
for grp_name, grp in [("SUPPORTED", sup), ("REFUTED", ref)]:
    ns_vals = np.array([r["nabla_S"] for r in grp if not np.isnan(r["nabla_S"])])
    coh_vals = np.array([r["mean_coh"] for r in grp if not np.isnan(r["nabla_S"])])
    if len(ns_vals) > 10:
        r, p = stats.pearsonr(ns_vals, coh_vals)
        print(f"  {grp_name}: corr(nabla_S, mean_coh) = {r:.4f} (p={p:.4f})")

# ---- Semiotic mass: does higher mass -> higher phase coherence? ----
print("\n=== Semiotic mass -> phase coherence? ===")
for grp_name, grp in [("ALL", data), ("SUPPORTED", sup), ("REFUTED", ref)]:
    sm_vals = np.array([r["sem_mass"] for r in grp if not np.isnan(r["sem_mass"])])
    coh_vals = np.array([r["mean_coh"] for r in grp if not np.isnan(r["sem_mass"])])
    if len(sm_vals) > 10:
        r, p = stats.pearsonr(sm_vals, coh_vals)
        print(f"  {grp_name}: corr(sem_mass, mean_coh) = {r:.4f} (p={p:.4f})")

# ---- GR: does c_sem relate to curvature-like quantities? ----
print("\n=== GR check: c_sem vs density ===")
for grp_name, grp in [("SUPPORTED", sup), ("REFUTED", ref)]:
    cs_vals = np.array([r["c_sem"] for r in grp if not np.isnan(r["c_sem"])])
    dens_vals = np.array([r["density"] for r in grp if not np.isnan(r["c_sem"])])
    if len(cs_vals) > 10:
        r, p = stats.pearsonr(cs_vals, dens_vals)
        print(f"  {grp_name}: corr(c_sem, density) = {r:.4f} (p={p:.4f})")
