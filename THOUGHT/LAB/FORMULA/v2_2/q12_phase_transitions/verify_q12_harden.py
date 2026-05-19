"""Q12 Hardening: MiniLM K=96, focused angles for speed."""
import os, math
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
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
climate = load_dataset("climate_fever",split="test",trust_remote_code=True)

def build_sf():
    data = []
    for claim in scifact:
        ids = claim.get("cited_doc_ids",[])
        if not ids or ids[0] not in doc_to_sents: continue
        s = doc_to_sents[ids[0]]; s = [t for t in s if isinstance(t,str)]
        if len(s) < 8: continue
        cl = str(claim.get("evidence_label",""))
        data.append({"claim": claim["claim"], "evs": s[:10], "label": 1 if "SUPPORT" in cl else 0})
    return data

def build_cf():
    data = []
    for claim in climate:
        evs = claim.get("evidences",[])
        if len(evs) < 5: continue
        cl = claim.get("claim_label","")
        if cl not in [0,2]: continue
        data.append({"claim": claim["claim"], "evs": [e["evidence"] for e in evs[:8]], "label":1 if cl==2 else 0})
    return data

print("Building datasets...")
sf = build_sf(); cf = build_cf()
print(f"SF={len(sf)}, CF={len(cf)}")

def check_transition(K, claim_text, ev_texts):
    ce = m.encode(claim_text, normalize_embeddings=True)
    ee = [m.encode(t, normalize_embeddings=True) for t in ev_texts]
    all_e = np.vstack([ce.reshape(1,-1), np.array(ee)])
    x = (all_e - ae.mean(axis=0)) @ evecs[:, :K]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    M_seq = []
    for k in range(2, len(ee)+1):
        ev_arr = np.array(ee[:k]); mu = ev_arr.mean(axis=0)
        cs = [np.dot(e, mu) for e in ev_arr]
        dS = np.std(cs) + 1e-6
        z_val = abs(np.dot(ce, mu) - np.mean(cs)) / dS
        E = math.exp(-z_val**2/2); R = E/dS
        M_seq.append(math.log(max(R, 1e-6)))
    M_seq = np.array(M_seq)
    dM = np.diff(M_seq); d2M = np.diff(dM)
    return np.max(np.abs(d2M)) > 2*np.std(dM) if len(d2M)>0 else False

# ANGLE 1: PCA sweep
print("\n=== ANGLE 1: PCA sweep (SF) ===")
np.random.seed(42)
sf_sub = np.random.choice(len(sf), 60, replace=False)
for K in [32, 64, 96, 128, 192, 384]:
    n_t = sum(check_transition(K, sf[i]["claim"], sf[i]["evs"]) for i in sf_sub)
    print(f"  K={K:4d}: {n_t}/60 ({n_t/60:.0%})")

# ANGLE 2: Cross-domain
print("\n=== ANGLE 2: Cross-domain (K=96) ===")
np.random.seed(42)
cf_sub = np.random.choice(len(cf), 80, replace=False)
n_cf = sum(check_transition(96, cf[i]["claim"], cf[i]["evs"]) for i in cf_sub)
print(f"  CF: {n_cf}/80 ({n_cf/80:.0%})")

# ANGLE 3: Seed stability
print("\n=== ANGLE 3: Seed stability (SF, K=96) ===")
for seed in range(5):
    np.random.seed(seed)
    idx = np.random.choice(len(sf), 60, replace=False)
    n_t = sum(check_transition(96, sf[i]["claim"], sf[i]["evs"]) for i in idx)
    print(f"  seed={seed}: {n_t}/60 ({n_t/60:.0%})")

# ANGLE 4: Permuted control
print("\n=== ANGLE 4: Permuted evidence (SF, K=96) ===")
np.random.seed(99)
perm_sub = np.random.choice(len(sf), 50, replace=False)
n_perm = 0
for i in perm_sub:
    evs = sf[i]["evs"].copy(); np.random.shuffle(evs)
    if check_transition(96, sf[i]["claim"], evs): n_perm += 1
print(f"  Permuted: {n_perm}/50 ({n_perm/50:.0%})")

# ANGLE 5: Kuramoto threshold correlation
print("\n=== ANGLE 5: Kuramoto threshold (K=96, 100 claims) ===")
np.random.seed(42)
k_sub = np.random.choice(len(sf), 100, replace=False)
n_kura = 0; n_jump = 0; n_both = 0
for i in k_sub:
    ce = m.encode(sf[i]["claim"], normalize_embeddings=True)
    ee = [m.encode(t, normalize_embeddings=True) for t in sf[i]["evs"]]
    all_e = np.vstack([ce.reshape(1,-1), np.array(ee)])
    x = (all_e - ae.mean(axis=0)) @ evecs[:, :96]
    nrm = np.linalg.norm(x, axis=1, keepdims=True); nrm[nrm==0]=1; x = x / nrm
    z = hilbert(x, axis=0).astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    M_seq, sigma_seq, nabla_seq = [], [], []
    for k in range(2, len(ee)+1):
        ev_arr = np.array(ee[:k]); mu = ev_arr.mean(axis=0)
        cs = [np.dot(e, mu) for e in ev_arr]; dS = np.std(cs) + 1e-6
        z_val = abs(np.dot(ce, mu) - np.mean(cs)) / dS
        E = math.exp(-z_val**2/2); R = E/dS
        M_seq.append(math.log(max(R,1e-6)))
        rho = ev_arr.T@ev_arr/k; r_ev = np.linalg.eigvalsh(rho)
        r_ev = np.maximum(r_ev,1e-15); r_ev /= r_ev.sum()
        s = 1.0/max(r_ev.sum()**2/(r_ev**2).sum(),1e-10)
        nb = -np.sum(r_ev*np.log(r_ev+1e-15))
        sigma_seq.append(s); nabla_seq.append(nb)
    M_seq=np.array(M_seq); sigma_seq=np.array(sigma_seq); nabla_seq=np.array(nabla_seq)
    dM=np.diff(M_seq); d2M=np.diff(dM)
    has_jump = np.max(np.abs(d2M)) > 2*np.std(dM) if len(d2M)>0 else False
    has_kura = (sigma_seq > 2*nabla_seq).any()
    if has_jump: n_jump += 1
    if has_kura: n_kura += 1
    if has_jump and has_kura: n_both += 1
print(f"  M jumps: {n_jump}/100, Kuramoto: {n_kura}/100, Both: {n_both}/100")
if n_kura > 0:
    print(f"  Kuramoto -> jump accuracy: {n_both}/{n_kura} ({n_both/n_kura:.0%})")
print(f"\nDone.")
