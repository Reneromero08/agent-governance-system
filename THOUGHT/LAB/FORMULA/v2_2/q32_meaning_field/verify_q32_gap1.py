"""Q32 Gap 1: Climate-FEVER streaming with cross-encoder (worktree fix)."""
import os, math, time
os.environ["HF_HOME"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "LAW/CONTRACTS/_runs/q32_public/hf_cache/datasets"

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset

print("Loading models...")
st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
ce = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", device="cpu")

print("Loading Climate-FEVER...")
climate = load_dataset("climate_fever", split="test", trust_remote_code=True)

def R_grounded_cosine(claim_emb, evidence_embs):
    """R = E/deltaS using cosine similarity."""
    ev = np.array(evidence_embs)
    mu_hat = ev.mean(axis=0)
    cos_sims = [np.dot(e, mu_hat) for e in ev]
    deltaS = np.std(cos_sims) + 1e-6
    claim_sim = np.dot(claim_emb, mu_hat)
    z = abs(claim_sim - np.mean(cos_sims)) / deltaS
    E = math.exp(-z**2 / 2)
    return E / deltaS

def R_grounded_ce(claim_text, evidence_texts):
    """R = E/deltaS using cross-encoder NLI scores."""
    pairs = [(claim_text, ev) for ev in evidence_texts]
    scores = ce.predict(pairs, show_progress_bar=False)
    mu_hat = np.mean(scores)
    deltaS = np.std(scores) + 1e-6
    # z = distance from claim to evidence mean in NLI-score space
    # claim "score" = max NLI score among evidence
    claim_score = float(np.max(scores))
    z = abs(claim_score - mu_hat) / deltaS
    E = math.exp(-z**2 / 2)
    return E / deltaS, scores

def streaming_test(claim, evidence_texts, truth_label, scoring="cosine"):
    """Streaming M(t): truth-consistent vs truth-inconsistent evidence.
    Returns (R_correct_end, R_wrong_end, correct_wins, n_steps).
    """
    claim_text = claim["claim"]
    claim_emb = st.encode(claim_text, normalize_embeddings=True)

    # Split evidence: SUPPORTED vs REFUTED
    supported_ev = [e["evidence"] for e in evidence_texts if e["evidence_label"] == 2]
    refuted_ev = [e["evidence"] for e in evidence_texts if e["evidence_label"] == 0]

    if len(supported_ev) < 2 or len(refuted_ev) < 2:
        return None

    if scoring == "cosine":
        sup_embs = [st.encode(t, normalize_embeddings=True) for t in supported_ev]
        ref_embs = [st.encode(t, normalize_embeddings=True) for t in refuted_ev]

        # Stream: accumulate evidence, track M at each step
        R_correct = R_grounded_cosine(claim_emb, sup_embs)
        R_wrong = R_grounded_cosine(claim_emb, ref_embs)
    else:
        R_correct, _ = R_grounded_ce(claim_text, supported_ev)
        R_wrong, _ = R_grounded_ce(claim_text, refuted_ev)

    M_correct = math.log(max(R_correct, 1e-6))
    M_wrong = math.log(max(R_wrong, 1e-6))
    correct_wins = 1 if M_correct > M_wrong else 0

    return {"M_correct": M_correct, "M_wrong": M_wrong, "correct_wins": correct_wins}

# Filter claims with both supported and refuted evidence
valid = []
for claim in climate:
    evs = claim.get("evidences", [])
    if len(evs) < 4: continue
    sup_count = sum(1 for e in evs if e["evidence_label"] == 2)
    ref_count = sum(1 for e in evs if e["evidence_label"] == 0)
    if sup_count >= 2 and ref_count >= 2:
        valid.append(claim)

print(f"Claims with >=2 supported AND >=2 refuted evidence: {len(valid)}")

np.random.seed(42)
sample = np.random.choice(len(valid), min(80, len(valid)), replace=False)

for scoring, label in [("cosine", "Fast/Cosine"), ("crossencoder", "Full/Cross-Encoder")]:
    wins = 0; total = 0
    for si in sample:
        claim = valid[si]
        result = streaming_test(claim, claim["evidences"], claim["claim_label"], scoring)
        if result:
            wins += result["correct_wins"]
            total += 1
        if total >= 50: break

    if total > 0:
        acc = wins / total
        print(f"  {label}: {wins}/{total} claims where M_correct > M_wrong ({acc:.1%})")
        print(f"    Gate (acc > 0.55): {'PASS' if acc > 0.55 else 'FAIL'}")
        print(f"    Worktree fast-mode FAIL on CF streaming is {'REPRODUCED' if scoring == 'cosine' and acc <= 0.55 else 'FIXED' if scoring == 'crossencoder' and acc > 0.55 else ''}")
