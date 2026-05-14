"""Phase 3: Symbol Survival — lightweight, no GPU required.

Uses sentence-transformers for semantic similarity. Transmission chain
via text corruption (random word swap/drop/repeat) rather than LLM generation.
This isolates the compression signal from the LLM's own priors.
"""

import json, math, random, time, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(20260513)
np.random.seed(20260513)

symbols = [
    {"id": "S1_const", "text": "Maximize the phase coherence of all sentient beings",
     "sigma": 5, "Df": 5},
    {"id": "S2_arc", "text": "The arc of the moral universe is long but it bends toward justice",
     "sigma": 4, "Df": 3},
    {"id": "S3_res", "text": "Resonance amplifies coherence across scales",
     "sigma": 4, "Df": 4},
    {"id": "S4_just", "text": "Justice is the constant and perpetual will to render to each his due",
     "sigma": 3, "Df": 3},
    {"id": "S5_gold", "text": "Do unto others as you would have them do unto you",
     "sigma": 3, "Df": 2},
    {"id": "S6_reas", "text": "Everything happens for a reason",
     "sigma": 2, "Df": 2},
    {"id": "S7_emc2", "text": "Energy equals mass times the speed of light squared",
     "sigma": 5, "Df": 1},
    {"id": "S8_hurt", "text": "Do not hurt people",
     "sigma": 1, "Df": 1},
    {"id": "S9_wash", "text": "Wash your hands before eating",
     "sigma": 1, "Df": 1},
    {"id": "S10_good", "text": "Be good",
     "sigma": 1, "Df": 1},
]

noise_levels = [
    ("LOW", 0.1, 0.0),
    ("MED", 0.25, 0.1),
    ("HIGH", 0.5, 0.25),
]
generations = 8
num_chains = 5  # multiple independent chains per symbol

print("Loading embedder...", flush=True)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Loaded.", flush=True)

def corrupt(text, drop_rate, swap_rate):
    words = text.split()
    if not words: return text
    
    # Random word drops
    kept = [w for w in words if random.random() > drop_rate]
    if not kept: kept = words[:1]  # keep at least one
    
    # Random adjacent swaps
    i = 0
    while i < len(kept) - 1:
        if random.random() < swap_rate:
            kept[i], kept[i+1] = kept[i+1], kept[i]
            i += 2
        else:
            i += 1
    
    return " ".join(kept)

def similarity(original, corrupted):
    emb = embedder.encode([original, corrupted])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-12))

results = []
for sym in symbols:
    print(f"\n{sym['id']} (sigma={sym['sigma']}, Df={sym['Df']})", flush=True)
    
    for nl_name, drop_rate, swap_rate in noise_levels:
        chain_sims = []
        for chain in range(num_chains):
            current = sym["text"]
            sims = [1.0]
            for gen in range(generations):
                current = corrupt(current, drop_rate, swap_rate)
                sim = similarity(sym["text"], current)
                sims.append(sim)
            chain_sims.append(sims)
        
        # Average across chains
        avg_sims = [float(np.mean([c[g] for c in chain_sims])) for g in range(generations + 1)]
        final = avg_sims[-1]
        
        results.append({
            "symbol_id": sym["id"], "sigma": sym["sigma"], "Df": sym["Df"],
            "noise": nl_name, "trajectory": avg_sims, "final_sim": final,
        })
        print(f"  [{nl_name}] final_sim={final:.4f}", flush=True)

# Save
out = RESULTS / "phase3_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

# Analysis
print(f"\n{'='*60}")
print("SURVIVAL BY sigma*Df PRODUCT")
print(f"{'='*60}")

for nl_name, _, _ in noise_levels:
    print(f"\nNoise: {nl_name}")
    by_product = []
    for sym in symbols:
        sym_res = [r for r in results if r["symbol_id"] == sym["id"] and r["noise"] == nl_name]
        if sym_res:
            fs = sym_res[0]["final_sim"]
            product = sym["sigma"] * sym["Df"]
            by_product.append((product, sym["id"], sym["sigma"], sym["Df"], fs))
            print(f"  {sym['id']:>10s} sig={sym['sigma']} Df={sym['Df']} s*Df={product:>3d} final_sim={fs:.4f}")
    
    products = np.array([x[0] for x in by_product])
    finals = np.array([x[4] for x in by_product])
    corr = np.corrcoef(products, finals)[0, 1]
    
    # Also test sigma alone and Df alone
    sigmas = np.array([x[2] for x in by_product])
    dfs = np.array([x[3] for x in by_product])
    corr_s = np.corrcoef(sigmas, finals)[0, 1]
    corr_d = np.corrcoef(dfs, finals)[0, 1]
    
    print(f"  Corr(sigma*Df, final):  {corr:+.4f}")
    print(f"  Corr(sigma, final):     {corr_s:+.4f}")
    print(f"  Corr(Df, final):        {corr_d:+.4f}")
    
    mid = np.median(products)
    high = [x[4] for x in by_product if x[0] > mid]
    low = [x[4] for x in by_product if x[0] <= mid]
    print(f"  High s*Df: mean={np.mean(high):.4f}  Low s*Df: mean={np.mean(low):.4f}  delta={np.mean(high)-np.mean(low):+.4f}")

print("\nDone.", flush=True)
