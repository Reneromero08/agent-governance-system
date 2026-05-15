"""Phase 3b: Proper symbol survival test -- LLM transmission, measured sigma, rated Df."""
import json, math, torch, numpy as np, time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...", flush=True)
# === LOCKED SYMBOLS ===
proverbs = [  # (id, proverb, literal_expansion)
    ("P1", "A stitch in time saves nine",
     "If you fix a small problem early, you prevent a much larger problem later"),
    ("P2", "Actions speak louder than words",
     "What people actually do matters more than what they say they will do"),
    ("P3", "Birds of a feather flock together",
     "People who are similar to each other tend to associate with each other"),
    ("P4", "Don't count your chickens before they hatch",
     "Don't assume a positive outcome will happen before it actually occurs"),
    ("P5", "Every cloud has a silver lining",
     "Even difficult or negative situations contain some element of hope or benefit"),
    ("P6", "Fortune favors the bold",
     "Those who take decisive action and risk are more likely to achieve success"),
    ("P7", "Honesty is the best policy",
     "Being truthful and straightforward in your dealings produces the best long-term outcomes"),
    ("P8", "Look before you leap",
     "Consider the potential consequences carefully before taking action"),
    ("P9", "Rome wasn't built in a day",
     "Significant achievements require sustained effort over a long period of time"),
    ("P10", "The pen is mightier than the sword",
     "Communication and persuasion through writing are more effective at creating lasting change than violence"),
]

controls = [  # (id, text)
    ("C1", "The weather today is partly cloudy with a chance of afternoon showers"),
    ("C2", "She walked to the store to buy groceries for the week"),
    ("C3", "The meeting was scheduled for three o'clock on Thursday afternoon"),
    ("C4", "He decided to take the train instead of driving to work today"),
    ("C5", "The garden needed watering after a week of hot dry weather"),
    ("C6", "They ordered pizza and watched a movie on Friday night"),
    ("C7", "The professor explained the concept using simple diagrams"),
    ("C8", "She found her keys under the couch cushions this morning"),
    ("C9", "The package arrived two days earlier than expected"),
    ("C10", "He practiced the piano for an hour before dinner each evening"),
]

# Df scores (rated, 1-4 layers)
df_scores = {
    "P1": 3, "P2": 3, "P3": 2, "P4": 2, "P5": 3,
    "P6": 3, "P7": 3, "P8": 2, "P9": 3, "P10": 3,
}
# Literal expansions inherit proverb Df (same meaning)
# Controls: Df = 1 (literal only)

# Tokenizer for sigma computation
print("Computing sigma...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)

symbols = []
for pid, proverb, literal in proverbs:
    tok_p = len(tokenizer.encode(proverb))
    tok_l = len(tokenizer.encode(literal))
    sigma = round(tok_l / tok_p, 2)
    df = df_scores[pid]
    symbols.append({"id": pid, "type": "proverb", "text": proverb, "sigma": sigma, "Df": df,
                    "tokens": tok_p, "literal": literal, "literal_tokens": tok_l})
    symbols.append({"id": f"L{pid[1:]}", "type": "literal", "text": literal, "sigma": 1.0, "Df": df,
                    "tokens": tok_l, "proverb": proverb, "proverb_sigma": sigma})

for cid, text in controls:
    tok_c = len(tokenizer.encode(text))
    symbols.append({"id": cid, "type": "control", "text": text, "sigma": 1.0, "Df": 1,
                    "tokens": tok_c})

# Print symbol table
print(f"\n{'ID':>5s} {'type':>8s} {'sigma':>6s} {'Df':>3s} {'tokens':>6s} {'text'}", flush=True)
for s in symbols:
    print(f"{s['id']:>5s} {s['type']:>8s} {s['sigma']:6.2f} {s['Df']:3d} {s['tokens']:6d} {s['text'][:60]}", flush=True)

# Lock and save
lock_path = ROOT / "PREREGISTRATION_PHASE3B.md"
print(f"\nPreregistration: {lock_path}", flush=True)
print(f"Symbols locked: {len(symbols)}", flush=True)
print(f"Noise levels: LOW(0.3) MED(0.7) HIGH(1.2)", flush=True)
print(f"Chains per condition: 5, Generations: 10", flush=True)
print(f"Total chains: {len(symbols) * 3 * 5}", flush=True)

# === LOAD MODELS ===
print("\nLoading models...", flush=True)
try:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
except:
    pass

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto", dtype=torch.float16, trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
print("Models loaded.", flush=True)
print("Starting transmission chains (first call compiles CUDA, ~60s)...", flush=True)

noise_levels = [("LOW", 0.3), ("MED", 0.7), ("HIGH", 1.2)]
generations = 10
num_chains = 5

def transmit(text, temperature, gen_num):
    prompt = f"Paraphrase the following text while preserving its core meaning:\n\n\"{text}\"\n\nParaphrase (generation {gen_num}):"
    msgs = [{"role": "user", "content": prompt}]
    t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    ilen = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=temperature, top_p=0.9, pad_token_id=tokenizer.eos_token_id)  # type: ignore
    ids = out.sequences[0] if hasattr(out, "sequences") else out[0]
    return tokenizer.decode(ids[ilen:], skip_special_tokens=True).strip()

def similarity(original, text):
    emb = embedder.encode([original, text])
    return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-12))

results = []
total = len(symbols) * len(noise_levels)
t_start = time.time()
for si, sym in enumerate(symbols):
    for ni, (nl_name, nl_temp) in enumerate(noise_levels):
        idx = si * len(noise_levels) + ni + 1
        print(f"\n[{idx}/{total}] {sym['id']} {sym['type']} sigma={sym['sigma']} Df={sym['Df']} {nl_name}  (elapsed: {time.time()-t_start:.0f}s)", flush=True)

        chain_finals = []
        for chain in range(num_chains):
            current = sym["text"]
            for gen in range(1, generations + 1):
                t_gen = time.time()
                current = transmit(current, nl_temp, gen)
                print(f"  chain {chain+1}/{num_chains} gen {gen}/{generations} ({time.time()-t_gen:.1f}s)", flush=True)
            final_sim = similarity(sym["text"], current)
            chain_finals.append(final_sim)

        mean_sim = float(np.mean(chain_finals))
        std_sim = float(np.std(chain_finals))
        results.append({**sym, "noise": nl_name, "final_sim": mean_sim, "std": std_sim, "chains": chain_finals})

        print(f"  -> mean={mean_sim:.4f} std={std_sim:.4f}", flush=True)

# Save
out = RESULTS / "phase3b_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")

# === ANALYSIS ===
print(f"\n{'='*60}", flush=True)
print("ANALYSIS", flush=True)
print(f"{'='*60}", flush=True)

for nl_name, _ in noise_levels:
    print(f"\nNoise: {nl_name}", flush=True)
    nr = [r for r in results if r["noise"] == nl_name]

    # Group by type
    prov = [r for r in nr if r["type"] == "proverb"]
    lit = [r for r in nr if r["type"] == "literal"]
    ctrl = [r for r in nr if r["type"] == "control"]

    for label, group in [("Proverbs", prov), ("Literals", lit), ("Controls", ctrl)]:
        sims = [r["final_sim"] for r in group]
        print(f"  {label:>12s}: mean={np.mean(sims):.4f} std={np.std(sims):.4f} n={len(sims)}", flush=True)

    # Proverb vs literal (same meaning, different sigma)
    pair_diffs = []
    for rp in prov:
        rl = [r for r in lit if r["id"] == f"L{rp['id'][1:]}"]
        if rl:
            pair_diffs.append(rp["final_sim"] - rl[0]["final_sim"])
    if pair_diffs:
        t_stat, p_val = stats.ttest_1samp(pair_diffs, 0)
        d = float(np.mean(pair_diffs)) / max(float(np.std(pair_diffs)), 1e-12)
        print(f"  Proverb - Literal (paired): diff={np.mean(pair_diffs):+.4f} p={p_val:.4f} d={d:.2f}", flush=True)

    # sigma*Df correlation
    products = [r["sigma"] * r["Df"] for r in nr if r["type"] != "control"]
    sims_p  = [r["final_sim"] for r in nr if r["type"] != "control"]
    if len(products) > 3:
        r_val, p_val = stats.pearsonr(products, sims_p)
        print(f"  Corr(sigma*Df, survival) for non-controls: r={r_val:+.4f} p={p_val:.4f}", flush=True)

    # High vs low sigma*Df
    all_pairs = [(r["sigma"] * r["Df"], r["final_sim"]) for r in nr if r["type"] != "control"]
    if len(all_pairs) > 4:
        all_pairs.sort()
        mid = len(all_pairs) // 2
        low = [x[1] for x in all_pairs[:mid]]
        high = [x[1] for x in all_pairs[mid:]]
        t_stat, p_val = stats.ttest_ind(high, low)
        d = (np.mean(high) - np.mean(low)) / math.sqrt((np.var(high) + np.var(low)) / 2)
        print(f"  High vs Low sigma*Df: diff={np.mean(high)-np.mean(low):+.4f} p={p_val:.4f} d={d:.2f}", flush=True)

print("\nDone.", flush=True)
