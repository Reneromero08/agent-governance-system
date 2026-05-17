"""Phase A: Truth Attractor — GPT-2 fragments.

Build 4 verification fragments for GPT-2, test calibration, measure
independence, falsify, acquire fragments, and track dynamics.

All phases iterate on GPT-2 (124M, CPU, fast).
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
import json, math, time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "truth_attractor_results"
RESULTS.mkdir(parents=True, exist_ok=True)

# ---- Load GPT-2 ----
print("Loading GPT-2...", flush=True)
model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Loaded.", flush=True)

# ---- Phase A: Build Fragments ----
facts_db = {
    "capital of France": "Paris",
    "largest planet": "Jupiter", 
    "water freezes at": "0 degrees Celsius",
    "first US president": "George Washington",
    "speed of light": "300000 km/s",
    "mars moons": "Phobos and Deimos",
    "largest ocean": "Pacific",
    "earth orbits": "the Sun",
    "human chromosomes": "46",
    "chemical symbol gold": "Au",
}

def factual_fragment(output, prompt):
    """Check if output contains known facts correctly."""
    score = 0; total = 0
    output_lower = output.lower()
    for fact, answer in facts_db.items():
        if fact.lower() in prompt.lower():
            total += 1
            if answer.lower() in output_lower:
                score += 1
    return {"pass": score >= max(total/2, 1) if total > 0 else True, 
            "confidence": score/max(total,1), "details": f"{score}/{total}"}

def self_consistency_fragment(output, prompt):
    """Generate again, check semantic similarity."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out1 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        out2 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    resp1 = tokenizer.decode(out1[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    resp2 = tokenizer.decode(out2[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    emb = embedder.encode([resp1, resp2])
    sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-12))
    return {"pass": sim > 0.6, "confidence": sim, "details": f"sim={sim:.3f}"}

def logical_fragment(output, prompt):
    """Quick heuristic: does output contain obvious contradictions."""
    text = output.lower()
    contradictions = [
        ("is not", "is"), ("never", "always"), ("none", "all"),
        ("cannot", "can"), ("impossible", "possible")
    ]
    for a, b in contradictions:
        if a in text and b in text:
            return {"pass": False, "confidence": 0.2, "details": f"contradiction: {a}/{b}"}
    return {"pass": True, "confidence": 0.7, "details": "no obvious contradictions"}

def coherence_fragment(output, prompt):
    """Basic check: complete sentences, addresses prompt."""
    # Complete sentences?
    has_period = "." in output
    has_subject = len(output.split()) > 3
    # Addresses prompt? Check keyword overlap
    prompt_words = set(prompt.lower().split()) - {"the","a","an","is","of","in","to","and","what","who","where","when","why","how"}
    output_words = set(output.lower().split())
    overlap = len(prompt_words & output_words) / max(len(prompt_words), 1)
    score = (has_period + has_subject + (overlap > 0.1)) / 3.0
    return {"pass": score > 0.5, "confidence": score, "details": f"period={has_period} subj={has_subject} overlap={overlap:.2f}"}

fragments = [
    ("factual", factual_fragment),
    ("self_consistent", self_consistency_fragment),
    ("logical", logical_fragment),
    ("coherent", coherence_fragment),
]

print(f"Fragments defined: {[f[0] for f in fragments]}", flush=True)

# ---- Phase B: Calibrate Thresholds ----
print("\n=== PHASE B: Calibrate Thresholds ===", flush=True)

calibration_prompts = [
    ("What is the capital of France?", True),
    ("What is the largest planet in our solar system?", True),
    ("At what temperature does water freeze?", True),
    ("Who was the first president of the United States?", True),
    ("Name the largest ocean on Earth.", True),
    ("What planet do we live on?", True),
    ("What is the chemical symbol for gold?", True),
    ("How many chromosomes do humans have?", True),
    ("Do cats lay eggs?", False),
    ("Is the Earth flat?", False),
    ("Can humans breathe underwater?", False),
    ("Is the moon made of cheese?", False),
    ("Can fish climb trees?", False),
    ("Does the sun orbit the Earth?", False),
    ("What is French toast made of? Describe it in detail.", True),  # open-ended
    ("What color is the sky and why?", True),
]

R_values = []
accuracies = []
for prompt, is_true in calibration_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Run fragments
    passes = 0; conf_sum = 0
    for name, frag_fn in fragments:
        result = frag_fn(output, prompt)
        if result["pass"]: passes += 1
        conf_sum += result["confidence"]
    
    R_truth = conf_sum / len(fragments)  # simple average for now
    R_values.append(R_truth)
    
    # Simple accuracy: for factual prompts, check if output matches known answer
    if is_true:
        correct = any(a.lower() in output.lower() for a in facts_db.values())
    else:
        correct = not any(a.lower() in output.lower() for a in facts_db.values())
    accuracies.append(float(correct))
    
    print(f"  R={R_truth:.3f} correct={correct} | {output[:60]}...", flush=True)

# Find optimal threshold
best_f1 = 0; best_theta = 0.5
for theta in np.arange(0.3, 0.9, 0.02):
    preds = [r > theta for r in R_values]
    tp = sum(p and a for p, a in zip(preds, accuracies))
    fp = sum(p and not a for p, a in zip(preds, accuracies))
    fn = sum(not p and a for p, a in zip(preds, accuracies))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    if f1 > best_f1:
        best_f1 = f1; best_theta = float(theta)

R_random = float(np.mean(R_values))
theta_high = best_theta
theta_low = theta_high - (theta_high - R_random) / 2

print(f"\n  theta_high={theta_high:.3f} theta_low={theta_low:.3f} R_random={R_random:.3f} F1={best_f1:.3f}", flush=True)

calibration = {"theta_high": theta_high, "theta_low": theta_low, "R_random": R_random, "F1": best_f1}
(RESULTS / "calibration.json").write_text(json.dumps(calibration, indent=2))

# ---- Phase C: Fragment Independence ----
print("\n=== PHASE C: Fragment Independence ===", flush=True)

fragment_decisions = {name: [] for name, _ in fragments}
fragment_confs = {name: [] for name, _ in fragments}

for prompt, is_true in calibration_prompts[:12]:  # factual only for clean measurement
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    for name, frag_fn in fragments:
        result = frag_fn(output, prompt)
        fragment_decisions[name].append(1 if result["pass"] else 0)
        fragment_confs[name].append(result["confidence"])

# Pairwise Cohen's kappa
print("  Pairwise kappa:")
frag_names = [f[0] for f in fragments]
for i in range(len(frag_names)):
    for j in range(i+1, len(frag_names)):
        k = cohen_kappa_score(fragment_decisions[frag_names[i]], fragment_decisions[frag_names[j]])
        print(f"    {frag_names[i]:>16s} vs {frag_names[j]:16s}: kappa={k:+.3f} {'REDUNDANT' if k > 0.8 else ''}")

# I(S:F) — mutual information with ground truth
print("  I(S:F) (mutual info with ground truth):")
I_SF = {}
for name in frag_names:
    decisions = np.array(fragment_decisions[name])
    gt = np.array([a for _,a in calibration_prompts[:12]])
    # MI = H(S) - H(S|F)
    p_match = np.mean(decisions == gt)
    I = max(0.0, p_match - 0.5) * 2  # scaled
    I_SF[name] = I
    print(f"    {name:>16s}: I(S:F)={I:.3f}")

# Weights
total_I = sum(I_SF.values())
weights = {n: I_SF[n]/max(total_I, 1e-12) for n in frag_names}
print("  Weights:")
for n, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {n:>16s}: w={w:.3f}")

(RESULTS / "fragments.json").write_text(json.dumps({
    "kappa": {f"{a}_{b}": float(cohen_kappa_score(fragment_decisions[a], fragment_decisions[b])) 
              for a in frag_names for b in frag_names if a < b},
    "I_SF": I_SF, "weights": weights
}, indent=2))

# ---- Phase D: Falsification ----
print("\n=== PHASE D: Falsification ===", flush=True)

test_prompts = [
    ("What planet is closest to the Sun?", True),
    ("How many continents are there?", True),
    ("What gas do plants use for photosynthesis?", True),
    ("Who wrote Romeo and Juliet?", True),
    ("Can penguins fly?", False),
    ("Do fish have lungs?", False),
    ("Is plastic biodegradable?", False),
    ("Can you see the Great Wall from space?", False),
]

R_test = []; acc_test = []
for prompt, is_true in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    conf_sum = sum(frag_fn(output, prompt)["confidence"] for _, frag_fn in fragments)
    R_test.append(conf_sum / len(fragments))
    
    correct = any(a.lower() in output.lower() for a in facts_db.values()) if is_true \
              else not any(a.lower() in output.lower() for a in facts_db.values())
    acc_test.append(float(correct))

r_val, p_val = stats.pearsonr(R_test, acc_test)
print(f"  r(R_truth, accuracy) = {r_val:+.4f}, p = {p_val:.4f}")
print(f"  FALSIFICATION: {'PASSED' if r_val > 0.3 and p_val < 0.05 else 'FAILED'}", flush=True)

(RESULTS / "falsification.json").write_text(json.dumps({
    "r": float(r_val), "p": float(p_val), "passed": r_val > 0.3 and p_val < 0.05,
    "R_test": R_test, "acc_test": acc_test
}, indent=2))

print("\nDone.", flush=True)
