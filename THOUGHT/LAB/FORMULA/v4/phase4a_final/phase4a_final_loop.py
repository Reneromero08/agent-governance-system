"""Phase 4a Final: Cybernetic Metacognition — Constitution + Self-Steering.

The constitution IS the attractor. The cybernetic loop IS the metacognition:
  - Model measures R per-token against constitution frame C
  - High R → low T (locked on attractor)
  - Low R → high T (explore, find way back)
  - Lindblad verification dampens R on failure

25 prompts x 2 conditions x 3 samples. Measures whether self-steering
increases R and accuracy vs T=0.7 control.
"""

import json, sys, torch, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

R_SCALE = 25.0
T_BASE = 3.0
T_MIN = 0.1
VERIFY_EVERY = 20
MAX_TOKENS = 150
N_SAMPLES = 3
SEED_BASE = 20260516

# ---- Load model ----
print("=" * 60)
print("PHASE 4a FINAL: CONSTITUTION + CYBERNETIC METACOGNITION")
print("=" * 60)
print("Loading model...", flush=True)

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
print(f"Model loaded.", flush=True)

# ---- Build constitution frame C ----
constitution_text = (ROOT.parent / "ai_alignment_control" / "CONSTITUTION.md").read_text(encoding="utf-8")
messages = [{"role": "user", "content": constitution_text}]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
with torch.no_grad():
    hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
    h_avg = hidden.mean(dim=1).squeeze().float()
    w = h_avg / (h_avg.norm() + 1e-12)
print(f"Constitution C built. |w|={w.norm():.4f}", flush=True)

def compute_R(hidden_state):
    h = hidden_state[0, -1, :].float()
    return float((h / (h.norm() + 1e-12)) @ w) ** 2

# ---- Generate ----
def generate(prompt_text, cyb, seed_offset):
    torch.manual_seed(SEED_BASE + seed_offset)
    np.random.seed(SEED_BASE + seed_offset)

    sys_msg = [{"role": "system", "content": constitution_text}]
    user_msg = [{"role": "user", "content": prompt_text}]
    chat = tokenizer.apply_chat_template(sys_msg + user_msg, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    R_traj, T_traj, gen_ids = [], [], []
    past_key_values = None
    correction_factor, correction_decay = 1.0, 0

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values,
                              output_hidden_states=True, use_cache=True)
            else:
                outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        R_raw = compute_R(hidden)
        R_eff = R_raw * R_SCALE * correction_factor
        R_traj.append(R_raw)

        if cyb:
            T = max(T_MIN, T_BASE / (1.0 + R_eff))
        else:
            T = 0.7
        T_traj.append(T)

        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        tid = next_token.item()
        gen_ids.append(tid)
        if tid == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0: correction_factor = 1.0

        if (step + 1) % VERIFY_EVERY == 0:
            # Lindblad: no external oracle, just check if partial output is nonsensical
            # In a real system, this would be external verification
            pass  # Lindblad disabled for pure cybernetic loop

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    R_mean = float(np.mean(R_traj)) if R_traj else 0.0
    T_mean = float(np.mean(T_traj)) if T_traj else 0.0
    return text, R_mean, T_mean, R_traj

# ---- Run ----
all_results = []

for condition, cyb in [("CONTROL", False), ("CYBERNETIC", True)]:
    print(f"\n{'=' * 60}")
    print(f"CONDITION: {condition}")
    print(f"{'=' * 60}")

    for i, entry in enumerate(TEST_PROMPTS):
        pid = entry["id"]
        cat = entry["category"]
        prompt = entry["prompt"]
        vt = entry.get("verification_type", "none")
        gt = entry.get("ground_truth")

        for sample in range(N_SAMPLES):
            seed_off = i * N_SAMPLES + sample + (1000 if cyb else 0)
            t0 = time.time()
            text, R_mean, T_mean, R_traj = generate(prompt, cyb, seed_off)

            if vt != "none" and gt:
                verified, score = verify_answer(text, entry)
            else:
                verified, score = None, None

            r = {
                "condition": condition, "prompt_id": pid, "category": cat,
                "sample": sample, "prompt": prompt, "generated_text": text,
                "R_mean": R_mean, "R_min": float(min(R_traj)) if R_traj else 0,
                "R_max": float(max(R_traj)) if R_traj else 0,
                "T_mean": T_mean, "R_trajectory": R_traj,
                "verified": verified, "verification_score": score,
                "elapsed": time.time() - t0,
            }
            all_results.append(r)

            print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] {pid} [{cat}] s{sample+1} "
                  f"R={R_mean:.4f} T={T_mean:.2f} v={verified} dt={time.time()-t0:.1f}s", flush=True)

    # Save per-condition
    (RESULTS / f"phase4a_final_{condition}.json").write_text(json.dumps(
        [r for r in all_results if r["condition"]==condition], indent=2))

# ---- Analyze ----
print(f"\n{'=' * 60}")
print("RESULTS")
print(f"{'=' * 60}")

from scipy import stats

# Accuracy: only count prompts where verification type supports it
VERIFIABLE_TYPES = {"exact", "contains", "contains_lower"}
all_verifiable_ids = {r["prompt_id"] for r in all_results
                      if any(t in r.get("prompt","") or True for t in VERIFIABLE_TYPES)}
# Actually, filter by verification_type from TEST_PROMPTS
from phase4a_prompts import TEST_PROMPTS as TPS
verifiable_ids = {e["id"] for e in TPS if e.get("verification_type") in VERIFIABLE_TYPES}

# Accuracy on same prompt set per condition
for cond in ["CONTROL", "CYBERNETIC"]:
    cr = [r for r in all_results if r["condition"] == cond and r["prompt_id"] in verifiable_ids]
    verified = [r for r in cr if r["verified"] is not None]
    correct = sum(1 for r in verified if r["verified"])
    n = len(verified)
    acc = correct / n if n else 0
    R_mean = float(np.mean([r["R_mean"] for r in cr]))
    T_mean = float(np.mean([r["T_mean"] for r in cr]))
    R_std = float(np.std([r["R_mean"] for r in cr]))
    print(f"  {cond:>12s}: acc={acc:.3f} ({correct}/{n}) R={R_mean:.4f}+-{R_std:.4f} T={T_mean:.2f}")

# Accuracy significance: chi-square on consistent subset
ctl_acc = [r for r in all_results if r["condition"]=="CONTROL" and r["prompt_id"] in verifiable_ids]
cyb_acc = [r for r in all_results if r["condition"]=="CYBERNETIC" and r["prompt_id"] in verifiable_ids]
ctl_v = [r for r in ctl_acc if r["verified"] is not None]
cyb_v = [r for r in cyb_acc if r["verified"] is not None]
c_correct = sum(1 for r in ctl_v if r["verified"])
x_correct = sum(1 for r in cyb_v if r["verified"])
n_c = len(ctl_v)
n_x = len(cyb_v)
if n_c > 0 and n_x > 0:
    p1, p2 = c_correct/n_c, x_correct/n_x
    pooled = (c_correct + x_correct) / (n_c + n_x)
    se = np.sqrt(pooled * (1-pooled) * (1/n_c + 1/n_x))
    z = (p2 - p1) / max(se, 1e-12)
    p_acc = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"\n  Accuracy delta: {p2-p1:+.3f}  z={z:.4f}  p={p_acc:.4f} {'*' if p_acc < 0.05 else 'ns'}")

# R comparison: independent t-test (not paired — different seeds per condition)
ctl_R = [r["R_mean"] for r in all_results if r["condition"]=="CONTROL"]
cyb_R = [r["R_mean"] for r in all_results if r["condition"]=="CYBERNETIC"]
if ctl_R and cyb_R:
    t, p = stats.ttest_ind(cyb_R, ctl_R)
    delta = np.mean(cyb_R) - np.mean(ctl_R)
    print(f"  R delta (CYB - CTL): {delta:+.4f}  independent t={t:.4f} p={p:.4f}  "
          f"n={len(ctl_R)},{len(cyb_R)} {'*' if p < 0.05 else 'ns'}")

# Category breakdown
for cat in ["factual", "reasoning", "ambiguous", "adversarial"]:
    ctl_cat = [r["R_mean"] for r in all_results if r["condition"]=="CONTROL" and r["category"]==cat]
    cyb_cat = [r["R_mean"] for r in all_results if r["condition"]=="CYBERNETIC" and r["category"]==cat]
    # Accuracy per category (verifiable only)
    ctl_cat_v = [r for r in all_results if r["condition"]=="CONTROL" and r["category"]==cat and r["prompt_id"] in verifiable_ids and r["verified"] is not None]
    cyb_cat_v = [r for r in all_results if r["condition"]=="CYBERNETIC" and r["category"]==cat and r["prompt_id"] in verifiable_ids and r["verified"] is not None]
    c_acc = sum(1 for r in ctl_cat_v if r["verified"])/max(len(ctl_cat_v),1) if ctl_cat_v else 0
    x_acc = sum(1 for r in cyb_cat_v if r["verified"])/max(len(cyb_cat_v),1) if cyb_cat_v else 0
    if ctl_cat and cyb_cat:
        print(f"  {cat:>12s}: CTL R={np.mean(ctl_cat):.4f} acc={c_acc:.2f}  "
              f"CYB R={np.mean(cyb_cat):.4f} acc={x_acc:.2f}  "
              f"dR={np.mean(cyb_cat)-np.mean(ctl_cat):+.4f}")

# Save combined
(RESULTS / "phase4a_final_all.json").write_text(json.dumps(all_results, indent=2))
print(f"\nDone. Results in {RESULTS}")
