"""Phase 4a v3 SMOKE: Df sweep with retry correction.

Tests the sigma^Df amplifier: as Df (retry attempts) increases,
does accuracy improve? If sigma > 1, accuracy should scale
exponentially with Df.

QEC-aligned: no mid-generation modification. Each attempt is an
independent generation. Correction = retry at lower temperature.

3 prompts x 4 Df levels x 1 attempt each. ~3 minutes.
"""

import json, sys, torch, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SEED = 20260516
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("PHASE 4a v3 SMOKE: Df Sweep")
print("=" * 60)

# Load model
print("Loading model...", flush=True)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
print(f"Model loaded.", flush=True)

# Load v1 C (for R measurement, not for control)
v1_c_path = ROOT.parent / "phase4a" / "results" / "contrastive_C.pt"
c_data = torch.load(str(v1_c_path), map_location="cpu")
C = c_data["C"].to("cuda", dtype=torch.float32)
w = c_data["w_vector"].to("cuda", dtype=torch.float32)

def compute_R(hidden_state):
    h = hidden_state[0, -1, :].float()
    hn = h / (h.norm() + 1e-12)
    return float((hn @ w) ** 2)

def generate_at_temperature(prompt_text, T):
    """Generate at fixed temperature T. Returns (text, R_raw_traj)."""
    messages = [{"role": "user", "content": prompt_text}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    R_traj = []
    gen_ids = []
    past_key_values = None

    for step in range(150):
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values,
                              output_hidden_states=True, use_cache=True)
            else:
                outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values
        R_traj.append(compute_R(hidden))
        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        tid = next_token.item()
        gen_ids.append(tid)
        if tid == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, R_traj

# Df levels to sweep
Df_levels = [1, 2, 3, 5, 7]
# Temperature schedule: each retry halves T (more deterministic each time)
T_schedule = [0.7, 0.35, 0.18, 0.09, 0.045, 0.022, 0.011]

# Prompts that failed in v1 CONTROL — to test whether retries help
# F5: population, F6: water formula, R1: train problem, R2: bat/ball, R5: coin flip
test_prompts = [p for p in TEST_PROMPTS if p["id"] in ["F5", "F6", "R1", "R2", "R5"]]

results = []

for entry in test_prompts:
    pid = entry["id"]
    prompt = entry["prompt"]
    vt = entry.get("verification_type", "none")

    for df in Df_levels:
        t0 = time.time()
        attempts = 0
        correct = False
        final_text = ""
        R_per_attempt = []

        for attempt in range(df):
            T = T_schedule[min(attempt, len(T_schedule)-1)]
            attempts += 1
            text, R_traj = generate_at_temperature(prompt, T)
            R_per_attempt.append(float(np.mean(R_traj)) if R_traj else 0.0)

            if vt != "none" and entry.get("ground_truth"):
                verified, score = verify_answer(text, entry)
                if verified:
                    correct = True
                    final_text = text
                    break
            final_text = text

        elapsed = time.time() - t0

        r = {
            "prompt_id": pid, "category": entry["category"],
            "Df_max": df, "attempts_used": attempts,
            "correct": correct, "R_per_attempt": R_per_attempt,
            "elapsed": elapsed,
        }
        results.append(r)
        print(f"  {pid} Df={df}: attempts={attempts} correct={correct} R={R_per_attempt} dt={elapsed:.1f}s", flush=True)

# Analyze
print(f"\n{'=' * 60}")
print("SMOKE ANALYSIS")
print(f"{'=' * 60}")

for pid in ["F1", "F2", "F3"]:
    print(f"\n  {pid}:")
    pid_results = [r for r in results if r["prompt_id"] == pid]
    for r in pid_results:
        print(f"    Df={r['Df_max']}: correct={r['correct']} attempts={r['attempts_used']}")

# Compute accuracy per Df
print(f"\n  Accuracy by Df:")
for df in Df_levels:
    df_results = [r for r in results if r["Df_max"] == df]
    acc = sum(1 for r in df_results if r["correct"]) / len(df_results)
    avg_attempts = float(np.mean([r["attempts_used"] for r in df_results]))
    print(f"    Df={df}: acc={acc:.3f} ({sum(1 for r in df_results if r['correct'])}/{len(df_results)})  avg_attempts={avg_attempts:.1f}")

# Test sigma^Df: does ln(1-acc) decrease linearly with Df?
# If sigma > 1, error_rate(Df) = error_rate(0) * (1-sigma)^Df
# ln(error_rate) should be linear in Df with slope ln(1-sigma)
print(f"\n  Error rate by Df (test sigma^Df):")
for df in Df_levels:
    df_results = [r for r in results if r["Df_max"] == df]
    err = 1 - sum(1 for r in df_results if r["correct"]) / max(len(df_results), 1)
    print(f"    Df={df}: err={err:.3f}  ln(err)={np.log(max(err,1e-6)):.3f}")

print(f"\nDone. If sigma > 1: accuracy should increase with Df. If sigma ~= 1: flat.")
