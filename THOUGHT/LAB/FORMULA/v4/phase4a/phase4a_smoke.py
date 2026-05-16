"""Phase 4a smoke test: quick validation of R scaling and T modulation.
Runs 2 prompts x 30 tokens per condition to verify the loop works correctly."""

import json, torch, numpy as np, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))

from phase4a_prompts import TEST_PROMPTS, verify_answer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

C_PATH = RESULTS / "contrastive_C.pt"
EPSILON = 0.1
R_SCALE = 500.0
T_BASE = 3.0
T_MIN = 0.2
MAX_TOKENS = 45

print("=" * 50)
print("PHASE 4a SMOKE TEST")
print("=" * 50)

# Load model
print("Loading model...", flush=True)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

# Load C
c_data = torch.load(C_PATH, map_location="cpu")
C = c_data["C"].to("cuda", dtype=torch.float32)
w_vector = c_data["w_vector"].to("cuda", dtype=torch.float32)
print(f"C loaded. Separation p={c_data['separation_pval']:.6f}", flush=True)

def compute_R(hidden_state):
    h = hidden_state[0, -1, :].float()
    hn = h / (h.norm() + 1e-12)
    rho = torch.outer(hn, hn)
    return float(torch.trace(rho @ C))

def smoke_test_one(entry, condition, max_tokens=MAX_TOKENS):
    """Run one prompt through the loop and report diagnostics."""
    messages = [{"role": "user", "content": entry["prompt"]}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    R_traj = []
    T_traj = []
    gen_ids = []
    past_key_values = None
    correction_factor = 1.0
    correction_decay = 0

    for step in range(max_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(input_ids=input_ids[:, -1:], past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
            else:
                outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)

        logits = outputs.logits[:, -1, :]
        hidden = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        R_raw = compute_R(hidden)
        R_scaled = R_raw * R_SCALE
        R_eff = R_scaled * correction_factor
        R_traj.append(R_eff)

        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0:
                correction_factor = 1.0

        if condition == "CYBERNETIC":
            T = T_BASE / (1.0 + R_eff)
            T = max(T_MIN, T)
        else:
            T = 0.7
        T_traj.append(T)

        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        tid = next_token.item()
        gen_ids.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text_ascii = text.encode('ascii', errors='replace').decode('ascii')
    return text, R_traj, T_traj, text_ascii

# Run on 2 prompts in each condition
test_entries = [TEST_PROMPTS[0], TEST_PROMPTS[9]]  # F1 (factual) and R1 (reasoning)

for condition in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    print(f"\n--- {condition} ---")
    for entry in test_entries:
        text, R_traj, T_traj, text_ascii = smoke_test_one(entry, condition)
        r_raw_vals = [r / R_SCALE for r in R_traj]
        print(f"  {entry['id']}: R_raw=[{min(r_raw_vals):.4f}..{max(r_raw_vals):.4f}]  "
              f"R_eff=[{min(R_traj):.3f}..{max(R_traj):.3f}]  "
              f"T=[{min(T_traj):.2f}..{max(T_traj):.2f}]  "
              f"text='{text_ascii[:60]}...'")

    # Diagnostic
    t_min, t_max = min(T_traj), max(T_traj)
    if condition == "CYBERNETIC":
        # With R_SCALE=500, T_BASE=3.0: T range should be ~0.3-2.5
        if t_min <= T_MIN * 1.1:
            print(f"  INFO: T_min={t_min:.2f} near floor (deterministic attractor active)")
        if t_max > 2.5:
            print(f"  INFO: T_max={t_max:.2f} exploration regime entered")
        print(f"  T range: [{t_min:.2f}, {t_max:.2f}] (target: 0.2-3.0)")
    elif condition == "CYBERNETIC" and (t_max >= T_CLAMP[1] or t_min <= T_CLAMP[0] * 1.5):
        print(f"  WARN: T range [{t_min:.2f}, {t_max:.2f}] near clamp bounds [{T_CLAMP[0]}, {T_CLAMP[1]}]")
    elif condition == "CYBERNETIC":
        print(f"  OK: T range [{t_min:.2f}, {t_max:.2f}] within operating range")

print(f"\nSmoke test complete. Verify T values above are reasonable (~0.3-10.0).")
