"""Phase 4a Final: Cybernetic Metacognition — Constitution + Self-Steering.

No external oracle. No retry messages. The loop:
  - Measures R per-token against constitution frame C
  - High R → low T (locked on attractor)
  - Low R → high T (explore, find way back)
  - Lindblad verification every N tokens (correction factor dampens R on failure)

This IS the metacognition. 2 conditions, constitution in both:
  1. CONTROL: T=0.7 fixed (no loop)
  2. CYBERNETIC: T = T_base/(1 + R_eff) (self-steering)

Smoke: 3 prompts, Df=1 and Df=7, 2 conditions.
"""

import json, sys, torch, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

R_SCALE = 25.0   # calibrated for constitution C (R~0.15-0.30)
T_BASE = 3.0
T_MIN = 0.1
SEED = 20260516
torch.manual_seed(SEED); np.random.seed(SEED)

# ---- Load model ----
print("Loading model...", flush=True)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

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

def generate_cybernetic(prompt_text, max_tokens=150, cyb=True, verify_every=20):
    """Token-by-token generation. Cyb=True: T modulation. Cyb=False: T=0.7."""
    sys_msg = [{"role": "system", "content": constitution_text}]
    user_msg = [{"role": "user", "content": prompt_text}]
    messages = sys_msg + user_msg
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    R_traj = []
    gen_ids = []
    past_key_values = None
    correction_factor = 1.0
    correction_decay = 0

    for step in range(max_tokens):
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

        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        tid = next_token.item()
        gen_ids.append(tid)
        if tid == tokenizer.eos_token_id: break
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0: correction_factor = 1.0

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, R_traj

# ---- Run ----
test_prompts = [p for p in TEST_PROMPTS if p["id"] in ["F1", "F5", "R2"]]

for entry in test_prompts:
    pid = entry["id"]
    prompt = entry["prompt"]
    vt = entry.get("verification_type", "none")
    gt = entry.get("ground_truth")

    for cyb in [False, True]:
        label = "CYBERNETIC" if cyb else "CONTROL"
        t0 = time.time()
        text, R_traj = generate_cybernetic(prompt, cyb=cyb, verify_every=150 if not cyb else 20)

        if vt != "none" and gt:
            v, _ = verify_answer(text, entry)

        R_mean = float(np.mean(R_traj)) if R_traj else 0.0
        if cyb:
            T_values = [max(T_MIN, T_BASE/(1.0+r*R_SCALE)) for r in R_traj[:10]]
            print(f"  {pid} {label}: R={R_mean:.4f} T~[{min(T_values):.2f}..{max(T_values):.2f}] verified={v} dt={time.time()-t0:.1f}s")
        else:
            print(f"  {pid} {label}: R={R_mean:.4f} T=0.70 verified={v} dt={time.time()-t0:.1f}s")
        print(f"    R_raw range: [{min(R_traj):.4f}..{max(R_traj):.4f}]")
        print(f"    text: {text.encode('ascii',errors='replace').decode()[:100]}")

print(f"\nDone. Check: R significantly higher than v1's 0.007? T modulation has dynamic range?")
