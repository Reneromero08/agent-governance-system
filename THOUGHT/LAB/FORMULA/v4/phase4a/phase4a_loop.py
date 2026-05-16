"""Phase 4a: Cybernetic Truth Token-Level Control Loop Experiment.

Three conditions:
  1. CONTROL:   Standard generation at T=0.7. Observer-only R tracking.
  2. CYBERNETIC: T = 1/(R + epsilon). C from contrastive pairs. Lindblad verification.
  3. VERIFY:     Fixed T=0.7 with verification but no R-driven temperature modulation.

Tests whether the loop finds truth WITHOUT a constitution.
"""

import json, torch, math, time, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
C_PATH = RESULTS / "contrastive_C.pt"

import sys
sys.path.insert(0, str(ROOT))
from phase4a_prompts import TEST_PROMPTS, verify_answer

# ---- Config ----
R_SCALE = 500.0              # scale factor: 0.007 raw -> 3.5 eff -> T≈0.67 (baseline)
T_BASE = 3.0                 # base temperature (T when R=0)
T_MIN = 0.2                  # minimum temperature (high R, deterministic attractor)
VERIFY_EVERY = 20            # Lindblad verification interval (tokens)
MAX_NEW_TOKENS = 150         # max tokens per prompt
SEED = 20260516

# Conditions to skip (if already run successfully)
SKIP_CONDITIONS = {"CONTROL", "VERIFY"}  # Already have valid data

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Step 1: Load model ----
print("=" * 60)
print("PHASE 4a: TOKEN-LEVEL CYBERNETIC TRUTH EXPERIMENT")
print("=" * 60)
print("\n[1/5] Loading Gemma 4B E4B-it...", flush=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it",
    quantization_config=quant_config,
    device_map="auto",
    dtype=torch.float16,
    output_hidden_states=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()
print(f"Model loaded. Device: {next(model.parameters()).device}", flush=True)

# ---- Step 2: Load C ----
print(f"\n[2/5] Loading contrastive alignment frame C...", flush=True)
c_data = torch.load(C_PATH, map_location="cpu")
C = c_data["C"].to("cuda", dtype=torch.float32)
w_vector = c_data["w_vector"].to("cuda", dtype=torch.float32)
print(f"  C shape: {C.shape}  Separation p={c_data['separation_pval']:.6f}", flush=True)

# ---- Step 3: Helper functions ----
def compute_R(hidden_state):
    """Compute resonance R = Tr(rho C) from last-token hidden state."""
    h = hidden_state[0, -1, :].float()  # last token position
    hn = h / (h.norm() + 1e-12)
    rho = torch.outer(hn, hn)
    return float(torch.trace(rho @ C))

def tokenize_prompt(prompt_text):
    """Tokenize a user prompt as chat template with generation prompt."""
    messages = [{"role": "user", "content": prompt_text}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    return inputs.input_ids

def _generate_core(input_ids, max_tokens, condition, prompt_entry):
    """Token-by-token generation with R tracking and Lindblad verification.

    Lindblad operators: when verification detects a wrong answer, a correction
    factor dampens R for subsequent tokens, simulating quantum decoherence.
    The dampened R forces higher T, causing the model to explore alternatives.

    Returns: (text, R_traj, T_traj, verifications, R_raw_traj)
    """
    R_trajectory = []      # Effective R (after Lindblad correction)
    R_raw_trajectory = []  # Raw R from hidden state (before Lindblad correction)
    T_trajectory = []
    verifications = []
    generated_ids = []

    # Lindblad correction state
    correction_factor = 1.0   # multiplies R (1.0 = no correction)
    correction_decay = 0      # remaining tokens to decay back to 1.0
    CORRECTION_HALFLIFE = 10  # tokens for correction to decay by half

    past_key_values = None

    for step in range(max_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                outputs = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )

        logits = outputs.logits[:, -1, :]  # [1, vocab_size]
        hidden = outputs.hidden_states[-1]
        past_key_values = outputs.past_key_values

        # Compute raw resonance from hidden state
        R_raw = compute_R(hidden)
        R_raw_trajectory.append(R_raw)

        # Scale R for meaningful temperature modulation
        # R_raw is typically 0.001-0.02 (weak per-token projection onto C)
        # Scale to 0.1-2.0 range for the T = 1/(R + epsilon) formula
        R_scaled = R_raw * R_SCALE

        # Apply Lindblad correction factor
        R_effective = R_scaled * correction_factor
        R_trajectory.append(R_effective)

        # Decay correction toward 1.0
        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0:
                correction_factor = 1.0
            else:
                # Smooth decay: 1/e halflife
                correction_factor = correction_factor + (1.0 - correction_factor) / correction_decay

        # Determine temperature based on condition
        if condition == "CYBERNETIC":
            T = T_BASE / (1.0 + R_effective)
            T = max(T_MIN, T)  # floor only, T_BASE is the ceiling
        else:
            T = 0.7  # fixed for CONTROL and VERIFY
        T_trajectory.append(T)

        # Sample next token
        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        token_id = next_token.item()
        generated_ids.append(token_id)

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break

        # Lindblad verification every N tokens
        if (step + 1) % VERIFY_EVERY == 0 and condition != "CONTROL":
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            verified, vscore = verify_answer(partial_text, prompt_entry)

            verifications.append({
                "step": step + 1,
                "tokens_generated": len(generated_ids),
                "partial_text": partial_text[:200],
                "verified": verified,
                "score": vscore,
                "R_raw_before": R_raw,
                "R_effective_before": R_effective,
                "correction_before": correction_factor,
            })

            # Lindblad decoherence: if verification clearly fails
            if verified is False and vscore is not None and vscore < 0.5:
                correction_factor = 0.5
                correction_decay = CORRECTION_HALFLIFE

        # Prepare for next iteration
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text, R_trajectory, T_trajectory, verifications, R_raw_trajectory


def run_condition(condition_name, prompts):
    """Run all prompts under one condition."""
    print(f"\n{'=' * 60}")
    print(f"CONDITION: {condition_name}")
    print(f"{'=' * 60}")

    results = []
    for i, entry in enumerate(prompts):
        pid = entry["id"]
        cat = entry["category"]
        prompt = entry["prompt"]

        print(f"\n[{i+1}/{len(prompts)}] {pid} [{cat}] '{prompt[:60]}...'", flush=True)

        input_ids = tokenize_prompt(prompt)

        t0 = time.time()
        text, R_traj, T_traj, verifs, R_raw_traj = _generate_core(
            input_ids, MAX_NEW_TOKENS, condition_name, entry
        )
        elapsed = time.time() - t0

        # End-of-generation ground truth verification
        vt = entry.get("verification_type", "none")
        if vt != "none" and entry.get("ground_truth"):
            final_verified, final_score = verify_answer(text, entry)
        else:
            final_verified, final_score = None, None

        # Metrics computed from effective R (scaled)
        R_mean = float(np.mean(R_traj)) if R_traj else 0.0
        R_final = R_traj[-1] if R_traj else 0.0
        R_delta = R_traj[-1] - R_traj[0] if len(R_traj) > 1 else 0.0
        T_mean = float(np.mean(T_traj)) if T_traj else 0.0
        T_range = (float(min(T_traj)), float(max(T_traj))) if T_traj else (0.0, 0.0)

        # Raw R metrics (unscaled, for comparison to CONTROL)
        R_raw_mean = float(np.mean(R_raw_traj)) if R_raw_traj else 0.0

        # For dR/dt: use effective R trajectory
        if len(R_traj) > 5:
            steps = np.arange(len(R_traj))
            slope = np.polyfit(steps, R_traj, 1)[0]
        else:
            slope = 0.0

        result = {
            "prompt_id": pid,
            "category": cat,
            "condition": condition_name,
            "prompt": prompt,
            "generated_text": text,
            "tokens_generated": len(R_traj),
            "R_mean": R_mean,
            "R_final": R_final,
            "R_initial": R_traj[0] if R_traj else 0.0,
            "R_delta": R_delta,
            "dR_dt": float(slope),
            "R_raw_mean": R_raw_mean,
            "T_mean": T_mean,
            "T_min": T_range[0],
            "T_max": T_range[1],
            "R_trajectory": R_traj,
            "R_raw_trajectory": R_raw_traj,
            "T_trajectory": T_traj,
            "verifications": verifs,
            "final_verified": final_verified,
            "final_verification_score": final_score,
            "elapsed_seconds": elapsed,
        }
        results.append(result)

        print(f"  tokens={len(R_traj)}  R_mean={R_mean:.4f}  R_final={R_final:.4f}"
              f"  dR/dt={slope:+.4e}  T_mean={T_mean:.4f}  "
              f"verified={final_verified}  dt={elapsed:.1f}s", flush=True)

    # Save per-condition results
    out_path = RESULTS / f"phase4a_{condition_name}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved {out_path}", flush=True)

    return results

# ---- Step 4: Run all conditions ----
print(f"\n[3/5] Running experiment: {len(TEST_PROMPTS)} prompts x 3 conditions", flush=True)

all_results = {}

for cond_name in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    if cond_name in SKIP_CONDITIONS:
        # Load existing data
        existing_path = RESULTS / f"phase4a_{cond_name}.json"
        if existing_path.exists():
            all_results[cond_name] = json.loads(existing_path.read_text())
            print(f"\n  [{cond_name}] Loaded existing results ({len(all_results[cond_name])} prompts)", flush=True)
        else:
            print(f"\n  [{cond_name}] SKIPPED (no existing data)", flush=True)
    else:
        results = run_condition(cond_name, TEST_PROMPTS)
        all_results[cond_name] = results

# ---- Step 5: Compare and summarize ----
print(f"\n\n{'=' * 60}")
print("RESULTS SUMMARY")
print(f"{'=' * 60}")

# Truth accuracy: fraction of verifiable claims correct
def compute_accuracy(results):
    verified = [r for r in results if r["final_verified"] is not None]
    if not verified:
        return 0.0, 0
    correct = sum(1 for r in verified if r["final_verified"])
    return correct / len(verified), len(verified)

for cond in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    res = all_results[cond]
    acc, n = compute_accuracy(res)
    print(f"  {cond:>12s}: accuracy={acc:.3f} ({n} verifiable prompts)", flush=True)

# R metrics by condition
print(f"\n{'Condition':>12s}  {'R_mean':>8s}  {'R_final':>8s}  {'dR/dt':>10s}  {'T_mean':>8s}")
print(f"{'-'*55}")
for cond in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    res = all_results[cond]
    r_mean = np.mean([r["R_mean"] for r in res])
    r_final = np.mean([r["R_final"] for r in res])
    drdt = np.mean([r["dR_dt"] for r in res])
    t_mean = np.mean([r["T_mean"] for r in res])
    print(f"  {cond:>12s}: {r_mean:8.4f}  {r_final:8.4f}  {drdt:+10.2e}  {t_mean:8.4f}")

# By category
for cat in ["factual", "reasoning", "ambiguous", "adversarial"]:
    cat_items = [r["id"] for r in TEST_PROMPTS if r["category"] == cat]
    print(f"\n  --- {cat.upper()} ({len(cat_items)} prompts) ---")
    for cond in ["CONTROL", "CYBERNETIC", "VERIFY"]:
        res = [r for r in all_results[cond] if r["category"] == cat]
        r_mean = np.mean([r["R_mean"] for r in res])
        r_final = np.mean([r["R_final"] for r in res])
        drdt = np.mean([r["dR_dt"] for r in res])
        verified = [r for r in res if r["final_verified"] is not None]
        acc = np.mean([1.0 if r["final_verified"] else 0.0 for r in verified]) if verified else 0.0
        print(f"    {cond:>12s}: R={r_mean:.4f}  R_f={r_final:.4f}  dR/dt={drdt:+.2e}  acc={acc:.3f}")

# Statistical tests
print(f"\n  Statistical Tests:")
from scipy import stats
for metric in ["R_mean", "R_final", "dR_dt"]:
    c_vals = [r[metric] for r in all_results["CONTROL"]]
    x_vals = [r[metric] for r in all_results["CYBERNETIC"]]
    t_stat, p_val = stats.ttest_ind(c_vals, x_vals)
    print(f"    {metric:>12s}: CONTROL vs CYBERNETIC: t={t_stat:+.4f} p={p_val:.4f}")

# dR/dt test: is it > 0 for cybernetic?
drdt_x = np.array([r["dR_dt"] for r in all_results["CYBERNETIC"]])
t_stat, p_val = stats.ttest_1samp(drdt_x, 0.0)
print(f"    CYBERNETIC dR/dt > 0: t={t_stat:+.4f} p={p_val:.4f}  mean={np.mean(drdt_x):+.2e}")

# Resonance by truth: do true answers have higher R than false answers?
all_r_traj = []
all_truth_labels = []
for cond in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    for r in all_results[cond]:
        if r["final_verified"] is not None:
            all_r_traj.append(r["R_final"])
            all_truth_labels.append(1.0 if r["final_verified"] else 0.0)
if len(set(all_truth_labels)) > 1:
    true_Rs = [all_r_traj[i] for i, l in enumerate(all_truth_labels) if l == 1.0]
    false_Rs = [all_r_traj[i] for i, l in enumerate(all_truth_labels) if l == 0.0]
    t_stat, p_val = stats.ttest_ind(true_Rs, false_Rs)
    print(f"\n    R tracks truth: true_R={np.mean(true_Rs):.4f} false_R={np.mean(false_Rs):.4f}  t={t_stat:+.4f} p={p_val:.4f}")

# Save combined results
combined_path = RESULTS / "phase4a_all_results.json"
combined = {
    "conditions": {k: v for k, v in all_results.items()},
    "config": {
        "R_scale": R_SCALE,
        "T_formula": "T_BASE / (1 + R_eff)",
        "T_base": T_BASE,
        "T_min": T_MIN,
        "verify_every": VERIFY_EVERY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "model": "google/gemma-4-E4B-it",
        "n_prompts": len(TEST_PROMPTS),
    }
}
combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"\n  Combined results saved to {combined_path}", flush=True)

print(f"\nDone. Phase 4a experiment complete.", flush=True)
