"""Phase 4a v2: Cybernetic Truth — Dynamic C + Context Feedback

Condition 1 (FULL): Dynamic C updates across runs + context injection on errors.
  - Run 1: uses v1 C (comprehension-time), collects generation states
  - After each run: builds new C from verified-true vs verified-false generation states
  - Every 20 tokens: verify. If fail, rebuild full context with correction message.
  - T = T_BASE / (1 + R_eff), R_eff = R_raw * R_SCALE * correction_factor

Same model, same verification pipeline as v1. No constitution.
"""

import json, sys, torch, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---- Config ----
R_SCALE = 500.0
T_BASE = 3.0
T_MIN = 0.2
VERIFY_EVERY = 20
MAX_TOKENS = 150
N_RUNS = 5
SEED = 20260516

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Load model ----
print("=" * 60)
print("PHASE 4a v2: DYNAMIC C + CONTEXT FEEDBACK")
print("=" * 60)
print("\n[1/4] Loading model...", flush=True)

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

hidden_dim = model.config.text_config.hidden_size
print(f"Model loaded. Hidden dim: {hidden_dim}", flush=True)

# ---- Load v1 C ----
print("\n[2/4] Loading v1 C (comprehension-time)...", flush=True)
v1_c_path = ROOT.parent / "phase4a" / "results" / "contrastive_C.pt"
c_data = torch.load(str(v1_c_path), map_location="cpu")
C_cuda = c_data["C"].to("cuda", dtype=torch.float32)
w_cuda = c_data["w_vector"].to("cuda", dtype=torch.float32)
print(f"  C shape: {C_cuda.shape}  Sep p={c_data['separation_pval']:.6f}", flush=True)

# ---- Helpers ----
def compute_R(hidden_state):
    """R = (h_norm . w)^2 — equivalent to Tr(rho @ C)"""
    h = hidden_state[0, -1, :].float()
    hn = h / (h.norm() + 1e-12)
    return float((hn @ w_cuda) ** 2)

def build_C_from_generation_states(states_list, labels_list):
    """Fisher discriminant on generation-time mean hidden states.
    Returns (C_tensor, w_tensor, sep_ratio, metadata_dict) or (None,None,None,{})"""
    true_states = np.array([s for s, l in zip(states_list, labels_list) if l])
    false_states = np.array([s for s, l in zip(states_list, labels_list) if not l])

    n_t, n_f = len(true_states), len(false_states)
    if n_t < 2 or n_f < 2:
        return None, None, None, {"n_true": n_t, "n_false": n_f, "reason": "insufficient samples"}

    mu_true = np.mean(true_states, axis=0)
    mu_false = np.mean(false_states, axis=0)
    diff = mu_true - mu_false

    var_true = np.var(true_states, axis=0)
    var_false = np.var(false_states, axis=0)
    pooled_var = (var_true * n_t + var_false * n_f) / (n_t + n_f)
    ridge = np.median(pooled_var) * 0.1

    w_new = diff / (pooled_var + ridge)
    w_new_norm = np.linalg.norm(w_new) + 1e-12
    w_new = w_new / w_new_norm
    C_new = np.outer(w_new, w_new)

    # Validate
    true_scores = [float(np.dot(s, w_new) ** 2) for s in true_states]
    false_scores = [float(np.dot(s, w_new) ** 2) for s in false_states]
    sep_ratio = np.mean(true_scores) / max(np.mean(false_scores), 1e-12)

    # Cosine similarity to previous C
    old_w = w_cuda.cpu().numpy()
    cos_sim = float(np.dot(old_w, w_new) / (np.linalg.norm(old_w) * w_new_norm))

    return (torch.tensor(C_new, dtype=torch.float32),
            torch.tensor(w_new, dtype=torch.float32),
            sep_ratio,
            {"n_true": n_t, "n_false": n_f, "sep_ratio": sep_ratio,
             "cos_sim_prev": cos_sim, "diff_norm": float(np.linalg.norm(diff))})

def generate_with_feedback(prompt_entry, collect_states=True):
    """Token-by-token generation with T modulation + context injection.
    Returns: (text, R_raw_traj, verifications, collected_states, corrections_list)
    """
    prompt = prompt_entry["prompt"]
    prompt_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    R_raw_traj = []
    verifications = []
    collected_states = []
    corrections = []
    generated_ids = []

    past_key_values = None
    correction_factor = 1.0
    correction_decay = 0
    CORRECTION_HALFLIFE = 10

    for step in range(MAX_TOKENS):
        verified = None
        vscore = None
        context_rebuilt = False

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
        R_raw_traj.append(R_raw)

        if collect_states:
            collected_states.append(hidden[0, -1, :].float().cpu().numpy())

        R_eff = R_raw * R_SCALE * correction_factor
        T = max(T_MIN, T_BASE / (1.0 + R_eff))
        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0:
                correction_factor = 1.0

        if (step + 1) % VERIFY_EVERY == 0:
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            verified, vscore = verify_answer(partial_text, prompt_entry)

            verifications.append({
                "step": step + 1, "verified": verified, "score": vscore,
                "partial": partial_text[:150],
                "R_raw_at_check": R_raw,
            })

            if verified is False and vscore is not None and vscore < 0.5:
                # CONTEXT INJECTION: full context reconstruction with correction
                existing_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                correction_msg = (
                    "[VERIFICATION FAILED: The previous output contained factual errors. "
                    "Correct the information and continue with accurate facts.]"
                )
                reconstructed = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": existing_text},
                    {"role": "user", "content": correction_msg},
                ]
                new_chat = tokenizer.apply_chat_template(reconstructed, tokenize=False, add_generation_prompt=True)
                new_input_ids = tokenizer(new_chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

                with torch.no_grad():
                    new_outputs = model(input_ids=new_input_ids, output_hidden_states=True, use_cache=True)
                    past_key_values = new_outputs.past_key_values

                input_ids = new_input_ids
                context_rebuilt = True

                corrections.append({
                    "step": step + 1, "tokens_generated": len(generated_ids),
                    "R_raw_before": R_raw, "context_rebuilt": True,
                })

                correction_factor = 0.5
                correction_decay = CORRECTION_HALFLIFE

        if not context_rebuilt:
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text, R_raw_traj, verifications, collected_states, corrections


# ---- Run experiment ----
print(f"\n[3/4] Running {N_RUNS} runs x {len(TEST_PROMPTS)} prompts...", flush=True)

all_run_results = []
all_gen_states = []
C_history = [{"run": 0, "source": "comprehension", "sep_pval": c_data["separation_pval"]}]

for run_idx in range(N_RUNS):
    print(f"\n{'=' * 60}")
    print(f"RUN {run_idx + 1}/{N_RUNS}", flush=True)
    print(f"{'=' * 60}")

    run_states = []
    run_labels = []
    run_results = []

    for i, entry in enumerate(TEST_PROMPTS):
        pid = entry["id"]
        cat = entry["category"]

        t0 = time.time()
        text, R_traj, verifs, states, corrs = generate_with_feedback(entry)
        elapsed = time.time() - t0

        # Final verification
        vt = entry.get("verification_type", "none")
        if vt != "none" and entry.get("ground_truth"):
            final_verified, final_score = verify_answer(text, entry)
        else:
            final_verified, final_score = None, None

        # Per-prompt mean generation state for C building
        if states:
            mean_state = np.mean(states, axis=0)
            run_states.append(mean_state)
            run_labels.append(final_verified)

        # Metrics
        R_mean = float(np.mean(R_traj)) if R_traj else 0.0
        n_corr = len(corrs)

        result = {
            "run": run_idx + 1, "prompt_id": pid, "category": cat,
            "prompt": entry["prompt"], "generated_text": text,
            "R_raw_mean": R_mean, "R_raw_final": R_traj[-1] if R_traj else 0.0,
            "R_raw_trajectory": R_traj,
            "verifications": verifs, "corrections": n_corr,
            "final_verified": final_verified, "final_verification_score": final_score,
            "elapsed_seconds": elapsed,
        }
        run_results.append(result)

        print(f"  [{i+1:2d}/{len(TEST_PROMPTS)}] {pid} [{cat}] "
              f"R={R_mean:.6f}  T_corr={n_corr}  verified={final_verified}  dt={elapsed:.1f}s", flush=True)

    # Save per-run results
    run_path = RESULTS / f"phase4a_v2_run{run_idx+1}.json"
    run_path.write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    all_run_results.append(run_results)

    # Build new C from generation states
    n_t = sum(1 for l in run_labels if l is True)
    n_f = sum(1 for l in run_labels if l is False)
    n_u = len(run_labels) - n_t - n_f
    print(f"\n  Run {run_idx+1} states: {n_t}T + {n_f}F + {n_u}U", flush=True)

    if n_t >= 2 and n_f >= 2:
        valid_states = [s for s, l in zip(run_states, run_labels) if l is not None]
        valid_labels = [l for s, l in zip(run_states, run_labels) if l is not None]
        C_new, w_new, sep, meta = build_C_from_generation_states(valid_states, valid_labels)

        if C_new is not None:
            # Save C
            C_save = {"C": C_new, "w_vector": w_new, **meta, "run": run_idx + 1}
            torch.save(C_save, RESULTS / f"dynamic_C_run{run_idx+1}.pt")

            # Update for next run
            C_cuda = C_new.to("cuda", dtype=torch.float32)
            w_cuda = w_new.to("cuda", dtype=torch.float32)

            C_history.append({"run": run_idx + 1, "source": "generation",
                            "sep_ratio": sep, "n_true": n_t, "n_false": n_f,
                            "cos_sim_prev": meta.get("cos_sim_prev", 0)})

            print(f"  NEW C: sep={sep:.2f}  cos_sim_prev={meta.get('cos_sim_prev',0):.4f}  "
                  f"({n_t}T + {n_f}F)", flush=True)
        else:
            print(f"  C not built: {meta.get('reason','?')}", flush=True)
    else:
        print(f"  Not enough labeled states (need >=2 per class)", flush=True)

    all_gen_states.append((run_states, run_labels))

# ---- Save combined results ----
print(f"\n[4/4] Saving combined results...", flush=True)

combined = {
    "config": {
        "R_scale": R_SCALE, "T_formula": f"{T_BASE}/(1+R_eff)", "T_base": T_BASE, "T_min": T_MIN,
        "verify_every": VERIFY_EVERY, "max_tokens": MAX_TOKENS, "n_runs": N_RUNS,
        "model": "google/gemma-4-E4B-it", "n_prompts": len(TEST_PROMPTS),
        "conditions": ["FULL (dynamic C + context feedback)"],
    },
    "runs": all_run_results,
    "C_history": C_history,
}
combined_path = RESULTS / "phase4a_v2_all_results.json"
combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"  Saved {combined_path}", flush=True)

# ---- Summary ----
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")

for run_idx in range(N_RUNS):
    res = all_run_results[run_idx]
    verified = [r for r in res if r["final_verified"] is not None]
    correct = sum(1 for r in verified if r["final_verified"])
    acc = correct / len(verified) if verified else 0.0
    R_mean = float(np.mean([r["R_raw_mean"] for r in res]))
    n_corr = sum(r["corrections"] for r in res)
    print(f"  Run {run_idx+1}: acc={acc:.3f} ({correct}/{len(verified)})  R_mean={R_mean:.6f}  corrections={n_corr}", flush=True)

if C_history:
    print(f"\n  C convergence:")
    for ch in C_history:
        if ch["run"] == 0:
            print(f"    Run 0 (comprehension): sep_pval={ch['sep_pval']:.6f}")
        else:
            print(f"    Run {ch['run']} (generation): sep={ch.get('sep_ratio','?'):.2f}  "
                  f"cos_sim_prev={ch.get('cos_sim_prev','?'):.4f}  ({ch.get('n_true','?')}T+{ch.get('n_false','?')}F)")

print(f"\nDone. Phase 4a v2 experiment complete.", flush=True)
