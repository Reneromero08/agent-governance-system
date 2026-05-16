"""Phase 4a v2 smoke test: validate dynamic C building + context injection.

Runs 3 prompts across 2 runs:
  Run 1: uses v1 C (comprehension-time), collects generation states
  Run 2: uses C built from Run 1 generation states
  Both runs: context injection on verification failure

Validates:
  1. Context injection actually fires and model sees correction
  2. Dynamic C built from generation states differs from v1 C
  3. R changes between runs (indicates C is learning)
  4. Accuracy changes between runs
"""

import json, sys, torch, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

sys.path.insert(0, str(ROOT))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---- Config ----
R_SCALE = 500.0
T_BASE = 3.0
T_MIN = 0.2
VERIFY_EVERY = 20
MAX_TOKENS = 80  # shorter for smoke test
SEED = 20260516

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Load model ----
print("=" * 60)
print("PHASE 4a v2 SMOKE TEST: Dynamic C + Context Feedback")
print("=" * 60)
print("\nLoading model...", flush=True)

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=quant_config, device_map="auto",
    dtype=torch.float16, output_hidden_states=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

hidden_dim = model.config.text_config.hidden_size
print(f"Hidden dim: {hidden_dim}", flush=True)

# ---- Load v1 C (comprehension-time) ----
v1_c_path = ROOT.parent / "phase4a" / "results" / "contrastive_C.pt"
c_data = torch.load(str(v1_c_path), map_location="cpu")
C = c_data["C"].to("cuda", dtype=torch.float32)
w = c_data["w_vector"].to("cuda", dtype=torch.float32)
print(f"Loaded v1 C (comprehension-time). Sep p={c_data['separation_pval']:.6f}", flush=True)

def compute_R(hidden_state):
    h = hidden_state[0, -1, :].float()
    hn = h / (h.norm() + 1e-12)
    return float((hn @ w) ** 2)  # equivalent to Tr(rho @ C) but faster

def build_C_from_states(states_list, labels_list):
    """Build C from labeled generation-time hidden states using Fisher discriminant.
    states_list: list of mean hidden states (each [hidden_dim] numpy or tensor)
    labels_list: list of booleans (True=verified, False=failed)
    """
    true_states = [s for s, l in zip(states_list, labels_list) if l]
    false_states = [s for s, l in zip(states_list, labels_list) if not l]

    if len(true_states) < 2 or len(false_states) < 2:
        return None, None, None

    true_arr = np.array(true_states)
    false_arr = np.array(false_states)

    mu_true = np.mean(true_arr, axis=0)
    mu_false = np.mean(false_arr, axis=0)
    diff = mu_true - mu_false

    var_true = np.var(true_arr, axis=0)
    var_false = np.var(false_arr, axis=0)
    pooled_var = (var_true * len(true_arr) + var_false * len(false_arr)) / (len(true_arr) + len(false_arr))
    ridge = np.median(pooled_var) * 0.1

    w_new = diff / (pooled_var + ridge)
    w_new = w_new / (np.linalg.norm(w_new) + 1e-12)
    C_new = np.outer(w_new, w_new)

    # Validate separation
    true_scores = [np.dot(s, w_new) ** 2 for s in true_arr]
    false_scores = [np.dot(s, w_new) ** 2 for s in false_arr]
    sep_ratio = np.mean(true_scores) / max(np.mean(false_scores), 1e-12)

    return torch.tensor(C_new, dtype=torch.float32), torch.tensor(w_new, dtype=torch.float32), sep_ratio

def generate_with_feedback(prompt_entry, C_local, w_local, collect_states=True):
    """Generate with T modulation + context injection on verification failure.
    Returns: (text, R_raw_traj, verifications, collected_states, corrections)
    """
    prompt = prompt_entry["prompt"]
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

    # Save the original prompt tokens for context reconstruction
    prompt_ids = input_ids.clone()
    prompt_len = prompt_ids.shape[1]

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

        # Compute R
        R_raw = compute_R(hidden)
        R_raw_traj.append(R_raw)

        # Collect generation-time hidden state (mean over current token position)
        if collect_states:
            h_vec = hidden[0, -1, :].float().cpu().numpy()
            collected_states.append(h_vec)

        # Temperature
        R_eff = R_raw * R_SCALE * correction_factor
        T = max(T_MIN, T_BASE / (1.0 + R_eff))
        probs = torch.softmax(logits.float() / T, dim=-1)
        next_token = torch.multinomial(probs[0], 1)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        # Decay Lindblad correction
        if correction_decay > 0:
            correction_decay -= 1
            if correction_decay == 0:
                correction_factor = 1.0

        # Verification + context injection every N tokens
        if (step + 1) % VERIFY_EVERY == 0:
            partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            verified, vscore = verify_answer(partial_text, prompt_entry)

            verifications.append({
                "step": step + 1,
                "verified": verified,
                "score": vscore,
                "partial": partial_text[:150],
            })

            if verified is False and vscore is not None and vscore < 0.5:
                # CONTEXT INJECTION: rebuild full conversation with correction
                # 1. Decode existing generation
                existing_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # 2. Build correction message
                correction_msg = "[VERIFICATION FAILED: The previous output contained factual errors. Correct the information and continue with accurate facts.]"

                # 3. Reconstruct chat: original prompt + model output + correction as user
                reconstructed = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": existing_text},
                    {"role": "user", "content": correction_msg},
                ]
                new_chat = tokenizer.apply_chat_template(reconstructed, tokenize=False, add_generation_prompt=True)
                new_input_ids = tokenizer(new_chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda").input_ids

                # 4. FULL RECOMPUTE — discard old KV cache, start fresh
                # Process the entire reconstructed chat through the model
                with torch.no_grad():
                    new_outputs = model(input_ids=new_input_ids, output_hidden_states=True, use_cache=True)
                    past_key_values = new_outputs.past_key_values

                # 5. Update state
                input_ids = new_input_ids
                context_rebuilt = True

                # Record correction
                corrections.append({
                    "step": step + 1,
                    "tokens_before": step + 1,
                    "context_rebuilt": True,
                })

                # Lindblad decoherence
                correction_factor = 0.5
                correction_decay = CORRECTION_HALFLIFE

        # Prepare for next iteration
        if not context_rebuilt:
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text, R_raw_traj, verifications, collected_states, corrections


# ---- SMOKE TEST ----
test_prompts = TEST_PROMPTS[:3]  # F1, F2, F3

all_gen_states = []  # (state, label) pairs across all runs

for run_idx in range(2):
    print(f"\n{'=' * 60}")
    print(f"RUN {run_idx + 1} (C from {'comprehension' if run_idx == 0 else 'generation states'})")
    print(f"{'=' * 60}")

    run_states = []
    run_labels = []

    for i, entry in enumerate(test_prompts):
        pid = entry["id"]
        print(f"\n  [{i+1}/3] {pid}: '{entry['prompt'][:50]}...'", flush=True)

        text, R_traj, verifs, states, corrs = generate_with_feedback(entry, C, w, collect_states=True)

        # Final verification
        vt = entry.get("verification_type", "none")
        if vt != "none" and entry.get("ground_truth"):
            final_verified, final_score = verify_answer(text, entry)
        else:
            final_verified, final_score = None, None

        # Per-prompt mean hidden state (for C building)
        if states:
            mean_state = np.mean(states, axis=0)
            if final_verified is True:
                run_states.append(mean_state)
                run_labels.append(True)
            elif final_verified is False:
                run_states.append(mean_state)
                run_labels.append(False)

        R_mean = float(np.mean(R_traj)) if R_traj else 0.0
        print(f"    R_raw_mean={R_mean:.6f}  R_eff_typical={R_mean*R_SCALE:.1f}  verified={final_verified}", flush=True)
        print(f"    verifs: {len(verifs)}  corrections: {len(corrs)}", flush=True)
        if corrs:
            print(f"    CORRECTION FIRED at step(s): {[c['step'] for c in corrs]}", flush=True)
        text_safe = text.encode('ascii', errors='replace').decode('ascii')
        print(f"    text: {text_safe[:120]}...", flush=True)

    # After run: build new C from generation states
    n_true = sum(run_labels)
    n_false = len(run_labels) - n_true
    print(f"\n  Run {run_idx+1} states collected: {n_true} true-gen, {n_false} false-gen", flush=True)

    if n_true >= 2 and n_false >= 2:
        C_new, w_new, sep = build_C_from_states(run_states, run_labels)
        if C_new is not None:
            # Compare to previous C
            old_w = w.cpu().numpy()
            new_w = w_new.cpu().numpy()
            cosine_sim = float(np.dot(old_w, new_w) / (np.linalg.norm(old_w) * np.linalg.norm(new_w) + 1e-12))
            print(f"  New C built: sep_ratio={sep:.2f}  cos_sim_with_prev={cosine_sim:.4f}", flush=True)

            # Save C for this run
            torch.save({"C": C_new, "w_vector": w_new, "sep_ratio": sep, "n_true": n_true, "n_false": n_false},
                       RESULTS / f"dynamic_C_run{run_idx+1}.pt")

            # Update for next run
            C = C_new.to("cuda", dtype=torch.float32)
            w = w_new.to("cuda", dtype=torch.float32)
        else:
            print(f"  Failed to build new C (need >= 2 per class)", flush=True)
    else:
        print(f"  Not enough labeled states to build new C", flush=True)

    all_gen_states.append((run_states, run_labels))

print(f"\n{'=' * 60}")
print("SMOKE TEST SUMMARY")
print(f"{'=' * 60}")
for run_idx in range(min(2, len(all_gen_states))):
    states, labels = all_gen_states[run_idx]
    n_t = sum(labels)
    n_f = len(labels) - n_t
    print(f"  Run {run_idx+1}: {n_t}T + {n_f}F generation states")

cs = sorted(RESULTS.glob("dynamic_C_*.pt"))
print(f"\n  Dynamic C files: {len(cs)}")
for c in cs:
    d = torch.load(str(c), map_location="cpu")
    print(f"    {c.name}: sep={d.get('sep_ratio','?'):.2f}  {d.get('n_true',0)}T + {d.get('n_false',0)}F")

print(f"\nDone. Verify: context injection fired, R changed between runs, C evolved.")
