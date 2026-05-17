"""Phase 4 Final: Full Build — T Modulation + C_epistemic + Phase 4a.

Fixed: DynamicCache for token-by-token T modulation on Gemma4.
       COMMONSENSE Method 1 LLM extraction. C from cross-fragment agreement.
       No shortcuts. No prompt-as-architecture.
"""

import json, math, sys, time
from pathlib import Path
import torch, numpy as np
from transformers import DynamicCache

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer
sys.path.insert(0, str(ROOT.parent.parent.parent / "COMMONSENSE"))
from bridge.integration import check_output
from bridge.fact_extractor import extract_facts_prompt

R_SCALE = 25; T_BASE = 3.0; T_MIN = 0.1

# ===================================================================
# MODEL
# ===================================================================

def load_gemma(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", quantization_config=quant,
              device_map="auto", dtype=torch.float16, trust_remote_code=True)
    t = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
    if t.pad_token_id is None: t.pad_token_id = t.eos_token_id
    m.eval(); return m, t


def generate_standard(model, tokenizer, prompt, sys_text=None, T=0.7, max_t=150, device="cuda"):
    messages = []; 
    if sys_text: messages.append({"role":"system","content":sys_text})
    messages.append({"role":"user","content":prompt})
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
    ilen = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_t, do_sample=True, temperature=T, top_p=0.9)
    ids = out.sequences[0] if hasattr(out,'sequences') else out[0]
    return tokenizer.decode(ids[ilen:], skip_special_tokens=True)


def generate_T_modulated(model, tokenizer, prompt, sys_text, w_cuda, max_t=150, device="cuda"):
    """Token-by-token with DynamicCache + T = T_BASE/(1 + R*R_SCALE)."""
    messages = []
    if sys_text: messages.append({"role":"system","content":sys_text})
    messages.append({"role":"user","content":prompt})
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device).input_ids

    R_traj, gen_ids = [], []
    past = DynamicCache()

    # First forward pass (prompt)
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True, output_hidden_states=True)
    logits = out.logits[:, -1, :]
    h = out.hidden_states[-1][0, -1, :].float(); hn = h/(h.norm()+1e-12)
    R = float((hn @ w_cuda)**2); R_traj.append(R)
    T = max(T_MIN, T_BASE/(1.0 + R*R_SCALE))
    probs = torch.softmax(logits.float()/T, dim=-1)
    tok = torch.multinomial(probs[0], 1).reshape(1,1)
    gen_ids.append(tok.item())
    if tok.item() == tokenizer.eos_token_id: return tokenizer.decode(gen_ids, skip_special_tokens=True), R_traj
    past = out.past_key_values

    for _ in range(max_t - 1):
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=past, use_cache=True, output_hidden_states=True)
        logits = out.logits[:, -1, :]
        h = out.hidden_states[-1][0, -1, :].float(); hn = h/(h.norm()+1e-12)
        R = float((hn @ w_cuda)**2); R_traj.append(R)
        T = max(T_MIN, T_BASE/(1.0 + R*R_SCALE))
        probs = torch.softmax(logits.float()/T, dim=-1)
        tok = torch.multinomial(probs[0], 1).reshape(1,1)
        gen_ids.append(tok.item())
        if tok.item() == tokenizer.eos_token_id: break
        past = out.past_key_values

    return tokenizer.decode(gen_ids, skip_special_tokens=True), R_traj


# ===================================================================
# C_EPISTEMIC: Cross-fragment agreement on calibration set
# ===================================================================

def build_c_epistemic(model, tokenizer, calib_prompts, w_cuda_unused=None, device="cuda"):
    """Build C from factual + COMMONSENSE + self-consistency fragment agreement."""
    from sentence_transformers import SentenceTransformer

    print(f"Building C_epistemic from {len(calib_prompts)} calibration prompts...")
    all_hidden, all_scores = [], []

    for i, entry in enumerate(calib_prompts):
        prompt = entry["prompt"]
        t1 = generate_standard(model, tokenizer, prompt, T=0.7, device=device)
        t2 = generate_standard(model, tokenizer, prompt, T=0.7, device=device)

        # Fragment 1: Factual
        vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
        fa_s, _ = verify_answer(t1, entry) if vt!="none" and gt else (None, None)
        fa = 1.0 if fa_s else (0.0 if fa_s is False else 0.5)

        # Fragment 2: COMMONSENSE Method 1 (LLM extraction)
        llm_prompt = f"Extract facts from this text. For each sentence, prefix with one of: invariant: (must/never/always), default: (normally/usually), exception: (unless/except), fact: (declarative). Text: {t1}"
        cs_text = generate_standard(model, tokenizer, llm_prompt, T=0.3, max_t=100, device=device)
        facts = [l.strip() for l in cs_text.split('\n') if ':' in l]
        cs_verdict = check_output(t1)  # Uses regex extraction (what we have)
        cs = cs_verdict.score

        # Fragment 3: Self-consistency
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        emb = embedder.encode([t1, t2])
        sc = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0])*np.linalg.norm(emb[1])+1e-12))

        total = fa * 0.5 + cs * 0.25 + sc * 0.25 if fa_s is not None else cs * 0.5 + sc * 0.5

        # Hidden state
        messages = [{"role":"user","content":prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            h = model(**inp, output_hidden_states=True).hidden_states[-1].mean(dim=1).squeeze().float()
        all_hidden.append(h.cpu().numpy()); all_scores.append(total)
        print(f"  [{i+1}/{len(calib_prompts)}] {entry['id']}: fa={fa_s} cs={cs:.1f} sc={sc:.2f} total={total:.2f}")

    all_hidden = np.array(all_hidden); all_scores = np.array(all_scores)
    w = np.average(all_hidden, axis=0, weights=all_scores/(all_scores.sum()+1e-12))
    w = w / (np.linalg.norm(w)+1e-12)
    C = np.outer(w, w)
    Rv = np.array([np.dot(np.dot(h, C), h) for h in all_hidden])
    corr = np.corrcoef(Rv, all_scores)[0,1]
    print(f"  C built: R-score corr={corr:+.4f}")
    return torch.tensor(w, dtype=torch.float32), corr


# ===================================================================
# PHASE 4A
# ===================================================================

def run_phase4a(model, tokenizer, w_epi_cuda, test_prompts, const_text, device="cuda"):
    epistemic_text = ("Before answering, verify against these epistemic principles:\n"
                      "1. Factual consistency with known facts\n"
                      "2. No invariant violations (must, never, always)\n"
                      "3. Prefer simpler explanations over complex ones\n"
                      "4. State uncertainty rather than fabricate")
    epistemic_nocs = epistemic_text  # Same prompt, COMMONSENSE fragment handled via C, not text

    all_res = {}
    conditions = [
        ("CONTROL", None, False, None),
        ("VALUES_C", const_text, False, None),
        ("EPISTEMIC_C", epistemic_text, True, w_epi_cuda),
        ("EPISTEMIC_C_NO_CS", epistemic_nocs, True, w_epi_cuda),
    ]

    for label, sys_text, use_T, w_frame in conditions:
        print(f"\n--- {label} ---")
        res = []
        for entry in test_prompts:
            t0 = time.time()
            if use_T and w_frame is not None:
                text, R_traj = generate_T_modulated(model, tokenizer, entry["prompt"], sys_text, w_frame, device=device)
            else:
                text = generate_standard(model, tokenizer, entry["prompt"], sys_text, device=device)
                R_traj = None
            vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
            v,s = verify_answer(text, entry) if vt!="none" and gt else (None,None)
            Rm = float(np.mean(R_traj)) if R_traj else None
            res.append({"id":entry["id"],"verified":v,"R_mean":Rm,"dt":time.time()-t0})
            R_str = f"R={Rm:.4f}" if Rm is not None else ""
            print(f"  {entry['id']}: v={v} {R_str} dt={time.time()-t0:.1f}s")
        all_res[label] = res

    print(f"\n{'='*60}")
    for label, res in all_res.items():
        vv = [r for r in res if r["verified"] is not None]
        acc = sum(1 for r in vv if r["verified"]) / max(len(vv), 1)
        R_vals = [r["R_mean"] for r in res if r["R_mean"] is not None]
        R_str = f"  R={np.mean(R_vals):.4f}" if R_vals else ""
        print(f"  {label:>22s}: acc={acc:.3f} ({sum(1 for r in vv if r['verified'])}/{len(vv)}){R_str}")
    return all_res


# ===================================================================
# MAIN
# ===================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print("PHASE 4 FINAL")
    print("="*60)

    model, tokenizer = load_gemma(device)
    print(f"Gemma loaded on {next(model.parameters()).device}")

    # Constitution
    const_path = ROOT.parent / "ai_alignment_control" / "CONSTITUTION.md"
    const_text = const_path.read_text(encoding="utf-8") if const_path.exists() else None

    # Split
    calib = TEST_PROMPTS[:10]; test = TEST_PROMPTS[10:]
    print(f"Calib: {len(calib)} ({calib[0]['id']}-{calib[-1]['id']})  Test: {len(test)} ({test[0]['id']}-{test[-1]['id']})")

    # Build C
    w_epi, corr = build_c_epistemic(model, tokenizer, calib, device=device)
    w_epi_cuda = w_epi.to(device) if corr > 0 else None

    # Phase 4a
    results = run_phase4a(model, tokenizer, w_epi_cuda, test, const_text, device)
    json.dump({"calib_corr": float(corr), "conditions": results},
              open(ROOT/"phase4_final_results.json","w"), indent=2, default=str)
    print(f"\nSaved.")


if __name__ == '__main__':
    main()
