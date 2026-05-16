"""Phase 4a: Build alignment frame C from contrastive factual pairs.

No constitution. C is built from external verification data:
hidden-state representations of true vs false factual claims.

C = projector onto the subspace that separates true from false.
"""

import json, torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
C_PATH = RESULTS / "contrastive_C.pt"

# Import prompt data
import sys
sys.path.insert(0, str(ROOT))
from phase4a_prompts import CTRASTIVE_CLAIMS

print("=" * 60)
print("PHASE 4a: BUILDING CONTRASTIVE ALIGNMENT FRAME C")
print("=" * 60)

# ---- Step 1: Load model ----
print("\n[1/5] Loading Gemma 4B E4B-it in 4-bit...", flush=True)
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
hidden_dim = model.config.text_config.hidden_size
print(f"Hidden dimension: {hidden_dim}", flush=True)

# ---- Step 2: Collect hidden states for true and false claims ----
print(f"\n[2/5] Processing {len(CTRASTIVE_CLAIMS)} contrastive claims...", flush=True)

true_hidden = []
false_hidden = []
claim_results = []

for i, entry in enumerate(CTRASTIVE_CLAIMS):
    claim = entry["claim"]
    truth = entry["truth"]
    label = "TRUE" if truth else "FALSE"

    # Format as user message (no system prompt, no generation prompt)
    messages = [{"role": "user", "content": f"Please verify: {claim} Is this statement true or false?"}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = tokenizer(chat_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        h_mean = hidden.mean(dim=1).squeeze().to(torch.float32)  # [hidden_dim]

    if truth:
        true_hidden.append(h_mean.cpu().numpy())
    else:
        false_hidden.append(h_mean.cpu().numpy())

    claim_results.append({"claim": claim, "truth": truth, "hidden_norm": float(h_mean.norm())})
    print(f"  [{i+1:2d}/{len(CTRASTIVE_CLAIMS)}] {label:5s} |h|={h_mean.norm():.4f}  '{claim[:50]}...'", flush=True)

true_hidden = np.array(true_hidden)    # [N_true, hidden_dim]
false_hidden = np.array(false_hidden)  # [N_false, hidden_dim]

print(f"\n  Collected: {len(true_hidden)} true, {len(false_hidden)} false representations", flush=True)

# ---- Step 3: Build the separation direction ----
print(f"\n[3/5] Computing Fisher discriminant direction...", flush=True)

mu_true = np.mean(true_hidden, axis=0)
mu_false = np.mean(false_hidden, axis=0)
diff = mu_true - mu_false

# Pooled covariance (diagonal approximation for stability)
var_true = np.var(true_hidden, axis=0)
var_false = np.var(false_hidden, axis=0)
pooled_var = (var_true * len(true_hidden) + var_false * len(false_hidden)) / (
    len(true_hidden) + len(false_hidden)
)
# Ridge regularization for stability
ridge = np.median(pooled_var) * 0.1
w = diff / (pooled_var + ridge)  # Fisher direction

# Normalize
w_norm = w / (np.linalg.norm(w) + 1e-12)

# Build C = w w^T (rank-1 projector)
C_matrix = np.outer(w_norm, w_norm)  # [hidden_dim, hidden_dim]

print(f"  separation norm: |diff| = {np.linalg.norm(diff):.4f}", flush=True)
print(f"  direction norm: |w| = {np.linalg.norm(w_norm):.4f}", flush=True)
print(f"  C shape: {C_matrix.shape}", flush=True)

# ---- Step 4: Validate ----
print(f"\n[4/5] Validating separation...", flush=True)

true_scores = [np.dot(np.dot(h, C_matrix), h) for h in true_hidden]
false_scores = [np.dot(np.dot(h, C_matrix), h) for h in false_hidden]

from scipy import stats
t_stat, p_val = stats.ttest_ind(true_scores, false_scores)
print(f"  True claims  R: {np.mean(true_scores):.6f} +/- {np.std(true_scores):.6f}", flush=True)
print(f"  False claims R: {np.mean(false_scores):.6f} +/- {np.std(false_scores):.6f}", flush=True)
print(f"  t-test: t={t_stat:.4f}, p={p_val:.6f}", flush=True)
print(f"  Separation: {'PASS' if p_val < 0.05 else 'FAIL'} (p < 0.05 required)", flush=True)

# ---- Step 5: Save ----
print(f"\n[5/5] Saving C to {C_PATH}...", flush=True)

torch.save({
    "C": torch.tensor(C_matrix, dtype=torch.float32),
    "w_vector": torch.tensor(w_norm, dtype=torch.float32),
    "diff_vector": torch.tensor(diff, dtype=torch.float32),
    "true_mean": torch.tensor(mu_true, dtype=torch.float32),
    "false_mean": torch.tensor(mu_false, dtype=torch.float32),
    "n_true": len(true_hidden),
    "n_false": len(false_hidden),
    "separation_tstat": float(t_stat),
    "separation_pval": float(p_val),
    "true_R_mean": float(np.mean(true_scores)),
    "false_R_mean": float(np.mean(false_scores)),
    "hidden_dim": hidden_dim,
    "claims": claim_results,
}, C_PATH)

# Save metadata separately for inspection
metadata = {
    "n_true": len(true_hidden),
    "n_false": len(false_hidden),
    "hidden_dim": hidden_dim,
    "separation_tstat": float(t_stat),
    "separation_pval": float(p_val),
    "true_R_mean": float(np.mean(true_scores)),
    "false_R_mean": float(np.mean(false_scores)),
    "diff_norm": float(np.linalg.norm(diff)),
    "w_norm": float(np.linalg.norm(w_norm)),
}
(RESULTS / "contrastive_C_meta.json").write_text(json.dumps(metadata, indent=2))
print(f"  Metadata saved to {RESULTS / 'contrastive_C_meta.json'}", flush=True)

print(f"\nDone. C built from external verification data only.", flush=True)
