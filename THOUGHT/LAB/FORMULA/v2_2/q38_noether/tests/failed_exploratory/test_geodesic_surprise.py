"""Proper geodesic test: correlate token surprise with hidden state displacement.

Theory: If the geodesic follows the training distribution, then:
- Low-surprise tokens (expected) = smooth hidden state trajectory (low action)
- High-surprise tokens (unexpected) = rough trajectory (high action)

Method: Process a text sequentially. At each token, compute:
- NLL (negative log-likelihood) = how unexpected the token is
- Hidden state displacement |h_t - h_{t-1}| = how far the state jumped

Correlation should be positive: unexpected tokens cause larger state jumps.
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 65)
print("GEODESIC = SURPRISE CORRELATION")
print("=" * 65)

print("\nLoading gpt2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", output_hidden_states=True, torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Loaded.")

# Test texts: mix of factual, fluent + forced-weird
texts = [
    # Fluent, expected
    "The capital of France is Paris. It is known for the Eiffel Tower and its rich history of art and culture.",
    "Water boils at 100 degrees Celsius at sea level. This is a fundamental property of the substance.",
    "Shakespeare wrote Romeo and Juliet, a tragedy about two young lovers from feuding families.",
    "The Earth orbits the Sun once every 365 days. This orbit is elliptical, not circular.",
    # Factual but varied
    "Gravity is a fundamental force that attracts objects with mass toward each other. Einstein described it as curvature of spacetime.",
    "Photosynthesis converts sunlight into chemical energy in plants, releasing oxygen as a byproduct during the process.",
    # Weird/unexpected
    "The moon is made of green cheese and the president of Mars visited Earth yesterday to discuss trade agreements with dolphins.",
    "Water freezes at 500 degrees and elephants can fly by flapping their ears at supersonic speeds through quantum tunneling.",
    "Shakespeare wrote the US Constitution while riding a bicycle across the Atlantic Ocean during the Bronze Age collapse.",
    # Neutral narrative
    "She walked to the store to buy groceries. The weather was pleasant and the streets were quiet that afternoon.",
    "The train arrived at the station precisely on time. Passengers disembarked and went about their daily business in the city.",
]

results = []
for text in texts:
    tokens = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
    
    # Per-token NLL
    logits = outputs.logits[0]  # (seq_len, vocab_size)
    shift_logits = logits[:-1]
    shift_labels = tokens[0, 1:]
    nlls = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction='none'
    ).cpu().numpy()
    
    # Per-token hidden state (last layer, at each position)
    hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
    displacements = np.sqrt(np.sum(np.diff(hidden.cpu().numpy(), axis=0)**2, axis=1))
    # displacements[i] = |h_{i+1} - h_i| for token at position i+1
    
    # Align: nlls[i] is for token i+1, displacements[i] is also for token i+1
    for i in range(len(nlls)):
        results.append({
            "token": tokenizer.decode([tokens[0, i+1].item()]),
            "nll": float(nlls[i]),
            "displacement": float(displacements[i]),
        })

nlls = np.array([r["nll"] for r in results])
disps = np.array([r["displacement"] for r in results])

# Remove extreme outliers (> 3 sigma)
mask_nll = np.abs(nlls - np.mean(nlls)) < 3 * np.std(nlls)
mask_disp = np.abs(disps - np.mean(disps)) < 3 * np.std(disps)
mask = mask_nll & mask_disp

nlls_f = nlls[mask]
disps_f = disps[mask]

print(f"\n{'='*65}")
print(f"RESULTS ({np.sum(mask)} tokens after outlier removal)")
print(f"{'='*65}")

# Overall correlation
from scipy.stats import pearsonr, spearmanr
r, p = pearsonr(nlls_f, disps_f)
rho, prho = spearmanr(nlls_f, disps_f)
print(f"\n  Pearson r  = {r:.4f} (p = {p:.6f})")
print(f"  Spearman rho = {rho:.4f} (p = {prho:.6f})")

# Binned analysis: are high-NLL tokens higher displacement?
bins = np.percentile(nlls_f, [0, 25, 50, 75, 100])
print(f"\n--- Binned analysis ---")
print(f"  {'NLL range':>20s}  {'Mean displ':>12s}  {'Std displ':>12s}  {'Count':>6s}")
for i in range(len(bins)-1):
    bm = (nlls_f >= bins[i]) & (nlls_f < bins[i+1])
    if i == len(bins)-2: bm = (nlls_f >= bins[i])
    if np.sum(bm) > 0:
        print(f"  [{bins[i]:6.2f}, {bins[i+1]:6.2f}]  {np.mean(disps_f[bm]):12.2f}  {np.std(disps_f[bm]):12.2f}  {np.sum(bm):6d}")

# Top surprising tokens
print(f"\n--- Most surprising tokens ---")
top_idx = np.argsort(nlls_f)[-10:]
for idx in reversed(top_idx):
    orig_idx = np.where(mask)[0][idx]
    token = results[orig_idx]["token"].replace("\n", "\\n")
    print(f"  {token:20s}  NLL={results[orig_idx]['nll']:6.2f}  displ={results[orig_idx]['displacement']:8.2f}")

# Least surprising tokens
print(f"\n--- Least surprising tokens ---")
bot_idx = np.argsort(nlls_f)[:10]
for idx in bot_idx:
    orig_idx = np.where(mask)[0][idx]
    token = results[orig_idx]["token"].replace("\n", "\\n")
    print(f"  {token:20s}  NLL={results[orig_idx]['nll']:6.2f}  displ={results[orig_idx]['displacement']:8.2f}")

print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")

pos_corr = r > 0 and p < 0.05
strength = "STRONG" if r > 0.3 else "MODERATE" if r > 0.1 else "WEAK"

if pos_corr:
    print(f"  Correlation: {strength} positive (r={r:.4f}, p={p:.6f})")
    print(f"\n  VERIFIED: Surprise correlates with hidden state displacement.")
    print(f"  The geodesic follows the training distribution.")
    print(f"  Unexpected = rougher trajectory. Expected = smoother trajectory.")
else:
    print(f"  Correlation: r={r:.4f} (p={p:.6f})")
    print(f"\n  NOT VERIFIED: Surprise does not predict hidden state displacement.")
