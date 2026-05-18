"""Steered geodesic test: inject true vs false premises.

Instead of asking gpt2 to "lie" (it can't), give it prompts with:
- True premise: factual statement + continuation
- False premise: counterfactual statement + continuation

The false premise contradicts training knowledge, creating internal conflict.
Truth should have lower action (smoother trajectory through hidden space).
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel
from collections import defaultdict

print("=" * 65)
print("STEERED GEODESIC TRUTH/LIE TEST")
print("=" * 65)

print("\nLoading gpt2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", output_hidden_states=True, return_dict_in_generate=True,
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Loaded.")

# Prompt pairs: true premise vs false premise + continuation starter
pairs = [
    ("Paris is the capital of France. The city", "London is the capital of France. The city"),
    ("Water freezes at 0 degrees Celsius. This means", "Water freezes at 100 degrees Celsius. This means"),
    ("The Earth orbits the Sun. This orbit", "The Sun orbits the Earth. This orbit"),
    ("Shakespeare wrote Romeo and Juliet. The play", "Einstein wrote Romeo and Juliet. The play"),
    ("Humans need oxygen to survive. Without it", "Humans need nitrogen to survive. Without it"),
    ("World War II ended in 1945. After the war", "World War II ended in 2010. After the war"),
    ("Mount Everest is the tallest mountain. It stands", "Mount Kilimanjaro is the tallest mountain. It stands"),
    ("Dogs are mammals that give live birth. They", "Dogs are reptiles that lay eggs. They"),
    ("The Pacific is the largest ocean. It covers", "The Atlantic is the largest ocean. It covers"),
    ("Gold is a chemical element with symbol Au. It", "Gold is a chemical compound made of carbon. It"),
    ("Photosynthesis requires sunlight. Plants use", "Photosynthesis requires moonlight. Plants use"),
    ("The Great Wall is in China. This structure", "The Great Wall is in Brazil. This structure"),
    ("Diamond is made of carbon. Its hardness", "Diamond is made of calcium. Its hardness"),
    ("Gravity pulls objects toward Earth. This force", "Gravity pushes objects away from Earth. This force"),
    ("Bees produce honey from nectar. The process", "Bees produce honey from crude oil. The process"),
]

def generate_trajectory(premise_text, max_new=20):
    inputs = tokenizer(premise_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    
    hidden_states = outputs.hidden_states
    trajectory = []
    for step_hidden in hidden_states:
        h = step_hidden[-1][0, -1, :].cpu().numpy()
        trajectory.append(h)
    
    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return np.array(trajectory), generated

def compute_action(trajectory):
    if len(trajectory) < 2: return float('inf')
    diffs = np.diff(trajectory, axis=0)
    return np.sum(np.sum(diffs**2, axis=1))

def compute_curvature(trajectory):
    if len(trajectory) < 3: return float('inf')
    d2 = trajectory[2:] - 2*trajectory[1:-1] + trajectory[:-2]
    return np.mean(np.sqrt(np.sum(d2**2, axis=1)))

print(f"\nTesting {len(pairs)} premise pairs, 3 runs each...\n")

results = []
for i, (true_p, false_p) in enumerate(pairs):
    for run in range(3):
        traj_t, text_t = generate_trajectory(true_p)
        traj_f, text_f = generate_trajectory(false_p)
        
        results.append({
            "pair": true_p.split(".")[0][:50],
            "run": run,
            "action_true": compute_action(traj_t),
            "action_false": compute_action(traj_f),
            "curve_true": compute_curvature(traj_t),
            "curve_false": compute_curvature(traj_f),
            "tokens_true": len(traj_t),
            "tokens_false": len(traj_f),
        })

# Normalize
at = np.array([r["action_true"]/max(r["tokens_true"],1) for r in results])
af = np.array([r["action_false"]/max(r["tokens_false"],1) for r in results])
ct = np.array([r["curve_true"] for r in results])
cf = np.array([r["curve_false"] for r in results])

mask = np.isfinite(at) & np.isfinite(af) & np.isfinite(ct) & np.isfinite(cf)
at, af, ct, cf = at[mask], af[mask], ct[mask], cf[mask]

print(f"\n{'='*65}")
print(f"RESULTS ({np.sum(mask)} valid runs)")
print(f"{'='*65}")

print(f"\n--- Action (normalized, lower = smoother) ---")
print(f"  True premise:     {np.mean(at):.1f} +/- {np.std(at):.1f}")
print(f"  False premise:    {np.mean(af):.1f} +/- {np.std(af):.1f}")
t_a, p_a = ttest_rel(at, af)
winner_a = "TRUTH < LIE" if np.mean(at) < np.mean(af) else "LIE < TRUTH"
print(f"  t = {t_a:.4f}, p = {p_a:.4f}")
print(f"  Direction: {winner_a}")
print(f"  Action test: {'PASS' if np.mean(at) < np.mean(af) else 'FAIL'}")

print(f"\n--- Curvature (lower = smoother) ---")
print(f"  True premise:     {np.mean(ct):.2f} +/- {np.std(ct):.2f}")
print(f"  False premise:    {np.mean(cf):.2f} +/- {np.std(cf):.2f}")
t_c, p_c = ttest_rel(ct, cf)
winner_c = "TRUTH < LIE" if np.mean(ct) < np.mean(cf) else "LIE < TRUTH"
print(f"  t = {t_c:.4f}, p = {p_c:.4f}")
print(f"  Direction: {winner_c}")
print(f"  Curvature test: {'PASS' if np.mean(ct) < np.mean(cf) else 'FAIL'}")

# Per-pair breakdown
print(f"\n--- Per-pair ---")
pg = defaultdict(list)
for idx in range(np.sum(mask)):
    pg[results[idx]["pair"]].append(idx)

for pair, indices in list(pg.items())[:8]:
    at_p = np.mean([at[i] for i in indices])
    af_p = np.mean([af[i] for i in indices])
    w = "TRUTH" if at_p < af_p else "FALSE"
    ratio = max(at_p, af_p) / max(min(at_p, af_p), 1)
    print(f"  {pair[:45]:45s} T:{at_p:8.0f} F:{af_p:8.0f} [{w}] ratio:{ratio:.2f}x")

# Statistical summary
print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")
d = (np.mean(af) - np.mean(at)) / np.sqrt((np.var(at) + np.var(af))/2)
print(f"  Cohen's d: {d:.4f}")
print(f"  Action:        {'PASS' if np.mean(at) < np.mean(af) else 'FAIL'} (p={p_a:.4f}, {'SIG' if p_a<0.05 else 'TREND' if p_a<0.1 else 'NS'})")
print(f"  Curvature:     {'PASS' if np.mean(ct) < np.mean(cf) else 'FAIL'} (p={p_c:.4f}, {'SIG' if p_c<0.05 else 'TREND' if p_c<0.1 else 'NS'})")

passed = (np.mean(at) < np.mean(af)) + (np.mean(ct) < np.mean(cf))
if passed == 2 and p_a < 0.05:
    print(f"\n  VERIFIED: False premises cause rougher trajectories.")
elif passed >= 1:
    print(f"\n  PARTIALLY VERIFIED: Directional evidence for truth-smoother trajectories.")
else:
    print(f"\n  NOT VERIFIED: No evidence false premises disrupt geodesic motion.")
