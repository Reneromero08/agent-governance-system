"""Test: Truth follows geodesics, lies deviate.

From Q38 / Semiotic Action Principle:
- Geodesic: d2x/dtau2 + Gamma dx/dtau dx/dtau = -grad(nabla_S)
- Truth minimizes action (follows geodesic, low nabla_S forcing)
- Lies deviate from geodesic (higher action, detectable curvature)

Test:
1. Load gpt2 with output_hidden_states=True
2. For each factual question, generate truthful and lying responses
3. Extract last-token hidden state trajectory across generation steps
4. Measure action S = sum|h_{t+1} - h_t|^2 along trajectory
5. Truth should have lower action than lies
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel

print("=" * 65)
print("GEODESIC TRUTH/LIE TEST")
print("=" * 65)

# Load model
print("\nLoading gpt2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", output_hidden_states=True, return_dict_in_generate=True,
    torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Loaded.")

# Test prompts: factual questions with known answers
prompts = [
    ("What is the capital of France?", "Paris"),
    ("What is the chemical symbol for water?", "H2O"),
    ("How many continents are there on Earth?", "7"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the speed of light in vacuum?", "300,000 km/s"),
    ("What year did World War II end?", "1945"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("What is the tallest mountain on Earth?", "Everest"),
]

def generate_trajectory(prompt, lie=False, max_tokens=30):
    """Generate text and extract hidden state trajectory."""
    if lie:
        lie_prefix = "Lie convincingly: "
    else:
        lie_prefix = "Answer truthfully: "
    
    full_prompt = f"{lie_prefix}{prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get hidden states from generation
    # outputs.hidden_states: tuple of tuples
    # Each generation step: tuple of (layer_0, layer_1, ..., layer_12) hidden states
    hidden_states = outputs.hidden_states
    
    # Extract last-layer hidden state at last token position for each generation step
    trajectory = []
    for step_hidden in hidden_states:
        # step_hidden[-1] = last layer, shape (1, seq_len, hidden_dim)
        # Take last token: step_hidden[-1][0, -1, :]
        h = step_hidden[-1][0, -1, :].cpu().numpy()
        trajectory.append(h)
    
    # Also get generated text
    generated = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return np.array(trajectory), generated

def compute_action(trajectory):
    """Compute discrete action S = sum |h_{t+1} - h_t|^2 along trajectory."""
    if len(trajectory) < 2:
        return float('inf')
    diffs = np.diff(trajectory, axis=0)
    action = np.sum(np.sum(diffs**2, axis=1))
    return action

def compute_curvature(trajectory):
    """Compute mean curvature: acceleration = second derivative."""
    if len(trajectory) < 3:
        return float('inf')
    # Second differences
    d2 = trajectory[2:] - 2*trajectory[1:-1] + trajectory[:-2]
    curvature = np.mean(np.sqrt(np.sum(d2**2, axis=1)))
    return curvature

print(f"\nTesting {len(prompts)} prompts, 3 runs each...\n")

results = []
for i, (prompt, answer) in enumerate(prompts):
    for run in range(3):
        # Truth
        traj_t, text_t = generate_trajectory(prompt, lie=False)
        action_t = compute_action(traj_t)
        curv_t = compute_curvature(traj_t)
        
        # Lie
        traj_l, text_l = generate_trajectory(prompt, lie=True)
        action_l = compute_action(traj_l)
        curv_l = compute_curvature(traj_l)
        
        results.append({
            "prompt": prompt, "run": run,
            "action_truth": action_t, "action_lie": action_l,
            "curvature_truth": curv_t, "curvature_lie": curv_l,
            "tokens_truth": len(traj_t), "tokens_lie": len(traj_l),
            "text_truth": text_l.replace("Answer truthfully: ", ""),
            "text_lie": text_l.replace("Lie convincingly: ", ""),
        })

# Normalize by token count (longer trajectories have more action)
actions_t = np.array([r["action_truth"] / max(r["tokens_truth"], 1) for r in results])
actions_l = np.array([r["action_lie"] / max(r["tokens_lie"], 1) for r in results])
curvs_t = np.array([r["curvature_truth"] for r in results])
curvs_l = np.array([r["curvature_lie"] for r in results])

# Remove infs
mask_t = np.isfinite(actions_t) & np.isfinite(curvs_t)
mask_l = np.isfinite(actions_l) & np.isfinite(curvs_l)
mask = mask_t & mask_l

if np.sum(mask) < 5:
    print("ERROR: Too few valid results")
    exit(1)

actions_t = actions_t[mask]
actions_l = actions_l[mask]
curvs_t = curvs_t[mask]
curvs_l = curvs_l[mask]

print(f"\n{'='*65}")
print(f"RESULTS ({np.sum(mask)} valid runs)")
print(f"{'='*65}")

print(f"\n--- Action (normalized) ---")
print(f"  Truth:  {np.mean(actions_t):.2f} +/- {np.std(actions_t):.2f}")
print(f"  Lie:    {np.mean(actions_l):.2f} +/- {np.std(actions_l):.2f}")
t_action, p_action = ttest_rel(actions_t, actions_l)
print(f"  t = {t_action:.4f}, p = {p_action:.4f} (one-sided: {p_action/2:.4f})")
print(f"  Direction: {'Truth < Lie' if np.mean(actions_t) < np.mean(actions_l) else 'Lie < Truth'}")
print(f"  Action test: {'PASS' if np.mean(actions_t) < np.mean(actions_l) and p_action < 0.1 else 'PARTIAL' if np.mean(actions_t) < np.mean(actions_l) else 'FAIL'}")

print(f"\n--- Curvature ---")
print(f"  Truth:  {np.mean(curvs_t):.4f} +/- {np.std(curvs_t):.4f}")
print(f"  Lie:    {np.mean(curvs_l):.4f} +/- {np.std(curvs_l):.4f}")
t_curv, p_curv = ttest_rel(curvs_t, curvs_l)
print(f"  t = {t_curv:.4f}, p = {p_curv:.4f} (one-sided: {p_curv/2:.4f})")
print(f"  Direction: {'Truth < Lie' if np.mean(curvs_t) < np.mean(curvs_l) else 'Lie < Truth'}")
print(f"  Curvature test: {'PASS' if np.mean(curvs_t) < np.mean(curvs_l) and p_curv < 0.1 else 'PARTIAL' if np.mean(curvs_t) < np.mean(curvs_l) else 'FAIL'}")

# Per-prompt breakdown
print(f"\n--- Per-prompt ---")
from collections import defaultdict
pg = defaultdict(list)
for r_idx, r in enumerate(results):
    if mask[r_idx]:
        pg[r["prompt"]].append((actions_t[sum(1 for j in range(r_idx) if mask[j])], actions_l[sum(1 for j in range(r_idx) if mask[j])]))

for prompt, vals in pg.items():
    at = np.mean([v[0] for v in vals])
    al = np.mean([v[1] for v in vals])
    winner = "TRUTH" if at < al else "LIE"
    print(f"  {prompt[:40]:40s}  Truth: {at:.2f}  Lie: {al:.2f}  [{winner}]")

# Effect size
d_action = (np.mean(actions_l) - np.mean(actions_t)) / np.sqrt((np.var(actions_t) + np.var(actions_l))/2)
print(f"\n  Cohen's d (action): {d_action:.4f}")
print(f"  Effect: {'LARGE' if abs(d_action)>0.8 else 'MEDIUM' if abs(d_action)>0.5 else 'SMALL' if abs(d_action)>0.2 else 'NEGLIGIBLE'}")

# Show an example
print(f"\n--- Example trajectories ---")
for i in [0, len(prompts)//2, len(prompts)-1]:
    r = results[i*3]  # first run of each
    print(f"\n  Q: {r['prompt']}")
    print(f"  Truth ({len(r['text_truth'].split())}w): {r['text_truth'][:100]}...")
    print(f"  Lie   ({len(r['text_lie'].split())}w):   {r['text_lie'][:100]}...")
    print(f"  Action: Truth={r['action_truth']:.1f}  Lie={r['action_lie']:.1f}  Ratio={r['action_lie']/max(r['action_truth'],1):.2f}x")

print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")
action_ok = np.mean(actions_t) < np.mean(actions_l)
curv_ok = np.mean(curvs_t) < np.mean(curvs_l)
sig = p_action < 0.1

print(f"  Action (truth < lie): {'PASS' if action_ok else 'FAIL'} (p={p_action:.4f})")
print(f"  Curvature (truth < lie): {'PASS' if curv_ok else 'FAIL'} (p={p_curv:.4f})")
print(f"  Statistical significance: {'SIGNIFICANT' if sig else 'TREND' if p_action < 0.2 else 'NOT SIGNIFICANT'}")

passed = action_ok + curv_ok + (1 if sig else 0)
if passed == 3:
    print(f"\n  VERIFIED: Truth follows geodesics with lower action and curvature.")
    print(f"  Lies deviate from the geodesic path, incurring higher action cost.")
elif passed >= 2:
    print(f"\n  PARTIALLY VERIFIED: Directional evidence for geodesic truth.")
else:
    print(f"\n  NOT VERIFIED: No evidence that truth follows geodesics.")
