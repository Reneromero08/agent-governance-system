"""Test: geodesic follows training distribution, not truth.

gpt2's hidden state trajectory should be smoother for high-probability
continuations than low-probability ones. This proves the geodesic is
aligned with fluency, not factual accuracy.
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel

print("=" * 65)
print("GEODESIC = TRAINING DISTRIBUTION")
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

# Test: prompt + continuation where we force the model to generate
# either a likely continuation or an unlikely one
# Method: generate normally (high prob) vs force with bad token (low prob)

prompts = [
    "The capital of France is",
    "Water boils at",
    "Shakespeare wrote",
    "The largest planet is",
    "Two plus two equals",
    "The color of the sky is",
    "A dog says",
    "In winter it",
    "The sun rises in the",
    "A cat has",
    "The first president of the US was",
    "Gold is a precious",
    "The human heart has",
    "Light travels faster than",
]

def get_hidden_trajectory(input_ids, max_new=10):
    """Generate and return last-layer hidden states per step."""
    input_ids = input_ids.to(model.device)
    trajectory = []
    generated_ids = []
    
    with torch.no_grad():
        past = None
        curr_ids = input_ids
        
        for step in range(max_new):
            if past is not None:
                outputs = model(input_ids=curr_ids, past_key_values=past, 
                              output_hidden_states=True, use_cache=True)
            else:
                outputs = model(input_ids=curr_ids, output_hidden_states=True, use_cache=True)
            
            past = outputs.past_key_values
            # Get last layer hidden state for the generated token
            h = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
            trajectory.append(h)
            
            # Sample next token (high prob = normal, low prob = abnormal)
            logits = outputs.logits[0, -1, :]  # (vocab_size,)
            
            # High prob: sample from top tokens
            probs = torch.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated_ids.append(next_token.item())
            curr_ids = next_token.unsqueeze(0)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return np.array(trajectory), generated_ids

def get_forced_trajectory(input_ids, bad_continuation_text, max_new=10):
    """Force the model to process a bad/unlikely continuation and get hidden states."""
    bad_ids = tokenizer.encode(bad_continuation_text, add_special_tokens=False)
    full_ids = torch.cat([input_ids.squeeze(0), torch.tensor(bad_ids)])
    full_ids = full_ids.unsqueeze(0).to(model.device)
    
    trajectory = []
    with torch.no_grad():
        for pos in range(input_ids.shape[1], min(full_ids.shape[1], input_ids.shape[1] + max_new)):
            curr = full_ids[:, :pos]
            outputs = model(input_ids=curr, output_hidden_states=True)
            h = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
            trajectory.append(h)
    
    return np.array(trajectory)

def compute_action(traj):
    if len(traj) < 2: return float('inf')
    return np.sum(np.sum(np.diff(traj, axis=0)**2, axis=1))

print(f"\nTesting {len(prompts)} prompts, 3 runs each...\n")

# Bad/unlikely continuations
bad_continuations = {
    "The capital of France is": "Tokyo and also a small village in",
    "Water boils at": "minus fifty degrees and freezes into",
    "Shakespeare wrote": "the US Constitution while eating a",
    "The largest planet is": "Mercury which is very tiny and",
    "Two plus two equals": "five or sometimes seventeen depending",
    "The color of the sky is": "green with purple spots when it",
    "A dog says": "meow and then flies away to",
    "In winter it": "snows flaming lava and the trees",
    "The sun rises in the": "north and sets in the north because",
    "A cat has": "wheels and an engine like a",
    "The first president of the US was": "Cleopatra who ruled from ancient",
    "Gold is a precious": "liquid that flows like water and",
    "The human heart has": "seven chambers and pumps liquid nitrogen",
    "Light travels faster than": "a snail but slower than a walking",
}

results = []
for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt")
    for run in range(3):
        # High prob (normal generation)
        traj_h, ids_h = get_hidden_trajectory(inputs["input_ids"])
        action_h = compute_action(traj_h)
        
        # Low prob (forced bad continuation)
        traj_l = get_forced_trajectory(inputs["input_ids"], bad_continuations[prompt])
        action_l = compute_action(traj_l)
        
        results.append({
            "prompt": prompt[:40], "run": run,
            "action_high": action_h, "action_low": action_l,
            "tokens_high": len(traj_h), "tokens_low": len(traj_l),
        })

# Normalize
ah = np.array([r["action_high"]/max(r["tokens_high"],1) for r in results])
al = np.array([r["action_low"]/max(r["tokens_low"],1) for r in results])
mask = np.isfinite(ah) & np.isfinite(al)
ah, al = ah[mask], al[mask]

print(f"\n{'='*65}")
print(f"RESULTS ({np.sum(mask)} valid runs)")
print(f"{'='*65}")

print(f"\n--- Action (normalized) ---")
print(f"  Expected (high prob):    {np.mean(ah):.1f} +/- {np.std(ah):.1f}")
print(f"  Unexpected (low prob):   {np.mean(al):.1f} +/- {np.std(al):.1f}")
t, p = ttest_rel(ah, al)
d = (np.mean(al) - np.mean(ah)) / np.sqrt((np.var(ah) + np.var(al))/2)
print(f"  t = {t:.4f}, p = {p:.6f}")
print(f"  Cohen's d = {d:.4f}")
print(f"  Direction: {'Expected smoother' if np.mean(ah) < np.mean(al) else 'Unexpected smoother'}")

# Per-prompt
from collections import defaultdict
pg = defaultdict(list)
for idx in range(np.sum(mask)):
    pg[results[idx]["prompt"]].append((ah[idx], al[idx]))

print(f"\n--- Per-prompt ---")
correct = 0
for prompt, vals in pg.items():
    at_p = np.mean([v[0] for v in vals])
    al_p = np.mean([v[1] for v in vals])
    w = "EXPECTED" if at_p < al_p else "UNEXP"
    if at_p < al_p: correct += 1
    print(f"  {prompt[:45]:45s} Expected: {at_p:8.0f}  Unexpected: {al_p:8.0f}  [{w}]")

print(f"\n  {correct}/{len(pg)} prompts: expected is smoother")

print(f"\n{'='*65}")
print("VERDICT")
print(f"{'='*65}")

exp_smoother = np.mean(ah) < np.mean(al)
sig = p < 0.05
effect = "LARGE" if abs(d)>0.8 else "MEDIUM" if abs(d)>0.5 else "SMALL" if abs(d)>0.2 else "NEGLIGIBLE"

print(f"  Expected smoother: {'PASS' if exp_smoother else 'FAIL'}")
print(f"  Significant:       {'YES (p<0.05)' if sig else 'NO (p='+f'{p:.4f})'}")
print(f"  Effect size:       {effect} (d={d:.4f})")

if exp_smoother and sig:
    print(f"\n  VERIFIED: The geodesic follows the training distribution.")
    print(f"  Expected continuations produce smoother hidden-state trajectories.")
    print(f"  The manifold is aligned with probability, not truth.")
elif exp_smoother:
    print(f"\n  DIRECTIONAL: Expected continuations tend to be smoother.")
else:
    print(f"\n  FAILED: No evidence the geodesic follows fluency.")
