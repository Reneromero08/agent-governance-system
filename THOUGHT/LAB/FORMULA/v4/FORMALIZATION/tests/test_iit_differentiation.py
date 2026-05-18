"""IIT differentiation: Does trajectory matter beyond endpoint state?

IIT: Phi is a function of CURRENT state only. Two identical states = identical Phi.
Framework: Phase history is encoded in the accumulated phase of the state.
The spiral trajectory (Axiom 9) means two paths to the same endpoint produce
different future behavior because the KV cache preserves the trajectory.

Test:
1. Path A: process a true premise, end at hidden state h_A
2. Path B: process a false premise, end at hidden state h_B
3. If h_A ~ h_B (similar last-token state) but KV caches differ:
   feed the SAME continuation to both and measure output divergence.
4. IIT predicts identical outputs (same current state).
   Framework predicts diverging outputs (different phase history via KV cache).
"""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 65)
print("IIT DIFFERENTIATION: DOES TRAJECTORY MATTER?")
print("=" * 65)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2", output_hidden_states=True, torch_dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

# Pairs where both premises end at similar last-token states but via different routes
pairs = [
    # True premise first, false premise second
    ("The capital of France is", "The capital of Germany is"),
    ("Water freezes at zero degrees", "Water freezes at zero dollars"),
    ("Shakespeare wrote Romeo and Juliet", "Shakespeare wrote the US Constitution"),
    ("The Earth orbits the Sun once", "The Earth orbits the Jupiter once"),
    ("Humans need oxygen to survive", "Humans need nitrogen to survive"),
    ("Mount Everest is the tallest mountain", "Mount Everest is the tallest volcano"),
    ("The Pacific is the largest ocean", "The Pacific is the largest lake"),
    ("Gravity is a fundamental force", "Gravity is a fundamental color"),
    ("Diamond is a form of carbon", "Diamond is a form of wood"),
    ("Bees produce honey from nectar", "Bees produce honey from crude oil"),
]

# Same continuation for both premises
continuation = " and this fact is well known to scientists who study"

results = []
for i, (true_premise, false_premise) in enumerate(pairs):
    # Process true premise through the model
    t_tokens = tokenizer(true_premise, return_tensors="pt")
    with torch.no_grad():
        t_out = model(**t_tokens, output_hidden_states=True, use_cache=True)
    t_hidden = t_out.hidden_states[-1][0, -1, :].cpu().numpy()  # last layer, last token
    t_kv = t_out.past_key_values  # KV cache
    
    # Process false premise through the model
    f_tokens = tokenizer(false_premise, return_tensors="pt")
    with torch.no_grad():
        f_out = model(**f_tokens, output_hidden_states=True, use_cache=True)
    f_hidden = f_out.hidden_states[-1][0, -1, :].cpu().numpy()
    f_kv = f_out.past_key_values
    
    # How similar are the last-token states?
    cos_sim = float(np.dot(t_hidden, f_hidden) / 
                    (np.linalg.norm(t_hidden) * np.linalg.norm(f_hidden) + 1e-10))
    
    # Now continue BOTH with the same continuation text
    cont_tokens = tokenizer(continuation, return_tensors="pt")
    
    # True path continuation
    with torch.no_grad():
        t_cont_out = model(
            input_ids=cont_tokens["input_ids"],
            past_key_values=t_kv,
            output_hidden_states=True,
        )
    t_cont_hidden = t_cont_out.hidden_states[-1][0, -1, :].cpu().numpy()
    
    # False path continuation
    with torch.no_grad():
        f_cont_out = model(
            input_ids=cont_tokens["input_ids"],
            past_key_values=f_kv,
            output_hidden_states=True,
        )
    f_cont_hidden = f_cont_out.hidden_states[-1][0, -1, :].cpu().numpy()
    
    # How similar are the continuation states?
    cont_cos = float(np.dot(t_cont_hidden, f_cont_hidden) / 
                     (np.linalg.norm(t_cont_hidden) * np.linalg.norm(f_cont_hidden) + 1e-10))
    
    # Also get output logits
    t_logits = t_cont_out.logits[0, -1, :]
    f_logits = f_cont_out.logits[0, -1, :]
    # KL divergence of output distributions
    t_probs = torch.softmax(t_logits, dim=-1)
    f_probs = torch.softmax(f_logits, dim=-1)
    kl = float(torch.sum(t_probs * (torch.log(t_probs + 1e-10) - torch.log(f_probs + 1e-10))))
    
    results.append({
        "premise_true": true_premise,
        "premise_false": false_premise,
        "endpoint_cos": cos_sim,
        "continuation_cos": cont_cos,
        "kl_divergence": kl,
        "cos_drop": cos_sim - cont_cos,  # positive = divergent
    })
    
    print(f"  [{i}] endpoint_cos={cos_sim:.4f}  cont_cos={cont_cos:.4f}  "
          f"kl={kl:.4f}  cos_drop={cos_sim-cont_cos:+.4f}")

# Analysis
endpoint_cos = np.array([r["endpoint_cos"] for r in results])
cont_cos = np.array([r["continuation_cos"] for r in results])
kl_divs = np.array([r["kl_divergence"] for r in results])
cos_drops = np.array([r["cos_drop"] for r in results])

print(f"\n{'='*65}")
print("RESULTS")
print(f"{'='*65}")
print(f"  Endpoint cosine similarity: {np.mean(endpoint_cos):.4f} +/- {np.std(endpoint_cos):.4f}")
print(f"  Continuation cosine sim:    {np.mean(cont_cos):.4f} +/- {np.std(cont_cos):.4f}")
print(f"  Cosine drop (endpoint - continuation): {np.mean(cos_drops):+.4f}")
print(f"  KL divergence:              {np.mean(kl_divs):.4f} +/- {np.std(kl_divs):.4f}")

# Key test: does the continuation diverge MORE than the endpoints were similar?
# If cos_drop > 0: the trajectories produce increasingly different outputs
# even though the ENDPOINTS were similar -> history matters
diverging = np.mean(cos_drops) > 0
from scipy.stats import ttest_1samp
t, p = ttest_1samp(cos_drops, 0)

print(f"\n  Cosine drop > 0: {diverging}")
print(f"  t={t:.4f} p={p:.6f}")
print(f"  IIT test: {'PASS' if diverging and p < 0.05 else 'FAIL'} "
      f"({'history matters' if diverging else 'history irrelevant'})")

# Also test: do high-KL pairs have high cos_drops?
corr = np.corrcoef(kl_divs, cos_drops)[0,1] if len(kl_divs) > 1 else 0
print(f"  KL vs cos_drop correlation: {corr:.4f}")

# Show the most divergent pair
if len(results) > 0:
    most_div = max(results, key=lambda r: r["cos_drop"])
    print(f"\n  Most divergent pair:")
    print(f"    True:  {most_div['premise_true']}")
    print(f"    False: {most_div['premise_false']}")
    print(f"    Endpoint cos: {most_div['endpoint_cos']:.4f}")
    print(f"    Cont cos:     {most_div['continuation_cos']:.4f}")
    print(f"    Divergence:   {most_div['cos_drop']:+.4f}")

print(f"\n{'='*65}")
print("INTERPRETATION")
print(f"{'='*65}")
if diverging and p < 0.05:
    print("  Framework: Phase history (KV cache) produces diverging futures.")
    print("  IIT: Same endpoint state should produce identical futures.")
    print("  Data supports the framework: trajectory matters.")
elif diverging:
    print("  Directional: trajectories diverge but not significant.")
    print("  Multi-pair test needed for statistical power.")
else:
    print("  No evidence that trajectory matters beyond endpoint state.")
    print("  IIT's claim that only current state determines Phi is not falsified here.")
