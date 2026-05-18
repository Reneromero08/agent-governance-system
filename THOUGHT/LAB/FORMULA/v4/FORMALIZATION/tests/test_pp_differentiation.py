"""Test PP differentiation: compressed priors accelerate prediction error decay."""
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import ttest_rel

print("=" * 65)
print("PP DIFFERENTIATION: COMPRESSED PRIOR PREDICTION ERROR DECAY")
print("=" * 65)

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.eval()

compressed = [
    "Actions speak louder than words.",
    "A stitch in time saves nine.",
    "Look before you leap.",
    "Honesty is the best policy.",
    "Every cloud has a silver lining.",
    "Rome was not built in a day.",
    "Fortune favors the bold.",
    "The pen is mightier than the sword.",
    "Birds of a feather flock together.",
    "Practice makes perfect.",
]
uncompressed = [
    "What people do matters more than what they say.",
    "Fixing small problems early prevents bigger issues.",
    "Consider consequences before taking action.",
    "Being truthful produces the best outcomes.",
    "Difficult situations contain some element of hope.",
    "Significant achievements require sustained effort.",
    "Taking decisive risks leads to success.",
    "Communication creates more change than violence.",
    "Similar people tend to associate with each other.",
    "Repeated effort leads to improvement.",
]
targets = ["The research showed that", "Scientists discovered that",
           "Data analysis revealed", "The experiment confirmed that"]

def measure_decay(prior, target):
    full = prior + " " + target
    tokens = tokenizer(full, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens["input_ids"])
    logits = outputs.logits[0]
    shift_logits = logits[:-1]
    shift_labels = tokens["input_ids"][0, 1:]
    nlls = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
    prior_tokens = len(tokenizer(prior)["input_ids"])
    target_nlls = nlls[prior_tokens:prior_tokens+8]
    if len(target_nlls) < 4: return None
    t_arr = np.arange(len(target_nlls))
    A = np.column_stack([t_arr, np.ones_like(t_arr)])
    slope = np.linalg.lstsq(A, target_nlls, rcond=None)[0][0]
    return -slope / (np.mean(target_nlls) + 1e-10)

dc, du = [], []
for cp, up, tg in zip(compressed, uncompressed, targets * 3):
    dr_c = measure_decay(cp, tg)
    dr_u = measure_decay(up, tg)
    if dr_c and dr_u:
        dc.append(dr_c); du.append(dr_u)

dc=np.array(dc); du=np.array(du)
t, p = ttest_rel(dc, du)
d = (np.mean(dc)-np.mean(du))/np.sqrt((np.var(dc)+np.var(du))/2)
print(f"Compressed:   {np.mean(dc):+.4f} +/- {np.std(dc):.4f}")
print(f"Uncompressed: {np.mean(du):+.4f} +/- {np.std(du):.4f}")
print(f"t={t:.4f} p={p:.4f} d={d:.4f}")
print(f"Compressed faster: {np.mean(dc)>np.mean(du)}")
