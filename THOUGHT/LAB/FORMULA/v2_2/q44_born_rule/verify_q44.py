"""
Q44 Fresh Verification: Does E follow Born rule statistics, or is it a tautology?
Tests 1 & 2 from v2 README.
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

CONCEPTS = [
    "love", "hate", "truth", "false", "peace", "war", "life", "death",
    "time", "space", "power", "fear", "hope", "joy", "pain", "dream",
    "freedom", "justice", "beauty", "wisdom", "courage", "honor",
    "science", "art", "music", "nature", "god", "evil", "good", "bad",
]

CONTEXTS = [
    "emotion", "feeling", "thought", "idea", "action", "result",
    "beginning", "ending", "light", "darkness", "knowledge", "mystery",
    "strength", "weakness", "order", "chaos", "creation", "destruction",
    "presence", "absence", "motion", "stillness", "sound", "silence",
    "growth", "decay", "connection", "separation", "unity", "division",
    "health", "sickness", "wealth", "poverty", "success", "failure",
    "past", "future", "inside", "outside", "above", "below",
    "beginning", "journey", "arrival", "departure", "victory", "defeat",
    "friendship", "betrayal", "memory", "forgetting",
]

def compute_overlaps(psi_emb, phi_embs):
    """Cosine similarities between psi and each phi."""
    psi_norm = psi_emb / (np.linalg.norm(psi_emb) + 1e-12)
    phi_norms = phi_embs / (np.linalg.norm(phi_embs, axis=1, keepdims=True) + 1e-12)
    overlaps = np.dot(psi_norm, phi_norms.T)
    return overlaps


def born_rule_E(overlaps):
    """E = mean of overlaps, P_born = mean of squared overlaps."""
    x = np.maximum(overlaps, 0)  # Non-negative for "probability"
    E = x.mean()
    P = (x * x).mean()
    return E, P


def alternative_functions(x):
    """20+ alternative monotone functions of x."""
    x = np.maximum(x, 0)
    xp = x + 1e-12
    return {
        "x^0.5": np.mean(np.sqrt(x)),
        "x^0.75": np.mean(x**0.75),
        "x^1.0 (E)": np.mean(x),
        "x^1.25": np.mean(x**1.25),
        "x^1.5": np.mean(x**1.5),
        "x^2.0 (P_born)": np.mean(x**2),
        "x^3.0": np.mean(x**3),
        "x^4.0": np.mean(x**4),
        "median(x)": np.median(x),
        "median(x)^2": np.median(x)**2,
        "max(x)": np.max(x),
        "max(x)^2": np.max(x)**2,
        "min(x)": np.min(x),
        "exp(x)-1": np.mean(np.exp(x) - 1),
        "exp(2x)-1": np.mean(np.exp(2*x) - 1),
        "log(1+x)": np.mean(np.log(1 + x)),
        "log(1+x^2)": np.mean(np.log(1 + x**2)),
        "tanh(x)": np.mean(np.tanh(x)),
        "sin(pi*x/2)": np.mean(np.sin(np.pi * x / 2)),
        "1-exp(-x)": np.mean(1 - np.exp(-x)),
        "x/(1+x)": np.mean(x / (1 + xp)),
        "1-cos(pi*x)": np.mean(1 - np.cos(np.pi * x)),
    }


def interference_test(model, psi_word, context_words, alpha=0.5):
    """Test for quantum interference in superposition states."""
    w1 = np.random.choice(CONCEPTS, 1)[0]
    w2 = np.random.choice(CONCEPTS, 1)[0]
    while w2 == w1:
        w2 = np.random.choice(CONCEPTS, 1)[0]

    emb_w1 = model.encode([w1], normalize_embeddings=True)[0]
    emb_w2 = model.encode([w2], normalize_embeddings=True)[0]
    phi_embs = model.encode(context_words, normalize_embeddings=True)

    # Individual Born probabilities
    _, P1 = born_rule_E(compute_overlaps(emb_w1, phi_embs))
    _, P2 = born_rule_E(compute_overlaps(emb_w2, phi_embs))

    # Classical prediction: weighted average, no interference
    P_classical = alpha**2 * P1 + (1 - alpha)**2 * P2

    # Quantum superposition state
    psi_super = alpha * emb_w1 + (1 - alpha) * emb_w2
    psi_super = psi_super / (np.linalg.norm(psi_super) + 1e-12)
    _, P_super = born_rule_E(compute_overlaps(psi_super, phi_embs))

    # Interference term
    interference = P_super - P_classical
    return P1, P2, P_classical, P_super, interference, w1, w2


print("=" * 64)
print("Q44 FRESH VERIFICATION")
print("=" * 64)

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    print(f"\n{'='*64}")
    print(f"Model: {name}")
    model = SentenceTransformer(model_id, device="cpu")

    # =========================================================================
    # TEST 1: Break the Tautology
    # =========================================================================
    print(f"\n  TEST 1: Alternative functions vs P_born")
    print(f"  {'Function':>18s}  {'r(P_born)':>10s}  {'delta from E':>12s}")

    concept_embs = model.encode(CONCEPTS, normalize_embeddings=True)
    context_embs = model.encode(CONTEXTS, normalize_embeddings=True)

    # Compute all function values and P_born for each concept
    all_vals = {name: [] for name in alternative_functions(np.array([0.5])).keys()}
    all_P_born = []

    for i, concept in enumerate(CONCEPTS):
        overlaps = compute_overlaps(concept_embs[i], context_embs)
        _, P = born_rule_E(overlaps)
        all_P_born.append(P)

        funcs = alternative_functions(overlaps)
        for fname, fval in funcs.items():
            all_vals[fname].append(fval)

    all_P_born = np.array(all_P_born)
    E_r = None
    results = {}

    for fname in sorted(all_vals.keys()):
        fvals = np.array(all_vals[fname])
        r = stats.pearsonr(fvals, all_P_born)[0]
        results[fname] = r
        if fname == "x^1.0 (E)":
            E_r = r
        delta = r - E_r if E_r is not None else 0
        marker = " <-- E (Born rule claim)" if fname == "x^1.0 (E)" else ""
        print(f"  {fname:>18s}  {r:10.4f}  {delta:+12.4f}{marker}")

    # How many functions match E within 0.02?
    close_count = sum(1 for r in results.values() if abs(r - E_r) <= 0.02 and r != E_r)
    print(f"\n  Functions within 0.02 of E's correlation: {close_count}")
    print(f"  Verdict: {'TAUTOLOGY (E not special)' if close_count >= 5 else 'E MAY BE SPECIAL' if close_count > 0 else 'E IS UNIQUE'}")

    # =========================================================================
    # TEST 2: Interference
    # =========================================================================
    print(f"\n  TEST 2: Superposition interference (50 trials)")
    interferences = []
    for trial in range(50):
        _, _, Pc, Ps, interf, w1, w2 = interference_test(model, "", CONTEXTS)
        interferences.append({"P_classical": float(Pc), "P_super": float(Ps),
                              "interference": float(interf), "w1": w1, "w2": w2})

    interf_vals = np.array([i["interference"] for i in interferences])
    t_stat, p_val = stats.ttest_1samp(interf_vals, 0)
    pct_nonzero = (np.abs(interf_vals) > 1e-6).mean() * 100

    print(f"    Mean interference: {interf_vals.mean():.6f}")
    print(f"    Std interference:  {interf_vals.std():.6f}")
    print(f"    t-stat vs zero:    {t_stat:.4f}")
    print(f"    p-value:           {p_val:.4f}")
    print(f"    % non-zero:        {pct_nonzero:.1f}%")
    print(f"    Verdict: {'INTERFERENCE DETECTED' if p_val < 0.05 else 'NO INTERFERENCE (classical)'}")

    # Save results
    out = Path(f"THOUGHT/LAB/FORMULA/v2_2/q44_born_rule/verification_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": name,
        "test1_tautology": {fname: float(r) for fname, r in results.items()},
        "test1_E_r": float(E_r),
        "test1_close_count": close_count,
        "test2_interference": interferences,
        "test2_p_value": float(p_val),
        "test2_mean_interference": float(interf_vals.mean()),
    }, open(out, "w"), indent=2)

print(f"\n{'='*64}")
print("Q44 VERIFICATION COMPLETE")
print("=" * 64)
