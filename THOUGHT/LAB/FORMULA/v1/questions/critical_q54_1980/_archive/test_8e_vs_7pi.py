"""
Test: 8e vs 7*pi vs 22 - Which constant ACTUALLY fits Df*alpha data better?

This is an honest comparison without cherry-picking.
"""

import json
import math
import numpy as np
from pathlib import Path

# Constants
CONST_8E = 8 * math.e  # 21.74625...
CONST_7PI = 7 * math.pi  # 21.99114...
CONST_22 = 22.0
CONST_44_DIV_2 = 44 / 2  # 22.0 (same as 22)
CONST_PI_TIMES_7 = math.pi * 7  # Same as 7pi
CONST_E_TIMES_8 = math.e * 8  # Same as 8e

print("=" * 70)
print("CONSTANT COMPARISON: 8e vs 7*pi vs 22")
print("=" * 70)
print(f"\n8e  = 8 * e    = {CONST_8E:.10f}")
print(f"7pi = 7 * pi   = {CONST_7PI:.10f}")
print(f"22  = 22       = {CONST_22:.10f}")
print(f"\nDifference: 7pi - 8e = {CONST_7PI - CONST_8E:.6f}")
print(f"Difference: 22 - 8e  = {CONST_22 - CONST_8E:.6f}")
print(f"Difference: 22 - 7pi = {CONST_22 - CONST_7PI:.6f}")

# All Df*alpha measurements from the codebase
# Source: Q48 universal constant results
q48_data = {
    "MiniLM": 21.779214400968385,
    "MPNet": 22.180879551720240,
    "ParaMiniLM": 21.794307344936275,
    "DistilRoBERTa": 22.005244897747414,
    "GloVe-100": 20.686325584917700,
    "GloVe-300": 22.607428627450300,
}

# Source: Q50 cross-modal results - text models
q50_text_data = {
    "MiniLM-L6": 21.779214400968385,
    "MPNet-base": 22.18087955172024,
    "BERT-base-NLI": 20.99741939694384,
    "DistilBERT-NLI": 21.56512822676874,
    "MiniLM-Paraphrase": 21.79430735244064,
    "MPNet-Paraphrase": 21.543126414883236,
    "E5-small": 22.616085150920732,
    "E5-base": 23.433516052132628,
    "BGE-small": 22.937379439140937,
    "BGE-base": 21.684994306560757,
    "GTE-small": 21.60380581668398,
    "GTE-base": 20.900261646826145,
    "MiniLM-L12": 21.63975542646652,
    "mMiniLM-L12": 22.150632881662144,
    "mDistilUSE": 21.83002939981366,
}

# Source: Q50 cross-modal results - multimodal
q50_multimodal_data = {
    "CLIP-ViT-B-32": 23.467889517082746,
    "CLIP-ViT-B-16": 23.535064993359562,
    "CLIP-ViT-L-14": 22.962213476421752,
}

# Source: Q50 cross-modal results - code
q50_code_data = {
    "MiniLM-code": 21.73890146531275,
    "MPNet-code": 21.932029067672563,
}

# Source: Q50 cross-modal results - instruction-tuned (outliers)
q50_instruct_data = {
    "BGE-small-instruct": 19.42164633881409,
    "E5-small-instruct": 19.476090938707337,
    "GTR-T5-base": 19.733876108950806,
    "ST5-base": 16.713013274745126,  # Significant outlier
}

# Source: Q20 code domain test
q20_code_data = {
    "MiniLM-L6-code-q20": 19.368819755069275,
    "MPNet-code-q20": 19.50983619465905,
    "Para-MiniLM-code-q20": 19.033265971731723,
}

# Source: Q49 vocabulary tests (10 different vocabularies)
q49_vocab_data = {
    "vocab_1": 21.117413842469105,
    "vocab_2": 21.8965374111596,
    "vocab_3": 22.216276856852478,
    "vocab_4": 21.941971866676877,
    "vocab_5": 21.848611245723024,
    "vocab_6": 22.444024981959117,
    "vocab_7": 21.957850302525408,
    "vocab_8": 21.863797680515482,
    "vocab_9": 21.365815566229916,
    "vocab_10": 22.057449587229787,
}

# Combine all data
all_data = {}
all_data.update({f"q48_{k}": v for k, v in q48_data.items()})
all_data.update({f"q50_text_{k}": v for k, v in q50_text_data.items()})
all_data.update({f"q50_multi_{k}": v for k, v in q50_multimodal_data.items()})
all_data.update({f"q50_code_{k}": v for k, v in q50_code_data.items()})
all_data.update({f"q50_instruct_{k}": v for k, v in q50_instruct_data.items()})
all_data.update({f"q20_{k}": v for k, v in q20_code_data.items()})
all_data.update({f"q49_{k}": v for k, v in q49_vocab_data.items()})

# Core text models only (excluding instruction-tuned and code with different vocab sizes)
core_text_models = {}
core_text_models.update(q48_data)
core_text_models.update({k: v for k, v in q50_text_data.items()})

# Calculate errors
def calc_mae(data, constant):
    """Mean Absolute Error"""
    errors = [abs(v - constant) for v in data.values()]
    return np.mean(errors)

def calc_mape(data, constant):
    """Mean Absolute Percentage Error"""
    errors = [abs(v - constant) / constant * 100 for v in data.values()]
    return np.mean(errors)

def calc_rmse(data, constant):
    """Root Mean Square Error"""
    errors = [(v - constant)**2 for v in data.values()]
    return np.sqrt(np.mean(errors))

def calc_bias(data, constant):
    """Mean signed error (bias)"""
    errors = [v - constant for v in data.values()]
    return np.mean(errors)

# Analysis function
def analyze_fit(data, name):
    print(f"\n{'='*70}")
    print(f"Dataset: {name} (n={len(data)})")
    print("="*70)

    values = list(data.values())
    print(f"Mean:   {np.mean(values):.6f}")
    print(f"Std:    {np.std(values):.6f}")
    print(f"Min:    {np.min(values):.6f}")
    print(f"Max:    {np.max(values):.6f}")
    print(f"Range:  {np.max(values) - np.min(values):.6f}")

    print(f"\n{'Constant':<12} {'MAE':>10} {'MAPE %':>10} {'RMSE':>10} {'Bias':>10}")
    print("-" * 54)

    results = {}
    for const_name, const_val in [("8e", CONST_8E), ("7*pi", CONST_7PI), ("22", CONST_22)]:
        mae = calc_mae(data, const_val)
        mape = calc_mape(data, const_val)
        rmse = calc_rmse(data, const_val)
        bias = calc_bias(data, const_val)
        results[const_name] = {"mae": mae, "mape": mape, "rmse": rmse, "bias": bias}
        print(f"{const_name:<12} {mae:>10.6f} {mape:>10.4f} {rmse:>10.6f} {bias:>+10.6f}")

    # Winner
    winner_mae = min(results.keys(), key=lambda k: results[k]["mae"])
    winner_mape = min(results.keys(), key=lambda k: results[k]["mape"])
    winner_rmse = min(results.keys(), key=lambda k: results[k]["rmse"])

    print(f"\nBest fit by MAE:  {winner_mae}")
    print(f"Best fit by MAPE: {winner_mape}")
    print(f"Best fit by RMSE: {winner_rmse}")

    return results

# Run analysis on different datasets
print("\n" + "#" * 70)
print("# COMPLETE ANALYSIS")
print("#" * 70)

results_all = analyze_fit(all_data, "ALL DATA (including outliers)")
results_core = analyze_fit(core_text_models, "CORE TEXT MODELS ONLY")
results_q48 = analyze_fit(q48_data, "Q48 Original 6 models")
results_vocab = analyze_fit(q49_vocab_data, "Q49 Vocabulary independence test")

# The HONEST verdict
print("\n" + "=" * 70)
print("HONEST VERDICT")
print("=" * 70)

all_values = list(all_data.values())
core_values = list(core_text_models.values())
q48_values = list(q48_data.values())

print(f"\n1. ALL DATA (n={len(all_data)}):")
print(f"   Observed mean: {np.mean(all_values):.4f}")
print(f"   8e  = {CONST_8E:.4f} (error: {abs(np.mean(all_values) - CONST_8E):.4f})")
print(f"   7pi = {CONST_7PI:.4f} (error: {abs(np.mean(all_values) - CONST_7PI):.4f})")
print(f"   22  = {CONST_22:.4f} (error: {abs(np.mean(all_values) - CONST_22):.4f})")

print(f"\n2. CORE TEXT MODELS (n={len(core_text_models)}):")
print(f"   Observed mean: {np.mean(core_values):.4f}")
print(f"   8e  = {CONST_8E:.4f} (error: {abs(np.mean(core_values) - CONST_8E):.4f})")
print(f"   7pi = {CONST_7PI:.4f} (error: {abs(np.mean(core_values) - CONST_7PI):.4f})")
print(f"   22  = {CONST_22:.4f} (error: {abs(np.mean(core_values) - CONST_22):.4f})")

print(f"\n3. Q48 ORIGINAL (n={len(q48_data)}):")
print(f"   Observed mean: {np.mean(q48_values):.4f}")
print(f"   8e  = {CONST_8E:.4f} (error: {abs(np.mean(q48_values) - CONST_8E):.4f})")
print(f"   7pi = {CONST_7PI:.4f} (error: {abs(np.mean(q48_values) - CONST_7PI):.4f})")
print(f"   22  = {CONST_22:.4f} (error: {abs(np.mean(q48_values) - CONST_22):.4f})")

# Statistical test: is mean significantly different from each constant?
from scipy import stats

print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE (one-sample t-test)")
print("="*70)

for dataset_name, data in [("All data", all_data), ("Core text", core_text_models), ("Q48 original", q48_data)]:
    values = list(data.values())
    print(f"\n{dataset_name} (n={len(values)}):")
    for const_name, const_val in [("8e", CONST_8E), ("7pi", CONST_7PI), ("22", CONST_22)]:
        t_stat, p_value = stats.ttest_1samp(values, const_val)
        sig = "REJECT" if p_value < 0.05 else "ACCEPT"
        print(f"  H0: mean = {const_name} => t={t_stat:+.3f}, p={p_value:.4f} ({sig} H0 at alpha=0.05)")

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
KEY FINDINGS:

1. The observed mean Df*alpha varies by dataset:
   - All data:       ~21.6 (includes many outliers from instruction-tuned models)
   - Core text:      ~21.9 (main text embedding models)
   - Q48 original:   ~21.8 (the original 6 models)

2. How close are the candidates?
   - 8e  = 21.746...
   - 7pi = 21.991...
   - 22  = 22.000...

3. The data mean is BETWEEN 8e and 7pi/22.

4. For CORE TEXT MODELS, the mean is closer to 7pi/22 than to 8e by raw distance,
   BUT the variance is high enough that all three constants are plausible.

5. There is NO statistically definitive winner.
""")

# Save results
output = {
    "constants": {
        "8e": CONST_8E,
        "7pi": CONST_7PI,
        "22": CONST_22
    },
    "datasets": {
        "all_data": {
            "n": len(all_data),
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "mae_8e": calc_mae(all_data, CONST_8E),
            "mae_7pi": calc_mae(all_data, CONST_7PI),
            "mae_22": calc_mae(all_data, CONST_22),
        },
        "core_text": {
            "n": len(core_text_models),
            "mean": float(np.mean(core_values)),
            "std": float(np.std(core_values)),
            "mae_8e": calc_mae(core_text_models, CONST_8E),
            "mae_7pi": calc_mae(core_text_models, CONST_7PI),
            "mae_22": calc_mae(core_text_models, CONST_22),
        },
        "q48_original": {
            "n": len(q48_data),
            "mean": float(np.mean(q48_values)),
            "std": float(np.std(q48_values)),
            "mae_8e": calc_mae(q48_data, CONST_8E),
            "mae_7pi": calc_mae(q48_data, CONST_7PI),
            "mae_22": calc_mae(q48_data, CONST_22),
        }
    },
    "raw_data": all_data
}

with open(Path(__file__).parent / "8e_vs_7pi_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to 8e_vs_7pi_results.json")
