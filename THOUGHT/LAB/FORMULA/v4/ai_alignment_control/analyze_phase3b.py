import json, numpy as np
from scipy import stats
from pathlib import Path

d = json.loads((Path(__file__).resolve().parent / "results" / "phase3b_results.json").read_text())

for nl in ["LOW", "MED", "HIGH"]:
    nr = [r for r in d if r["noise"] == nl]
    prov = [r for r in nr if r["type"] == "proverb"]
    lit = [r for r in nr if r["type"] == "literal"]
    ctrl = [r for r in nr if r["type"] == "control"]

    print(f"\n=== {nl} ===")
    for label, group in [("Proverbs", prov), ("Literals", lit), ("Controls", ctrl)]:
        sims = [r["final_sim"] for r in group]
        print(f"  {label:>12s}: mean={np.mean(sims):.4f} std={np.std(sims):.4f}")

    # Paired t-test: proverb vs literal
    pair_diffs = []
    for rp in prov:
        pid = rp["id"]
        literal_id = "L" + pid[1:]
        rl = [r for r in lit if r["id"] == literal_id]
        if rl:
            pair_diffs.append(rp["final_sim"] - rl[0]["final_sim"])
    if pair_diffs:
        tstat, pval = stats.ttest_1samp(pair_diffs, 0)
        dval = np.mean(pair_diffs) / max(np.std(pair_diffs), 1e-12)
        print(f"  Proverb - Literal (paired): diff={np.mean(pair_diffs):+.4f} p={pval:.4f} d={dval:.2f}")

    # sigma*Df correlation
    products = [r["sigma"] * r["Df"] for r in nr if r["type"] != "control"]
    sims_p = [r["final_sim"] for r in nr if r["type"] != "control"]
    r_val, p_val = stats.pearsonr(products, sims_p)
    print(f"  Corr(sigma*Df, survival): r={r_val:+.4f} p={p_val:.4f}")

    # High vs low sigma*Df
    all_pairs = sorted([(r["sigma"] * r["Df"], r["final_sim"]) for r in nr if r["type"] != "control"])
    mid = len(all_pairs) // 2
    low = [x[1] for x in all_pairs[:mid]]
    high = [x[1] for x in all_pairs[mid:]]
    d_size = (np.mean(high) - np.mean(low)) / np.sqrt((np.var(high) + np.var(low)) / 2)
    print(f"  High vs Low sigma*Df: diff={np.mean(high)-np.mean(low):+.4f} d={d_size:.2f}")

# Overall
print(f"\n=== OVERALL ===")
prov_all = np.mean([r["final_sim"] for r in d if r["type"] == "proverb"])
lit_all = np.mean([r["final_sim"] for r in d if r["type"] == "literal"])
ctrl_all = np.mean([r["final_sim"] for r in d if r["type"] == "control"])
print(f" Proverbs: {prov_all:.4f}  Literals: {lit_all:.4f}  Controls: {ctrl_all:.4f}")
