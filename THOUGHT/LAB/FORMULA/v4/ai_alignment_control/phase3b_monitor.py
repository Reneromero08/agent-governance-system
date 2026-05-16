"""Phase 3b monitor — runs in its own terminal window. Shows live progress bar."""
import json, time, sys, math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
F = RESULTS / "phase3b_results.json"

print("Phase 3b Monitor — checking every 30 seconds...")
print()

try:
    while True:
        try:
            if F.exists():
                data = json.loads(F.read_text())
                n = len(data)
                pct = n / 90 * 100
                bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
                print(f"\r[{bar}] {n}/90 ({pct:.0f}%)", end="")
                if n >= 90:
                    print("\n\n=== PHASE 3b COMPLETE ===")
                    from scipy import stats
                    import numpy as np
                    
                    for nl in ["LOW", "MED", "HIGH"]:
                        nr = [r for r in data if r["noise"] == nl]
                        prov = [r for r in nr if r["type"] == "proverb"]
                        lit = [r for r in nr if r["type"] == "literal"]
                        ctrl = [r for r in nr if r["type"] == "control"]
                        
                        print(f"\n=== {nl} NOISE ===")
                        for label, group in [("Proverbs", prov), ("Literals", lit), ("Controls", ctrl)]:
                            sims = [r["final_sim"] for r in group]
                            print(f"  {label:>12s}: mean={np.mean(sims):.4f} std={np.std(sims):.4f}")
                        
                        pair_diffs = []
                        for rp in prov:
                            rl = [r for r in lit if r["id"] == f"L{rp['id'][1:]}"]
                            if rl: pair_diffs.append(rp["final_sim"] - rl[0]["final_sim"])
                        if pair_diffs:
                            t_stat, p_val = stats.ttest_1samp(pair_diffs, 0)
                            d = float(np.mean(pair_diffs)) / max(np.std(pair_diffs), 1e-12)
                            print(f"  Proverb - Literal (paired): diff={np.mean(pair_diffs):+.4f} p={p_val:.4f} d={d:.2f}")
                        
                        products = [r["sigma"] * r["Df"] for r in nr if r["type"] != "control"]
                        sims_p = [r["final_sim"] for r in nr if r["type"] != "control"]
                        if len(products) > 3:
                            r_val, p_val = stats.pearsonr(products, sims_p)
                            print(f"  Corr(sigma*Df, survival): r={r_val:+.4f} p={p_val:.4f}")
                            all_pairs = sorted(zip(products, sims_p))
                            mid = len(all_pairs)//2
                            low = [x[1] for x in all_pairs[:mid]]
                            high = [x[1] for x in all_pairs[mid:]]
                            d = (np.mean(high)-np.mean(low)) / math.sqrt((np.var(high)+np.var(low))/2)
                            print(f"  High vs Low sigma*Df: diff={np.mean(high)-np.mean(low):+.4f} d={d:.2f}")
                    
                    print("\nPress Enter to close this window...")
                    input()
                    break
            else:
                print(f"\rWaiting for results file... ({F})", end="")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"\rFile not ready, retrying...", end="")
        time.sleep(30)
except KeyboardInterrupt:
    print("\n Monitor stopped.")
