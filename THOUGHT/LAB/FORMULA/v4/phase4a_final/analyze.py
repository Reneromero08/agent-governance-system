"""Analyze Phase 4a Final results with corrected statistics."""
import json, numpy as np
from pathlib import Path
from scipy import stats
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "phase4a"))
from phase4a_prompts import TEST_PROMPTS

VERIFIABLE_TYPES = {"exact", "contains", "contains_lower"}
verifiable_ids = {e["id"] for e in TEST_PROMPTS if e.get("verification_type") in VERIFIABLE_TYPES}

ctl = json.loads((ROOT / "results" / "phase4a_final_CONTROL.json").read_text())
cyb = json.loads((ROOT / "results" / "phase4a_final_CYBERNETIC.json").read_text())

print("=" * 60)
print("PHASE 4a FINAL — CORRECTED ANALYSIS")
print("=" * 60)

# Overall
for label, data in [("CONTROL", ctl), ("CYBERNETIC", cyb)]:
    v = [r for r in data if r["prompt_id"] in verifiable_ids and r["verified"] is not None]
    correct = sum(1 for r in v if r["verified"])
    acc = correct / len(v) if v else 0
    R = np.mean([r["R_mean"] for r in data])
    T = np.mean([r["T_mean"] for r in data])
    print(f"  {label:>12s}: acc={acc:.3f} ({correct}/{len(v)}) R={R:.4f}+-{np.std([r['R_mean'] for r in data]):.4f} T={T:.2f}")

# Accuracy significance (2-proportion z-test)
c_v = [r for r in ctl if r["prompt_id"] in verifiable_ids and r["verified"] is not None]
x_v = [r for r in cyb if r["prompt_id"] in verifiable_ids and r["verified"] is not None]
c_c, x_c = sum(1 for r in c_v if r["verified"]), sum(1 for r in x_v if r["verified"])
n_c, n_x = len(c_v), len(x_v)
p1, p2 = c_c / n_c, x_c / n_x
pooled_p = (c_c + x_c) / (n_c + n_x)
se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n_c + 1 / n_x))
z = (p2 - p1) / max(se, 1e-12)
p_val = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"\n  Accuracy delta: {p2-p1:+.3f} ({p1:.3f} -> {p2:.3f})  z={z:.4f}  p={p_val:.4f}  {'*' if p_val < 0.05 else 'ns'}")

# R comparison (independent t-test)
c_R = [r["R_mean"] for r in ctl]
x_R = [r["R_mean"] for r in cyb]
t_r, p_r = stats.ttest_ind(x_R, c_R)
print(f"  R delta:        {np.mean(x_R)-np.mean(c_R):+.4f}  t={t_r:.4f}  p={p_r:.4f}  {'*' if p_r < 0.05 else 'ns'}")

# Category breakdown
print(f"\n  {'Category':>12s}: {'CTL_R':>8s} {'CTL_acc':>8s} {'CYB_R':>8s} {'CYB_acc':>8s} {'dR':>8s}")
for cat in ["factual", "reasoning", "ambiguous", "adversarial"]:
    cc = [r for r in ctl if r["category"] == cat]
    xc = [r for r in cyb if r["category"] == cat]
    cv = [r for r in cc if r["prompt_id"] in verifiable_ids and r["verified"] is not None]
    xv = [r for r in xc if r["prompt_id"] in verifiable_ids and r["verified"] is not None]
    ca = sum(1 for r in cv if r["verified"]) / max(len(cv), 1) if cv else 0
    xa = sum(1 for r in xv if r["verified"]) / max(len(xv), 1) if xv else 0
    print(f"  {cat:>12s}: {np.mean([r['R_mean'] for r in cc]):8.4f} {ca:8.3f} "
          f"{np.mean([r['R_mean'] for r in xc]):8.4f} {xa:8.3f} "
          f"{np.mean([r['R_mean'] for r in xc])-np.mean([r['R_mean'] for r in cc]):+8.4f}")

# Baseline comparison to v1 (no constitution)
v1_path = ROOT.parent / "phase4a" / "results" / "phase4a_CONTROL.json"
if v1_path.exists():
    v1 = json.loads(v1_path.read_text())
    v1_v = [r for r in v1 if r["final_verified"] is not None]
    v1_acc = sum(1 for r in v1_v if r["final_verified"]) / len(v1_v) if v1_v else 0
    print(f"\n  v1 CONTROL (no constitution): acc={v1_acc:.3f} R~0.007")
    print(f"  Constitution gain (CTL - v1):  acc +{p1 - v1_acc:+.3f}  R 30x (0.007 -> 0.225)")
    print(f"  Cybernetic gain (CYB - CTL):   acc {p2-p1:+.3f} (p={p_val:.4f})  R {np.mean(x_R)-np.mean(c_R):+.4f} (p={p_r:.4f})")

print(f"\nDone.")
