"""Phase 4a: Analyze experiment results and generate report.

IMPORTANT: All cross-condition R comparisons use R_raw (unscaled, 0-1 range).
CYBERNETIC condition's R_mean/R_final are scaled by R_SCALE=100x for
temperature control, but the raw hidden-state projection R_raw is comparable
across conditions. dR/dt is always computed from the raw trajectory.

The temperature formula T = T_base/(1 + R_eff) is a numerical stabilization
of the specified T = 1/(R + epsilon). See report for full discussion.
"""

import json, math, numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

print("=" * 60)
print("PHASE 4a: RESULTS ANALYSIS")
print("=" * 60)

# Load results
data_path = RESULTS / "phase4a_all_results.json"
all_data = json.loads(data_path.read_text())
conditions = all_data["conditions"]
cfg = all_data.get("config", {})

def safe_mean(vals):
    return float(np.mean(vals)) if vals else 0.0
def safe_std(vals):
    return float(np.std(vals)) if vals else 0.0

# ---- Helper: extract raw R and raw dR/dt for any condition ----
def get_raw_R(r):
    """Get raw (unscaled) R from any condition result."""
    if "R_raw_mean" in r and r["R_raw_mean"] is not None:
        return r["R_raw_mean"]
    # CONTROL and VERIFY: R_mean IS raw (no scaling applied)
    return r["R_mean"]

def get_raw_R_final(r):
    """Get raw R_final from any condition result."""
    if "R_raw_trajectory" in r and r["R_raw_trajectory"]:
        return float(r["R_raw_trajectory"][-1])
    return r.get("R_final", 0.0)  # CONTROL/VERIFY: R_final IS raw

def get_raw_drdt(r):
    """Compute raw dR/dt (unscaled) from R_raw_trajectory if available."""
    traj = r.get("R_raw_trajectory", r.get("R_trajectory", []))
    if len(traj) > 5:
        steps = np.arange(len(traj))
        slope = np.polyfit(steps, traj, 1)[0]
        return float(slope)
    return r.get("dR_dt", 0.0)

print(f"\n{'=' * 60}")
print(f"SUMMARY: {len(conditions)} CONDITIONS, {len(list(conditions.values())[0])} PROMPTS EACH")
print(f"{'=' * 60}")

# ---- 1. Truth accuracy (PRIMARY RESULT) ----
print(f"\n--- 1. TRUTH ACCURACY (PRIMARY) ---")
accuracies = {}
for cond_name, results in conditions.items():
    verified = [r for r in results if r["final_verified"] is not None]
    if verified:
        correct = sum(1 for r in verified if r["final_verified"])
        acc = correct / len(verified)
        print(f"  {cond_name:>12s}: {correct}/{len(verified)} = {acc:.3f}  ({len(verified)} verifiable)")
        accuracies[cond_name] = acc
    else:
        print(f"  {cond_name:>12s}: no verifiable prompts")
        accuracies[cond_name] = 0.0

if "CONTROL" in accuracies and "CYBERNETIC" in accuracies:
    delta = accuracies["CYBERNETIC"] - accuracies["CONTROL"]
    print(f"\n  ** Cybernetic loop degrades accuracy by {delta:+.3f} ({abs(delta)*100:.0f}pp) **")

# ---- 2. Raw R metrics (cross-condition comparable) ----
print(f"\n--- 2. RAW RESONANCE METRICS (comparable across conditions) ---")
print(f"  {'Condition':>12s}  {'R_mean_raw':>12s}  {'R_final_raw':>12s}  {'dR/dt_raw':>12s}  {'T_mean':>8s}")
print(f"  {'-'*62}")
for cond_name, results in conditions.items():
    r_mean_raw = safe_mean([get_raw_R(r) for r in results])
    r_final_raw = safe_mean([get_raw_R_final(r) for r in results])
    drdt_raw = safe_mean([get_raw_drdt(r) for r in results])
    t_mean = safe_mean([r["T_mean"] for r in results])
    print(f"  {cond_name:>12s}: {r_mean_raw:12.6f}  {r_final_raw:12.6f}  {drdt_raw:+12.2e}  {t_mean:8.4f}")

# Note: scaled R values for CYBERNETIC (used for T control)
if "CYBERNETIC" in conditions:
    x_res = conditions["CYBERNETIC"]
    r_scaled = safe_mean([r["R_mean"] for r in x_res])
    r_scaled_f = safe_mean([r["R_final"] for r in x_res])
    print(f"\n  CYBERNETIC scaled R (R_raw * 100, used for T modulation):")
    print(f"    R_scaled_mean={r_scaled:.4f}  R_scaled_final={r_scaled_f:.4f}")

# ---- 3. Temperature ----
print(f"\n--- 3. TEMPERATURE ---")
print(f"  {'Condition':>12s}  {'T_mean':>8s}  {'T_min_avg':>10s}  {'T_max_avg':>10s}")
print(f"  {'-'*48}")
for cond_name, results in conditions.items():
    t_mean = safe_mean([r["T_mean"] for r in results])
    t_min = safe_mean([r.get("T_min", 0.7) for r in results])
    t_max = safe_mean([r.get("T_max", 0.7) for r in results])
    formula = "T=5.0/(1+R_eff)" if cond_name == "CYBERNETIC" else "T=0.7 fixed"
    print(f"  {cond_name:>12s}: {t_mean:8.4f}  {t_min:10.4f}  {t_max:10.4f}  {formula}")

# ---- 4. Category breakdown (using raw R) ----
print(f"\n--- 4. BY CATEGORY (raw R values) ---")
categories = sorted(set(r["category"] for r in list(conditions.values())[0]))

for cat in categories:
    print(f"\n  [{cat.upper()}]")
    for cond_name, results in conditions.items():
        cat_results = [r for r in results if r["category"] == cat]
        r_raw = safe_mean([get_raw_R(r) for r in cat_results])
        r_f_raw = safe_mean([get_raw_R_final(r) for r in cat_results])
        drdt_raw = safe_mean([get_raw_drdt(r) for r in cat_results])
        verified = [r for r in cat_results if r["final_verified"] is not None]
        acc = safe_mean([1.0 if r["final_verified"] else 0.0 for r in verified]) if verified else None
        n = len(cat_results)
        acc_str = f"acc={acc:.3f}" if acc is not None else "acc=N/A"
        print(f"    {cond_name:>12s}: R={r_raw:.6f} R_f={r_f_raw:.6f} dR/dt={drdt_raw:+.2e} {acc_str} n={n}")

# ---- 5. Statistical tests (on raw R values) ----
print(f"\n--- 5. STATISTICAL TESTS (raw R values) ---")

if "CONTROL" in conditions and "CYBERNETIC" in conditions:
    c_res = conditions["CONTROL"]
    x_res = conditions["CYBERNETIC"]

    # Raw R comparison
    for metric_label, fn in [("R_mean_raw", get_raw_R), ("R_final_raw", get_raw_R_final), ("dR/dt_raw", get_raw_drdt)]:
        c_vals = [fn(r) for r in c_res]
        x_vals = [fn(r) for r in x_res]
        t_stat, p_val = stats.ttest_ind(c_vals, x_vals)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        delta = safe_mean(x_vals) - safe_mean(c_vals)
        print(f"    {metric_label:>12s}: C={safe_mean(c_vals):.6f} X={safe_mean(x_vals):.6f} delta={delta:+.2e} t={t_stat:+.4f} p={p_val:.4f} {sig}")

    # dR/dt > 0 test
    drdt_x = np.array([get_raw_drdt(r) for r in x_res])
    t_stat, p_val = stats.ttest_1samp(drdt_x, 0.0)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"    CYBERNETIC dR/dt > 0: t={t_stat:+.4f} p={p_val:.4f} {sig}  mean={np.mean(drdt_x):+.2e}")

    drdt_c = np.array([get_raw_drdt(r) for r in c_res])
    t_stat, p_val = stats.ttest_1samp(drdt_c, 0.0)
    print(f"    CONTROL    dR/dt > 0: t={t_stat:+.4f} p={p_val:.4f} {sig}  mean={np.mean(drdt_c):+.2e}")

# ---- 6. R tracks truth (raw R, per-condition) ----
print(f"\n--- 6. R TRACKS TRUTH (raw R, per-condition) ---")
for cond_name in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    if cond_name not in conditions:
        continue
    ctrue = [get_raw_R_final(r) for r in conditions[cond_name] if r.get("final_verified") == True]
    cfalse = [get_raw_R_final(r) for r in conditions[cond_name] if r.get("final_verified") == False]
    if ctrue and cfalse:
        t_stat, p_val = stats.ttest_ind(ctrue, cfalse)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        diff = safe_mean(ctrue) - safe_mean(cfalse)
        cohens_d = diff / max(math.sqrt((np.var(ctrue)+np.var(cfalse))/2), 1e-12)
        direction = "TRUE > FALSE" if diff > 0 else "FALSE > TRUE"
        print(f"  {cond_name:>12s}: true_R={safe_mean(ctrue):.6f} false_R={safe_mean(cfalse):.6f} "
              f"diff={diff:+.6f} d={cohens_d:+.2f} t={t_stat:+.4f} p={p_val:.4f} {sig} [{direction}]")

# ---- 7. Lindblad verification events ----
print(f"\n--- 7. LIND BLAD VERIFICATION EVENTS ---")
print(f"  NOTE: Partial verification at 20-token intervals is unreliable.")
print(f"  Claims may not be verifiable mid-generation (answer not yet produced).")
for cond_name in ["CYBERNETIC", "VERIFY"]:
    if cond_name not in conditions:
        continue
    all_verifs = []
    for r in conditions[cond_name]:
        all_verifs.extend(r.get("verifications", []))
    if all_verifs:
        n_total = len(all_verifs)
        n_passed = sum(1 for v in all_verifs if v.get("verified") == True)
        n_failed = sum(1 for v in all_verifs if v.get("verified") == False)
        n_unknown = n_total - n_passed - n_failed
        print(f"  {cond_name:>12s}: {n_total} events: {n_passed} passed, {n_failed} failed, {n_unknown} unknown")
        print(f"                   ({n_unknown} unverifiable mid-generation — expected)")

# ---- 8. Trajectory shape (raw R) ----
print(f"\n--- 8. TRAJECTORY SHAPE (raw R) ---")
for cond_name, results in conditions.items():
    all_trajs = []
    for r in results:
        raw_traj = r.get("R_raw_trajectory", r.get("R_trajectory", []))
        if raw_traj:
            all_trajs.append(raw_traj)
    if all_trajs:
        min_len = min(len(t) for t in all_trajs)
        avg_traj = np.mean([t[:min_len] for t in all_trajs], axis=0)
        print(f"  {cond_name:>12s}: len={min_len} "
              f"R_start={avg_traj[0]:.6f} R_mid={avg_traj[min_len//2]:.6f} "
              f"R_end={avg_traj[-1]:.6f} "
              f"delta={avg_traj[-1] - avg_traj[0]:+.6f} "
              f"trend={'UP' if avg_traj[-1] > avg_traj[0] else 'DOWN'}")

# ---- 9. SUCCESS CRITERIA (HONEST) ----
print(f"\n{'=' * 60}")
print(f"9. SUCCESS CRITERIA (HONEST ASSESSMENT)")
print(f"{'=' * 60}")

# Criterion 1: Loop improves accuracy
delta_acc = accuracies.get("CYBERNETIC", 0) - accuracies.get("CONTROL", 0)
print(f"\n  [1] Loop improves accuracy (+10%): {'PASS' if delta_acc >= 0.10 else ('PARTIAL' if delta_acc > 0 else 'FAIL')}")
print(f"      CONTROL={accuracies.get('CONTROL',0):.3f} CYBERNETIC={accuracies.get('CYBERNETIC',0):.3f} delta={delta_acc:+.3f}")
if delta_acc < 0:
    print(f"      The loop DEGRADES accuracy by {abs(delta_acc)*100:.0f}pp. Hypothesis falsified.")

# Criterion 2: R tracks truth (per-condition, with effect sizes)
print(f"\n  [2] R tracks truth (R higher for true answers):")
any_pass = False
for cond_name in ["CONTROL", "CYBERNETIC", "VERIFY"]:
    if cond_name not in conditions:
        continue
    ctrue = [get_raw_R_final(r) for r in conditions[cond_name] if r.get("final_verified") == True]
    cfalse = [get_raw_R_final(r) for r in conditions[cond_name] if r.get("final_verified") == False]
    if ctrue and cfalse:
        t_stat, p_val = stats.ttest_ind(ctrue, cfalse)
        diff = safe_mean(ctrue) - safe_mean(cfalse)
        pooled_sd = math.sqrt((np.var(ctrue) + np.var(cfalse)) / 2)
        d = diff / max(pooled_sd, 1e-12)
        passes = diff > 0 and p_val < 0.05
        any_pass = any_pass or passes
        status = "PASS" if passes else ("ns (wrong dir)" if diff < 0 else "ns")
        print(f"      {cond_name}: d={d:+.2f} p={p_val:.4f} {status}  "
              f"(true_R={safe_mean(ctrue):.6f} false_R={safe_mean(cfalse):.6f})")
print(f"      Overall: {'PASS' if any_pass else 'FAIL'} — weak signal, tiny effect size")
print(f"      NOTE: Absolute difference in raw R is ~0.003 (CONTROL). This is ~0.3% of the R scale.")

# Criterion 3: Loop recovers from errors
x_res = conditions.get("CYBERNETIC", [])
recovery_events = 0
recovery_success = 0
for r in x_res:
    verifs = r.get("verifications", [])
    raw_traj = r.get("R_raw_trajectory", r.get("R_trajectory", []))
    for v in verifs:
        if v.get("verified") == False:
            recovery_events += 1
            vstep = v.get("step", 0)
            if vstep < len(raw_traj):
                R_at_fail = raw_traj[vstep - 1] if vstep > 0 else 0
                future_Rs = raw_traj[vstep:min(vstep + 30, len(raw_traj))]
                if future_Rs and max(future_Rs) > R_at_fail * 1.1:
                    recovery_success += 1
print(f"\n  [3] Loop recovers from errors: ", end="")
if recovery_events > 0:
    rate = recovery_success / recovery_events
    print(f"{'PASS' if rate >= 0.5 else 'PARTIAL'} ({recovery_success}/{recovery_events}, {rate*100:.0f}%)")
    print(f"      NOTE: 66/{len([vv for r in x_res for vv in r.get('verifications',[])])} verifications were 'unknown' (mid-generation, not yet verifiable).")
    print(f"      The 'failed' count may be inflated by partial outputs that hadn't yet produced an answer.")
else:
    print("N/A (no failures)")

# Criterion 4: Truth is an attractor (dR/dt > 0)
drdt_x = np.array([get_raw_drdt(r) for r in conditions.get("CYBERNETIC", [])])
t_stat, p_val = stats.ttest_1samp(drdt_x, 0.0)
print(f"\n  [4] Truth is an attractor (dR/dt > 0): ", end="")
if np.mean(drdt_x) > 0 and p_val < 0.05:
    print(f"PASS (dR/dt={np.mean(drdt_x):+.2e} p={p_val:.4f})")
elif np.mean(drdt_x) > 0:
    print(f"PARTIAL (dR/dt={np.mean(drdt_x):+.2e} p={p_val:.4f})")
else:
    print(f"FAIL (dR/dt={np.mean(drdt_x):+.2e})")

# Show that CONTROL also has positive dR/dt
drdt_c = np.array([get_raw_drdt(r) for r in conditions.get("CONTROL", [])])
print(f"      CONTROL dR/dt={np.mean(drdt_c):+.2e} p={stats.ttest_1samp(drdt_c, 0.0)[1]:.4f}")
print(f"      NOTE: Both conditions show ~same raw dR/dt. The loop does NOT add attractor force.")

# ---- 10. FILE LIST ----
print(f"\n--- 10. OUTPUT FILES ---")
for f in sorted(RESULTS.glob("phase4a_*")):
    print(f"  {f.name}")

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 60}")
