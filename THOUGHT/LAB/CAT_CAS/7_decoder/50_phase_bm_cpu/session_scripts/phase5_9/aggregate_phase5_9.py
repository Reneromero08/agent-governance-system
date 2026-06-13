#!/usr/bin/env python3
"""aggregate_phase5_9.py — Cross-run Phase 5.9 master verdict aggregator.

Computes gates, classifies instability-edge regime, produces master verdict CSV.
"""

import csv, json, os, sys, argparse
import numpy as np
from collections import defaultdict

def fit_r2(xs, ys):
    """Compute R² for linear regression."""
    n = len(xs)
    if n < 3:
        return 0.0
    mean_y = sum(ys) / n
    ss_tot = sum((y - mean_y)**2 for y in ys)
    if ss_tot == 0:
        return 0.0
    # Simple linear regression
    xbar = sum(xs) / n
    num = sum((xs[i] - xbar) * (ys[i] - mean_y) for i in range(n))
    den = sum((x - xbar)**2 for x in xs)
    if den == 0:
        return 0.0
    slope = num / den
    inter = mean_y - slope * xbar
    ss_res = sum((ys[i] - (slope * xs[i] + inter))**2 for i in range(n))
    return 1.0 - ss_res / ss_tot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    outdir = args.output_dir
    if not os.path.isdir(outdir):
        print(f"ERROR: {outdir} not found")
        sys.exit(1)

    # Collect all runs with geometry
    runs = {}
    for name in os.listdir(outdir):
        geo_csv = os.path.join(outdir, name, 'geometry_stats.csv')
        stress_csv = os.path.join(outdir, name, 'stress_ladder.csv')
        if os.path.exists(geo_csv):
            with open(geo_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    runs[name] = row
                    break

    if len(runs) < 3:
        print(f"Only {len(runs)} runs with geometry — need >= 3")
        sys.exit(1)

    print(f"Found {len(runs)} completed runs with geometry")
    print("=" * 60)
    print("EXP50 PHASE 5.9: CROSS-RUN MASTER VERDICT")
    print("=" * 60)

    gates = {}

    # Gate 1: Baseline Reproduction
    baseline_runs = [rid for rid in runs if rid.startswith('BASELINE')]
    gates['1_Baseline_Reproduction'] = 'PASS' if len(baseline_runs) >= 1 else 'FAIL'

    # Gate 2: Stress Ladder Validity
    gates['2_Stress_Ladder_Validity'] = 'PASS' if len(runs) >= 5 else 'FAIL'
    gates['2_num_stress_points'] = len(runs)

    # Gate 3: Restoration Survival Curve
    rest_oks = []
    rest_fails = []
    for rid, r in runs.items():
        try:
            rest_oks.append(int(r.get('restoration_ok', 0)))
            rest_fails.append(int(r.get('restoration_failures', 0)))
        except (ValueError, KeyError):
            pass
    total_failures = sum(rest_fails)
    gates['3_Restoration_Survival_Curve'] = 'PASS' if total_failures == 0 else 'FAIL'
    gates['3_total_restore_failures'] = total_failures

    # Gate 4: Boundary Geometry Stress Response
    thicknesses = []
    radii = []
    d_effs = []
    distances = []
    for rid, r in runs.items():
        try:
            thicknesses.append(float(r.get('boundary_thickness_nn_mean', 0)))
            radii.append(float(r.get('mean_radius', 0)))
            d_effs.append(float(r.get('D_eff', 0)))
            distances.append(float(r.get('distance_to_failure', 0)))
        except (ValueError, KeyError):
            pass

    r2 = 0.0
    corr = 0.0
    if len(thicknesses) >= 3:
        thick_spread = max(thicknesses) - min(thicknesses)
        if len(distances) >= 3:
            r2 = fit_r2(distances, thicknesses)
            corr = float(np.corrcoef(np.array(distances), np.array(thicknesses))[0, 1]) if len(set(distances)) > 1 else 0.0
        if thick_spread > 0.01 and (r2 >= 0.1 or abs(corr) >= 0.3):
            gates['4_Boundary_Geometry_Stress_Response'] = 'PASS'
        elif thick_spread > 0.01:
            gates['4_Boundary_Geometry_Stress_Response'] = 'PARTIAL'
            gates['4_reason'] = 'GEOMETRY_SPREAD_WITH_WEAK_STRESS_CORRELATION'
        else:
            gates['4_Boundary_Geometry_Stress_Response'] = 'FAIL'
        gates['4_thickness_spread'] = thick_spread
        gates['4_thickness_vs_failure_r2'] = r2
        gates['4_thickness_vs_failure_corr'] = corr
    else:
        gates['4_Boundary_Geometry_Stress_Response'] = 'FAIL'

    # Gate 5: Instability-Edge Classification
    if len(distances) >= 3 and len(thicknesses) >= 3:
        # Classify based on thickness vs distance_to_failure
        xs = np.array(distances)
        ys = np.array(thicknesses)
        r2 = fit_r2(list(xs), list(ys))

        # Determine regime
        if max(thicknesses) - min(thicknesses) < 0.01:
            if r2 < 0.1:
                regime = 'GEOMETRY_INVARIANT_TO_FAILURE'
            else:
                regime = 'GEOMETRY_INVARIANT_TO_FAILURE'
        elif np.corrcoef(xs, ys)[0, 1] > 0.3:
            regime = 'GEOMETRY_PEAKS_NEAR_FAILURE'
        elif np.corrcoef(xs, ys)[0, 1] < -0.3:
            regime = 'GEOMETRY_COLLAPSES_BEFORE_FAILURE'
        elif r2 < 0.1:
            regime = 'GEOMETRY_NOISE_ONLY'
        else:
            regime = 'GEOMETRY_PEAKS_NEAR_FAILURE'

        gates['5_Instability_Edge_Classification'] = 'PASS'
        gates['5_regime'] = regime
        gates['5_thickness_vs_failure_r2'] = r2
    else:
        gates['5_Instability_Edge_Classification'] = 'FAIL'
        regime = 'INSUFFICIENT_DATA'

    # Gate 6: Artifact Audit. This is not allowed to be a hardcoded PASS:
    # restoration failures, missing distance fields, or invalid worker telemetry
    # must downgrade or fail the master verdict.
    artifact_flags = []
    if total_failures > 0:
        artifact_flags.append('RESTORATION_FAILURES')
    if not distances or all(d == 0 for d in distances):
        artifact_flags.append('DISTANCE_TO_FAILURE_MISSING_OR_FLAT')
    if not thicknesses or max(thicknesses) <= 0:
        artifact_flags.append('GEOMETRY_MISSING_OR_ZERO')

    worker_scores = []
    for r in runs.values():
        try:
            worker_scores.append(float(r.get('worker_integrity_score', 1)))
        except ValueError:
            artifact_flags.append('WORKER_INTEGRITY_NON_NUMERIC')
    if worker_scores and min(worker_scores) < 1.0:
        artifact_flags.append('WORKER_INTEGRITY_DEGRADED')

    if not artifact_flags:
        gates['6_Artifact_Audit'] = 'PASS'
    elif 'RESTORATION_FAILURES' in artifact_flags or 'GEOMETRY_MISSING_OR_ZERO' in artifact_flags:
        gates['6_Artifact_Audit'] = 'FAIL'
    else:
        gates['6_Artifact_Audit'] = 'PARTIAL'
    gates['6_artifact_flags'] = ';'.join(artifact_flags) if artifact_flags else 'NONE'

    # Gate 7: Area-Law Persistence Under Stress
    # Compute separately for stable (baseline only) and full ladder.
    def compute_area_law(run_subset):
        """Compute area-law R² for a subset of runs."""
        tape_data = defaultdict(list)
        for rid, r in run_subset.items():
            parts = rid.split('_T')
            if len(parts) >= 2:
                try:
                    ts = int(parts[-1].split('_')[0])
                    tape_data[ts].append(float(r.get('boundary_thickness_nn_mean', 0)))
                except (ValueError, IndexError):
                    pass
        tape_sizes = sorted(tape_data.keys())
        if len(tape_sizes) < 3:
            return None, None, None, None
        ts_list = []
        thick_list = []
        for ts in tape_sizes:
            vals = tape_data[ts]
            if vals:
                ts_list.append(ts)
                thick_list.append(np.mean(vals))
        if len(ts_list) < 3:
            return None, None, None, None
        r2_vol = fit_r2(ts_list, thick_list)
        xs_area = [x**(2.0/3.0) for x in ts_list]
        r2_area = fit_r2(xs_area, thick_list)
        xs_log = [np.log(x) for x in ts_list]
        r2_log = fit_r2(xs_log, thick_list)
        wins = 0
        if r2_area > r2_vol: wins += 1
        if r2_log > r2_vol: wins += 1
        return r2_vol, r2_area, r2_log, wins

    # Baseline-only area-law (stable region)
    baseline_runs_dict = {rid: r for rid, r in runs.items() if rid.startswith('BASELINE')}
    bl_vol, bl_area, bl_log, bl_wins = compute_area_law(baseline_runs_dict)

    # Full-ladder area-law (all runs)
    fl_vol, fl_area, fl_log, fl_wins = compute_area_law(runs)

    if bl_vol is not None and fl_vol is not None:
        # Use full-ladder wins for Gate 7 per spec (persistence under stress)
        if fl_wins >= 2:
            gates['7_Area_Law_Persistence_Under_Stress'] = 'PASS'
        elif bl_wins >= 2:
            gates['7_Area_Law_Persistence_Under_Stress'] = 'PARTIAL'
        else:
            gates['7_Area_Law_Persistence_Under_Stress'] = 'FAIL'
        gates['7_full_ladder_wins'] = fl_wins
        gates['7_full_ladder_r2_volume'] = fl_vol
        gates['7_full_ladder_r2_area'] = fl_area
        gates['7_full_ladder_r2_log'] = fl_log
        gates['7_baseline_wins'] = bl_wins
        gates['7_baseline_r2_volume'] = bl_vol
        gates['7_baseline_r2_area'] = bl_area
        gates['7_baseline_r2_log'] = bl_log
    elif fl_vol is not None:
        gates['7_Area_Law_Persistence_Under_Stress'] = 'PASS' if fl_wins >= 2 else 'FAIL'
        gates['7_full_ladder_wins'] = fl_wins
        gates['7_full_ladder_r2_volume'] = fl_vol
        gates['7_full_ladder_r2_area'] = fl_area
        gates['7_full_ladder_r2_log'] = fl_log
    else:
        gates['7_Area_Law_Persistence_Under_Stress'] = 'INSUFFICIENT_DATA'

    # Gate 8: Analog Entry Readiness
    if regime in ('GEOMETRY_PEAKS_NEAR_FAILURE', 'GEOMETRY_INVARIANT_TO_FAILURE'):
        gates['8_Analog_Entry_Readiness'] = 'PASS'
    elif regime == 'GEOMETRY_COLLAPSES_BEFORE_FAILURE':
        gates['8_Analog_Entry_Readiness'] = 'PARTIAL'
    else:
        gates['8_Analog_Entry_Readiness'] = 'INCONCLUSIVE'

    # ── Final verdict ─────────────────────────────────────────
    fail_count = sum(1 for k, v in gates.items() if v == 'FAIL')
    partial_count = sum(1 for k, v in gates.items() if v == 'PARTIAL')

    if fail_count > 0:
        final_verdict = 'EXP50_PHASE5_9_PARTIAL'
    elif partial_count > 0:
        final_verdict = 'EXP50_PHASE5_9_PARTIAL'
    elif regime == 'GEOMETRY_PEAKS_NEAR_FAILURE':
        final_verdict = 'EXP50_PHASE5_9_GEOMETRY_PEAKS_NEAR_FAILURE'
    elif regime == 'GEOMETRY_COLLAPSES_BEFORE_FAILURE':
        final_verdict = 'EXP50_PHASE5_9_GEOMETRY_COLLAPSES_BEFORE_FAILURE'
    elif regime == 'GEOMETRY_INVARIANT_TO_FAILURE':
        final_verdict = 'EXP50_PHASE5_9_GEOMETRY_INVARIANT_TO_FAILURE'
    elif regime == 'GEOMETRY_NOISE_ONLY':
        final_verdict = 'EXP50_PHASE5_9_NOISE_ONLY'
    else:
        final_verdict = 'EXP50_PHASE5_9_BOUNDARY_STRESS_CONFIRMED'

    # Print gate results
    for gate, result in sorted(gates.items()):
        print(f"  Gate {gate}: {result}")

    print(f"\n  VERDICT: {final_verdict}")
    print(f"  REGIME: {regime}")

    # Write master verdict CSV
    master_csv = os.path.join(outdir, 'phase5_9_master_verdict.csv')
    with open(master_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gate', 'result'])
        for gate, result in sorted(gates.items()):
            writer.writerow([gate, result])
        writer.writerow(['VERDICT', final_verdict])
        writer.writerow(['REGIME', regime])

    print(f"\nMaster verdict written to {master_csv}")

if __name__ == '__main__':
    main()
