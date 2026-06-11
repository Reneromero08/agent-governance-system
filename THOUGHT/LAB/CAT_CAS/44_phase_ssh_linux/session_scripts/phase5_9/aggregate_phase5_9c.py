#!/usr/bin/env python3
"""aggregate_phase5_9c.py — Phase 5.9C 9-gate master verdict aggregator."""

import csv, json, os, sys, argparse
import numpy as np
from collections import defaultdict

def fit_r2(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mean_y = sum(ys) / n
    ss_tot = sum((y - mean_y)**2 for y in ys)
    if ss_tot == 0: return 0.0
    xbar = sum(xs) / n
    num = sum((xs[i] - xbar) * (ys[i] - mean_y) for i in range(n))
    den = sum((x - xbar)**2 for x in xs)
    if den == 0: return 0.0
    slope = num / den
    inter = mean_y - slope * xbar
    ss_res = sum((ys[i] - (slope * xs[i] + inter))**2 for i in range(n))
    return 1.0 - ss_res / ss_tot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--audit-dir', default=None)
    args = parser.parse_args()

    outdir = args.output_dir
    if not os.path.isdir(outdir):
        print(f"ERROR: {outdir} not found")
        sys.exit(1)

    # Collect runs
    runs = {}
    for name in os.listdir(outdir):
        geo_csv = os.path.join(outdir, name, 'geometry_stats.csv')
        if os.path.exists(geo_csv):
            with open(geo_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    runs[name] = row
                    break

    if len(runs) < 5:
        print(f"Only {len(runs)} runs — need >= 5")
        sys.exit(1)

    print(f"Found {len(runs)} completed runs")
    print("=" * 60)
    print("EXP44 PHASE 5.9C: CROSS-RUN MASTER VERDICT")
    print("=" * 60)

    gates = {}
    n_runs = len(runs)
    gates['total_runs'] = n_runs

    # Gate 1: Baseline Reproduction
    bl_runs = [r for rid, r in runs.items() if 'BASELINE' in rid or 'S0_BASELINE' in rid]
    flicker_oks = sum(1 for r in bl_runs if r.get('flicker_detected', 'YES') == 'NO')
    gates['1_Baseline_Reproduction'] = 'PASS' if len(bl_runs) >= 1 and flicker_oks == len(bl_runs) else 'FAIL'

    # Gate 2: Effective Frequency Audit
    eff_audit = args.audit_dir
    freq_ok = 0
    if eff_audit and os.path.exists(os.path.join(eff_audit, 'frequency_effective_audit.csv')):
        with open(os.path.join(eff_audit, 'frequency_effective_audit.csv')) as f:
            for row in csv.DictReader(f):
                if row.get('effective_change', 'NO') == 'YES':
                    freq_ok += 1
    gates['2_Effective_Frequency_Audit'] = 'PASS' if freq_ok >= 3 else ('PARTIAL' if freq_ok >= 1 else 'FAIL')
    gates['2_effective_pstates'] = freq_ok

    # Gate 3: All-Core Control
    all_core_ok = 0
    if eff_audit and os.path.exists(os.path.join(eff_audit, 'per_core_pstate_audit.csv')):
        with open(os.path.join(eff_audit, 'per_core_pstate_audit.csv')) as f:
            for row in csv.DictReader(f):
                if row.get('control_ok', 'NO') == 'YES':
                    all_core_ok += 1
    gates['3_All_Core_Control'] = 'PASS' if all_core_ok >= 3 else ('PARTIAL' if all_core_ok >= 1 else 'FAIL')
    gates['3_cores_controlled'] = all_core_ok

    # Gate 4: Monotonic Stress Ladder
    gates['4_Monotonic_Stress_Ladder'] = 'PASS' if n_runs >= 10 else ('PARTIAL' if n_runs >= 5 else 'FAIL')
    gates['4_stress_points'] = n_runs

    # Gate 5: Long-Duration Edge Search
    long_runs = [r for rid, r in runs.items() if 'LONG_DURATION' in rid]
    long_flicker = any(r.get('flicker_detected', 'NO') == 'YES' for r in long_runs)
    gates['5_Long_Duration_Edge_Search'] = 'PARTIAL' if len(long_runs) >= 1 and not long_flicker else ('PASS' if long_flicker else 'FAIL')
    gates['5_long_runs'] = len(long_runs)

    # Gate 6: Restoration Flicker Search
    flicker_total = sum(1 for r in runs.values() if r.get('flicker_detected', 'NO') == 'YES')
    gates['6_Restoration_Flicker_Search'] = 'PARTIAL' if flicker_total == 0 else 'PASS'
    gates['6_flicker_runs'] = flicker_total

    edge_reached = flicker_total > 0 or any(float(r.get('p99_p50_ratio', 0)) > 5.0 for r in runs.values())

    # Gate 7: Boundary vs failure-adjacent timing response. A timing-CV/spike
    # correlation is not a direct failure-boundary response unless an edge was
    # actually approached.
    thicknesses = [float(r.get('boundary_thickness_nn_mean', 0)) for r in runs.values()]
    spike_rates = [float(r.get('spike_rate', 0)) for r in runs.values()]
    cv_vals = [float(r.get('cycle_cv', 0)) for r in runs.values()]
    p99p50 = [float(r.get('p99_p50_ratio', 0)) for r in runs.values()]

    if len(thicknesses) >= 5:
        r_cv = np.corrcoef(cv_vals, thicknesses)[0,1] if len(set(cv_vals)) > 1 else 0
        r_spike = np.corrcoef(spike_rates, thicknesses)[0,1] if len(set(spike_rates)) > 1 else 0
        timing_response = abs(r_cv) >= 0.3 or abs(r_spike) >= 0.3
        if timing_response and edge_reached:
            gates['7_Boundary_vs_Timing_Response'] = 'PASS'
        elif timing_response:
            gates['7_Boundary_vs_Timing_Response'] = 'PARTIAL'
            gates['7_reason'] = 'TIMING_RESPONSE_ONLY_EDGE_NOT_REACHED'
        else:
            gates['7_Boundary_vs_Timing_Response'] = 'PARTIAL'
            gates['7_reason'] = 'NO_STRONG_TIMING_OR_SPIKE_CORRELATION'
        gates['7_r_thickness_vs_cv'] = r_cv
        gates['7_r_thickness_vs_spike'] = r_spike
    else:
        gates['7_Boundary_vs_Timing_Response'] = 'INCONCLUSIVE'

    # Gate 8: Artifact-Separated Geometry
    has_raw = all('raw_thickness' in r for r in runs.values())
    has_sf = all('spike_free_thickness' in r for r in runs.values())
    has_stable = all('stable_thickness' in r for r in runs.values())
    if has_raw and has_sf and has_stable:
        raw_vals = [float(r.get('raw_thickness', 0)) for r in runs.values()]
        sf_vals = [float(r.get('spike_free_thickness', 0)) for r in runs.values()]
        stable_vals = [float(r.get('stable_thickness', 0)) for r in runs.values()]
        r_raw_sf = np.corrcoef(raw_vals, sf_vals)[0,1] if len(set(raw_vals)) > 1 and len(set(sf_vals)) > 1 else 0
        stable_spread = max(stable_vals) - min(stable_vals)
        if abs(r_raw_sf) >= 0.8 and stable_spread > 0:
            gates['8_Artifact_Separated_Geometry'] = 'PASS'
        else:
            gates['8_Artifact_Separated_Geometry'] = 'PARTIAL'
            gates['8_reason'] = 'RAW_SPIKE_FREE_STABLE_CHANNELS_PRESENT_BUT_WEAKLY_SEPARATED'
        gates['8_r_raw_vs_sf'] = r_raw_sf
        gates['8_stable_spread'] = stable_spread
    else:
        gates['8_Artifact_Separated_Geometry'] = 'PARTIAL'

    # Gate 9: Final Boundary Classification
    # Determine if edge was approached
    platform_hard = all_core_ok < 2 and freq_ok < 3

    if edge_reached:
        gates['9_Final_Classification'] = 'EDGE_APPROACHED'
        final_verdict = 'EXP44_PHASE5_9C_EDGE_APPROACHED'
    elif platform_hard:
        gates['9_Final_Classification'] = 'PLATFORM_HARD_BOUNDARY'
        final_verdict = 'EXP44_PHASE5_9C_PLATFORM_HARD_BOUNDARY'
    else:
        gates['9_Final_Classification'] = 'INSTABILITY_EDGE_NOT_REACHED'
        final_verdict = 'EXP44_PHASE5_9C_INSTABILITY_EDGE_NOT_REACHED'

    gates['9_edge_reached'] = 'YES' if edge_reached else 'NO'
    gates['9_platform_hard'] = 'YES' if platform_hard else 'NO'

    # Print
    for gate, result in sorted(gates.items()):
        print(f"  Gate {gate}: {result}")

    print(f"\n  VERDICT: {final_verdict}")

    # Write master CSV
    master_csv = os.path.join(outdir, 'phase5_9c_master_verdict.csv')
    with open(master_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gate', 'result'])
        for gate, result in sorted(gates.items()):
            writer.writerow([gate, result])
        writer.writerow(['VERDICT', final_verdict])

    print(f"\nMaster verdict written to {master_csv}")

if __name__ == '__main__':
    main()
