#!/usr/bin/env python3
"""analyze_phase5_9c.py — Phase 5.9C analyzer: per-window geometry, flicker detection, artifact separation."""

import csv, json, math, sys, os, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--window-size', type=int, default=256)
    args = parser.parse_args()

    indir = args.input_dir
    raw_csv = os.path.join(indir, 'raw_cycles.csv')
    restore_csv = os.path.join(indir, 'restoration_integrity.csv')

    if not os.path.exists(raw_csv):
        print(f"SKIP: {raw_csv} not found")
        return

    cycles = []
    restores = []
    skipped_rows = 0
    with open(raw_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corrected = row.get('rdtsc_cycles_corrected')
            restore_ok = row.get('restore_ok')
            if corrected in (None, ''):
                skipped_rows += 1
                continue
            try:
                cycles.append(float(corrected))
                restores.append(int(restore_ok if restore_ok not in (None, '') else 1))
            except (ValueError, TypeError):
                skipped_rows += 1

    if len(cycles) < 64:
        print(f"SKIP: only {len(cycles)} trials")
        return

    cycles = np.array(cycles)
    restores = np.array(restores)
    n = len(cycles)
    wsize = min(args.window_size, max(n // 4, 16))

    n_windows = n // wsize
    run_id = os.path.basename(indir.rstrip('/'))

    # ── Per-window analysis ──────────────────────────────────
    window_data = []
    for wi in range(n_windows):
        wc = cycles[wi*wsize:(wi+1)*wsize]
        wr = restores[wi*wsize:(wi+1)*wsize]
        w_mean = float(np.mean(wc))
        w_std = float(np.std(wc))
        w_p50 = float(np.percentile(wc, 50))
        w_p99 = float(np.percentile(wc, 99))
        w_p999 = float(np.percentile(wc, 99.9)) if len(wc) >= 1000 else w_p99
        w_cv = w_std / max(w_mean, 1e-6)
        w_spike_rate = float(np.sum(np.abs(wc - w_mean) > 3 * w_std)) / max(len(wc), 1)
        w_restore_ok = int(np.sum(wr))
        w_restore_fail = len(wr) - w_restore_ok
        w_mismatch = w_restore_fail
        window_data.append({
            'window': wi, 'mean': w_mean, 'std': w_std,
            'p50': w_p50, 'p99': w_p99, 'p999': w_p999,
            'cv': w_cv, 'spike_rate': w_spike_rate,
            'restore_ok': w_restore_ok, 'restore_fail': w_restore_fail,
            'mismatch': w_mismatch, 'p99_p50': w_p99 / max(w_p50, 1e-6)
        })

    # ── Global geometry ──────────────────────────────────────
    window_means = [w['mean'] for w in window_data]
    window_stds = [w['std'] for w in window_data]

    # Boundary thickness
    if n_windows >= 2:
        wm = np.array(window_means)
        diffs = []
        for i in range(len(wm)):
            for j in range(i+1, len(wm)):
                diffs.append(abs(wm[i] - wm[j]))
        thickness = float(np.mean(diffs)) if diffs else 0.0
    else:
        thickness = 0.0

    centroid = float(np.mean(window_means))
    mean_radius = float(np.mean([abs(m - centroid) for m in window_means])) if window_means else 0.0

    # D_eff
    if n_windows >= 4:
        features = np.column_stack([window_means, window_stds])
        features = features - np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        try:
            eigenvals = np.linalg.eigvalsh(cov)
            eigenvals = eigenvals[eigenvals > 1e-10]
            if len(eigenvals) > 0:
                total = np.sum(eigenvals)
                pk = eigenvals / total
                h = -np.sum(pk * np.log(pk + 1e-12))
                d_eff = float(np.exp(h)) if h > 0 else 1.0
            else:
                d_eff = 1.0
        except Exception:
            d_eff = 1.0
    else:
        d_eff = 1.0

    # ── Artifact-separated geometry ──────────────────────────
    # Raw timing geometry: use raw cycles
    raw_means = []
    with open(raw_csv, 'r') as f:
        reader = csv.DictReader(f)
        raw_i = 0
        for row in reader:
            raw_value = row.get('rdtsc_cycles_raw')
            if raw_value in (None, ''):
                continue
            if raw_i % wsize == 0:
                raw_means.append([])
            try:
                raw_means[-1].append(float(raw_value))
                raw_i += 1
            except (ValueError, TypeError):
                continue

    raw_thickness = 0.0
    if len(raw_means) >= 2:
        rm = np.array([np.mean(w) for w in raw_means])
        rdiffs = [abs(rm[i] - rm[j]) for i in range(len(rm)) for j in range(i+1, len(rm))]
        raw_thickness = float(np.mean(rdiffs)) if rdiffs else 0.0

    # Spike-filtered geometry (remove top 1% outliers)
    p99_val = np.percentile(cycles, 99)
    spike_free = cycles[cycles <= p99_val]
    sf_windows = max(len(spike_free) // wsize, 1)
    sf_means = [float(np.mean(spike_free[i*wsize:min((i+1)*wsize, len(spike_free))]))
                for i in range(sf_windows)]
    sf_thickness = 0.0
    if len(sf_means) >= 2:
        sfm = np.array(sf_means)
        sfdiffs = [abs(sfm[i] - sfm[j]) for i in range(len(sfm)) for j in range(i+1, len(sfm))]
        sf_thickness = float(np.mean(sfdiffs)) if sfdiffs else 0.0

    # Stable-window geometry (low-spike windows)
    spike_rates = [w['spike_rate'] for w in window_data]
    median_sr = np.median(spike_rates) if spike_rates else 0
    stable_windows = [window_means[i] for i, sr in enumerate(spike_rates) if sr <= median_sr]
    stable_thickness = 0.0
    if len(stable_windows) >= 2:
        swm = np.array(stable_windows)
        swdiffs = [abs(swm[i] - swm[j]) for i in range(len(swm)) for j in range(i+1, len(swm))]
        stable_thickness = float(np.mean(swdiffs)) if swdiffs else 0.0

    # High-spike-window geometry
    high_spike = [window_means[i] for i, sr in enumerate(spike_rates) if sr > median_sr]
    hs_thickness = 0.0
    if len(high_spike) >= 2:
        hsm = np.array(high_spike)
        hsdiffs = [abs(hsm[i] - hsm[j]) for i in range(len(hsm)) for j in range(i+1, len(hsm))]
        hs_thickness = float(np.mean(hsdiffs)) if hsdiffs else 0.0

    # ── Restoration flicker per run ──────────────────────────
    flicker_mismatches = int(np.sum([w['mismatch'] for w in window_data]))
    flicker_detected = "YES" if flicker_mismatches > 0 else "NO"

    # ── Write geometry stats ─────────────────────────────────
    geo_csv = os.path.join(indir, 'geometry_stats.csv')
    with open(geo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'n_windows', 'boundary_thickness_nn_mean',
                         'mean_radius', 'D_eff', 'spectral_entropy',
                         'restoration_ok', 'restoration_failures',
                         'distance_to_failure',
                         'raw_thickness', 'spike_free_thickness',
                         'stable_thickness', 'high_spike_thickness',
                         'flicker_mismatches', 'flicker_detected',
                         'p99_p50_ratio', 'cycle_cv', 'spike_rate',
                         'skipped_malformed_rows'])
        p99 = float(np.percentile(cycles, 99))
        p50 = float(np.percentile(cycles, 50))
        p99p50 = p99 / max(p50, 1e-6)
        cv_global = float(np.std(cycles)) / max(float(np.mean(cycles)), 1e-6)
        spike_rate = float(np.sum(np.abs(cycles - np.mean(cycles)) > 3 * np.std(cycles))) / max(n, 1)
        total_restore_ok = int(np.sum(restores))
        total_restore_fail = n - total_restore_ok
        writer.writerow([run_id, n_windows, thickness, mean_radius, d_eff, 0.0,
                         total_restore_ok, total_restore_fail, 0.0,
                         raw_thickness, sf_thickness, stable_thickness, hs_thickness,
                         flicker_mismatches, flicker_detected,
                         p99p50, cv_global, spike_rate, skipped_rows])

    # ── Per-window CSV ───────────────────────────────────────
    win_csv = os.path.join(indir, 'window_boundary_geometry.csv')
    with open(win_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'window', 'mean_cycles', 'std_cycles',
                         'p50', 'p99', 'p999', 'cv', 'spike_rate',
                         'p99_p50', 'restore_ok', 'restore_fail'])
        for w in window_data:
            writer.writerow([run_id, w['window'], w['mean'], w['std'],
                             w['p50'], w['p99'], w['p999'], w['cv'], w['spike_rate'],
                             w['p99_p50'], w['restore_ok'], w['restore_fail']])

    print(f"  Geometry: thickness={thickness:.2f} raw={raw_thickness:.2f} sf={sf_thickness:.2f} stable={stable_thickness:.2f}")
    print(f"  Flicker: mismatches={flicker_mismatches} detected={flicker_detected}")
    print(f"  p99/p50={p99p50:.4f} cv={cv_global:.4f} spike_rate={spike_rate:.6f}")
    if skipped_rows:
        print(f"  Skipped malformed rows: {skipped_rows}")

if __name__ == '__main__':
    main()
