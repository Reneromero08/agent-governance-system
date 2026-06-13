#!/usr/bin/env python3
"""analyze_phase5_9.py — Per-run boundary geometry analyzer for Phase 5.9."""

import csv, json, math, sys, os, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--window-size', type=int, default=256)
    args = parser.parse_args()

    indir = args.input_dir
    raw_csv = os.path.join(indir, 'raw_cycles.csv')
    stress_csv = os.path.join(indir, 'stress_ladder.csv')

    if not os.path.exists(raw_csv):
        print(f"SKIP: {raw_csv} not found")
        return

    # Read raw cycles
    cycles = []
    with open(raw_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cycles.append(float(row.get('rdtsc_cycles_corrected', 0)))
            except (ValueError, KeyError):
                cycles.append(0.0)

    if len(cycles) < 64:
        print(f"SKIP: only {len(cycles)} trials (< 64)")
        return

    cycles = np.array(cycles)
    n = len(cycles)
    wsize = min(args.window_size, n // 4)
    if wsize < 16: wsize = 16

    # Windowed features
    n_windows = n // wsize
    window_means = []
    window_stds = []
    for i in range(n_windows):
        w = cycles[i*wsize:(i+1)*wsize]
        window_means.append(float(np.mean(w)))
        window_stds.append(float(np.std(w)))

    # Boundary thickness: NN mean distance between windows
    if n_windows >= 2:
        wm = np.array(window_means)
        diffs = []
        for i in range(len(wm)):
            for j in range(i+1, len(wm)):
                diffs.append(abs(wm[i] - wm[j]))
        thickness = float(np.mean(diffs)) if diffs else 0.0
    else:
        thickness = 0.0

    # Mean radius: mean distance from centroid
    centroid = float(np.mean(window_means))
    radii = [abs(m - centroid) for m in window_means]
    mean_radius = float(np.mean(radii)) if radii else 0.0

    # D_eff via PCA on window features
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

    # Spectral entropy
    if n_windows >= 4:
        spectral = np.fft.fft(np.array(window_means) - np.mean(window_means))
        power = np.abs(spectral[:n_windows//2]) ** 2
        power = power[power > 1e-12]
        if len(power) > 0:
            total_p = np.sum(power)
            pk_fft = power / total_p
            spectral_entropy = float(-np.sum(pk_fft * np.log(pk_fft + 1e-12)))
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0

    # Restoration from stress_ladder.csv
    restoration_ok = 0
    restoration_failures = 0
    distance_to_failure = 0.0
    if os.path.exists(stress_csv):
        with open(stress_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    restoration_ok = int(row.get('restore_ok', 0))
                    restoration_failures = int(row.get('restore_failures', 0))
                    distance_to_failure = float(row.get('distance_to_failure', 0.0))
                except (ValueError, KeyError):
                    pass
                break  # first row

    # Write geometry stats
    geo_csv = os.path.join(indir, 'geometry_stats.csv')
    with open(geo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id', 'n_windows', 'boundary_thickness_nn_mean',
                         'mean_radius', 'D_eff', 'spectral_entropy',
                         'restoration_ok', 'restoration_failures',
                         'distance_to_failure'])
        run_id = os.path.basename(indir.rstrip('/'))
        writer.writerow([run_id, n_windows, thickness, mean_radius,
                         d_eff, spectral_entropy,
                         restoration_ok, restoration_failures,
                         distance_to_failure])

    # Verdict per run
    verdict = "GEOMETRY_EXISTS" if thickness > 0 else "NO_GEOMETRY"
    if restoration_failures > 0:
        verdict = "RESTORATION_FAILED"

    print(f"  Geometry: thickness={thickness:.6f} radius={mean_radius:.6f} D_eff={d_eff:.4f}")
    print(f"  Restoration: {restoration_ok} OK, {restoration_failures} failures")
    print(f"  distance_to_failure: {distance_to_failure:.6f}")
    print(f"  VERDICT: {verdict}")

if __name__ == '__main__':
    main()
