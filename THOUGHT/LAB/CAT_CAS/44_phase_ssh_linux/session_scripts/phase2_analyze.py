#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import statistics
import sys

PERIOD = 1024.0

def load_csv(path):
    rows = []
    meta = {}
    with open(path, newline="") as f:
        first = f.readline().strip()
        if first.startswith("#"):
            for part in first[1:].strip().split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k] = v
        else:
            f.seek(0)
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: int(v) for k, v in row.items()})
    return meta, rows

def deltas(rows, key):
    out = []
    prev = rows[0][key]
    for row in rows[1:]:
        cur = row[key]
        out.append((cur - prev) & ((1 << 64) - 1))
        prev = cur
    return out

def pearson(a, b):
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)

def phase(v):
    return 2.0 * math.pi * ((v % int(PERIOD)) / PERIOD)

def circ_r(angles):
    if not angles:
        return 0.0
    sx = sum(math.cos(a) for a in angles)
    sy = sum(math.sin(a) for a in angles)
    return math.hypot(sx, sy) / len(angles)

def analyze(path, shuffled=False):
    meta, rows = load_csv(path)
    if shuffled:
        rrows = rows[:]
        random.Random(12345).shuffle(rrows)
        rows = rrows
    d3 = deltas(rows, "c3")
    d4 = deltas(rows, "c4")
    d5 = deltas(rows, "c5")
    dt = deltas(rows, "tsc")
    valid = [x for x in dt if 0 < x < (statistics.median(dt) * 10 if dt else 0)]
    sample_rate = 3.2e9 / statistics.median(valid) if valid else 0.0

    theta35 = [phase(r["c3"]) - phase(r["c5"]) for r in rows]
    theta45 = [phase(r["c4"]) - phase(r["c5"]) for r in rows]
    theta34 = [phase(r["c3"]) - phase(r["c4"]) for r in rows]
    kuramoto = []
    for a, b in zip(theta35, theta45):
        kuramoto.append(abs((complex(math.cos(a), math.sin(a)) +
                             complex(math.cos(b), math.sin(b))) / 2.0))

    windows = 32
    size = max(1, len(theta34) // windows)
    phase_lock_windows = [circ_r(theta34[i:i + size]) for i in range(0, len(theta34), size)]
    k_windows = [statistics.fmean(kuramoto[i:i + size]) for i in range(0, len(kuramoto), size)]

    return {
        "file": os.path.basename(path),
        "label": meta.get("label", os.path.basename(path)),
        "shuffled": shuffled,
        "sample_rate_hz": sample_rate,
        "corr_d3_d4": pearson(d3, d4),
        "corr_d3_d5": pearson(d3, d5),
        "corr_d4_d5": pearson(d4, d5),
        "phase34_r_mean": statistics.fmean(phase_lock_windows),
        "phase34_r_stdev": statistics.pstdev(phase_lock_windows),
        "kuramoto_r_mean": statistics.fmean(k_windows),
        "kuramoto_r_stdev": statistics.pstdev(k_windows),
        "d3_mean": statistics.fmean(d3) if d3 else 0,
        "d4_mean": statistics.fmean(d4) if d4 else 0,
        "d5_mean": statistics.fmean(d5) if d5 else 0,
    }

def spacing_ratio(matrix):
    # Jacobi eigenvalue solver for small symmetric matrices.
    n = len(matrix)
    a = [row[:] for row in matrix]
    for _ in range(100):
        p, q, m = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > m:
                    p, q, m = i, j, abs(a[i][j])
        if m < 1e-12:
            break
        phi = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(phi), math.sin(phi)
        app = c*c*a[p][p] - 2*s*c*a[p][q] + s*s*a[q][q]
        aqq = s*s*a[p][p] + 2*s*c*a[p][q] + c*c*a[q][q]
        a[p][q] = a[q][p] = 0.0
        a[p][p], a[q][q] = app, aqq
        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
    vals = sorted(a[i][i] for i in range(n))
    spacings = [vals[i + 1] - vals[i] for i in range(n - 1) if vals[i + 1] - vals[i] > 1e-12]
    ratios = [min(spacings[i], spacings[i + 1]) / max(spacings[i], spacings[i + 1])
              for i in range(len(spacings) - 1) if max(spacings[i], spacings[i + 1]) > 0]
    return statistics.fmean(ratios) if ratios else 0.0

def goe_from_results(results):
    keys = ["corr_d3_d4", "corr_d3_d5", "corr_d4_d5", "phase34_r_mean",
            "kuramoto_r_mean", "d3_mean", "d4_mean", "d5_mean"]
    vectors = [[r[k] for k in keys] for r in results if not r["shuffled"]]
    n = len(vectors)
    if n < 3:
        return 0.0
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = pearson(vectors[i], vectors[j])
    return spacing_ratio(mat)

if __name__ == "__main__":
    results = []
    for path in sys.argv[1:]:
        results.append(analyze(path, False))
        results.append(analyze(path, True))
    final = {"results": results, "goe_spacing_r": goe_from_results(results)}
    print(json.dumps(final, indent=2, sort_keys=True))
