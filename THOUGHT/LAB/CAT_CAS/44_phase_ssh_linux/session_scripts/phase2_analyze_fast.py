#!/usr/bin/env python3
import csv, json, math, os, random, statistics, sys

PERIOD = 1024
MAX_ROWS = 4096

def read_rows(path):
    meta = {}
    rows = []
    with open(path, newline="") as f:
        first = f.readline().strip()
        if first.startswith("#"):
            for part in first[1:].split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k] = v
        else:
            f.seek(0)
        data = list(csv.DictReader(f))
    if len(data) > MAX_ROWS:
        step = max(1, len(data) // MAX_ROWS)
        data = data[::step][:MAX_ROWS]
    for r in data:
        rows.append({k: int(v) for k, v in r.items()})
    return meta, rows

def delta(rows, key):
    return [((rows[i][key] - rows[i-1][key]) & ((1 << 64) - 1)) for i in range(1, len(rows))]

def corr(a, b):
    n = min(len(a), len(b))
    if n < 4: return 0.0
    a, b = a[:n], b[:n]
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0: return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)

def angle(v):
    return 2 * math.pi * ((v % PERIOD) / PERIOD)

def circ_r(vals):
    if not vals: return 0.0
    sx = sum(math.cos(v) for v in vals)
    sy = sum(math.sin(v) for v in vals)
    return math.hypot(sx, sy) / len(vals)

def summarize(path, shuffled=False):
    meta, rows = read_rows(path)
    if shuffled:
        rows = rows[:]
        random.Random(42).shuffle(rows)
    d3, d4, d5, dt = delta(rows, "c3"), delta(rows, "c4"), delta(rows, "c5"), delta(rows, "tsc")
    med_dt = statistics.median([x for x in dt if x > 0]) if dt else 0
    sample_rate = 3.2e9 / med_dt if med_dt else 0
    theta34 = [angle(r["c3"]) - angle(r["c4"]) for r in rows]
    theta35 = [angle(r["c3"]) - angle(r["c5"]) for r in rows]
    theta45 = [angle(r["c4"]) - angle(r["c5"]) for r in rows]
    size = max(8, len(rows) // 32)
    p34_windows = [circ_r(theta34[i:i+size]) for i in range(0, len(theta34), size)]
    k_windows = []
    for i in range(0, min(len(theta35), len(theta45)), size):
        vals = []
        for a, b in zip(theta35[i:i+size], theta45[i:i+size]):
            vals.append(abs((complex(math.cos(a), math.sin(a)) + complex(math.cos(b), math.sin(b))) / 2))
        if vals: k_windows.append(statistics.fmean(vals))
    return {
        "label": meta.get("label", os.path.basename(path)),
        "file": os.path.basename(path),
        "shuffled": shuffled,
        "sample_rate_hz": sample_rate,
        "corr_d3_d4": corr(d3, d4),
        "corr_d3_d5": corr(d3, d5),
        "corr_d4_d5": corr(d4, d5),
        "phase34_r_mean": statistics.fmean(p34_windows) if p34_windows else 0.0,
        "phase34_r_stdev": statistics.pstdev(p34_windows) if len(p34_windows) > 1 else 0.0,
        "kuramoto_r_mean": statistics.fmean(k_windows) if k_windows else 0.0,
        "kuramoto_r_stdev": statistics.pstdev(k_windows) if len(k_windows) > 1 else 0.0,
        "d3_mean": statistics.fmean(d3) if d3 else 0.0,
        "d4_mean": statistics.fmean(d4) if d4 else 0.0,
        "d5_mean": statistics.fmean(d5) if d5 else 0.0,
    }

def group_stats(results):
    groups = {}
    for r in results:
        if r["shuffled"]: continue
        name = "_".join(r["label"].split("_")[:-1]) or r["label"]
        groups.setdefault(name, []).append(r)
    out = {}
    for k, vals in groups.items():
        out[k] = {
            "n": len(vals),
            "kuramoto_r_mean": statistics.fmean(v["kuramoto_r_mean"] for v in vals),
            "kuramoto_r_stdev": statistics.pstdev([v["kuramoto_r_mean"] for v in vals]) if len(vals) > 1 else 0,
            "phase34_r_mean": statistics.fmean(v["phase34_r_mean"] for v in vals),
            "phase34_r_stdev": statistics.pstdev([v["phase34_r_mean"] for v in vals]) if len(vals) > 1 else 0,
            "corr_d3_d4_mean": statistics.fmean(v["corr_d3_d4"] for v in vals),
        }
    return out

def spacing_ratio_from_groups(groups):
    vals = sorted(v["kuramoto_r_mean"] + v["phase34_r_mean"] for v in groups.values())
    spacings = [vals[i+1] - vals[i] for i in range(len(vals)-1) if vals[i+1] - vals[i] > 1e-9]
    ratios = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1]) for i in range(len(spacings)-1)]
    return statistics.fmean(ratios) if ratios else 0.0

if __name__ == "__main__":
    results = []
    for p in sys.argv[1:]:
        results.append(summarize(p, False))
        results.append(summarize(p, True))
    groups = group_stats(results)
    print(json.dumps({"results": results, "groups": groups, "goe_spacing_r": spacing_ratio_from_groups(groups)}, indent=2, sort_keys=True))
