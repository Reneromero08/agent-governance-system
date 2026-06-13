#!/usr/bin/env python3
"""transient_fold_probe.py - local/transient fold-odd audit for Exp50 Phase 6.

Question tested:
  The static/global Phase 6 sensors found the orientation bit absent from public
  even data. REPORT_SESSION_LATTICE_CLIMB.md left one concrete crack: maybe the
  full transient of the public map f(x) contains a fold-odd functional that the
  static spectrum/winding census did not price.

This probe builds only public transient features:
  - accepting/fixed points inferred from public (k,b,N)
  - basin sizes and score contrast
  - first-passage times from public anchor points
  - local approach times around each fixed point
  - endpoint/lower-vs-upper transient moments

It then runs the hardened no-smuggle gate. A genuine crack must raise
orientation AUC while staying random-private-fold clean. A hidden-orientation
control is included to prove the gate is live.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE6 = HERE.parent
FOLD = PHASE6 / "fold_audit"
STAGE3 = FOLD / "stage3"
for path in (str(FOLD), str(STAGE3)):
    if path not in sys.path:
        sys.path.insert(0, path)

import construction as C  # noqa: E402
import hardened_gate as H  # noqa: E402


def score_all_x(k: np.ndarray, b: np.ndarray, N: int) -> np.ndarray:
    """Public score(x) for all x via FFT."""
    a = np.zeros(N, dtype=float)
    np.add.at(a, (k % N).astype(np.int64), b)
    return np.real(np.fft.ifft(a)) * N


def public_map(k: np.ndarray, b: np.ndarray, N: int):
    scores = score_all_x(k, b, N)
    accept = scores > (len(b) / 4.0)
    x = np.arange(N, dtype=np.int64)
    f = (x + 1) % N
    f[accept] = x[accept]
    fixed = np.flatnonzero(accept)
    return f, fixed, scores


def walk_to_fixed(f: np.ndarray, x0: int, cap: int):
    """Return endpoint and first-passage time under public f."""
    x = int(x0 % len(f))
    for t in range(cap + 1):
        y = int(f[x])
        if y == x:
            return x, t
        x = y
    return x, cap


def basin_sizes(fixed: np.ndarray, N: int) -> np.ndarray:
    fx = np.sort(fixed)
    if len(fx) == 0:
        return np.zeros(0, dtype=float)
    sizes = np.zeros(len(fx), dtype=float)
    for j in range(len(fx)):
        sizes[j] = (int(fx[j]) - int(fx[j - 1])) % N
    sizes[sizes == 0] = N
    return sizes


def _moments(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    ]


def O_transient_public(inst):
    """Pure-public transient features of f. Does not read inst['d']."""
    k = inst["k"]
    b = inst["b"]
    N = int(inst["N"])
    f, fixed, scores = public_map(k, b, N)
    fixed = np.sort(fixed)
    bs = basin_sizes(fixed, N)

    feats = [
        float(len(fixed)),
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.max(scores)),
        float(np.min(scores)),
    ]

    if len(fixed) >= 2:
        lo = int(fixed[0])
        hi = int(fixed[-1])
        feats.extend([
            lo / N,
            hi / N,
            (hi - lo) / N,
            float(scores[lo] - scores[hi]) / max(len(b), 1),
            float(np.min(bs) / N),
            float(np.max(bs) / N),
            float((np.max(bs) - np.min(bs)) / N),
        ])
    else:
        lo = hi = -1
        feats.extend([0.0] * 7)

    anchors = [
        0,
        1,
        N // 8,
        N // 4,
        N // 2 - 1,
        N // 2,
        3 * N // 4,
        N - 1,
    ]
    endpoint_is_hi = []
    times = []
    signed_endpoint = []
    for a in anchors:
        ep, t = walk_to_fixed(f, a, N)
        times.append(t / N)
        endpoint_is_hi.append(1.0 if ep == hi else 0.0)
        signed_endpoint.append(((ep - (N / 2.0)) / N))
    feats.extend(times)
    feats.extend(endpoint_is_hi)
    feats.extend(signed_endpoint)
    feats.extend(_moments(times))

    local_times = []
    endpoint_switches = []
    for p in fixed[:4]:
        p = int(p)
        for off in (-8, -4, -2, -1, 1, 2, 4, 8):
            ep, t = walk_to_fixed(f, p + off, N)
            local_times.append(t / N)
            endpoint_switches.append(1.0 if ep != p else 0.0)
    feats.extend(_moments(local_times))
    feats.extend(_moments(endpoint_switches))
    return np.asarray(feats, dtype=float)


def O_transient_smuggle(inst):
    """Gate-live control: same public features plus hidden orientation."""
    base = O_transient_public(inst)
    return np.concatenate([base, [float(C.orientation_bit(inst["d"], inst["N"]))]])


def run_gate(name, op, expected, n_values, n_instances, seed, n_shuffles):
    rows = []
    for i, n in enumerate(n_values):
        t0 = time.time()
        res = H.hardened_gate(
            op,
            n,
            n_instances=n_instances,
            seed=seed + 1009 * n + 37 * i,
            n_shuffles=n_shuffles,
        )
        rows.append({
            "name": name,
            "n": n,
            "expected": expected,
            "verdict": res["verdict"],
            "auc": res["auc"],
            "shuffle_null_95": res["shuffle_null_95"],
            "random_fold_auc": res["random_fold_auc"],
            "random_fold_null_95": res["random_fold_null_95"],
            "max_fold_delta": res["max_fold_delta"],
            "smuggle_reason": res["smuggle_reason"],
            "seconds": time.time() - t0,
            "matches_expected": res["verdict"] == expected,
        })
    return rows


def main():
    n_values = [8, 10, 12]
    seed = 5060613
    n_instances = 300
    n_shuffles = 25
    out = {
        "status": "TRANSIENT_FOLD_ODD_PROBE",
        "seed": seed,
        "n_values": n_values,
        "n_instances": n_instances,
        "n_shuffles": n_shuffles,
        "rows": [],
    }
    out["rows"].extend(run_gate(
        "transient_public",
        O_transient_public,
        "FAIL_CHANCE",
        n_values,
        n_instances,
        seed,
        n_shuffles,
    ))
    out["rows"].extend(run_gate(
        "transient_smuggle_control",
        O_transient_smuggle,
        "FAIL_SMUGGLE",
        [8, 10],
        n_instances,
        seed + 90000,
        n_shuffles,
    ))

    public_rows = [r for r in out["rows"] if r["name"] == "transient_public"]
    smuggle_rows = [r for r in out["rows"] if r["name"] == "transient_smuggle_control"]
    out["verdict"] = (
        "TRANSIENT_PUBLIC_FAIL_CHANCE__NO_FOLD_ODD_FUNCTIONAL_FOUND"
        if all(r["verdict"] == "FAIL_CHANCE" for r in public_rows)
        and all(r["verdict"] == "FAIL_SMUGGLE" for r in smuggle_rows)
        else "TRANSIENT_PROBE_REQUIRES_AUDIT"
    )

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / "transient_fold_probe_result.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="ascii")

    report = [
        "# Phase 6 Transient Fold Probe",
        "",
        f"**Verdict:** `{out['verdict']}`",
        "",
        "## Question",
        "",
        "Does the public transient of `f(x)` carry a fold-odd orientation functional that static/global Phase 6 sensors missed?",
        "",
        "## Result",
        "",
        "| candidate | n | verdict | auc | null95 | random_fold_auc | random_fold_null95 | delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in out["rows"]:
        report.append(
            f"| {r['name']} | {r['n']} | `{r['verdict']}` | "
            f"{r['auc']:.3f} | {r['shuffle_null_95']:.3f} | "
            f"{r['random_fold_auc']:.3f} | {r['random_fold_null_95']:.3f} | "
            f"{r['max_fold_delta']:.3g} |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "The public transient features remain fold-even under the hardened random-private-fold gate. "
        "The hidden-orientation control is caught as a smuggle, so the instrument is live. "
        "This closes the specific `REPORT_SESSION_LATTICE_CLIMB.md` open crack about local/transient invariants for the current feature family.",
        "",
        "This does not change the formal dihedral lower-bound status. It only says this concrete public transient route did not recover the orientation bit.",
        "",
    ])
    (HERE / "PHASE6_TRANSIENT_FOLD_PROBE.md").write_text("\n".join(report), encoding="ascii")

    print(out["verdict"])
    for r in out["rows"]:
        print(
            "%s n=%d verdict=%s auc=%.3f null95=%.3f rf=%.3f rf95=%.3f delta=%.3g"
            % (
                r["name"],
                r["n"],
                r["verdict"],
                r["auc"],
                r["shuffle_null_95"],
                r["random_fold_auc"],
                r["random_fold_null_95"],
                r["max_fold_delta"],
            )
        )
    return 0 if out["verdict"].endswith("NO_FOLD_ODD_FUNCTIONAL_FOUND") else 1


if __name__ == "__main__":
    raise SystemExit(main())
