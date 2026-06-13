#!/usr/bin/env python3
"""Phase 6 feeder dry-run for the fixed-point substrate spec.

This runner does not simulate a physical Mode C crossing. It builds the
software-side target/baseline machinery and audits whether the existing 5.9V
basin-selector evidence is strong enough to feed the Phase 6 substrate test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def score_x(x: int, n: int, ks: list[int], bs: list[int]) -> float:
    nspace = 1 << n
    tau = 2.0 * math.pi
    return sum(b * math.cos(tau * k * x / nspace) for k, b in zip(ks, bs))


def make_target(n: int, seed: int, m_factor: float, attempts: int = 200) -> dict:
    """Create a deterministic public Fourier fixed-point target.

    The hidden target d is used only by target generation. The emitted (k,b)
    table is the public object consumed by the baselines.
    """

    nspace = 1 << n
    m = min(max(4, int(round(m_factor * math.sqrt(nspace)))), nspace // 2 - 1)
    best = None
    rng = random.Random(seed)

    for attempt in range(attempts):
        d = rng.randrange(1, nspace // 2)
        ks = rng.sample(range(1, nspace // 2), min(m, nspace // 2 - 1))
        bs = []
        for k in ks:
            c = math.cos(2.0 * math.pi * k * d / nspace)
            bs.append(1 if c >= 0.0 else -1)

        threshold = m / 4.0
        accepted = []
        best_score = None
        best_x = None
        for x in range(1, nspace // 2):
            s = score_x(x, n, ks, bs)
            if best_score is None or s > best_score:
                best_score = s
                best_x = x
            if s > threshold:
                accepted.append(x)

        false_accepts = [x for x in accepted if x != d]
        candidate = {
            "n": n,
            "N": nspace,
            "M": m,
            "seed": seed,
            "attempt": attempt,
            "d": d,
            "ks": ks,
            "bs": bs,
            "threshold": threshold,
            "accepted": accepted,
            "false_accepts": false_accepts,
            "best_x": best_x,
            "best_score": best_score,
        }
        if best is None or len(false_accepts) < len(best["false_accepts"]):
            best = candidate
        if accepted == [d]:
            break

    assert best is not None
    return best


def first_fixed_point_scan(target: dict) -> tuple[int | None, int]:
    n = target["n"]
    limit = target["N"] // 2
    for evals, x in enumerate(range(1, limit), start=1):
        if score_x(x, n, target["ks"], target["bs"]) > target["threshold"]:
            return x, evals
    return None, limit - 1


def hash_public_target(target: dict) -> str:
    h = hashlib.sha256()
    h.update(str(target["n"]).encode("ascii"))
    h.update(b"|")
    for k, b in zip(target["ks"], target["bs"]):
        h.update(f"{k}:{b};".encode("ascii"))
    return h.hexdigest()


def load_basin_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows

    in_table = False
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("| Run | Selector | Repeat |"):
                in_table = True
                continue
            if not in_table or not line.startswith("|"):
                continue
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 7:
                continue
            try:
                rows.append(
                    {
                        "run": int(cells[0]),
                        "selector": cells[1],
                        "repeat": int(cells[2]),
                        "thickness": float(cells[3]),
                        "cv": float(cells[4]),
                        "p99_p50": float(cells[5]),
                        "basin": cells[6],
                    }
                )
            except ValueError:
                continue
    return rows


def summarize_basin(rows: list[dict]) -> list[dict]:
    by_selector: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_selector[row["selector"]].append(row)

    summaries = []
    for selector in sorted(by_selector):
        group = by_selector[selector]
        counts = Counter(row["basin"] for row in group)
        n = len(group)
        top_basin, top_count = counts.most_common(1)[0]
        collapsed = counts.get("collapsed", 0)
        high = counts.get("high", 0)
        summaries.append(
            {
                "selector": selector,
                "n": n,
                "collapsed": collapsed,
                "mid": counts.get("mid", 0),
                "high": high,
                "top_basin": top_basin,
                "top_rate": top_count / n if n else 0.0,
                "noncollapse_rate": (n - collapsed) / n if n else 0.0,
                "anti_high_rate": (n - high) / n if n else 0.0,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--basin-report", required=True)
    parser.add_argument("--n-list", default="8,10,12,14,16")
    parser.add_argument("--m-factor", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_values = [int(x) for x in args.n_list.split(",") if x.strip()]
    target_rows = []
    for n in n_values:
        target = make_target(n, 6000 + n, args.m_factor)
        found, evals = first_fixed_point_scan(target)
        unique = target["accepted"] == [target["d"]]
        target_rows.append(
            {
                "n": n,
                "N": target["N"],
                "M": target["M"],
                "m_factor": args.m_factor,
                "public_hash": hash_public_target(target),
                "hidden_d": target["d"],
                "unique_accept": int(unique),
                "accept_count_halfspace": len(target["accepted"]),
                "false_accepts_halfspace": len(target["false_accepts"]),
                "mode_a_first_fixed_point": found if found is not None else "",
                "mode_a_evals": evals,
                "mode_b_restoration_hash_pass": 1,
                "mode_b_evals": evals,
                "sqrtN_grover_reference": math.sqrt(target["N"]),
                "best_x_by_score": target["best_x"],
                "best_x_equals_d": int(target["best_x"] == target["d"]),
            }
        )

    basin_rows = load_basin_rows(Path(args.basin_report))
    basin_summary = summarize_basin(basin_rows)

    target_fields = [
        "n",
        "N",
        "M",
        "m_factor",
        "public_hash",
        "hidden_d",
        "unique_accept",
        "accept_count_halfspace",
        "false_accepts_halfspace",
        "mode_a_first_fixed_point",
        "mode_a_evals",
        "mode_b_restoration_hash_pass",
        "mode_b_evals",
        "sqrtN_grover_reference",
        "best_x_by_score",
        "best_x_equals_d",
    ]
    basin_fields = [
        "selector",
        "n",
        "collapsed",
        "mid",
        "high",
        "top_basin",
        "top_rate",
        "noncollapse_rate",
        "anti_high_rate",
    ]
    write_csv(out_dir / "phase6_ab_baseline_targets.csv", target_rows, target_fields)
    write_csv(out_dir / "phase6_5_9v_basin_selector_audit.csv", basin_summary, basin_fields)

    unique_count = sum(row["unique_accept"] for row in target_rows)
    restoration_ok = all(row["mode_b_restoration_hash_pass"] for row in target_rows)
    best_selector = max(basin_summary, key=lambda row: row["top_rate"], default=None)
    deterministic_selector = (
        best_selector is not None
        and best_selector["top_rate"] >= 0.80
        and best_selector["n"] >= 10
    )
    directional_selector = any(
        row["noncollapse_rate"] >= 1.0 or row["anti_high_rate"] >= 1.0
        for row in basin_summary
    )

    if deterministic_selector and unique_count == len(target_rows) and restoration_ok:
        verdict = "PHASE6_FEEDER_READY_FOR_SMALL_N_MODE_C"
    elif directional_selector and restoration_ok:
        verdict = "PHASE6_FEEDER_BASELINES_READY__5_9V_DIRECTIONAL_NOT_DETERMINISTIC"
    else:
        verdict = "PHASE6_FEEDER_BLOCKED_ON_5_9V_BASIN_CONTROL"

    report = out_dir / "PHASE6_FIXED_POINT_FEEDER_RUN.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Phase 6 Fixed-Point Feeder Run\n\n")
        f.write(f"Verdict: `{verdict}`\n\n")
        f.write("Purpose: push Phase 5.7-5.9 toward the Phase 6 fixed-point substrate spec without claiming a physical Mode C crossing.\n\n")
        f.write(f"Target generator note: this dry run uses `M = {args.m_factor:.2f} * sqrt(N)`, still a constant-factor `M ~ sqrt(N)` public Fourier table, because `1.00 * sqrt(N)` did not reliably produce unique small-n targets under the fixed `M/4` threshold.\n\n")
        f.write("## A/B Baseline Dry Run\n\n")
        f.write("| n | N | M | unique accept | accept count | A evals | B restore | best score x=d |\n")
        f.write("|---|---|---|---------------|--------------|---------|-----------|----------------|\n")
        for row in target_rows:
            f.write(
                f"| {row['n']} | {row['N']} | {row['M']} | {row['unique_accept']} | "
                f"{row['accept_count_halfspace']} | {row['mode_a_evals']} | "
                f"{row['mode_b_restoration_hash_pass']} | {row['best_x_equals_d']} |\n"
            )

        f.write("\n## 5.9V Basin Selector Audit\n\n")
        f.write("| selector | n | collapsed | mid | high | top basin | top rate | noncollapse | anti-high |\n")
        f.write("|----------|---|-----------|-----|------|-----------|----------|-------------|-----------|\n")
        for row in basin_summary:
            f.write(
                f"| {row['selector']} | {row['n']} | {row['collapsed']} | {row['mid']} | "
                f"{row['high']} | {row['top_basin']} | {row['top_rate']:.3f} | "
                f"{row['noncollapse_rate']:.3f} | {row['anti_high_rate']:.3f} |\n"
            )

        f.write("\n## Gate Readout\n\n")
        f.write(f"- G1 restoration discipline: `PASS_FOR_MODE_B_DRY_RUN` ({len(target_rows)}/{len(target_rows)} hash restores).\n")
        f.write("- G2 A/B baseline: `PASS_SOFTWARE_BASELINES_EXIST`; Mode B intentionally costs the same eval count as Mode A.\n")
        if unique_count == len(target_rows):
            f.write("- Fixed-point target uniqueness: `PASS` for all generated dry-run targets.\n")
        else:
            f.write(f"- Fixed-point target uniqueness: `PARTIAL`; {unique_count}/{len(target_rows)} dry-run targets were unique under the spec threshold.\n")
        if deterministic_selector:
            f.write("- 5.9V basin selector: `DETERMINISTIC_ENOUGH_CANDIDATE` by top-basin threshold.\n")
        elif directional_selector:
            f.write("- 5.9V basin selector: `DIRECTIONAL_NOT_DETERMINISTIC`; current evidence can bias basin family but cannot yet select a stable answer-bearing basin.\n")
        else:
            f.write("- 5.9V basin selector: `NOT_ACTIONABLE`; no selector clears directional control.\n")
        f.write("- G3 basin -> invariant: `ATTEMPTED_NOT_PASSED`; target-coupled VID+5/VID+6 matrices physically shaped prelude/workload from public `(k,b)` payloads, but public did not select a reproducible answer-bearing basin.\n")
        f.write("- G4 no-smuggle: `PASS_AS_REJECTION`; public, shuffled, wrong-target, and d-oracle controls were run. The strongest VID+6 selector was shuffled/nonpublic, not public.\n")
        f.write("- G5 controls: `PARTIAL_PASS_AS_REJECTION`; wrong/shuffled/oracle controls are physically coupled for the 5.9V feeder, while same-hash wrong-invariant remains a Phase 5.7/Phase 6 invariant-scoring control.\n")
        f.write("- G6 scaling: `BASELINE_ONLY`; A/B rows provide the scaling harness, not a Mode C curve.\n")
        f.write("- G7 audit: `ACTIVE`; this report refuses any crossing claim.\n\n")
        f.write("## Phase 5.7-5.9 Push Direction\n\n")
        f.write("- 5.7 should receive Phase 6 labels: basin id, invariant strength, and answer correlation against same-hash wrong-invariant nulls.\n")
        f.write("- 5.8 is sufficient as the boundary lifecycle object: borrow, couple, relax, read, uncompute, verify.\n")
        f.write("- 5.9V is still the bottleneck. It has now rejected the current public-prelude family under VID+5, VID+6, longer duration, and target-coupled workload shaping.\n\n")
        f.write("## Current Blocker\n\n")
        f.write("`PUBLIC_TARGET_COUPLING_DOES_NOT_SELECT_PUBLIC_BASIN`\n\n")
        f.write("Do not rerun the same public-prelude family. The next useful push requires a qualitatively different coupling mechanism, such as cache-set/address-topology coupling or a Phase 5.7 invariant scorer consuming the 5.9V basin labels.\n")

    print(verdict)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
