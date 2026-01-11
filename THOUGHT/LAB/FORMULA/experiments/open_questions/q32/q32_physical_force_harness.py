"""
Q32 Phase 6 (additional track): Physical coupling harness

This is NOT a claim of a new fundamental force.
This is a deterministic, receipted test harness that can:
- detect a known synthetic coupling when present
- reject null controls when absent
- flag trivial “echo/leak” constructions

It exists so that any future “meaning is a fundamental physical field” claim
has a hard falsification scaffold instead of post-hoc storytelling.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import csv
import sys
from typing import Any, Dict, List, Optional, Tuple


def _utc_ts() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _mean(xs: List[float]) -> float:
    if not xs:
        raise ValueError("mean() on empty list")
    return sum(xs) / len(xs)


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson_r requires equal-length lists")
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = _mean(xs)
    my = _mean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0.0:
        return float("nan")
    return num / den


def _shift(xs: List[float], lag: int) -> Tuple[List[float], List[float]]:
    """
    Returns (xs_trunc, ys_trunc) alignment helper:
    - lag > 0 aligns xs[t-lag] with ys[t]
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag == 0:
        return xs[:], xs[:]
    if lag >= len(xs):
        return [], []
    return xs[:-lag], xs[lag:]


def _align_for_lag(x: List[float], y: List[float], lag: int) -> Tuple[List[float], List[float]]:
    if len(x) != len(y):
        raise ValueError("align_for_lag requires equal-length lists")
    if lag == 0:
        return x[:], y[:]
    if lag >= len(x):
        return [], []
    return x[:-lag], y[lag:]


@dataclasses.dataclass(frozen=True)
class ScenarioResult:
    name: str
    r_m_to_b_lag: float
    r_b_to_m_lag: float
    r_m_to_b_zero: float
    r_m_to_b_shuffled: float
    detects_coupling: bool
    flags_echo_leak: bool

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _rotate(xs: List[float], k: int) -> List[float]:
    if not xs:
        return []
    k = k % len(xs)
    return xs[k:] + xs[:k]


def _read_csv_mb(path: str) -> Tuple[List[float], List[float]]:
    """
    Reads a CSV with at least columns for M and B.
    Accepted headers (case-insensitive):
    - M column: m, M, meaning, meaning_field
    - B column: b, B, physical, physical_field, sensor
    Extra columns are ignored.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        def norm(s: str) -> str:
            return s.strip().lower()

        fields = {norm(x): x for x in reader.fieldnames}
        m_key = None
        b_key = None
        for candidate in ["m", "meaning", "meaning_field"]:
            if candidate in fields:
                m_key = fields[candidate]
                break
        for candidate in ["b", "physical", "physical_field", "sensor"]:
            if candidate in fields:
                b_key = fields[candidate]
                break
        if m_key is None or b_key is None:
            raise ValueError(f"CSV must include M and B columns; got headers={reader.fieldnames}")

        m: List[float] = []
        b: List[float] = []
        for row in reader:
            m.append(float(row[m_key]))
            b.append(float(row[b_key]))
        if len(m) != len(b):
            raise ValueError("M and B series lengths differ")
        if len(m) < 10:
            raise ValueError("CSV too short (need >= 10 rows)")
        return m, b


def _generate_m_series(rng: random.Random, n_steps: int) -> List[float]:
    m: List[float] = []
    level = 0.0
    for t in range(n_steps):
        if t == 0 or rng.random() < 0.15:
            level = rng.uniform(-1.0, 1.0)
        m.append(level + rng.gauss(0.0, 0.10))
    return m


def _generate_b_from_m_lagged(
    rng: random.Random,
    m: List[float],
    lag: int,
    coupling_strength: float,
    noise_std: float,
) -> List[float]:
    b: List[float] = []
    for t in range(len(m)):
        src = m[t - lag] if t - lag >= 0 else 0.0
        b.append(coupling_strength * src + rng.gauss(0.0, noise_std))
    return b


def _generate_b_independent(rng: random.Random, n_steps: int, noise_std: float) -> List[float]:
    return [rng.gauss(0.0, noise_std) for _ in range(n_steps)]


def _generate_b_echo_leak(rng: random.Random, m: List[float], noise_std: float) -> List[float]:
    return [x + rng.gauss(0.0, noise_std) for x in m]


def _scenario_metrics(
    name: str,
    m: List[float],
    b: List[float],
    lag: int,
    r_detect_threshold: float,
    echo_threshold: float,
    shuffle_seed: int,
) -> ScenarioResult:
    m_l, b_l = _align_for_lag(m, b, lag)
    b_l2, m_l2 = _align_for_lag(b, m, lag)

    r_m_to_b_lag = _pearson_r(m_l, b_l) if m_l else float("nan")
    r_b_to_m_lag = _pearson_r(b_l2, m_l2) if b_l2 else float("nan")
    r_m_to_b_zero = _pearson_r(m, b)

    rng = random.Random(shuffle_seed)
    idx = list(range(len(m)))
    rng.shuffle(idx)
    m_shuf = [m[i] for i in idx]
    m_shuf_l, b_shuf_l = _align_for_lag(m_shuf, b, lag)
    r_m_to_b_shuffled = _pearson_r(m_shuf_l, b_shuf_l) if m_shuf_l else float("nan")

    detects_coupling = bool(
        (not math.isnan(r_m_to_b_lag))
        and abs(r_m_to_b_lag) >= r_detect_threshold
        and (math.isnan(r_m_to_b_shuffled) or abs(r_m_to_b_shuffled) < abs(r_m_to_b_lag))
    )

    flags_echo_leak = bool((not math.isnan(r_m_to_b_zero)) and abs(r_m_to_b_zero) >= echo_threshold)

    return ScenarioResult(
        name=name,
        r_m_to_b_lag=r_m_to_b_lag,
        r_b_to_m_lag=r_b_to_m_lag,
        r_m_to_b_zero=r_m_to_b_zero,
        r_m_to_b_shuffled=r_m_to_b_shuffled,
        detects_coupling=detects_coupling,
        flags_echo_leak=flags_echo_leak,
    )


def _default_receipt_path() -> str:
    return os.path.join(
        "LAW",
        "CONTRACTS",
        "_runs",
        "q32_public",
        "datatrail",
        f"physical_force_receipt_p6_synth_{_utc_ts()}.json",
    )


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _write_csv_mb(path: str, m: List[float], b: List[float]) -> None:
    if len(m) != len(b):
        raise ValueError("M and B length mismatch")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "b"])
        for mi, bi in zip(m, b):
            w.writerow([f"{mi:.10f}", f"{bi:.10f}"])
    os.replace(tmp, path)


def run_synthetic_validator_suite(
    seed: int,
    n_steps: int,
    lag: int,
    coupling_strength: float,
    noise_std: float,
    r_detect_threshold: float,
    echo_threshold: float,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    m = _generate_m_series(rng, n_steps)

    pos_rng = random.Random(seed + 1)
    b_pos = _generate_b_from_m_lagged(pos_rng, m, lag=lag, coupling_strength=coupling_strength, noise_std=noise_std)

    null_rng = random.Random(seed + 2)
    b_null = _generate_b_independent(null_rng, n_steps=n_steps, noise_std=noise_std)

    leak_rng = random.Random(seed + 3)
    b_leak = _generate_b_echo_leak(leak_rng, m, noise_std=noise_std)

    pos = _scenario_metrics(
        name="positive_lagged_coupling",
        m=m,
        b=b_pos,
        lag=lag,
        r_detect_threshold=r_detect_threshold,
        echo_threshold=echo_threshold,
        shuffle_seed=seed + 1001,
    )
    null = _scenario_metrics(
        name="null_independent",
        m=m,
        b=b_null,
        lag=lag,
        r_detect_threshold=r_detect_threshold,
        echo_threshold=echo_threshold,
        shuffle_seed=seed + 1002,
    )
    leak = _scenario_metrics(
        name="echo_leak",
        m=m,
        b=b_leak,
        lag=lag,
        r_detect_threshold=r_detect_threshold,
        echo_threshold=echo_threshold,
        shuffle_seed=seed + 1003,
    )

    passed = bool(pos.detects_coupling and (not null.detects_coupling) and leak.flags_echo_leak)

    return {
        "type": "PhysicalForceCouplingReceipt",
        "version": 1,
        "passed": passed,
        "run": {
            "mode": "synthetic_validator_suite",
            "seed": seed,
            "n_steps": n_steps,
            "lag": lag,
            "coupling_strength": coupling_strength,
            "noise_std": noise_std,
            "r_detect_threshold": r_detect_threshold,
            "echo_threshold": echo_threshold,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": [pos.to_json(), null.to_json(), leak.to_json()],
    }


def run_csv_coupling(
    csv_path: str,
    max_lag: int,
    r_detect_threshold: float,
    echo_threshold: float,
    null_rotations: List[int],
    seed_for_rotations: int,
    null_n: int,
    null_kind: str,
    null_quantile: float,
) -> Dict[str, Any]:
    m, b = _read_csv_mb(csv_path)

    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    max_lag = min(max_lag, max(0, len(m) - 2))

    lag_scan: List[Dict[str, Any]] = []
    best = {"lag": 0, "r": float("nan")}
    best_rev = {"lag": 0, "r": float("nan")}
    for lag in range(max_lag + 1):
        m_l, b_l = _align_for_lag(m, b, lag)
        r = _pearson_r(m_l, b_l) if m_l else float("nan")
        b_l2, m_l2 = _align_for_lag(b, m, lag)
        r_rev = _pearson_r(b_l2, m_l2) if b_l2 else float("nan")
        lag_scan.append({"lag": lag, "r_m_to_b": r, "r_b_to_m": r_rev})
        if math.isnan(best["r"]) or (not math.isnan(r) and abs(r) > abs(best["r"])):
            best = {"lag": lag, "r": r}
        if math.isnan(best_rev["r"]) or (not math.isnan(r_rev) and abs(r_rev) > abs(best_rev["r"])):
            best_rev = {"lag": lag, "r": r_rev}

    if null_n < 3:
        raise ValueError("null_n must be >= 3")
    if not (0.5 < null_quantile < 1.0):
        raise ValueError("null_quantile must be in (0.5, 1.0)")

    # Null distribution: rotation or permutation of M (deterministic).
    rng = random.Random(seed_for_rotations)
    rotations = null_rotations[:]
    if not rotations:
        rotations = [len(m) // 3, (2 * len(m)) // 3]
    rotations = [max(1, int(abs(k))) for k in rotations]
    rotations = [k % len(m) for k in rotations if (k % len(m)) != 0]
    if not rotations:
        rotations = [1]

    null_rs: List[float] = []
    if null_kind == "rotate":
        # Sample k from the provided rotation set (with replacement)
        for _ in range(null_n):
            k = rotations[rng.randrange(0, len(rotations))]
            m_rot = _rotate(m, k)
            m_l, b_l = _align_for_lag(m_rot, b, int(best["lag"]))
            null_rs.append(_pearson_r(m_l, b_l) if m_l else float("nan"))
    elif null_kind == "permute":
        idx = list(range(len(m)))
        for _ in range(null_n):
            rng.shuffle(idx)
            m_perm = [m[i] for i in idx]
            m_l, b_l = _align_for_lag(m_perm, b, int(best["lag"]))
            null_rs.append(_pearson_r(m_l, b_l) if m_l else float("nan"))
    else:
        raise ValueError("null_kind must be one of: rotate, permute")

    null_abs = sorted(abs(r) for r in null_rs if not math.isnan(r))
    if not null_abs:
        raise ValueError("null distribution produced no finite correlations")
    q_idx = int(math.floor(null_quantile * (len(null_abs) - 1)))
    null_threshold = null_abs[q_idx]
    p_value = (sum(1 for r in null_abs if r >= abs(best["r"])) + 1) / (len(null_abs) + 1)

    r0 = _pearson_r(m, b)
    flags_echo_leak = bool((not math.isnan(r0)) and abs(r0) >= echo_threshold and int(best["lag"]) == 0)

    directionality_ok = True
    if int(best["lag"]) > 0 and (not math.isnan(best_rev["r"])):
        directionality_ok = abs(best["r"]) > abs(best_rev["r"])

    detects_coupling = bool(
        (not math.isnan(best["r"]))
        and abs(best["r"]) >= r_detect_threshold
        and abs(best["r"]) > null_threshold
        and directionality_ok
        and (not flags_echo_leak)
    )

    return {
        "type": "PhysicalForceCouplingReceipt",
        "version": 1,
        "passed": detects_coupling,
        "run": {
            "mode": "csv_coupling",
            "csv_path": csv_path,
            "max_lag": max_lag,
            "best_lag": int(best["lag"]),
            "r_detect_threshold": r_detect_threshold,
            "echo_threshold": echo_threshold,
            "null_kind": null_kind,
            "null_n": null_n,
            "null_quantile": null_quantile,
            "null_rotations": rotations,
            "seed_for_rotations": seed_for_rotations,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": {
            "r_m_to_b_zero": r0,
            "best_r_m_to_b_lag": best["r"],
            "best_r_b_to_m_lag": best_rev["r"],
            "null_threshold_abs_r": null_threshold,
            "p_value": p_value,
            "flags_echo_leak": flags_echo_leak,
            "directionality_ok": directionality_ok,
            "detects_coupling": detects_coupling,
            "lag_scan": lag_scan,
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q32 Phase 6: physical coupling harness (synthetic validators)")
    p.add_argument(
        "--mode",
        choices=["synthetic_validator_suite", "csv_coupling", "emit_synth_csv"],
        default="synthetic_validator_suite",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_steps", type=int, default=240)
    p.add_argument("--lag", type=int, default=3)
    p.add_argument("--coupling_strength", type=float, default=1.5)
    p.add_argument("--noise_std", type=float, default=0.35)
    p.add_argument("--r_detect_threshold", type=float, default=0.35)
    p.add_argument("--echo_threshold", type=float, default=0.80)
    p.add_argument("--threads", type=int, default=1, help="Accepted for CLI parity; not used (stdlib-only).")
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--max_lag", type=int, default=12)
    p.add_argument("--null_rotations", type=str, default="")
    p.add_argument("--seed_for_rotations", type=int, default=2026)
    p.add_argument("--null_kind", choices=["rotate", "permute"], default="rotate")
    p.add_argument("--null_n", type=int, default=250)
    p.add_argument("--null_quantile", type=float, default=0.99)
    p.add_argument("--csv_out", type=str, default=None)
    p.add_argument("--scenario", choices=["positive", "null", "echo_leak"], default="positive")
    p.add_argument("--receipt_out", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.lag < 0:
        raise SystemExit("--lag must be >= 0")
    if args.n_steps < 10:
        raise SystemExit("--n_steps too small (need >= 10)")

    receipt_out = args.receipt_out or _default_receipt_path()
    if args.mode == "synthetic_validator_suite":
        receipt = run_synthetic_validator_suite(
            seed=args.seed,
            n_steps=args.n_steps,
            lag=args.lag,
            coupling_strength=args.coupling_strength,
            noise_std=args.noise_std,
            r_detect_threshold=args.r_detect_threshold,
            echo_threshold=args.echo_threshold,
        )
    elif args.mode == "csv_coupling":
        if not args.csv_path:
            raise SystemExit("--csv_path is required for --mode csv_coupling")
        rotations: List[int] = []
        if args.null_rotations.strip():
            rotations = [int(x.strip()) for x in args.null_rotations.split(",") if x.strip()]
        receipt = run_csv_coupling(
            csv_path=args.csv_path,
            max_lag=args.max_lag,
            r_detect_threshold=args.r_detect_threshold,
            echo_threshold=args.echo_threshold,
            null_rotations=rotations,
            seed_for_rotations=args.seed_for_rotations,
            null_n=args.null_n,
            null_kind=args.null_kind,
            null_quantile=args.null_quantile,
        )
    elif args.mode == "emit_synth_csv":
        if not args.csv_out:
            raise SystemExit("--csv_out is required for --mode emit_synth_csv")
        rng = random.Random(args.seed)
        m = _generate_m_series(rng, args.n_steps)
        if args.scenario == "positive":
            b = _generate_b_from_m_lagged(
                random.Random(args.seed + 1),
                m,
                lag=args.lag,
                coupling_strength=args.coupling_strength,
                noise_std=args.noise_std,
            )
        elif args.scenario == "null":
            b = _generate_b_independent(random.Random(args.seed + 2), n_steps=args.n_steps, noise_std=args.noise_std)
        elif args.scenario == "echo_leak":
            b = _generate_b_echo_leak(random.Random(args.seed + 3), m, noise_std=args.noise_std)
        else:
            raise SystemExit(f"Unknown scenario: {args.scenario}")
        _write_csv_mb(args.csv_out, m, b)
        print(args.csv_out)
        print(_sha256_file(args.csv_out))
        return 0
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")
    receipt["run"]["threads"] = args.threads

    _write_json(receipt_out, receipt)
    print(receipt_out)
    print(_sha256_file(receipt_out))
    return 0 if receipt.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
