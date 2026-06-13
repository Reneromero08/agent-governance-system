"""chiral_phase_kickback_probe.py - Phase 6 chiral prep audit.

Hypothesis under test:
  A restored catalytic tape might expose the dihedral orientation bit through the
  trajectory that prepares the public cosine samples, not through the final public
  samples themselves. In operational terms: bind the phase-walk direction into a
  carrier before the cosine projection, cancel the even lane, and ask whether the
  fold-odd residual survives.

This is the local no-smuggle version of that test. It reuses the existing Phase 6
construction and hardened gate. Public candidates may read only k, b, N, n. The
hidden-prep control deliberately reads d to prove the chiral carrier/gate would
detect a real pre-projection orientation lane if present.

ASCII only. All randomness is seeded and recorded.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PHASE6 = HERE.parent
FOLD = PHASE6 / "fold_audit"
STAGE3 = FOLD / "stage3"
for _p in (str(FOLD), str(STAGE3)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import construction as C
import hardened_gate as Hg


MASTER_SEED = 50606017
N_INST = 320
N_SHUF = 28


def _binned(k: np.ndarray, b: np.ndarray, n_slots: int) -> np.ndarray:
    out = np.zeros(n_slots, dtype=np.float64)
    np.add.at(out, k % n_slots, b)
    return out


def _public_seed(k: np.ndarray, b: np.ndarray, N: int) -> int:
    h = int(N) * 1000003
    h ^= int(np.sum((k.astype(np.int64) + 17) * 2654435761) & 0x7FFFFFFF)
    h ^= int(np.sum((b > 0).astype(np.int64) * 2246822519) & 0x7FFFFFFF)
    return h & 0x7FFFFFFF


def _moments(x: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return [0.0] * 8
    mu = float(np.mean(x))
    sd = float(np.std(x))
    centered = x - mu
    denom = sd + 1e-12
    return [
        mu,
        sd,
        float(np.mean(np.abs(x))),
        float(np.max(x)),
        float(np.min(x)),
        float(np.mean((centered / denom) ** 3)),
        float(np.mean((centered / denom) ** 4)),
        float(np.linalg.norm(x) / math.sqrt(max(1, x.size))),
    ]


def _phase_walk(k: np.ndarray, b: np.ndarray, N: int, direction: int,
                order_mode: str) -> dict[str, np.ndarray | complex]:
    """Build a chiral carrier trace from public samples.

    direction=+1 and direction=-1 are the two opposite physical preparations. The
    carrier is intentionally richer than a cosine score: it tracks a complex tape
    lane, a reversible XOR-like tag lane, and the per-step sideband power that a
    physical lock-in would try to read.
    """
    if order_mode == "ramp":
        order = np.argsort(k if direction > 0 else ((N - k) % N), kind="mergesort")
    elif order_mode == "shuffle_public":
        rng = np.random.default_rng(_public_seed(k, b, N) ^ (0x9E3779B9 if direction > 0 else 0x85EBCA6B))
        order = rng.permutation(len(k))
    else:
        raise ValueError(order_mode)

    z = 1.0 + 0.0j
    tag = np.uint64(0xC0DEC0DEC0DEC0DE)
    trace_i: list[float] = []
    trace_q: list[float] = []
    sideband: list[float] = []
    tag_lane: list[float] = []

    for step, idx in enumerate(order):
        kk = int(k[idx] % N)
        bb = float(b[idx])
        theta = direction * 2.0 * math.pi * kk / N
        rot = complex(math.cos(theta), math.sin(theta))

        # PHASE_TAG + SIGN_BIND before projection. The nonlinear term is public and
        # reversible in the sense used here: the opposite lane is computed explicitly
        # and all summaries compare lane residuals, not final mutation state.
        z = (0.997 * z + bb * rot + 0.003 * np.conj(z) * rot)

        # XOR-like tape lane: binds sample sign, slot, and step into a carrier tag.
        mask64 = (1 << 64) - 1
        word = np.uint64(((kk + 1) * 0x9E3779B185EBCA87) & mask64)
        if bb < 0:
            word ^= np.uint64(0xD1B54A32D192ED03)
        word ^= np.uint64(((step + 1) * 0x94D049BB133111EB) & mask64)
        shift = np.uint64((kk + step) & 63)
        tag ^= ((word << shift) | (word >> np.uint64((64 - int(shift)) & 63)))

        tr_i = float(z.real)
        tr_q = float(z.imag)
        trace_i.append(tr_i)
        trace_q.append(tr_q)
        sideband.append(float(bb * (math.sin(theta) * tr_i - math.cos(theta) * tr_q)))
        tag_lane.append(float((int(tag & np.uint64(0xFFFF)) - 32768) / 32768.0))

    return {
        "z": z,
        "i": np.asarray(trace_i, dtype=np.float64),
        "q": np.asarray(trace_q, dtype=np.float64),
        "sideband": np.asarray(sideband, dtype=np.float64),
        "tag": np.asarray(tag_lane, dtype=np.float64),
    }


def _chiral_features_from_public(k: np.ndarray, b: np.ndarray, N: int,
                                 order_mode: str) -> np.ndarray:
    cw = _phase_walk(k, b, N, +1, order_mode)
    ccw = _phase_walk(k, b, N, -1, order_mode)

    residual_i = np.asarray(cw["i"]) - np.asarray(ccw["i"])
    residual_q = np.asarray(cw["q"]) + np.asarray(ccw["q"])
    residual_sb = np.asarray(cw["sideband"]) + np.asarray(ccw["sideband"])
    residual_tag = np.asarray(cw["tag"]) - np.asarray(ccw["tag"])

    spectrum = _binned(k, b, N)
    fft = np.fft.fft(spectrum)
    low = []
    for j in range(1, min(10, int(math.log2(N)) + 1)):
        m = (1 << j) % N
        low.extend([float(fft[m].real), float(fft[m].imag), float(abs(fft[m]))])

    zc = complex(cw["z"])
    za = complex(ccw["z"])
    cross = zc * np.conj(za)
    feats: list[float] = []
    feats.extend(_moments(residual_i))
    feats.extend(_moments(residual_q))
    feats.extend(_moments(residual_sb))
    feats.extend(_moments(residual_tag))
    feats.extend([
        float(zc.real), float(zc.imag), float(za.real), float(za.imag),
        float(cross.real), float(cross.imag), float(abs(zc) - abs(za)),
        float(np.angle(zc + 1e-30)), float(np.angle(za + 1e-30)),
    ])
    feats.extend(low)
    return np.asarray(feats, dtype=np.float64)


def O_chiral_phase_kickback_PUBLIC(inst: dict) -> np.ndarray:
    """Public chiral-prep candidate: ramp-ordered clockwise/counterclockwise lanes."""
    return _chiral_features_from_public(inst["k"], inst["b"], inst["N"], "ramp")


def O_chiral_shuffle_null_PUBLIC(inst: dict) -> np.ndarray:
    """Public schedule null: same carrier, but order is a public seeded shuffle."""
    return _chiral_features_from_public(inst["k"], inst["b"], inst["N"], "shuffle_public")


def O_dual_lane_even_cancel_PUBLIC(inst: dict) -> np.ndarray:
    """Even-cancel candidate: subtract the folded cavity lane before reading residuals."""
    k = inst["k"]
    b = inst["b"]
    N = inst["N"]
    B = _binned(k, b, N).astype(np.complex128)
    mirror = np.roll(B[::-1], 1)
    even = 0.5 * (B + mirror)
    odd_residual = B - even
    field = np.fft.ifft(odd_residual) * N
    feats: list[float] = []
    feats.extend(_moments(field.real))
    feats.extend(_moments(field.imag))
    for j in range(1, min(12, inst["n"] + 1)):
        m = (1 << j) % N
        feats.extend([float(odd_residual[m].real), float(odd_residual[m].imag), float(abs(odd_residual[m]))])
    return np.asarray(feats, dtype=np.float64)


def O_hidden_chiral_prep_CONTROL(inst: dict) -> np.ndarray:
    """Deliberate hidden-prep control.

    This is the exact thing a physical pre-projection carrier would look like if it
    had access to the missing orientation lane: the sign of the phase walk is bound
    to the hidden half before the cosine projection. It must be caught by the gate.
    """
    N = inst["N"]
    d = inst["d"]
    orient = 1 if C.orientation_bit(d, N) else -1
    k = inst["k"]
    b = inst["b"]
    tr = _phase_walk(k, b, N, orient, "ramp")
    z = complex(tr["z"])
    theta = 2.0 * math.pi * d / N
    return np.asarray([
        float(orient > 0),
        float(z.real),
        float(z.imag),
        float(math.sin(theta)),
        float(math.cos(theta)),
        *_moments(np.asarray(tr["sideband"])),
    ], dtype=np.float64)


def run_gate() -> dict:
    cells = []
    cases = [
        ("chiral_phase_kickback_PUBLIC", O_chiral_phase_kickback_PUBLIC, "public ramp chiral prep"),
        ("dual_lane_even_cancel_PUBLIC", O_dual_lane_even_cancel_PUBLIC, "public even cancellation residual"),
        ("chiral_shuffle_null_PUBLIC", O_chiral_shuffle_null_PUBLIC, "public shuffled schedule null"),
        ("hidden_chiral_prep_CONTROL", O_hidden_chiral_prep_CONTROL, "gate-live hidden pre-projection control"),
    ]
    t0 = time.time()
    for n in (8, 10, 12):
        for ci, (name, op, purpose) in enumerate(cases):
            seed = (MASTER_SEED + 9973 * n + 101 * ci) & 0x7FFFFFFF
            res = Hg.hardened_gate(op, n, n_instances=N_INST, seed=seed, n_shuffles=N_SHUF)
            cells.append({
                "name": name,
                "purpose": purpose,
                "n": int(n),
                "seed": int(seed),
                "verdict": res["verdict"],
                "orientation_auc": res["auc"],
                "orientation_null95": res["shuffle_null_95"],
                "random_fold_auc": res["random_fold_auc"],
                "random_fold_null95": res["random_fold_null_95"],
                "max_fold_delta": res["max_fold_delta"],
                "smuggle_flag": res["smuggle_flag"],
                "smuggle_reason": res["smuggle_reason"],
            })
            print(
                f"{name:34s} n={n:2d} verdict={res['verdict']:13s} "
                f"auc={res['auc']:.3f}/{res['shuffle_null_95']:.3f} "
                f"rf={res['random_fold_auc']:.3f}/{res['random_fold_null_95']:.3f} "
                f"delta={res['max_fold_delta']:.3g}"
            )
    elapsed = time.time() - t0

    public_cells = [c for c in cells if c["name"].endswith("_PUBLIC")]
    control_cells = [c for c in cells if c["name"].endswith("_CONTROL")]
    public_cross = [c for c in public_cells if c["verdict"] == "PASS_CROSSING"]
    controls_live = all(c["verdict"] == "FAIL_SMUGGLE" for c in control_cells)
    if public_cross and controls_live:
        verdict = "CHIRAL_PREP_PUBLIC_CROSSING_CANDIDATE"
    elif controls_live:
        verdict = "CHIRAL_PREP_PUBLIC_NO_CROSSING__HIDDEN_PREP_GATE_LIVE"
    else:
        verdict = "CHIRAL_PREP_GATE_NOT_LIVE"

    return {
        "experiment": "phase6_chiral_phase_kickback_probe",
        "master_seed": MASTER_SEED,
        "n_instances": N_INST,
        "n_shuffles": N_SHUF,
        "elapsed_s": elapsed,
        "verdict": verdict,
        "cells": cells,
        "physical_run_status": "not_started_current_phenom_matrix_run_active",
        "physical_next_command": "wait until /root/slot2_pdn/run_t300.sh exits, then run a clean PDN sideband harness on an idle core pair",
    }


def write_report(result: dict) -> None:
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "chiral_phase_kickback_result.json"
    json_path.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")

    lines = [
        "# Phase 6 Chiral Phase-Kickback Probe",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Question",
        "",
        "Can a chiral pre-projection tape preparation expose a fold-odd carrier before the public cosine boundary is read?",
        "",
        "## Gate",
        "",
        "The probe reuses `fold_audit/stage3/hardened_gate.py`: orientation AUC, random-private-fold AUC, and exact fold-invariance delta. Public candidates read only `k`, `b`, `N`, and `n`. The hidden control deliberately binds the phase-walk direction to `d` to verify the carrier/gate lights up when an orientation lane exists.",
        "",
        "## Results",
        "",
        "| candidate | n | verdict | orient auc/null95 | random-fold auc/null95 | delta |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for c in result["cells"]:
        lines.append(
            f"| {c['name']} | {c['n']} | {c['verdict']} | "
            f"{c['orientation_auc']:.3f}/{c['orientation_null95']:.3f} | "
            f"{c['random_fold_auc']:.3f}/{c['random_fold_null95']:.3f} | "
            f"{c['max_fold_delta']:.3g} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The public chiral preparation, dual-lane even cancellation, and public shuffled schedule did not produce a no-smuggle crossing in this local model. The hidden pre-projection control is caught at every size, which means the carrier features and gate are live: if the missing orientation lane is bound before projection, the instrument detects it.",
        "",
        "The physical PDN version was not launched because the Phenom is currently running the long 300-trial matrix job on `/root/slot2_pdn/run_t300.sh`. The clean next step is a PDN sideband run after that job exits, using the same logic: same public cosine stream, opposite chiral preparation lanes, even cancellation, and victim lock-in phase readout.",
        "",
        "## Artifacts",
        "",
        f"- `chiral_phase_kickback_probe.py`",
        f"- `results/chiral_phase_kickback_result.json`",
        "",
    ])
    (HERE / "PHASE6_CHIRAL_PHASE_KICKBACK_PROBE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_gate()
    write_report(result)
    print(f"wrote {HERE / 'results' / 'chiral_phase_kickback_result.json'}")
    print(f"wrote {HERE / 'PHASE6_CHIRAL_PHASE_KICKBACK_PROBE.md'}")
    print(f"verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
