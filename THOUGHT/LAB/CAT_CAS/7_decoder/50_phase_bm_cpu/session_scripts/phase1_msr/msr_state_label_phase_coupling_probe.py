#!/usr/bin/env python3
"""Read-only state-label phase/coupling probe for Exp50.

The probe bins rows by observed COFVID/PSTATE labels and timing features while
running a fixed-cost reversible workload. It does not change platform settings.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import struct
import time
from collections import Counter, defaultdict
from statistics import mean, pstdev


PSTATE_STATUS = 0xC0010063
COFVID_STATUS = 0xC0010071


def read_msr(core: int, msr: int) -> int | None:
    try:
        with open(f"/dev/cpu/{core}/msr", "rb", buffering=0) as handle:
            os.lseek(handle.fileno(), msr, os.SEEK_SET)
            return struct.unpack("<Q", handle.read(8))[0]
    except OSError:
        return None


def parse_cores(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return out


def decode(value: int | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "raw": f"0x{value:016x}",
        "fid": value & 0x3F,
        "did": (value >> 6) & 0x07,
        "vid": (value >> 9) & 0x7F,
        "pstate": value & 0x07,
    }


def label_of(item: dict[str, object] | None) -> str:
    if not item:
        return "missing"
    return f"fid={item['fid']:02x}/did={item['did']}/vid={item['vid']:02x}/p={item['pstate']}"


def fnv64(words: list[int]) -> int:
    h = 0xCBF29CE484222325
    for word in words:
        value = word & 0xFFFFFFFFFFFFFFFF
        for _ in range(8):
            h ^= value & 0xFF
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            value >>= 8
    return h


def make_tape(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(64) for _ in range(n)]


def expected_answer(tape: list[int]) -> int:
    acc = 0
    for i, word in enumerate(tape):
        rotated = ((word << (i % 17)) | (word >> (64 - (i % 17)))) & 0xFFFFFFFFFFFFFFFF
        acc ^= rotated ^ ((word * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    return (acc.bit_count() ^ (acc >> 7) ^ (acc >> 19)) & 1


def reversible_workload(tape: list[int], rounds: int) -> tuple[int, int, bool]:
    original_hash = fnv64(tape)
    answer = expected_answer(tape)
    carrier = 0xD6E8FEB86659FD93
    n = len(tape)
    for r in range(rounds):
        idx = (r * 13 + 7) % n
        jdx = (r * 29 + 3) % n
        mask = ((tape[jdx] << 7) ^ (tape[jdx] >> 11) ^ carrier ^ r) & 0xFFFFFFFFFFFFFFFF
        tape[idx] ^= mask
        carrier = ((carrier << 9) ^ (carrier >> 3) ^ tape[idx] ^ (answer << (r % 31))) & 0xFFFFFFFFFFFFFFFF
        tape[idx] ^= mask
    restored_hash = fnv64(tape)
    return answer, carrier, original_hash == restored_hash


def busy_worker(core: int, stop_event: mp.Event) -> int:
    os.sched_setaffinity(0, {core})
    value = 0xA5A5A5A5
    while not stop_event.is_set():
        value = ((value * 22695477) + 1) & 0xFFFFFFFF
        value ^= value >> 9
    return value


def start_workers(mode: str, sample_core: int, cores: list[int]) -> tuple[mp.Event, list[mp.Process]]:
    stop_event = mp.Event()
    if mode == "baseline":
        worker_cores: list[int] = []
    elif mode == "self_load":
        worker_cores = [sample_core]
    elif mode == "neighbor_load":
        idx = cores.index(sample_core)
        worker_cores = [cores[(idx + 1) % len(cores)]]
    elif mode == "other_load":
        worker_cores = [core for core in cores if core != sample_core]
    elif mode == "all_load":
        worker_cores = list(cores)
    else:
        raise ValueError(f"unknown mode: {mode}")
    workers = []
    for core in worker_cores:
        proc = mp.Process(target=busy_worker, args=(core, stop_event))
        proc.start()
        workers.append(proc)
    if workers:
        time.sleep(0.05)
    return stop_event, workers


def stop_workers(stop_event: mp.Event, workers: list[mp.Process]) -> None:
    stop_event.set()
    for proc in workers:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


def row(core: int, mode: str, seed: int, tape_words: int, rounds: int) -> dict[str, object]:
    os.sched_setaffinity(0, {core})
    before_pstate = decode(read_msr(core, PSTATE_STATUS))
    before_cofvid = decode(read_msr(core, COFVID_STATUS))
    tape = make_tape(seed, tape_words)
    start = time.perf_counter_ns()
    answer, carrier, restored = reversible_workload(tape, rounds)
    elapsed = time.perf_counter_ns() - start
    after_pstate = decode(read_msr(core, PSTATE_STATUS))
    after_cofvid = decode(read_msr(core, COFVID_STATUS))
    return {
        "core": core,
        "mode": mode,
        "seed": seed,
        "answer": answer,
        "carrier_low_bit": carrier & 1,
        "elapsed_ns": elapsed,
        "restored": restored,
        "before_pstate": label_of(before_pstate),
        "after_pstate": label_of(after_pstate),
        "before_cofvid": label_of(before_cofvid),
        "after_cofvid": label_of(after_cofvid),
        "state_label": f"{label_of(after_cofvid)}|{label_of(after_pstate)}",
    }


def accuracy(pred: list[int], truth: list[int]) -> float:
    if not truth:
        return 0.0
    return sum(int(a == b) for a, b in zip(pred, truth)) / len(truth)


def balanced_accuracy(pred: list[int], truth: list[int]) -> float:
    parts = []
    for cls in (0, 1):
        idx = [i for i, value in enumerate(truth) if value == cls]
        if idx:
            parts.append(sum(int(pred[i] == truth[i]) for i in idx) / len(idx))
    return sum(parts) / len(parts) if parts else 0.0


def majority_by_label(train: list[dict[str, object]], test: list[dict[str, object]], key: str) -> tuple[float, float]:
    global_majority = Counter(int(r["answer"]) for r in train).most_common(1)[0][0]
    bins: dict[str, Counter[int]] = defaultdict(Counter)
    for r in train:
        bins[str(r[key])][int(r["answer"])] += 1
    pred = []
    truth = []
    for r in test:
        label = str(r[key])
        if label in bins:
            pred.append(bins[label].most_common(1)[0][0])
        else:
            pred.append(global_majority)
        truth.append(int(r["answer"]))
    return accuracy(pred, truth), balanced_accuracy(pred, truth)


def threshold_accuracy(train: list[dict[str, object]], test: list[dict[str, object]]) -> tuple[float, float]:
    values = sorted({int(r["elapsed_ns"]) for r in train})
    if not values:
        return 0.0
    thresholds = values[:: max(1, len(values) // 32)]
    best = (0.0, thresholds[0], 0)
    for t in thresholds:
        for direction in (0, 1):
            pred_train = [int((int(r["elapsed_ns"]) >= t) ^ bool(direction)) for r in train]
            acc = accuracy(pred_train, [int(r["answer"]) for r in train])
            if acc > best[0]:
                best = (acc, t, direction)
    _, threshold, direction = best
    pred_test = [int((int(r["elapsed_ns"]) >= threshold) ^ bool(direction)) for r in test]
    truth_test = [int(r["answer"]) for r in test]
    return accuracy(pred_test, truth_test), balanced_accuracy(pred_test, truth_test)


def threshold_balanced_accuracy_with_truth(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    train_truth: list[int],
    test_truth: list[int],
) -> float:
    values = sorted({int(r["elapsed_ns"]) for r in train})
    if not values:
        return 0.0
    thresholds = values[:: max(1, len(values) // 32)]
    best = (0.0, thresholds[0], 0)
    for t in thresholds:
        for direction in (0, 1):
            pred_train = [int((int(r["elapsed_ns"]) >= t) ^ bool(direction)) for r in train]
            acc = balanced_accuracy(pred_train, train_truth)
            if acc > best[0]:
                best = (acc, t, direction)
    _, threshold, direction = best
    pred_test = [int((int(r["elapsed_ns"]) >= threshold) ^ bool(direction)) for r in test]
    return balanced_accuracy(pred_test, test_truth)


def shuffled_answer_null(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    repeats: int = 32,
) -> dict[str, object]:
    train_truth = [int(r["answer"]) for r in train]
    test_truth = [int(r["answer"]) for r in test]
    combined = train_truth + test_truth
    values = []
    for i in range(repeats):
        rng = random.Random(0xC0FFEE + i)
        shuffled = list(combined)
        rng.shuffle(shuffled)
        st = shuffled[: len(train_truth)]
        sv = shuffled[len(train_truth):]
        values.append(threshold_balanced_accuracy_with_truth(train, test, st, sv))
    values_sorted = sorted(values)
    return {
        "repeats": repeats,
        "mean": round(mean(values), 6),
        "max": round(max(values), 6) if values else 0.0,
        "p95": round(values_sorted[int(0.95 * (len(values_sorted) - 1))], 6) if values_sorted else 0.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    train = [r for r in rows if int(r["seed"]) % 4 != 0]
    test = [r for r in rows if int(r["seed"]) % 4 == 0]
    elapsed = [float(r["elapsed_ns"]) for r in rows]
    state_labels = sorted({str(r["state_label"]) for r in rows})
    mode_labels = sorted({str(r["mode"]) for r in rows})
    core_labels = sorted({str(r["core"]) for r in rows})
    carrier_pred = [int(r["carrier_low_bit"]) for r in test]
    truth = [int(r["answer"]) for r in test]
    elapsed_acc, elapsed_bacc = threshold_accuracy(train, test)
    shuffled_null = shuffled_answer_null(train, test)
    state_acc, state_bacc = majority_by_label(train, test, "state_label")
    mode_acc, mode_bacc = majority_by_label(train, test, "mode")
    core_acc, core_bacc = majority_by_label(train, test, "core")
    return {
        "row_count": len(rows),
        "train_count": len(train),
        "test_count": len(test),
        "restore_failures": sum(1 for r in rows if not r["restored"]),
        "answer_balance": dict(Counter(int(r["answer"]) for r in rows)),
        "state_label_count": len(state_labels),
        "mode_count": len(mode_labels),
        "core_count": len(core_labels),
        "elapsed_mean_ns": round(mean(elapsed), 3) if elapsed else None,
        "elapsed_cv": round((pstdev(elapsed) / mean(elapsed)), 9) if elapsed and mean(elapsed) else None,
        "carrier_low_bit_holdout_accuracy": round(accuracy(carrier_pred, truth), 6),
        "carrier_low_bit_holdout_balanced_accuracy": round(balanced_accuracy(carrier_pred, truth), 6),
        "elapsed_threshold_holdout_accuracy": round(elapsed_acc, 6),
        "elapsed_threshold_holdout_balanced_accuracy": round(elapsed_bacc, 6),
        "elapsed_shuffled_answer_null": shuffled_null,
        "elapsed_over_shuffle_p95": round(elapsed_bacc - float(shuffled_null["p95"]), 6),
        "state_label_holdout_accuracy": round(state_acc, 6),
        "state_label_holdout_balanced_accuracy": round(state_bacc, 6),
        "mode_label_holdout_accuracy": round(mode_acc, 6),
        "mode_label_holdout_balanced_accuracy": round(mode_bacc, 6),
        "core_label_holdout_accuracy": round(core_acc, 6),
        "core_label_holdout_balanced_accuracy": round(core_bacc, 6),
        "state_labels": state_labels[:32],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only state-label phase/coupling probe")
    parser.add_argument("--cores", default="0-5")
    parser.add_argument("--modes", default="baseline,self_load,neighbor_load,other_load,all_load")
    parser.add_argument("--trials-per-case", type=int, default=12)
    parser.add_argument("--tape-words", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=4096)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()

    cores = parse_cores(args.cores)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows: list[dict[str, object]] = []
    original_affinity = os.sched_getaffinity(0)
    try:
        seed = args.seed_start
        for mode in modes:
            for core in cores:
                stop_event, workers = start_workers(mode, core, cores)
                try:
                    for _ in range(args.trials_per_case):
                        rows.append(row(core, mode, seed, args.tape_words, args.rounds))
                        seed += 1
                finally:
                    stop_workers(stop_event, workers)
    finally:
        os.sched_setaffinity(0, original_affinity)

    result = {
        "mode": "read_only_state_label_phase_coupling_probe",
        "setting_changes": False,
        "cores": cores,
        "modes": modes,
        "trials_per_case": args.trials_per_case,
        "tape_words": args.tape_words,
        "rounds": args.rounds,
        "seed_start": args.seed_start,
        "acceptance": [
            "restore_failures must be zero",
            "state_label or elapsed holdout balanced accuracy must beat mode/core label controls by >=0.10",
            "claim is rejected if only the internally computed carrier bit predicts the answer",
        ],
        "summary": summarize(rows),
        "rows": rows,
    }
    s = result["summary"]
    best_observed = max(float(s["elapsed_threshold_holdout_balanced_accuracy"]), float(s["state_label_holdout_balanced_accuracy"]))
    best_control = max(float(s["mode_label_holdout_balanced_accuracy"]), float(s["core_label_holdout_balanced_accuracy"]))
    shuffle_margin = float(s["elapsed_over_shuffle_p95"])
    result["verdict"] = (
        "STATE_LABEL_PHASE_COUPLING_CANDIDATE"
        if s["restore_failures"] == 0 and best_observed >= best_control + 0.10 and best_observed >= 0.60 and shuffle_margin >= 0.05
        else "STATE_LABEL_PHASE_COUPLING_NOT_CONFIRMED"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
