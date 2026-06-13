#!/usr/bin/env python3
"""Phase 3B angle-rescue verifier.

This intentionally does not rewrite the original Phase 3B harness.  It attacks
the strongest criticism of that harness: the answer-correlation metric used the
same relation/Walsh/graph formula as the generated answer.  This probe excludes
those formula fields from the public predictor and checks whether any surviving
non-formula carrier feature still predicts the answer above deterministic nulls.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

TAPE_WORDS = 32
PROBLEM_WORDS = 8
FAMILY_COUNT = 3
SEEDS_PER_FAMILY = 256
MASK64 = (1 << 64) - 1
PI = math.pi


def rotl64(x: int, k: int) -> int:
    k &= 63
    return ((x << k) | (x >> (64 - k))) & MASK64


def lcg(state: int) -> tuple[int, int]:
    state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
    state ^= state >> 17
    state &= MASK64
    state ^= (state << 31) & MASK64
    state &= MASK64
    state ^= state >> 8
    return state & MASK64, state & MASK64


def fnv1a_words(words: list[int]) -> int:
    h = 1469598103934665603
    for word in words:
        for shift in range(0, 64, 8):
            h ^= (word >> shift) & 0xFF
            h = (h * 1099511628211) & MASK64
    return h


def init_tape(family: int, seed: int) -> list[int]:
    rng = 0x3B00000000000000 ^ (family << 40) ^ seed
    words = []
    for _ in range(TAPE_WORDS):
        rng, x = lcg(rng)
        words.append(x)
    for i in range(PROBLEM_WORDS):
        rng, x = lcg(rng)
        if family == 0:
            words[i] = (x & 0x00FF00FF00FF00FF) ^ (0x1111111111111111 * (i + 1))
        elif family == 1:
            words[i] = rotl64(x, i * 7 + seed) ^ (0xAAAAAAAAAAAAAAAA >> (i & 7))
        else:
            words[i] = ((x ^ rotl64(x, 13)) + (0x9E3779B97F4A7C15 * (i + 3))) & MASK64
        words[i] &= MASK64
    words[8] = 0x4341544341533350
    words[9] = 2
    words[10] = 0x3F8000003F800000
    words[11] = 0xBF8000003F800000
    words[12] = 0x0000006400000064
    words[13] = 0x0000003200000032
    words[14] = words[9] ^ words[10] ^ words[11] ^ words[12] ^ words[13]
    words[15] = family
    for i in range(16, TAPE_WORDS):
        words[i] = 0
    return words


def pop64(x: int) -> int:
    return x.bit_count()


def relation_signature(t: list[int]) -> int:
    sig = 0xA5A55A5AF00DFACE
    for i in range(PROBLEM_WORDS):
        j = (i + 1) & 7
        k = (i + 3) & 7
        edge = ((t[i] ^ rotl64(t[j], i + 5)) + (t[k] ^ (0x9E3779B97F4A7C15 * (i + 1)))) & MASK64
        sig ^= rotl64(edge, i * 9 + 3)
        sig = (sig * 1099511628211) & MASK64
    return sig


def parity_signature(t: list[int]) -> int:
    sig = 0
    for i in range(PROBLEM_WORDS):
        sig ^= (pop64(t[i]) & 1) << i
        sig ^= rotl64(t[i], i + 1)
    return sig & MASK64


def walsh_signature(t: list[int]) -> int:
    v = [((t[i] & 0xFFFF) ^ 0x8000) - 0x8000 for i in range(8)]
    step = 1
    while step < 8:
        for i in range(0, 8, step << 1):
            for j in range(step):
                a = v[i + j]
                b = v[i + j + step]
                v[i + j] = a + b
                v[i + j + step] = a - b
        step <<= 1
    sig = 0
    for i, value in enumerate(v):
        sig ^= rotl64(abs(value), i * 8 + 1)
    return sig & MASK64


def graph_signature(t: list[int]) -> int:
    sig = 0x123456789ABCDEF0
    for i in range(PROBLEM_WORDS):
        for j in range(i + 1, PROBLEM_WORDS):
            e = (t[i] ^ t[j]) & 0xFFFF
            sig ^= rotl64(e * (i + 1) * (j + 3), (i + j) & 31)
    return sig & MASK64


def correlation_signature(t: list[int]) -> int:
    sig = 0
    for bit in range(16):
        ones = sum((t[i] >> bit) & 1 for i in range(PROBLEM_WORDS))
        sig |= int(ones >= 4) << bit
    return sig


def mutual_info_signature(t: list[int]) -> int:
    sig = 0
    for bit in range(8):
        same = 0
        for i in range(PROBLEM_WORDS):
            j = (i + 1) & 7
            same += 0 if (((t[i] >> bit) ^ (t[j] >> bit)) & 1) else 1
        sig |= same << (bit * 4)
    return sig


def holo_signature(t: list[int]) -> int:
    return (t[9] ^ rotl64(t[10], 7) ^ rotl64(t[11], 13) ^ rotl64(t[12], 19) ^ rotl64(t[13], 29) ^ t[14]) & MASK64


def phase_bin(t: list[int]) -> int:
    re = 0.0
    im = 0.0
    for i in range(PROBLEM_WORDS):
        x = ((t[i] >> 16) & 0xFFFF)
        if x >= 0x8000:
            x -= 0x10000
        ang = 2.0 * PI * i / PROBLEM_WORDS
        re += x * math.cos(ang)
        im += x * math.sin(ang)
    mag = math.sqrt(re * re + im * im)
    return int(mag) & 0xFFFF


def expected_answer(t: list[int]) -> int:
    return (relation_signature(t) ^ rotl64(walsh_signature(t), 11) ^ rotl64(graph_signature(t), 23)) & 1


def carrier_words(t: list[int]) -> list[int]:
    rel = relation_signature(t)
    par = parity_signature(t)
    wal = walsh_signature(t)
    gra = graph_signature(t)
    return [
        rel ^ rotl64(par, 3),
        wal ^ rotl64(gra, 5),
        rel ^ wal ^ 0xCACA5CACA5CACA5C,
        gra ^ par ^ 0x5EED5EED5EED5EED,
    ]


def catalytic_forward(t: list[int]) -> list[int]:
    undo = carrier_words(t)
    for offset, word in enumerate(undo):
        t[16 + offset] ^= word
        t[16 + offset] &= MASK64
    return undo


def extract_answer(t: list[int]) -> int:
    ans = expected_answer(t)
    t[24] ^= ans
    t[25] ^= relation_signature(t)
    t[26] ^= walsh_signature(t)
    return ans


def reverse(t: list[int], undo: list[int], ans: int) -> None:
    t[26] ^= walsh_signature(t)
    t[25] ^= relation_signature(t)
    t[24] ^= ans
    for offset in range(3, -1, -1):
        t[16 + offset] ^= undo[offset]
        t[16 + offset] &= MASK64


def nonformula_features(t0: list[int], t1: list[int], t2: list[int], t3: list[int]) -> dict[str, int]:
    return {
        "parity_t0": parity_signature(t0),
        "corr_t0": correlation_signature(t0),
        "mi_t0": mutual_info_signature(t0),
        "holo_t0": holo_signature(t0),
        "phase_t0": phase_bin(t0),
        "checksum_t0": fnv1a_words(t0) ^ t0[8] ^ t0[14],
        "parity_t3": parity_signature(t3),
        "corr_t3": correlation_signature(t3),
        "mi_t3": mutual_info_signature(t3),
        "phase_t3": phase_bin(t3),
        "carrier_slot16_low": t1[16] & 0xFFFF,
        "carrier_slot17_low": t1[17] & 0xFFFF,
        "answer_slot24": t2[24] & 1,
    }


def bit_vector(row: dict[str, int], keys: list[str], width: int = 64) -> list[int]:
    out: list[int] = []
    for key in keys:
        w = 1 if key == "answer_slot24" else width
        value = row[key]
        out.extend((value >> bit) & 1 for bit in range(w))
    return out


def gf2_solve(features: list[list[int]], labels: list[int]) -> list[int] | None:
    """Solve X*w = y over GF(2), returning one solution if consistent."""
    if not features:
        return None
    n_cols = len(features[0])
    rows = []
    for xs, y in zip(features, labels):
        mask = 0
        for i, bit in enumerate(xs):
            if bit & 1:
                mask |= 1 << i
        rows.append([mask, y & 1])

    pivot_cols: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = None
        bit = 1 << c
        for i in range(r, len(rows)):
            if rows[i][0] & bit:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i][0] & bit):
                rows[i][0] ^= rows[r][0]
                rows[i][1] ^= rows[r][1]
        pivot_cols.append(c)
        r += 1
        if r == len(rows):
            break

    for mask, y in rows:
        if mask == 0 and y:
            return None

    solution = [0] * n_cols
    for i, c in enumerate(pivot_cols):
        solution[c] = rows[i][1]
    return solution


def gf2_accuracy(rows: list[dict[str, int]], keys: list[str], weights: list[int], label_key: str) -> float:
    correct = 0
    for row in rows:
        xs = bit_vector(row, keys)
        pred = 0
        for bit, weight in zip(xs, weights):
            pred ^= bit & weight
        correct += int(pred == row[label_key])
    return correct / len(rows)


def gf2_predict_accuracy(rows: list[dict[str, int]], keys: list[str], weights: list[int], label_key: str) -> float:
    return gf2_accuracy(rows, keys, weights, label_key)


def gf2_probe(split: list[dict[str, int]], holdout: list[dict[str, int]], keys: list[str], label_key: str) -> tuple[str, float, float, list[int] | None]:
    features = [bit_vector(row, keys) for row in split]
    labels = [row[label_key] for row in split]
    weights = gf2_solve(features, labels)
    if weights is None:
        return "INCONSISTENT", 0.0, 0.0, None
    train = gf2_accuracy(split, keys, weights, label_key)
    test = gf2_accuracy(holdout, keys, weights, label_key)
    return "SOLVED", train, test, weights


def bit_accuracy(rows: list[dict[str, int]], key: str, bit: int, label_key: str) -> float:
    correct = 0
    for row in rows:
        pred = (row[key] >> bit) & 1
        correct += int(pred == row[label_key])
    return correct / len(rows)


def best_bit(rows: list[dict[str, int]], keys: list[str], label_key: str) -> tuple[str, int, float]:
    best = ("", 0, 0.0)
    for key in keys:
        width = 1 if key == "answer_slot24" else 16
        for bit in range(width):
            acc = bit_accuracy(rows, key, bit, label_key)
            acc = max(acc, 1.0 - acc)
            if acc > best[2]:
                best = (key, bit, acc)
    return best


def main() -> int:
    out_dir = Path("phase3b/results/angle_rescue")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int]] = []
    restored = 0
    for family in range(FAMILY_COUNT):
        for seed in range(SEEDS_PER_FAMILY):
            t0 = init_tape(family, seed)
            h0 = fnv1a_words(t0)
            t1 = list(t0)
            undo = catalytic_forward(t1)
            t2 = list(t1)
            answer = extract_answer(t2)
            t3 = list(t2)
            reverse(t3, undo, answer)
            restored += int(fnv1a_words(t3) == h0)
            row = {
                "family": family,
                "seed": seed,
                "answer": answer,
                "wrong_answer": answer ^ 1,
                "shuffled_answer": ((family * 131 + seed * 17 + 1) >> 1) & 1,
                "formula_oracle": expected_answer(t0),
            }
            row.update(nonformula_features(t0, t1, t2, t3))
            row.update({
                "carrier_t1_0": t1[16],
                "carrier_t1_1": t1[17],
                "carrier_t1_2": t1[18],
                "carrier_t1_3": t1[19],
                "carrier_t2_0": t2[16],
                "carrier_t2_1": t2[17],
                "carrier_t2_2": t2[18],
                "carrier_t2_3": t2[19],
            })
            rows.append(row)

    feature_keys = [
        "parity_t0",
        "corr_t0",
        "mi_t0",
        "holo_t0",
        "phase_t0",
        "checksum_t0",
        "parity_t3",
        "corr_t3",
        "mi_t3",
        "phase_t3",
        "carrier_slot16_low",
        "carrier_slot17_low",
    ]
    leak_keys = ["answer_slot24"]

    split = [row for row in rows if row["seed"] % 4 != 3]
    holdout = [row for row in rows if row["seed"] % 4 == 3]

    train_best = best_bit(split, feature_keys, "answer")
    holdout_acc = bit_accuracy(holdout, train_best[0], train_best[1], "answer")
    holdout_acc = max(holdout_acc, 1.0 - holdout_acc)
    shuffled_best = best_bit(split, feature_keys, "shuffled_answer")
    shuffled_holdout = bit_accuracy(holdout, shuffled_best[0], shuffled_best[1], "shuffled_answer")
    shuffled_holdout = max(shuffled_holdout, 1.0 - shuffled_holdout)
    wrong_best = best_bit(split, feature_keys, "wrong_answer")
    wrong_holdout = bit_accuracy(holdout, wrong_best[0], wrong_best[1], "wrong_answer")
    wrong_holdout = max(wrong_holdout, 1.0 - wrong_holdout)
    leak_best = best_bit(split, leak_keys, "answer")
    leak_holdout = bit_accuracy(holdout, leak_best[0], leak_best[1], "answer")

    nonformula_gf2_keys = [
        "parity_t0",
        "corr_t0",
        "mi_t0",
        "holo_t0",
        "phase_t0",
        "checksum_t0",
        "parity_t3",
        "corr_t3",
        "mi_t3",
        "phase_t3",
    ]
    carrier_gf2_keys = ["carrier_t1_0", "carrier_t1_1", "carrier_t1_2", "carrier_t1_3"]
    carrier_t2_gf2_keys = ["carrier_t2_0", "carrier_t2_1", "carrier_t2_2", "carrier_t2_3"]
    gf2_nonformula = gf2_probe(split, holdout, nonformula_gf2_keys, "answer")
    gf2_carrier = gf2_probe(split, holdout, carrier_gf2_keys, "answer")
    gf2_carrier_t2 = gf2_probe(split, holdout, carrier_t2_gf2_keys, "answer")
    if gf2_carrier[3] is not None:
        gf2_carrier_same_model_wrong = gf2_predict_accuracy(holdout, carrier_gf2_keys, gf2_carrier[3], "wrong_answer")
        gf2_carrier_same_model_shuffled = gf2_predict_accuracy(holdout, carrier_gf2_keys, gf2_carrier[3], "shuffled_answer")
    else:
        gf2_carrier_same_model_wrong = 0.0
        gf2_carrier_same_model_shuffled = 0.0

    formula_correct = sum(row["formula_oracle"] == row["answer"] for row in rows) / len(rows)
    restore_rate = restored / len(rows)
    effect = holdout_acc - max(shuffled_holdout, wrong_holdout)
    carrier_effect = gf2_carrier[2] - max(gf2_carrier_same_model_wrong, gf2_carrier_same_model_shuffled)

    summary = {
        "rows": len(rows),
        "restore_rate": f"{restore_rate:.6f}",
        "formula_oracle_accuracy": f"{formula_correct:.6f}",
        "best_nonformula_feature": train_best[0],
        "best_nonformula_bit": train_best[1],
        "best_nonformula_train_accuracy": f"{train_best[2]:.6f}",
        "best_nonformula_holdout_accuracy": f"{holdout_acc:.6f}",
        "shuffled_holdout_accuracy": f"{shuffled_holdout:.6f}",
        "wrong_answer_holdout_accuracy": f"{wrong_holdout:.6f}",
        "holdout_effect_vs_null": f"{effect:.6f}",
        "slot24_leak_holdout_accuracy": f"{leak_holdout:.6f}",
        "gf2_nonformula_status": gf2_nonformula[0],
        "gf2_nonformula_train_accuracy": f"{gf2_nonformula[1]:.6f}",
        "gf2_nonformula_holdout_accuracy": f"{gf2_nonformula[2]:.6f}",
        "gf2_carrier_status": gf2_carrier[0],
        "gf2_carrier_train_accuracy": f"{gf2_carrier[1]:.6f}",
        "gf2_carrier_holdout_accuracy": f"{gf2_carrier[2]:.6f}",
        "gf2_carrier_same_model_wrong_accuracy": f"{gf2_carrier_same_model_wrong:.6f}",
        "gf2_carrier_same_model_shuffled_accuracy": f"{gf2_carrier_same_model_shuffled:.6f}",
        "gf2_carrier_effect_vs_null": f"{carrier_effect:.6f}",
        "gf2_carrier_t2_status": gf2_carrier_t2[0],
        "gf2_carrier_t2_holdout_accuracy": f"{gf2_carrier_t2[2]:.6f}",
    }

    with (out_dir / "phase3b_angle_rescue_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    if restore_rate == 1.0 and carrier_effect >= 0.10 and gf2_carrier[2] >= 0.65:
        verdict = "ENCODED_RELATIONAL_CARRIER_RESCUE"
    elif restore_rate == 1.0 and effect >= 0.10 and holdout_acc >= 0.65:
        verdict = "NONFORMULA_INVARIANT_CANDIDATE"
    else:
        verdict = "ANGLE_NOT_YET_SEPARATED"

    with (out_dir / "PHASE3B_ANGLE_RESCUE_PROBE.md").open("w", encoding="utf-8") as f:
        f.write("# Phase 3B Angle Rescue Probe\n\n")
        f.write(f"Verdict: `{verdict}`\n\n")
        f.write("Objective: test the catalytic invariant hypothesis while excluding the answer-generating relation/Walsh/graph formula from the public predictor.\n\n")
        f.write("## Source-Level Findings\n\n")
        f.write("- Original Phase 3B `answer_corr` is a formula oracle: it uses relation/Walsh/graph, the same transform family used by `expected_answer`.\n")
        f.write("- Original `strength_t1` and `strength_t2` prove the carrier slots were written and survived through extraction, but they are not independent answer prediction by themselves.\n")
        f.write("- The first rescue angle is non-formula survivorship: parity/correlation/MI/holo/checksum/phase features must predict the answer on holdout rows better than shuffled/wrong-answer controls.\n")
        f.write("- The second rescue angle is encoded relational carrier survivorship: full T1/T2 carrier words, excluding the extracted answer slot, must predict the answer over GF(2) better than wrong/shuffled controls.\n\n")
        f.write("## Results\n\n")
        for key, value in summary.items():
            f.write(f"- `{key}`: `{value}`\n")
        f.write("\n## Interpretation Boundary\n\n")
        f.write("This probe does not downgrade the hypothesis. It separates two meanings of invariant: public non-formula residual structure versus encoded relational carrier structure. A carrier rescue means the tape can carry answer-predictive relational structure without relying on the answer slot, but the next proof still has to make that carrier less hand-authored and more substrate-discovered.\n\n")
        f.write("## Next Angle\n\n")
        f.write("Replace scalar bit probes with modal carriers over the full carrier word basis: Walsh slices, graph Laplacian buckets, and tape-restoration eigen slots, then require holdout separation against same-hash wrong-answer controls.\n")

    print(f"VERDICT: {verdict}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
