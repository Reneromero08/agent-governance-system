#!/usr/bin/env python3
import json
import sys


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    if len(sys.argv) != 3:
        print("usage: phase4b_to_phase6_feeder_scorer.py input_summary.json output_features.json", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as fh:
        summary = json.load(fh)

    delays = [summary["by_delay"][k] for k in sorted(summary["by_delay"], key=lambda x: int(x))]
    rows = []
    for d in delays:
        c = d["canonical"]
        p = d["physical_baseline"]
        row = {
            "delay_class": d["delay_class"],
            "delay_pauses": d["delay_pauses"],
            "pass": bool(d["pass"]),
            "mode_score": clamp01(c["real_accuracy"]),
            "mode_floor_score": clamp01(c["real_mode_accuracy_floor"]),
            "wrong_schedule_score": clamp01(c["wrong_actual_match"] - c["wrong_declared_match"]),
            "pseudo_reject_score": clamp01(d["canonical_pseudo_reject_floor"]),
            "layout_gain_score": clamp01(c["real_accuracy"] - p["real_accuracy"]),
            "fixed_address_baseline": clamp01(p["real_accuracy"]),
            "pseudo_declared_match": clamp01(c["pseudo_declared_match"]),
        }
        rows.append(row)

    feature_summary = {
        "input": sys.argv[1],
        "source_verdict": summary["verdict"],
        "all_restore": bool(summary["all_restore"]),
        "all_delay_pass": bool(summary["all_delay_pass"]),
        "rows": summary["rows"],
        "delay_scores": rows,
        "features": {
            "mode_score_mean": mean([r["mode_score"] for r in rows]),
            "mode_score_floor": min(r["mode_score"] for r in rows),
            "mode_floor_score_floor": min(r["mode_floor_score"] for r in rows),
            "wrong_schedule_score_floor": min(r["wrong_schedule_score"] for r in rows),
            "pseudo_reject_score_floor": min(r["pseudo_reject_score"] for r in rows),
            "layout_gain_score_floor": min(r["layout_gain_score"] for r in rows),
            "fixed_address_baseline_ceiling": max(r["fixed_address_baseline"] for r in rows),
            "retention_delay_count": len(rows),
        },
        "phase6_allowed_use": [
            "same_core_scalar_holo_feature",
            "basin_label_covariate",
            "invariant_observability_score",
            "layout_normalized_timing_witness",
        ],
        "phase6_disallowed_use": [
            "phase_resolving_quadrature_claim",
            "phase6_crossing_claim",
            "cross_core_lockin_claim",
            "odd_channel_source",
            "thermodynamic_claim",
        ],
    }
    f = feature_summary["features"]
    feeder_ready = (
        feature_summary["all_restore"]
        and feature_summary["all_delay_pass"]
        and f["mode_score_floor"] >= 0.90
        and f["mode_floor_score_floor"] >= 0.65
        and f["wrong_schedule_score_floor"] >= 0.80
        and f["pseudo_reject_score_floor"] >= 0.95
        and f["layout_gain_score_floor"] >= 0.45
        and f["fixed_address_baseline_ceiling"] <= 0.45
    )
    feature_summary["verdict"] = (
        "PHASE4B_TO_PHASE6_FEEDER_SCORER_READY"
        if feeder_ready else
        "PHASE4B_TO_PHASE6_FEEDER_SCORER_PARTIAL"
    )

    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(feature_summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(feature_summary["verdict"])
    print(
        "mode_floor=%.6f wrong_floor=%.6f pseudo_floor=%.6f layout_gain_floor=%.6f fixed_ceiling=%.6f"
        % (
            f["mode_score_floor"],
            f["wrong_schedule_score_floor"],
            f["pseudo_reject_score_floor"],
            f["layout_gain_score_floor"],
            f["fixed_address_baseline_ceiling"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
