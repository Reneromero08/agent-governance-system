#!/usr/bin/env python3
"""
aggregate.py -- score every SLOT 2 PDN matrix CSV and emit the reproducibility
verdict across (seed, core pair). SUCCESS requires the witness to hold 6/6 seeds
on at least the primary pair (and reports all pairs).

Gates per run (matched-null battery + phase):
  all_rows_restore AND real_accuracy>=0.60 AND real_vs_pseudo floor>=0.95 AND
  pseudo_reject>=0.95 AND wrong_actual_match>=0.60 AND wrong_declared_match<=0.20
  AND (phase_corr_true - phase_corr_null) > 0.30

Usage: aggregate.py <dir-with-matrix_*.csv> <out.json>
ASCII only.
"""
import glob
import json
import os
import re
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("an", os.path.join(HERE, "slot2_pdn_analyze.py"))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def parse_name(path):
    b = os.path.basename(path)
    m = re.match(r"matrix_v(\d+)s(\d+)_seed(\d+)\.csv", b)
    if not m:
        return None
    return {"victim": int(m.group(1)), "sender": int(m.group(2)), "seed": int(m.group(3)),
            "pair": f"{m.group(1)}:{m.group(2)}"}


def run_gates(a):
    g = a["gates"]
    return all(g.values())


def main():
    if len(sys.argv) != 3:
        print("usage: aggregate.py <dir> <out.json>", file=sys.stderr)
        return 2
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "matrix_v*s*_seed*.csv")))
    runs = []
    for f in files:
        meta = parse_name(f)
        if meta is None:
            continue
        try:
            a = an.analyze(f)
        except Exception as e:  # noqa
            runs.append({**meta, "error": str(e), "pass": False})
            continue
        runs.append({
            **meta,
            "real_accuracy": a["real_accuracy"],
            "real_mode_floor": a["real_mode_floor"],
            "real_vs_pseudo_floor": a["real_vs_pseudo_floor"],
            "pseudo_reject_floor": a["pseudo_reject_floor"],
            "pseudo_declared_match": a["pseudo_declared_match"],
            "wrong_actual_match": a["wrong_actual_match"],
            "wrong_declared_match": a["wrong_declared_match"],
            "phase_corr_true": a["phase_corr_true"],
            "phase_corr_null": a["phase_corr_null"],
            "phase_delta": a["phase_delta"],
            "all_rows_restore": a["all_rows_restore"],
            "gates": a["gates"],
            "pass": run_gates(a),
            "verdict": a["verdict"],
        })

    by_pair = {}
    for r in runs:
        by_pair.setdefault(r["pair"], []).append(r)

    pair_summary = {}
    for pair, rs in by_pair.items():
        rs_sorted = sorted(rs, key=lambda x: x["seed"])
        npass = sum(1 for r in rs_sorted if r.get("pass"))
        pair_summary[pair] = {
            "n_seeds": len(rs_sorted),
            "n_pass": npass,
            "all_pass": npass == len(rs_sorted) and len(rs_sorted) >= 6,
            "seeds_pass": [r["seed"] for r in rs_sorted if r.get("pass")],
            "seeds_fail": [r["seed"] for r in rs_sorted if not r.get("pass")],
            "real_acc_range": [min((r.get("real_accuracy", 0) for r in rs_sorted), default=0),
                               max((r.get("real_accuracy", 0) for r in rs_sorted), default=0)],
            "rvp_range": [min((r.get("real_vs_pseudo_floor", 0) for r in rs_sorted), default=0),
                          max((r.get("real_vs_pseudo_floor", 0) for r in rs_sorted), default=0)],
            "phase_delta_range": [min((r.get("phase_delta", 0) for r in rs_sorted), default=0),
                                  max((r.get("phase_delta", 0) for r in rs_sorted), default=0)],
        }

    # ---- negative controls: must FAIL the witness gates (sit at chance) ----
    controls = {}
    for name in ("silent", "scramble"):
        cf = os.path.join(d, f"control_{name}.csv")
        if os.path.exists(cf):
            try:
                a = an.analyze(cf)
                passed = all(a["gates"].values())
                controls[name] = {
                    "real_accuracy": a["real_accuracy"],
                    "real_vs_pseudo_floor": a["real_vs_pseudo_floor"],
                    "pseudo_reject_floor": a["pseudo_reject_floor"],
                    "wrong_actual_match": a["wrong_actual_match"],
                    "phase_delta": a["phase_delta"],
                    "all_rows_restore": a["all_rows_restore"],
                    "witness_gates_pass": passed,     # MUST be False for a valid control
                    "control_ok": not passed,         # control behaves as a null
                }
            except Exception as e:  # noqa
                controls[name] = {"error": str(e), "control_ok": False}
    controls_present = len(controls) > 0
    controls_all_null = controls_present and all(c.get("control_ok") for c in controls.values())

    # overall: witness holds reproducibly iff at least one pair is 6/6 AND a second
    # pair is also run (>=2 core pairs requirement) AND the negative controls behave as
    # nulls (fail the witness gates). Report all three.
    pairs_6of6 = [p for p, s in pair_summary.items() if s["all_pass"]]
    n_pairs_run = len(pair_summary)
    witness_holds = (len(pairs_6of6) >= 1 and n_pairs_run >= 2
                     and (controls_all_null if controls_present else True))

    overall = {
        "witness_holds_reproducibly": witness_holds,
        "pairs_run": n_pairs_run,
        "pairs_6of6": pairs_6of6,
        "controls_present": controls_present,
        "controls_all_null": controls_all_null,
        "verdict": ("PHASE4B_CROSS_CORE_PDN_LOCKIN_WITNESS"
                    if witness_holds else "CLEAN_NEGATIVE_SLOT2_to_SLOT3"),
    }

    out = {"overall": overall, "by_pair": pair_summary, "controls": controls, "runs": runs}
    with open(sys.argv[2], "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(overall["verdict"])
    for pair, s in sorted(pair_summary.items()):
        print(f"pair {pair}: {s['n_pass']}/{s['n_seeds']} pass  "
              f"real_acc={s['real_acc_range'][0]:.3f}-{s['real_acc_range'][1]:.3f}  "
              f"rvp={s['rvp_range'][0]:.3f}-{s['rvp_range'][1]:.3f}  "
              f"phase_d={s['phase_delta_range'][0]:.3f}-{s['phase_delta_range'][1]:.3f}  "
              f"fail_seeds={s['seeds_fail']}")
    for name, c in controls.items():
        if "error" in c:
            print(f"control {name}: ERROR {c['error']}")
        else:
            print(f"control {name}: real_acc={c['real_accuracy']:.3f} rvp={c['real_vs_pseudo_floor']:.3f} "
                  f"wr_act={c['wrong_actual_match']:.3f} phase_d={c['phase_delta']:.3f} "
                  f"witness_gates_pass={c['witness_gates_pass']} -> control_ok(null)={c['control_ok']}")
    return 0 if witness_holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
