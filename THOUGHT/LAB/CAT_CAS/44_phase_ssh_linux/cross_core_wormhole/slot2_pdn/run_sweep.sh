#!/bin/bash
# SLOT 2 PDN full sweep: 6 independent seed windows x 2 core pairs.
# Foreground on the box. k10temp veto is built into the binary (abort >= 68C).
# REPRODUCIBILITY is the make-or-break: 6/6 seeds must hold the witness.
set -u
cd /root/slot2_pdn

TRIALS="${TRIALS:-48}"
SLOT_S="${SLOT_S:-0.5}"
READ_HZ="${READ_HZ:-4000}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
OUTDIR="${OUTDIR:-/root/slot2_pdn/matrix}"
mkdir -p "$OUTDIR"

# core pairs among the isolated cores 2-5 (victim:sender)
PAIRS="${PAIRS:-2:3 4:5}"

# NOTE: the binary restores the P-state on its own clean exit AND on SIGTERM/SIGINT
# (signal safety net). We run each invocation directly (no external `timeout` wrapper)
# so a clean restore always happens; the temp veto bounds runtime from inside.

echo "=== SLOT2 SWEEP trials=$TRIALS slot_s=$SLOT_S read_hz=$READ_HZ seeds=[$SEEDS] pairs=[$PAIRS] ==="
date +%s
for pair in $PAIRS; do
  v="${pair%%:*}"; s="${pair##*:}"
  for seed in $SEEDS; do
    T=$(cat /sys/class/hwmon/hwmon0/temp1_input 2>/dev/null)
    echo "--- pair v$v:s$s seed=$seed (preT=${T}mC) ---"
    out="$OUTDIR/matrix_v${v}s${s}_seed${seed}.csv"
    ./slot2 --mode matrix --victim "$v" --sender "$s" --seed "$seed" \
            --trials "$TRIALS" --slot-s "$SLOT_S" --read-hz "$READ_HZ" \
            --out-csv "$out" 2>"$OUTDIR/matrix_v${v}s${s}_seed${seed}.log"
    rc=$?
    tail -1 "$OUTDIR/matrix_v${v}s${s}_seed${seed}.log"
    echo "  rc=$rc out=$out rows=$(grep -vc '^#' "$out" 2>/dev/null)"
    # cool-down guard between runs if warm
    T2=$(cat /sys/class/hwmon/hwmon0/temp1_input 2>/dev/null)
    if [ "${T2:-0}" -gt 60000 ]; then echo "  cooling (T=${T2}mC)"; sleep 20; fi
  done
done
date +%s
echo "=== SWEEP DONE ==="
ls -la "$OUTDIR"/*.csv
