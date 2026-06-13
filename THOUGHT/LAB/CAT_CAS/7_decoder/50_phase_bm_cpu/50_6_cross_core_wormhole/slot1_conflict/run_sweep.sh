#!/bin/sh
# SLOT 1 conflict-displacement sweep driver (runs ON the Phenom box).
# Userspace only. k10temp veto: abort the whole sweep if temp1_input > 68000 (68C).
# Seeds are deterministic: SEED_WINDOW k => fixed tape/buf/perm draw (recorded in CSV name + stderr).
#
# Usage:  sh /tmp/run_sweep.sh CONFLICT  "N_list"  "W_list"  "seed_list"  "corepair_list"
#   corepair encoded as WRITER:OBSERVER  e.g. 2:3 4:5 3:2(swapped)
# Output: /tmp/slot1/conf<C>_N<N>_W<W>_cp<w>-<o>_s<seed>.csv  (+ .err)
TEMP=/sys/class/hwmon/hwmon0/temp1_input
SRC=/tmp/conf.c
OUT=/tmp/slot1
mkdir -p "$OUT"

CONFLICT="$1"; NS="$2"; WS="$3"; SEEDS="$4"; PAIRS="$5"
THRESH=68000

check_temp() {
  t=$(cat "$TEMP")
  if [ "$t" -gt "$THRESH" ]; then
    echo "TEMP_VETO: ${t}mC > ${THRESH}mC -- ABORTING SWEEP" >&2
    exit 99
  fi
}

for pair in $PAIRS; do
  W_CORE=$(echo "$pair" | cut -d: -f1)
  O_CORE=$(echo "$pair" | cut -d: -f2)
  for W in $WS; do
    for N in $NS; do
      bin="$OUT/bin_c${CONFLICT}_N${N}_W${W}_cp${W_CORE}-${O_CORE}"
      gcc -O2 -pthread -DCONFLICT=$CONFLICT -DEVSET_W=$W -DN_AVG=$N \
          -DWRITER_CORE=$W_CORE -DOBSERVER_CORE=$O_CORE -o "$bin" "$SRC" || { echo "COMPILE_FAIL c$CONFLICT N$N W$W $pair" >&2; exit 4; }
      for s in $SEEDS; do
        check_temp
        # SEED_WINDOW is compiled per-seed (it changes constants). Recompile with seed.
        sbin="${bin}_s${s}"
        gcc -O2 -pthread -DCONFLICT=$CONFLICT -DEVSET_W=$W -DN_AVG=$N \
            -DWRITER_CORE=$W_CORE -DOBSERVER_CORE=$O_CORE -DSEED_WINDOW=$s -o "$sbin" "$SRC" || { echo "COMPILE_FAIL_SEED" >&2; exit 4; }
        csv="$OUT/conf${CONFLICT}_N${N}_W${W}_cp${W_CORE}-${O_CORE}_s${s}.csv"
        err="${csv%.csv}.err"
        t0=$(cat "$TEMP")
        "$sbin" > "$csv" 2> "$err"
        rc=$?
        t1=$(cat "$TEMP")
        rest=$(grep -o 'restored=[0-9]*/[0-9]*' "$err")
        echo "DONE c$CONFLICT N$N W$W cp$W_CORE-$O_CORE s$s rc=$rc $rest temp ${t0}->${t1} -> $csv"
        check_temp
        rm -f "$sbin"
      done
      rm -f "$bin"
    done
  done
done
echo "SWEEP_COMPLETE conflict=$CONFLICT"
