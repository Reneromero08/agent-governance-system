# Exp 50 L4A Class B Implementation Plan

**Date:** 2026-06-18  
**Status:** `SOURCE_REPAIR_IN_PROGRESS__HARDWARE_RUN_DEFERRED`

---

## Goal

Implement the repaired Class B crossed-assignment calibration exactly as specified in `EXP50_L4A_CLASS_B_PDN_SCREEN_DESIGN.md`.

The implementation must measure a complex value-dependent PDN coordinate while separating fixed core/route bias. It must not claim fold orientation, restored physical state, a winning branch, or scalar recovery.

---

## Runtime contract

The runtime accepts public structural inputs:

```text
--N <power of two>
--a <public lower fold magnitude>
--tsc-hz <measured TSC frequency>
--out <JSON artifact path>
```

It derives `mirror = N-a`. It never samples or reads hidden `d`.

The acquisition matrix is:

```text
core4/a
core5/mirror
core4/mirror
core5/a
core4/idle
core5/idle
core4/dummy42
core5/dummy42
```

Each active acquisition uses the same number of operations, tone, duty cycle, capture duration, and memory footprint. The public orbit value changes integer switching activity inside the burst.

---

## Required calculations

For `Z = I+iQ`:

```text
D_normal = Z4(a) - Z5(mirror)
D_swap   = Z4(mirror) - Z5(a)
R_value  = (D_normal - D_swap)/2
R_core   = (D_normal + D_swap)/2
```

Also record:

```text
same_orbit_core_bias = Z4(a)-Z5(a)
dummy_core_bias      = Z4(42)-Z5(42)
carrier_off_bias     = Z4(idle)-Z5(idle)
```

No threshold or pass flag is embedded in the capture binary. Statistical adjudication belongs to the reviewed observability pipeline.

---

## Source files

| File | Role |
|---|---|
| `class_b_pdn_screen.c` | Hardware capture and crossed decomposition |
| `EXP50_L4A_CLASS_B_PDN_SCREEN_DESIGN.md` | Binding experiment design |
| `EXP50_L4A_CLASS_B_WB_CARRIER_REPORT.md` | Historical report and invalidation record |
| `results/class_b_crossover_measurement.json` | Future generated capture artifact; ignored until deliberately imported |

The old L4A `holo_record.*` files are not used as the canonical output schema. They are preserved only as scaffold history.

---

## Failure conditions

The run is invalid if any of the following occurs:

- orbit value is not consumed by the sender workload;
- active captures use unequal operation counts or durations;
- carrier-off values are synthesized rather than measured;
- any control is hardcoded as passing;
- hidden `d`, truth labels, or verifier scores enter the runtime;
- capture order or source configuration is missing from the artifact;
- temperature veto fires;
- thread creation/join or affinity fails;
- raw I/Q data is absent.

---

## Build and execution gate

Planned Phenom build:

```bash
gcc -O2 -std=gnu11 -pthread -march=amdfam10 -Wall -Wextra -Werror \
  class_b_pdn_screen.c -o class_b_pdn_screen -lm
```

Planned safe invocation:

```bash
sudo ./class_b_pdn_screen \
  --N 256 \
  --a 125 \
  --tsc-hz 3214823000 \
  --out results/class_b_crossover_measurement.json
```

The final SSH verification batch must preserve compiler output, runtime log, artifact SHA-256, host metadata, temperature readings, and source commit SHA.

---

## Claim boundary

A successful capture means only:

```text
crossed PDN calibration artifact generated
complex value/core coordinates measured
```

It does not mean:

```text
fold-odd orientation found
physical HoloGeometry found
physical path restored
catalytic closure demonstrated
```
