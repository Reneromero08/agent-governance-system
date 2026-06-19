# Exp 50 L4A Class B Historical Carrier Report

**Original status:** `L4A_CLASS_B_WB_CARRIER_PASS`
**Corrected status:** `INVALIDATED_AS_WB_SCREEN__RETAINED_AS_SMOKE_PROVENANCE`
**Date corrected:** 2026-06-18

---

## Correction

The original report interpreted three captures (`normal`, `label_swap`, and `carrier_off`) as evidence that Q-difference was core-dependent rather than value-dependent.

A source audit showed that the committed runtime did not perform the claimed W_B experiment:

- `a` and `N-a` were printed but did not alter the sender workload;
- the two sender windows were executed sequentially;
- carrier-off was written as literal zeros rather than physically captured;
- the label-swap control was marked passed without comparing signs;
- same-orbit, dummy-orbit, replay, wrong-restore, and session controls were absent.

Therefore the original measurements cannot adjudicate a value-dependent Class B coordinate. They are retained only as evidence that the scaffold could invoke per-core PDN captures and serialize output.

---

## What remains valid

Independent T300 evidence remains the positive control for a live PDN lock-in channel on selected routes. That evidence is not invalidated by this source audit because it came from a separate experiment, run matrix, and artifact set.

The correct evidence split is:

```text
T300: sender-owned mode/phase transport through PDN channel
old Class B scaffold: per-core smoke capture only
repaired Class B: crossed value/core calibration, not yet run
```

---

## What is not established

The original report does not establish:

```text
core bias dominance under value-dependent W_B
fold-odd residue absence
fold-odd residue presence
operand-dependent PDN response
physical HoloGeometry
physical path memory
physical restoration
```

---

## Replacement experiment

The replacement is specified by:

- `EXP50_L4A_CLASS_B_PDN_SCREEN_DESIGN.md`
- `EXP50_L4A_CLASS_B_IMPLEMENTATION_PLAN.md`
- `class_b_pdn_screen.c`

It uses crossed assignments and decomposes measured complex response into:

```text
R_value = value-dependent coordinate
R_core  = fixed core/route coordinate
```

The replacement source has not yet been executed on the Phenom. Its eventual artifact must be adjudicated by the reviewed observability pipeline rather than by hardcoded pass flags.
