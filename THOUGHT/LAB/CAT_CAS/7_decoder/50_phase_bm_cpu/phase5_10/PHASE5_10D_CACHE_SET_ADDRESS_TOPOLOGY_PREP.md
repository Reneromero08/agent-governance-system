# EXP50 PHASE 5.10D - CACHE/ADDRESS TOPOLOGY PREP

**Parent:** `PHASE5_10_BOUNDARY_STATE_PREPARATION.md`

**Status:** `PHASE5_10D_RUN_COMPLETE__VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL`

## Objective

Test a software-controlled topology carrier after the passive strobe/rail route failed to produce a Phase 6-ready preparation channel.

This route does not assume voltage movement, physical rail observability, or a hidden firmware table. It asks a narrower question:

Can fixed address-set families and matched layout controls prepare a reproducible timing basin that survives held-out layout tests and shuffled-label nulls?

## Why This Route Is Different

The prior 5.9V and 5.10 rail/strobe routes mixed several effects:

- public prelude selection;
- runtime VID request labels;
- memory/shared-resource contention;
- passive strobe interpretation;
- long basin scans that could lose partial evidence.

5.10D isolates the directly controllable software part:

- same work, different address layout;
- fixed address families;
- randomized layout nulls;
- compute-only aggressor control;
- memory aggressor control;
- progress-flushed output after every row.

## Harness

`phase5_10_cache_address_topology.c`

The harness is user-space only. It allocates aligned memory, defines deterministic address families, runs victim timing loops under matched aggressor modes, and writes one CSV row per repeat.

Required controls:

| Control | Purpose |
| --- | --- |
| `none` | baseline victim timing |
| `compute` | same CPU load shape without shared address layout |
| `same_address` | memory aggressor on the same address family |
| `different_address` | memory aggressor on a different fixed family |
| `random_address` | randomized-layout null |

## Acceptance

`PHASE5_10D_TOPOLOGY_PREP_CANDIDATE` only if all hold:

- same-address response separates from compute-only and no-aggressor controls;
- same-address response separates from randomized-layout null;
- effect direction reproduces across repeats;
- held-out address families preserve the classifier;
- shuffled labels do not match the real effect size.

Otherwise use:

- `PHASE5_10D_NO_TOPOLOGY_BASIN`
- `PHASE5_10D_ARTIFACT_DOMINANT`
- `PHASE5_10D_UNDERPOWERED`

## Live Result

Artifact:

`PHASE5_10D_CACHE_ADDRESS_TOPOLOGY_LIVE.md`

Result:

```text
VALID_SCALAR_WITNESS_BELOW_ENCODING_WALL
```

The run produced a reproducible same-address timing effect across two core layouts:

- primary run: 240 rows, same-vs-controls median effect `36.9428405`, permutation p-value `0.0004997501`, family sign agreement `1.0`;
- swapped-core run: 240 rows, same-vs-controls median effect `32.3227925`, permutation p-value `0.0004997501`, family sign agreement `1.0`.

This satisfies the software-readable topology witness goal, but it remains scalar/cache-address timing. It does not open Phase 6 and does not break the encoding wall.

## Reproduction Command

```text
gcc -O2 -pthread -Wall -Wextra phase5_10_cache_address_topology.c -o cache_addr_topology
./cache_addr_topology --reps 12 --iters 2000 --buf-mb 64 --output phase5_10d_cache_address_topology.csv
```

Use SSH only for compiling/running on the owned Phenom. No flash, no voltage writes, no hardware modification.
