# First-Light Transducer Audit

Date: 2026-07-12

Scope:

- `../orbit_query_runtime.c`
- `../orbit_query_runtime.h`
- `../orbit_query_public.py`
- `../orbit_query_model.py`
- `../orbit_query_target.py`
- `../runs/orbit_query_first_light_0/ADJUDICATION.json`
- `../runs/orbit_query_first_light_0/FEATURES_FROZEN.json`
- `../runs/orbit_query_first_light_0/RAW_CAPTURE.jsonl`

The retained first-light run remains unchanged and remains classified as:

```text
ORBITSTATE_PHYSICAL_QUERY_COUPLING_NOT_ESTABLISHED
```

This audit does not reinterpret that evidence. It localizes why the physical
work-to-observable transducer was not yet calibrated.

## Finding 1: Fresh Anonymous-Page Contamination

The first-light runtime allocates the measured positive and negative banks with
fresh anonymous private mappings and immediately applies source same-value stores to
the selected prefixes. The contract intentionally avoided measured-bank
initialization sweeps so private source work could not be hidden by a full-bank write.
That was correct for blinding, but it left the first source write as the page
materialization event.

On Linux, first writes to untouched anonymous pages can allocate and zero backing
pages. The physical effect seen by the later receiver sweep could therefore include
page allocation and zero-page materialization, not only the intended count of source
ownership transfers.

Calibration repair:

- materialize both banks symmetrically before any encoded source work;
- initialize both banks with identical deterministic bytes;
- verify equal digests before the source encoder runs;
- never let the public encoded source operation be the first write to a bank.

## Finding 2: Different Positive/Negative Line Geometry

The first-light source encoder calls the same byte-preserving store primitive for both
banks, but with different salts:

```text
positive salt = phase + iteration
negative salt = phase + iteration + 11
```

The two logical banks therefore receive different line permutations. The difference
between `M + q` and `M - q` is not the only changing physical variable: the line subset
geometry can also vary with logical bank identity.

Calibration repair:

- use one public line permutation for both physical banks;
- use one public seed for both logical roles;
- differ only by prefix length;
- require every line in each prefix to be unique;
- record the permutation constants and prefix checks in raw evidence.

## Finding 3: Old Bank Swap Was Not A Pointer Swap

The first-light `physical_bank_swap` condition negates the effective query allocation,
but it does not exchange the two physical bank pointers. The logical sign changes,
while the measured positive-bank and negative-bank allocations remain attached to the
same physical objects.

Calibration repair:

- separate logical roles from physical pointers;
- mapping 0 binds logical positive to physical A and logical negative to physical B;
- mapping 1 binds logical positive to physical B and logical negative to physical A;
- report both logical positive-minus-negative and raw physical A-minus-B responses;
- require logical invariance and physical A-minus-B sign reversal under pointer swap.

## Finding 4: One-Shot Measurement Was Insufficient

Each first-light phase captures one positive-bank receiver sweep and one negative-bank
receiver sweep. The design was adequate as a first source-owned query smoke, but it
does not estimate a stable physical transfer law.

Calibration repair:

- run repeated paired trials inside each fresh process;
- repeat every public `q` value under both physical mappings;
- balance measurement order exactly;
- build the null region from repeated `q = 0` controls, pointer-swap symmetry,
  order symmetry, restoration sentinels, and fresh-process variation.

## Finding 5: PMU Custody Was Incomplete

The first-light PMU group opened ordinary raw events but retained only raw values.
It did not retain `time_enabled`, `time_running`, or event IDs in the raw capture, and
it did not reject multiplexed windows. It also used `exclude_kernel = 0`.

Calibration repair:

- set `exclude_kernel = 1` and `exclude_hv = 1`;
- read group values with `PERF_FORMAT_TOTAL_TIME_ENABLED`,
  `PERF_FORMAT_TOTAL_TIME_RUNNING`, and `PERF_FORMAT_ID`;
- retain event IDs and reject event-order drift;
- reject any window where `time_enabled != time_running`;
- report raw counts first and cycle-normalized counts as secondary coordinates.

## New Calibration Wall

The first-light result should not be read as a proof that OrbitState coupling is
impossible. The logical query bridge, public freeze boundary, source-work receipts,
and restoration custody were useful. The unresolved wall is narrower:

```text
public signed work imbalance q
-> controlled ownership preparation
-> receiver reacquisition response F(q)
```

Only if this public transfer law becomes stable, odd, monotonic, pointer-swap
invariant, measurement-order invariant, and reproducible across fresh processes
should another private OrbitState query be mapped through it.
