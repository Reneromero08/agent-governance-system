# Phase 6B.6 Gate A Engineering Smoke Run Plan

**Status:** `FROZEN_PLAN__NO_EXECUTION_AUTHORITY`

```text
base main = 9c41637992536f43d10d152ec176a3577aef1623
architecture review = 4614574719
architecture merge = 9c41637992536f43d10d152ec176a3577aef1623
schedule digest = 418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116
```

## Purpose

Gate A proves only that a separately implemented and qualified adapter can cross the physical interface, execute one exact explicit-slot sequence, preserve raw custody, enforce vetoes, and cleanly stop.

It is not a scientific mini-campaign. It cannot fit an operator, access scientific train, validation, or test data, classify persistence, or support an observability claim.

This document does not authorize execution.

## Frozen geometry

```text
target = root@192.168.137.100
hostname = catcas
CPU = AMD Phenom II X6 1090T
route = v4s5
sender core = 4
receiver core = 5
boot states = 1
sessions = 1
read rate = 8000 Hz
slot duration = 0.5 seconds
nominal samples per slot = 4000
slot count = 16
nominal duration = 8.0 seconds
maximum execution count = 1
automatic retry = false
temperature veto = 68 C
```

The target identity must match sealed stdout SHA-256:

```text
10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4
```

The observed frequency must already equal `1600000 kHz`. Gate A may observe and veto, but may not write frequency, voltage, or MSR state.

## Exact sequence

| Slots | Token | Meaning |
|---|---|---|
| 0-3 | `I` | Sender absent, idle baseline |
| 4 | `C0` | Carrier-off control for tone 0 |
| 5 | `D0` | Declared anchor context, no physical drive |
| 6-9 | `S0E` | Four contiguous driven STEP slots in one sender epoch |
| 10-11 | `O0` | STEP-off transition, sender absent |
| 12 | `A0P` | Positive anchor |
| 13 | `A0N` | Negative anchor |
| 14-15 | `T` | Tail sender-off and cleanup confirmation |

The canonical schedule is `GATE_A_ENGINEERING_SMOKE_SCHEDULE.json`.

Each token stores `declared` and `executed` controls separately. Off and sham tokens retain declared analytical context while executed amplitude, phase, tone, sign, and sender epoch remain null.

## Preconditions

Stop before drive unless all are true:

```text
exact Gate A authority artifact validates
artifact binds this run plan and schedule digest
project-owner approval names the exact reviewed Gate A head
adapter and execution-bundle digests match
target identity matches
sender process is absent
temperature is below 68 C
observed frequency is 1600000 kHz
no frequency, voltage, or MSR writer is present
no scientific data namespace is available
output namespace is new and empty
automatic retry is disabled
```

The current candidate fails the adapter and execution-bundle requirements by design. It cannot authorize execution.

## Required custody

The eventual run must preserve:

```text
authority artifact and exact schedule
source, adapter, executable, and bundle digests
target identity before and after
process state before, during transitions, and after cleanup
session TSC origin and measured TSC frequency
requested and actual slot boundaries
raw lock-in I and Q
raw ring-oscillator periods
declared and executed controls
route, core, temperature, P-state, sample-rate, and gap telemetry
stdout, stderr, command ledger, file modes, sizes, and SHA-256 digests
cleanup proof
```

Raw acquisition remains primary.

## Stop conditions

Stop immediately and preserve partial evidence on any authority, digest, identity, sender-state, temperature, frequency, timing, capture-quality, process, file, network, raw-output, or slot failure.

There is no catch-up, padding, interpolation, synthetic replacement, silent repetition, or retry.

A failed or vetoed attempt requires independent review and a new project-owner decision.

## Cleanup

Cleanup must prove all sender and receiver processes are absent, temporary roots are removed after verified copy-back, no background process remains, no hardware control state changed, and retained evidence hashes verify.

## Claim boundary

A Gate A pass establishes only that the physical interface and custody path function under this exact smoke sequence.

It does not establish a relational state, predictive operator, persistence, restoration, target coupling, fold-odd recovery, or Small Wall crossing.

## Current blocking state

```text
run plan frozen = true
schedule frozen = true
run-plan review complete = false
candidate review complete = false
hardware adapter implemented = false
hardware adapter qualified = false
execution bundle ready = false
project-owner execution approval recorded = false
engineering smoke authorized = false
hardware ran = false
```

## Next boundary

The immediate boundary is independent review of the frozen plan, schedule, candidate, schema, and verifier.

Execution remains blocked after that review. The next implementation boundary is:

```text
GATE_A_HARDWARE_ADAPTER_IMPLEMENTATION_AND_NONEXECUTING_QUALIFICATION
```
