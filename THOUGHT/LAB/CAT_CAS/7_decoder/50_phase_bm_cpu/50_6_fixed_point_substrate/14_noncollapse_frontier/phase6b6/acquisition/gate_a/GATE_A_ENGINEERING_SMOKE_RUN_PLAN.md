# Phase 6B.6 Gate A Engineering Smoke Run Plan

**Status:** `FROZEN_PLAN__NO_EXECUTION_AUTHORITY`

**Base main:** `9c41637992536f43d10d152ec176a3577aef1623`

**Architecture review:** `4614574719`

**Architecture merge:** `9c41637992536f43d10d152ec176a3577aef1623`

**Schedule digest:** `b2d73afb5b4d3fd351abbb4aa7f7f76cfa532dcff0808f331682b456d3e6e6ed`

## 1. Purpose

Gate A proves only that a separately implemented and qualified hardware adapter can cross the physical interface, obey one exact explicit-slot sequence, preserve raw custody, enforce vetoes, and cleanly stop.

It is not a scientific mini-campaign. It cannot fit an operator, open train, validation, or test data, classify persistence, or support a physical observability claim.

This document does not authorize execution.

## 2. Frozen target and route

```text
target = root@192.168.137.100
hostname = catcas
CPU = AMD Phenom II X6 1090T
architecture = x86_64
route = v4s5
sender core = 4
receiver core = 5
boot states = 1
sessions = 1
```

The target identity must be revalidated immediately before execution and must match the sealed identity evidence whose stdout SHA-256 is:

```text
10618a70ceb3413d7507c22254d595d63632bb7ad9243dbe3dc6ebbaf13e19a4
```

## 3. Timing and execution ceiling

```text
read rate = 8000 Hz
slot duration = 0.5 seconds
nominal samples per slot = 4000
slot count = 16
nominal driven timeline = 8.0 seconds
maximum execution count = 1
automatic retry = false
temperature veto = 68 C
```

The observed frequency must already equal `1600000 kHz`. Gate A may observe and veto on mismatch, but it may not write frequency, voltage, or MSR state.

## 4. Exact slot sequence

| Slots | Token | Physical meaning |
|---|---|---|
| 0-3 | `I` | Sender absent, idle baseline |
| 4 | `C0` | Carrier-off control for physical tone 0 |
| 5 | `D0` | Declaration sham for tone 0, declared anchor but no drive |
| 6-9 | `S0E` | Four contiguous driven STEP slots, tone 0, amplitude 2, positive sign, one sender epoch |
| 10-11 | `O0` | STEP-off transition with sender absent |
| 12 | `A0P` | Positive anchor, tone 0, amplitude 2, phase 0 |
| 13 | `A0N` | Negative anchor, tone 0, amplitude 2, phase pi |
| 14-15 | `T` | Tail sender-off and cleanup confirmation |

The canonical machine-readable schedule is `GATE_A_ENGINEERING_SMOKE_SCHEDULE.json`.

The four `S0E` slots must share exactly one sender process epoch. The sender must be absent before slot 0, absent during all off and sham slots, and absent after the final driven slot.

## 5. Preconditions

Execution must stop before any drive if any precondition fails:

```text
exact Gate A authority artifact validates
artifact binds the reviewed run plan and schedule digest
project-owner approval exactly names Gate A and the reviewed head
hardware adapter digest matches the artifact
execution bundle digest matches the artifact
target identity matches catcas evidence
sender process is absent
temperature is below 68 C
observed frequency is exactly 1600000 kHz
no frequency, voltage, or MSR writer is present
no scientific train, validation, or test namespace is mounted
output namespace is new and empty
automatic retry is disabled
```

The current candidate fails the hardware-adapter and execution-bundle preconditions by design. Therefore it cannot authorize execution.

## 6. Required custody

The eventual one-shot run must preserve:

```text
exact authority artifact
exact schedule and digest
source, adapter, executable, and bundle digests
target identity before and after the run
process table before, during transitions, and after cleanup
session TSC origin
measured TSC frequency
requested and actual slot boundaries
raw lock-in I and Q samples
raw ring-oscillator period samples
executed controls
declared controls
route and core identity
temperature and observed P-state telemetry
empirical sample rate
maximum capture gap
stdout and stderr
complete command ledger
file sizes, modes, and SHA-256 digests
cleanup proof
```

Raw acquisition is primary. Summaries do not replace raw files.

## 7. Fail-closed stop conditions

Stop immediately and preserve partial evidence when any condition occurs:

```text
authority or digest mismatch
unexpected source or schedule head
target identity mismatch
sender present before authorized start
sender present during an off or sham slot
sender epoch discontinuity across slots 6-9
temperature reaches 68 C
observed frequency differs from 1600000 kHz
frequency, voltage, or MSR write attempt
requested versus actual TSC boundary violation
capture gap or empirical sample-rate veto
unexpected process, file, or network activity
missing raw output or command custody
any failed slot
```

There is no catch-up slot, padding, interpolation, synthetic replacement, silent repetition, or retry.

A failed or vetoed attempt consumes no implied second attempt. A new run requires a new evidence review and a new project-owner authority decision.

## 8. Cleanup obligations

Cleanup must prove:

```text
sender process absent
receiver process absent
temporary executable and extraction roots removed
no scheduled or background process remains
no frequency, voltage, or MSR state was changed
partial and final evidence copied back before deletion
all retained evidence hashes verify
target output namespace removed only after copy-back verification
```

## 9. Engineering-only pass conditions

Gate A evidence may pass only if:

```text
one exact execution occurred
all 16 slots executed in order
all requested and actual TSC boundaries are present
sender-off semantics are physically real
slots 6-9 preserve one sender epoch
positive and negative anchors execute distinctly
raw capture and telemetry are complete
capture-quality checks pass
temperature and observed-frequency predicates hold
cleanup is complete
evidence custody independently verifies
```

A pass establishes only that the physical interface and custody path function under this exact smoke sequence.

It does not establish predictive observability, a relational state, an operator, persistence, restoration, target coupling, fold-odd recovery, or Small Wall crossing.

## 10. Current blocking state

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

## 11. Next boundary

The immediate boundary is independent review of the frozen Gate A plan, schedule, candidate, and verifier.

Even after that review, execution remains blocked. The next implementation boundary is:

```text
GATE_A_HARDWARE_ADAPTER_IMPLEMENTATION_AND_NONEXECUTING_QUALIFICATION
```

Only after an exact adapter and execution bundle are implemented, qualified without driving hardware, independently reviewed, and bound into a new authority artifact may project-owner execution approval be requested.
