# Phase-QEMU V0 Contract

## Purpose

Phase-QEMU V0 is the first virtual-hardware calibration substrate after the
M257 equal-access deterministic-software obstruction. It preserves the frozen
P0 quartz process geometry while leaving the hardware backend replaceable.
QEMU execution is not physical execution and does not escape M257.

```text
guest/controller
    -> Phase-QEMU PCI/MMIO command-tag and boundary contract
        -> PhaseBackendOps
            -> P0 fixed-point reference backend
            -> future dynamical/co-simulation/external backends
```

`REFERENCE_HARDWARE_MODEL_0` is one backend. It is not the final phase
computer architecture.

## Process-object partition

The P0 backend carries separate machine-relevant domains:

| Domain | V0 state | P0 interpretation |
|---|---:|---|
| source | two Q30 quadratures | energized preparation source upstream of the barrier |
| carrier | two Q30 quadratures | quartz tuning-fork rotating-frame displacement |
| detector | two Q30 quadratures | finite-response high-impedance I/Q detector |
| environment | one Q30 energy accumulator | dissipated carrier energy |
| boundary | I, Q, and energy | final diagnostic, locked until release |
| controller/tags | owner, program, cursor/status, generation | virtual device control state |
| topology | carrier-present property plus migration guard | declared carrier configuration |

The state is therefore not represented as a lone `phase = double`. The fixed
point arithmetic is a deterministic development model, not a claim that the
physical system is exactly discrete or noiseless.

## Guest-visible V0 protocol

The PCI device exposes identity, capabilities, owner/program request fields,
command arguments, status/error/generation receipts, barrier and virtual-time
receipts, resource-shape receipts, and the released final boundary. It does
not expose source, carrier, detector, environment, or pending boundary
coordinates.

The owner/program fields provide nominal command-time tag consistency only.
They are guest-writable and do not authenticate a caller or establish
malicious-guest isolation, multi-controller custody, or a CATVM security
boundary.

The V0 command sequence is:

```text
OWNER/PROGRAM LEASE
-> PREPARE(0 or pi)
-> ISOLATE_SOURCE
-> EVOLVE
-> PROJECT_BOUNDARY internally
-> backend-native INVERT
-> backend-native RESTORE
-> RELEASE response
-> REUSE restored carrier
```

P0 advertises only owner/program custody, phase preparation, source isolation,
virtual evolution, and final diagnostic capabilities. It deliberately does
not advertise native invert, restore, or restored reuse. Diagnostic release
spends the P0 carrier transaction.

The source remains energized upstream after isolation. Ringdown includes a
small attenuated source-feedthrough term and a switching impulse. Consequently
the demonstrated law is source-isolated, upstream-energized ringdown, not
literal source absence.

## Backend interface

`PhaseBackendOps` is the replaceable front/back seam. A successor may change
the carrier, equations of motion, coupling, nonlinear law, measurement, and
native echo while preserving the outer custody and response-ordering
contract. A backend may advertise inversion/restoration/reuse only when those
operations act on its actual resident process state without a saved baseline,
trajectory, answer, or snapshot.

Backend state that can affect future computation belongs in canonical machine
state. Future timer, queue, interrupt, coupler, bath, pointer, entropy, and
external-backend state must be added to this law when introduced.

## Migration and reset distinction

`CANONICAL_MODEL_REINITIALIZE` is an explicit model-reset sham. Its
classification is `NO_RESTORATION_CLAIM`.

The separate migration control uses QEMU's actual migration stream to carry a
mid-ringdown hidden state into a second QEMU process. It restores PCI BAR state
without guest configuration replay and continues to the same boundary as an
uninterrupted run. Its classification is `SNAPSHOT_RELOAD`, because:

- a retained migration stream supplies state;
- a different host process/backing instance consumes it;
- no native inverse executes;
- no restoration generation advances; and
- the P0 backend still rejects restoration and reuse.

Migration state includes hidden process coordinates and is privileged trusted
backend material. Only its hash, size, and aggregate accounting may enter
durable evidence. V0 establishes guest-MMIO nonexposure; it does not establish
host or migration-stream no-smuggle enforcement.

## P0 numerical calibration

- Q30 signed fixed-point quadratures;
- 256 preparation steps;
- 64 carrier cycles per step;
- 0.99 carrier decay per 64-cycle ringdown step;
- guarded isolation receipt code 8;
- detector response and residual source feedthrough;
- exact integer reference recurrence.

The 0 and pi results are antipodal only within predeclared fixed-point
quantization bounds. They are not asserted bitwise negatives.

## Resource and claim law

The package counts ten numeric model cells, custody/topology metadata,
preparation/ringdown steps, virtual cycles, and the actual migration artifact.
It separately marks uninstrumented QEMU process memory, MMIO byte traffic,
primitive-operation totals, build cost, migration timing, process RSS/PSS, and
runtime/container overhead. Zero retained inverse history in P0 is not a
benefit: there is no accepted inverse.

Claim ceiling:

`DETERMINISTIC_FIXED_POINT_QEMU_PCI_MODEL_OF_SELECTED_P0_PROCESS_GEOMETRY_ONLY`

Not established:

- physical waveform execution or physical restoration;
- catalytic restoration or restored-carrier reuse;
- a distinct phase-native computational resource;
- computational advantage or Small Wall crossing;
- unbounded compute or replacement of physical bits with pi;
- P0 as the final architecture.

## Successor gate

V0 rejects the implemented deterministic selected-P0 natural-ringdown
recurrence as a catalytic-restoration mechanism. It does not rule out every
possible physical P0 echo law. The next backend must change the modeled
dynamics, not add more selected-P0 fixture arms.
It must provide a history-free native inverse/echo, restored same-carrier reuse,
a non-Gaussian or otherwise nonclassical interaction that changes the tested
boundary, and a growing family compared against exact sparse, tensor-network,
Gaussian, and controlled-approximation baselines. The QEMU implementation
remains an ordinary classical emulator; only an eventual physical resource
law can escape M257.
