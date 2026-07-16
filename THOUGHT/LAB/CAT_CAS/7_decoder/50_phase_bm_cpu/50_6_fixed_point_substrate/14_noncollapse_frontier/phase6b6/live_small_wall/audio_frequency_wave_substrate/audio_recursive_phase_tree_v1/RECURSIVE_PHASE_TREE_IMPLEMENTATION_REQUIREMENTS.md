# Recursive Phase Tree Implementation Requirements

Status: `FROZEN_R0_IMPLEMENTATION_LAW`

## Native Object

The only native state is the complete recursive phase tree:

```text
Phi_v(t) = 2*pi*f_v*t + theta_v
           + sum_(c in children(v)) beta_vc*sin(Phi_c(t))

B_T(t) = exp(i*Phi_root(t))
Psi_T(t) = exp(i*theta_spin)*B_T(t), theta_spin in {0,pi}
```

No decoded spin, energy, FFT magnitude, matched score, winner label, or restoration
diagnostic feeds a native update. This package implements no native temporal update.

## Strict Tree Envelope

```text
schema                         recursive_phase_tree_v1
sample rate                    exactly 48000 Hz
node count                     1 through 64
tree depth                     at most 16
node identifier                ASCII safe pattern, at most 64 bytes
frequency                      finite and strictly inside (0, Nyquist)
local phase                    finite and inside [-2*pi, 2*pi]
modulation index               finite and inside [0, 4]
global orientation             exactly 0 or pi
graph                          connected rooted tree
unknown or duplicate JSON key  reject
```

Duplicate nodes, cycles, malformed edges, multiple parents, disconnected nodes,
nonfinite values, oversized integers, Nyquist frequencies, unsafe identifiers, excess
depth/count, unexpected fields, and noncanonical committed serialization fail closed.

## Canonical Serialization

Tree and evidence JSON is UTF-8, sorted-key, two-space-indented, newline-terminated,
and excludes NaN/Infinity. Nodes sort by ID. Edges sort by parent, child, and modulation
index. Tree identity is SHA-256 over compact canonical semantics; committed file identity
is SHA-256 over the pretty canonical bytes.

## Committed Complex Fixtures

The declared fixture set is:

```text
hierarchy A, global +1
hierarchy B, global +1, same node multiset and different parent-child geometry
hierarchy A, global -1
```

Each fixture contains one complete tree as a minimal stereo I/Q IEEE float32 WAV with
exactly `fmt ` then `data` chunks. Verification regenerates the expected bytes from the
committed canonical tree and requires exact byte equality before parsing or scoring.

## Frozen Numeric Law

```text
native complex128 tolerance             1e-12
committed float32 complex tolerance     2e-7
principal energy relative tolerance     1e-7
hierarchy phase gap minimum             0.10 rad
exact matched response minimum          1 - 1e-12
wrong hierarchy response maximum        0.98
matched adversary gap minimum            0.02
borrowed tape mutation L2 minimum        1.0
wrong/reordered restoration error min    0.05
spectrum magnitude relative tolerance    1e-12
portable metric comparison atol/rtol     5e-12 / 5e-12
```

The environment receipt is informational. Scientific comparison requires identical
structure, identities, statuses, and nonnumeric fields; numeric leaves use the frozen
portable comparison. Any recomputed hard-gate status change fails verification.

## Operations

```text
build      generate the exact bounded schema, tree/WAV fixtures, manifest, tests, result
verify     recompute identities and scientific results from committed-format bytes
self-test  build and verify a disposable package independently
```

Stored `PASS` strings carry no authority. `verify` recomputes every observation and
status and separately closes the source binding.
