# Recursive Phase Tree Charter

**Status:** `FROZEN_BOUNDED_R0_REFERENCE_CONTRACT`<br>
**Package:** `audio_recursive_phase_tree_v1`<br>
**Operation:** software architecture and deterministic reference only<br>
**Physical authority:** none

## 1. Primitive

The primitive is a recursively nested phase object, not a flat multitone vector.

For every node `v`:

```text
Phi_v(t) =
    2*pi*f_v*t + theta_v
    + sum_(c in children(v)) beta_vc * sin(Phi_c(t))
```

The root renders:

```text
B_T(t) = exp(i * Phi_root(t))
```

The unit-modulus beam carries information in recursive phase relations. A later physical
carrier may add an amplitude field, but amplitude is not the hierarchy identity in this
reference.

## 2. Structural Identity

A tree identity binds:

```text
node IDs
frequencies
local phases
directed parent-child edges
modulation indices
global spin phase
canonical ordering and serialization
```

Two trees may use the same node multiset while representing different objects because
their parent-child geometry differs.

A Fourier magnitude or energy statistic is not sufficient tree identity.

## 3. Global Ising Action

A global spin acts on the complete beam:

```text
Psi_T,+ = +B_T
Psi_T,- = -B_T
```

Equivalent phase form:

```text
Psi_T(t) = exp(i*theta_spin) * B_T(t)
theta_spin in {0, pi}
```

The action must preserve all internal relative phase relations. It may not rebuild a
flat waveform from a decoded spin.

## 4. Thin Catalytic Operator

The software reference borrows a deterministic dirty complex tape `tau` and applies:

```text
tau' = tau * B_T
```

The correct inverse is:

```text
tau_restored = tau' * conjugate(B_T)
```

Because `|B_T| = 1`, this is a reversible diagonal phase operator.

The mechanism is non-ceremonial only when:

```text
tau' differs nontrivially from tau
the extracted hierarchy response is computed from tau' or B_T
the correct inverse restores within the frozen tolerance
a wrong hierarchy inverse fails restoration
```

This establishes at most a software catalytic reference. It is not physical R2.

## 5. Query

The first hierarchy-sensitive query is the normalized complex overlap:

```text
Q(B_state, B_query) = mean(conjugate(B_query) * B_state)
```

An exact hierarchy has unit response. A changed parent-child geometry must produce a
different response while the amplitude-only channel remains null.

This query is intentionally simple. Later packages may replace it with a richer sensor
only by declaring the changed observable and adversary class.

## 6. Collapse Boundary

No scalar diagnostic feeds the reference recurrence.

Allowed boundary outputs:

```text
complex matched response
tree digest
restoration error
wrong-inverse error
optional final global Z2 spin
```

The complete tree remains the process-object before that boundary.

## 7. Frozen Thin-Slice Envelope

```text
sample rate:                         48000 Hz
frame duration:                      0.125 s
samples:                             6000
native numeric field:                float64 / complex128
committed fixture field:             stereo I/Q IEEE float32 little-endian
tree depth:                          at least 3
carrier amplitude:                   unit modulus
native correct-inverse tolerance:    1e-12 max absolute error
native global-Z2 tolerance:          1e-12 max absolute error
committed-WAV complex tolerance:     2e-7 max absolute error
committed-WAV restoration tolerance: 2e-7 max absolute error
hierarchy phase gap:                 at least 0.10 rad
cross-query magnitude:               at most 0.98
wrong-inverse error:                 at least 0.05
```

The native complex128 envelope and committed float32 envelope are distinct. Results
scored from parsed WAV bytes use the committed-WAV tolerance; they may not be adjudicated
against the tighter in-memory envelope or promoted as exact physical restoration.

These are software-reference thresholds only. They are not physical thresholds.

## 8. Required Thin-Slice Tests

```text
recursive depth present
same node multiset under different hierarchy
unit-modulus phase carrier
global Z2 rotates complete tree
amplitude-only exact null
hierarchy changes phase geometry
matched query prefers exact hierarchy
borrowed tape is actually mutated
correct inverse restores
wrong hierarchy inverse fails
canonical identity is deterministic
strict schema and canonical parser close
committed tree bytes match the declared fixture role
committed WAV bytes exactly match deterministic tree rendering
duplicate keys, cycles, malformed edges, nonfinite and unsafe inputs reject
substituted or nonminimal committed WAVs reject
subtree, phase-scramble, flat-bank, amplitude and spectrum controls close
wrong and reordered inverses fail
metadata stripping is invariant without admitting metadata into committed fixtures
```

## 8A. Committed-Byte Custody

Every declared tree is stored as canonical, newline-terminated JSON and every complete
beam is stored as a minimal `fmt ` then `data` stereo I/Q IEEE float32 WAV. Verification
must prove all of the following before scoring:

```text
tree bytes equal the source-declared role tree
tree digest and geometry identity close
WAV bytes exactly equal the deterministic render of that committed tree
parsed render error is within the frozen float32 envelope
RIFF chunks are exactly fmt then data
manifest path, role, hashes, sizes, metrics, orientation, and geometry close
```

The generic parser may accept an added metadata chunk only for the stripping adversary.
Such a file is forbidden as a committed fixture.

## 9. Claim Ceiling

Allowed after committed-byte reproduction:

```text
AUDIO_RECURSIVE_PHASE_TREE_REFERENCE_ESTABLISHED
```

Meaning only:

```text
the software reference represents and distinguishes nested phase geometry,
performs a nonzero reversible phase operation on borrowed software state,
and restores under the correct inverse while the wrong hierarchy fails
```

Forbidden:

```text
PHYSICAL_AUDIO_COMPUTING_ESTABLISHED
AUDIO_R2_RESTORATION_ESTABLISHED
AUDIO_RECURSIVE_CATALYTIC_ISING_EMULATOR_ESTABLISHED
AUDIO_CATALYTIC_TRANSFORM_CANDIDATE
SMALL_WALL_CROSSED
```

## 10. Successor Boundary

The next package must implement an operator over complete trees:

```text
T[k+1] = F(T[k], relation, query)
```

It may not use a decoded scalar spin to synthesize the next native state. Scalar Ising
updates remain an ordinary baseline.
