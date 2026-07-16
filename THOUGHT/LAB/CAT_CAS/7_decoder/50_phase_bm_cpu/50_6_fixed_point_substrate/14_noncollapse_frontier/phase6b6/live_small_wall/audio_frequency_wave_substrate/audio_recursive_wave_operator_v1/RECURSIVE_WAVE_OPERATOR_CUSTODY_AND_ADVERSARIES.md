# Recursive Wave Operator Custody and Adversaries

**Status:** `FROZEN_R1_MECHANICAL_CUSTODY_LAW`

## Step Specification

`TemporalStepSpec` admits exactly nine fields: schema, step index, root identity,
carrier frequency, root phase, state modulation index, drive modulation index,
`state_child_index`, and `drive_child_index`. The two indices are distinct and exactly
cover positions `0` and `1` in the canonical serialized child order. They prospectively
bind which complete child is the predecessor and which is the drive, including when the
two modulation indices are equal. The spec uses canonical UTF-8 JSON and rejects
duplicate keys, unknown fields, nonfinite values, booleans as numbers, step/root
mismatch, Nyquist violations, envelope violations, duplicate role indices, and
noncanonical bytes. No answer-bearing field is admitted.

## Receipt

Every receipt binds the embedded exact step specification and both its canonical-file
SHA-256 and compact deterministic digest. It also binds the step index, complete result
root and digest, complete state root and digest, state-edge modulation index, complete
drive root and digest, drive-edge modulation index, and exactly two ordered roles:
`state`, then `drive`. It additionally binds both canonical serialized child indices to
the independently prospective positions embedded in the exact step specification.

Receipt validation checks the complete result before extraction, both child trees, both
edge values, role multiplicity, absence of an extra child, and exact step custody. Full
trajectory validation additionally compares the extracted roles with the committed prior
state and declared drive before reverse traversal.

`temporal_step` canonicalizes the emitted complete result and rejects unless the actual
state and drive child positions equal the prospective step positions. Receipt parsing and
validation then reject any position change. A full equal-beta swap of roots, tree digests,
and indices therefore fails against the independently committed step.

## Committed Packet

```text
states       T0, T1, T2, T3 canonical tree JSON
drives       D1, D2, D3 canonical tree JSON
steps        step1, step2, step3 canonical specification JSON
receipts     step1, step2, step3 canonical receipt JSON
wave charts  T0, T1, T2, T3 minimal stereo I/Q float32 WAV
```

The manifest orders all 17 files, binds their hashes and total bytes, and records every
state's root, depth, node count, predecessor, drive, step, receipt, WAV identity, and
complex metrics. Verification parses and scores the committed bytes rather than trusting
regenerated arrays or stored PASS strings.

## Closed Adversary Classes

```text
state/drive identity collision
new-root collision
wrong state digest
wrong drive digest
swapped roles
fully swapped equal-beta roots, digests, and child indices
duplicate or changed canonical child indices
changed state beta
changed drive beta
changed step specification
wrong result root
extra child
missing child
duplicate role
reordered receipt roles
drive-order reversal
flat-wave native and receipt admission
decoded-spin native and receipt admission
trajectory truncation
trajectory duplication
out-of-order trajectory construction
WAV fixture substitution
manifest trajectory-order mutation
forbidden scalar-name insertion
aliased, dynamic, module-helper, or class-method call insertion
digest-driven or node-count-driven recurrence control
module built-in shadowing
module-attribute runtime rebinding
decorated native-root replacement
indirect global native-root rebinding
import-alias native-root rebinding
loop-target native-root rebinding
function-default runtime rebinding
native runtime binding shadowing
receipt lifecycle rebinding
result-store rebinding
structural pattern-capture rebinding
```

The flat-wave similarity is diagnostic only. Structural ancestry, receipt admission, and
exact T0 recovery are the primary controls.
