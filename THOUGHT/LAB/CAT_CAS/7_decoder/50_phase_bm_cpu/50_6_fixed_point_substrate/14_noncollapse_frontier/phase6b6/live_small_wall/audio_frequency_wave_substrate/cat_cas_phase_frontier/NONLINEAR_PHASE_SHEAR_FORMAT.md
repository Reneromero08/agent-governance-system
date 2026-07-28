# Nonlinear Unit-Phase Torus Shear

## Status

This bounded phase-machine experiment establishes:

```text
BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_INTERFERENCE_PROJECTION
```

within:

```text
BOUNDED_LINUX_SOFTWARE_NONLINEAR_TORUS_SHEAR_DEPTHS3_32_128_512_2048_4096_DOUBLE_COMPLEX_PHASE_REFERENCE_ONLY
```

It establishes a nonlinear, noncommuting phase update law on the existing
unit-modulus carrier. It does not establish a phase resource unavailable to
compact classical computation, machine-enforced custody, advantage, a Small
Wall crossing, physical execution, or unlimited computation.

## Native state and law

Two baseline-relative unit phasors are the entire resident state. Public
topology alternates the target cell. For target `t`, source `s`, and public
strength `k`, one forward morphism is:

```text
z[t] <- z[t] * exp(i * k * Im(z[s]))
z[s] <- z[s]
```

The update consumes the actual `relative` source phase and changes the target
only through `multiply_cell`. No code path directly mutates carrier storage.
Because a shear leaves its source fixed, its lawful inverse conjugates the
factor while that same source is resident. Alternating axes make adjacent
shears noncommuting, so topology order matters.

Only a copied final two-phase boundary is decoded. Its interference
probability is:

```text
(1 + Re(z[0] * conjugate(z[1]))) / 2
```

The boundary copy, inverse shears in reverse order, and initial seal are then
removed. Sixteen additional transactions consume the same restored carrier.

## Controls and resource law

The bounded suite covers depths `3, 32, 128, 512, 2048, 4096`. It checks:

```text
unit modulus after every update
exact quantized parity with a two-angle reference
deterministic replay
coupling-disabled boundary change
baseline-neutralized boundary change
actual null-carrier execution rejection
intermediate-projection rejection
wrong boundary inverse
missing shear inverse
wrong shear inverse
reordered noncommuting inverse
snapshot reload distinct from actual inverse
actual restored-carrier reuse
analyzer and ASan/UBSan
```

At depth 4096, 8,200 native phase updates execute. Maximum unit-modulus error
is `3.33066907388e-16`; maximum repeated restoration error is
`7.58532257521e-14` against the predeclared `2e-12` tolerance.

The accepted-path accounting is:

```text
live carrier                         4 cells / 128 bytes
verification snapshot                         128 bytes
peak carrier-related storage                  256 bytes
final projection                               48 bytes
declared local shear temporaries               80 bytes
best matched two-angle state                   16 bytes
compiled shear list                             0 bytes
```

Both phase and classical paths are `O(depth)`. The classical reference stores
two doubles and applies the same public nonlinear recurrence. Therefore the
remaining obstruction is an equivalent fixed two-angle classical state.

## Reproduction

```bash
evidence_parent=$(mktemp -d /tmp/nonlinear-phase-shear.XXXXXX)
bash qualify_algebraic_nonlinear_phase_shear.sh \
    "$evidence_parent/evidence"
```

Reviewed local evidence:

```text
/tmp/nonlinear-phase-shear-final.uC5B2f/evidence
```
