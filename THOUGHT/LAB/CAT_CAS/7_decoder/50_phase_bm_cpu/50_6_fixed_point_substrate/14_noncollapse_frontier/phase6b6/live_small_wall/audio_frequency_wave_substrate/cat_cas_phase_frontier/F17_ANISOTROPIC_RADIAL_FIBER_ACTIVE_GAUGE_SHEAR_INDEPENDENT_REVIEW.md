# M147 independent review

Decision: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`.

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`.

Resident 51-angle carrier restoration:
`NUMERICAL_PHYSICAL_STATE_RESTORATION`.

Transient Cartesian, chart, comparison, and verification buffers:
`NO_RESTORATION_CLAIM`.

M147 makes the M146 phase fiber causally relevant to the final boundary.  At
each public step it applies a magnitude-preserving diagonal shear

```text
z_j -> z_j exp(i (3/16) sin(g_j + public_offset(j,step,family)))
```

to the decoded base value while retaining the actual resident gauge angle.
The shear is interleaved with the public phase module and the 136-coupler
local Givens transform.  Swapping phase/shear order or shear/Givens order
changes the final boundary.  The same transported gauge is consumed again at
later steps.

The bounded causality witness starts two carriers with the same 17 complex
base values to `5.888e-16` but a gauge-phasor separation of `0.714`.  The same
public depth-4 program separates their final base states by `0.0825` and their
final boundaries by `0.0493`.  Actual reverse execution restores both 51-angle
carriers within `7.241e-14`.  This establishes that the base-only 17-complex
quotient is insufficient for the declared variable-gauge carrier family.

Across the 21 declared cases through depth 4096, maximum state error against
the executed matched scalar recurrence is `5.361e-12`, maximum boundary error
is `6.908e-11`, and maximum single-transaction restoration error is
`2.839e-11`.  Unrelated depth-1537 reuse agrees with fresh execution within
`9.368e-13`; 100 same-backing depth-64 cycles restore within `2.007e-11`.
No snapshot, inverse history, retained restoration baseline, or post-inverse
reset is used.

The independent oracle imports neither M147 production nor M146.  It shares
only the established M145 public program and Givens-plan compiler, then
separately implements the weighted three-phasor chart, 51-angle
forward/inverse, gauge shear and transport, causality witness, compact scalar
recurrence, mutation controls, and reuse paths.  All 146 declared comparisons
pass.  A fresh qualifier replay reproduces both sealed JSON files
byte-for-byte.

Frozen package hashes are:

```text
production source  eb847912e10ec18ba00b03a920a3d7bd4f76063b55d54ba4515fd655e0533c52
oracle source      2bfd1e07d749718fb9fd807a93ed163eb90ace80eeb958d7339bf009cfdca0e1
production JSON    a6d2fb757dbb2e0184f414326c55a2cabdb9c8dc31ecbc71d3b5a5bcafdf3c2a
oracle JSON        508498167ffa6bcff7a967e578678aee7f483b1fb1fb4d96c336ec55dd36cb9a
qualifier          2e4b610dba9181879fc31e42d817d4b6dde8c1bc04606b0011b0f91c2e38fd18
predecessor        27a147d45415e97de5fafd9436cfce7a3f1cb0ae0cbd3410189560282a760590
```

The accepted path retains 51 phase angles or 408 bytes, the 2,312-byte public
Givens plan, and no complex state or dense kernel.  Its named warm peak is
3,223 bytes and it still uses 24 local Cartesian/chart scratch cells.  The
executed matched recurrence stores exactly 17 complex base values plus 17
gauge scalars: the same 51 `float64` scalar-equivalent cells and 408 resident
bytes.  Its named warm peak is 3,191 bytes, and it avoids phase-chart decode
and reencode.  No optimal compact classical baseline is claimed.

Controls detect zero shear, phase/shear and shear/Givens order mutations,
missing shear inverse, wrong shear inverse, reordered inverse, premature
projection, null carrier, and out-of-envelope input.  The result is limited
to the sealed direct-process Linux cases and the declared interior magnitude
envelope.

M147 establishes neither a resource beyond compact 51-scalar software, an
optimal classical lower bound, a global full-sphere chart, exact algebraic
semantics, CATVM custody, computational advantage, a Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, nor
unbounded catalytic computation.
