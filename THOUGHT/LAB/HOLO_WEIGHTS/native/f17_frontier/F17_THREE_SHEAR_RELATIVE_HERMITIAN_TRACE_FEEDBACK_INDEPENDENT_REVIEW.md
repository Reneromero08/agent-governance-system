# M117 independent review: three-shear relative-Hermitian feedback

## Decision

- Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`
- Verification level: `SEPARATE_REFERENCE_PARITY`
- Restoration class: `EXACT_ALGEBRAIC_RESTORATION`
- Hosted execution confirmation: `false`
- Scientific source parent: `367716267b83f2bcbcb6c3cd3d52f6209f70a582`

The reviewed source implements a bounded three-shear program on a 17-cell,
two-by-eight exact cyclotomic pair carrier.  The accepted path loads public
phase seeds, keeps all intermediate cells resident, projects one final integer
trace, applies the three arithmetic inverses in reverse order, unloads the
public seed, and reuses the same zeroed backing for the other public family.

## Independent reconstruction

The oracle does not import the M117 production module.  It reconstructs the
public seed phases and shear descriptors and executes two independent views:

1. a canonical degree-16 `Q(zeta_17)` power-basis implementation; and
2. the separate M116-oracle two-by-eight integer recurrence.

For `x=A+zeta B` and `y=C+zeta D`, production and the compact reference use

```text
p = A*C
q = B*D
r = (A+B)*(C+D)
h = 2*p + 2*q + s1*(r-p-q)
```

which equals `Tr_{Q(zeta)/Q(s1)}(x*conjugate(y))`.  Each shear applies
`zeta^k * x_left * h` to its target.  The `x_left` factor makes the law
covariant under a common root-of-unity rotation of all carrier cells.

The independent full-power and pair states agree after every forward and
inverse shear for both public families, one single-site phase perturbation,
three global rotations per family, and one alternate valid three-shear plan.

## Observations and controls

```text
family    coupled boundary    no-shear boundary
PRIMARY   197                 16
REUSE     112                 -1
```

- all three shear pairs have nonzero commutators on both public seeds;
- a same-order inverse fails restoration for both families;
- a wrong arithmetic inverse exponent fails restoration;
- descriptor mismatch is separately rejected before mutation;
- missing inverse leaves detectable resident state;
- premature projection is rejected;
- source/target aliasing is rejected;
- a resident mutation is detected at restoration;
- a single-site phase perturbation changes each tested boundary;
- global root-of-unity rotations preserve each tested boundary;
- primary and reuse transactions restore the borrowed zero carrier exactly;
- restored-carrier reuse and fresh reuse agree in boundary, rank, and arithmetic
  signature;
- the original carrier list backing is retained and no snapshot is loaded.

The zero cross-cell value is a declared-erasure diagnostic.  It is not an
executed physical dephasing model and is not used to claim a physical coherence
resource.

## Resource law

The accepted transaction uses 17 pair cells, or 272 integer coordinates.  Per
transaction it performs seven relative-Hermitian evaluations (21 real-subfield
multiplications), six source-scaling injections (12 more real-subfield
multiplications), and 28 fixed root-action steps.  The production package
counts carrier and control-state payload, public program and shear descriptors,
the retained 17-root table, seed load/unload buffers, named coupling work, the
final boundary payload, restoration, and reuse.

The named totals are 5,177 bits for `PRIMARY` and 5,123 bits for `REUSE`.
They are conservative sums of named logical component maxima, not simultaneous
or whole-process peaks.  Real-multiply internal accumulator scratch, Python
objects, allocator/native-library memory, bigint internals, and public-table
compilation work are not bounded.  The monotone generation and lease counters
make restored reuse two metadata bits wider than fresh reuse; rank and
arithmetic signatures otherwise agree.

The strongest matched classical implementation is the identical two-by-eight
integer recurrence.  The power-basis implementation is a semantic oracle, not
the baseline used to suggest a resource reduction.

## Review repairs applied before sealing

- The initial direct-trace injection was replaced by the source-scaled law
  after it failed global-phase covariance.
- The wrong-descriptor control was relabeled and a wrong arithmetic inverse was
  added.
- Pairwise commutators were executed instead of inferring noncommutation only
  from data dependency.
- Boundary payload was added to the named total; unbounded arithmetic scratch
  and compilation scope are now explicit.
- The zero control is labeled declared cross-cell erasure rather than executed
  dephasing.

## Strict claim ceiling

The evidence supports bounded exact, globally phase-covariant, non-affine
relative-Hermitian three-shear composition on the 17-cell two-by-eight carrier,
with a final integer trace boundary, reverse-order exact algebraic restoration,
and cross-family same-backing reuse in direct-process software.

It does not establish CATVM custody, a distinct phase resource, computational
advantage, a Small Wall crossing, catalytic inference, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.
