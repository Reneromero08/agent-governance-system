# M118 independent review

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

Scientific source parent: `fc8825af6eb392e5a4d143bcebd74b5354f4841d`

## Scope independently reconstructed

The review independently reconstructed both public four-site programs over
`Q(zeta_17)` in the canonical sixteen-coordinate power basis. It did not
import the M118 production module or the M116 two-by-eight backend. Exact
minor search and the independent full-basis execution reproduce the natural
tensor-train rank trace

```text
[1,1,1] -> [2,2,1] -> [2,2,1] -> [2,2,1] -> [2,4,2] -> [2,4,2].
```

Every final two-versus-two cut has rank four and every final one-site cut has
rank two for both public families. Mapping `zeta_17` to the order-seventeen
element 72 in `F103` gives nonzero final determinant residues:

```text
family    AB|CD  AC|BD  AD|BC
PRIMARY      59     42     86
REUSE        90     65     78
```

These certificates measure rank over `Q(zeta_17)`, not over the sixteen
rational coordinates or the M116 pair lanes.

The independently reconstructed final boundaries are:

```text
PRIMARY  [9,2,1,4,2,1,0,1,2,0,0,2,0,0,0,0]
REUSE    [9,0,2,0,3,0,4,0,2,1,0,0,0,1,0,2]
```

Reverse phase gates and the inverse shear recover the product seed exactly;
public seed unload returns the actual original backing to zero. Cross-program
reuse agrees with a fresh carrier in final boundary, exact rank trace, and the
full non-metadata arithmetic signature. No snapshot or retained inverse
history participates.

## Adversarial controls

The review reproduced failure or rejection for a wrong phase exponent,
reordered inversion across the noncommuting shear/coupling pair, a missing
inverse, premature projection, a resident mutation, an invalid descriptor,
and a null carrier. The zero-weight shear sham remains rank one; removing the
`BD` edge lowers at least one final two-versus-two cut; a forced rank-two cap
after `U_AD` contradicts an exact rank-four minor. An alternate weight family,
semantic weight perturbation, and shear-disabled path behave distinctly.

## Baseline and accounting corrections

Pre-seal review rejected two weaker baselines and one timing contamination.
The repaired matched baseline now compiles the public topology to a coalesced
sparse character map: eight nonconstant roots for `PRIMARY`, seven for
`REUSE`, zero general pair multiplications, and at most four live phase cells.
The independent oracle matches this recurrence to both a three-product
closed form and a four-assignment factor sum. Production boundary execution
no longer calls the compact comparator inside the accepted transaction.

Named payload totals include the actual sixteen-cell pair carrier, public
program, retained root table, seed handling, gate endpoints and intermediate
root-action steps, final projection, restoration, and reuse. Rank-verifier
tensor/minor costs are reported separately. Python objects, allocator and
bigint internals, native-library storage, and simultaneous whole-process peaks
remain outside the named logical accounting. The warm timing is observational:
the phase sample is a restoring transaction while the compact sample is a
boundary evaluator, so timing is not used for an advantage claim.

## Strict ceiling

The verified result is bounded to two public width-four, local-dimension-two
programs in direct-process Linux software. It establishes exact intra-program
rank variation, final-only projection, exact reverse restoration, and
same-backing reuse. Earlier `Q(zeta_5)` work already established bounded exact
TT-rank growth; M118 is an F17 carrier integration calibration, not a new
general TT mechanism or growing-rank family.

It does not establish compact variable-rank closure, a distinct phase
resource, computational advantage, Small Wall crossing, CATVM custody,
catalytic inference, physical waveform or silicon execution, replacement of
physical bits with pi, or unbounded computation.
