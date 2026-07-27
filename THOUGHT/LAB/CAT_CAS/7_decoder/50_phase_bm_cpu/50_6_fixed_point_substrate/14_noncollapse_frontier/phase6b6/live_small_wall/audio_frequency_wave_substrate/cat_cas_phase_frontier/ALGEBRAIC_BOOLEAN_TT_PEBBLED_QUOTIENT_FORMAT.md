# Reversible Pebbling of Boolean-TT Quotient Stages

## Status

This bounded phase-machine repair establishes:

```text
TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_REDUCES_RETAINED_PHASE_HISTORY
```

within:

```text
BOUNDED_LINUX_SOFTWARE_WIDTHS4_5_8_12_16_DEPTHS4_5_8_HOMOGENEOUS_NEIGHBOR_AND_OR_PHASE_PEBBLING_REFERENCE_ONLY
```

It reduces actual phase carrier allocation. It does not establish fixed-rank
unbounded-depth closure, a phase resource unavailable to compact classical
computation, advantage, a Small Wall crossing, or physical execution.

## Public reversible schedule

The quotient chain is compiled as path nodes:

```text
node j = H(j+1)
H1 is the permanent transaction leaf
```

With `p=ceil(log2(depth))` work slots, the compiler emits a public toggle
sequence from:

```text
R(1,a) = toggle(a+1)

R(p,a) =
    R(p-1,a)
    toggle(a + 2^(p-1))
    reverse(R(p-1,a))
    R(p-1,a + 2^(p-1))
```

Execution truncates the sequence when `Hd` is first resident. Slot choice is
the deterministic lowest clean slot. Every move records node, slot, exact
activation generation, and predecessor generation in the schedule hash.

A stage may be added or removed only while its predecessor and H1 are
actually resident. Addition requires the complete target slot to equal its
borrowed baseline. Removal applies the actual conjugate phase composition
and must restore the complete slot before rebinding.

Only `Hd` is copied and decoded. The boundary copy is then removed and the
exact slot-tagged tape is reversed. H1 is removed last. The comparison
snapshot is used only to verify restoration; it is never the accepted
restoration path.

## Measured trade

At width 16/depth 8, the public 13-move forward tape uses three slots with
capacities:

```text
3,548, 2,968, 1,848 cells
```

It leaves `H5`, `H7`, and `H8` resident at projection. The actual carrier
law changes from:

```text
retain all carrier             17,288 cells / 553,216 bytes
pebbled carrier                12,152 cells / 388,864 bytes
reduction                       5,136 cells / 29.708%

retain-all phase updates       34,576
pebbled phase updates          46,896
update multiplier               1.356x
```

The recomputation is counted rather than hidden: 39,320 stage-move cells,
six reconstruction additions, and 6,160 reconstruction cells per complete
forward/inverse transaction.

Exact final parity holds against both the retain-all phase machine and the
independent raw-product verifier. Wrong boundary inverse, missing inverse,
and noncommuting reordered inverse controls fail restoration. Dirty-slot
injection, live predecessor-generation tampering, and schedule-hash tampering
are rejected by custody before an accepted restoration. Snapshot reload
remains separate, and 18 same-carrier transactions restore and reuse the
actual allocation below the predeclared `2e-12` tolerance.

## Scientific boundary

This repairs the retained-history depth factor in the phase machine. It does
not repair the stronger obstruction found by the triad: homogeneous
neighbor-AND/OR quotient cores are still generated directly by a public
Boolean threshold recurrence. Further depth or width in this family is not
the selected frontier. The next phase-owned experiment must introduce a
broader non-affine relation signature or a phase coupling law whose useful
unresolved state is not immediately homomorphic to that recurrence.

## Reproduction

```bash
evidence_parent=$(mktemp -d /tmp/boolean-tt-pebble.XXXXXX)
bash qualify_algebraic_boolean_tt_pebbled_quotient.sh \
    "$evidence_parent/evidence"
```

Reviewed local evidence:

```text
/tmp/boolean-tt-pebble.gKMVMI/evidence
```
