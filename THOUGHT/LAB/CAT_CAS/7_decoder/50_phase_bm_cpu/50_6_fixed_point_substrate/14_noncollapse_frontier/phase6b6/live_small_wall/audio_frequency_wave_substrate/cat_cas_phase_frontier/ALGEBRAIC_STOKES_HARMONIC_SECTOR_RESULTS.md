# Parity-closed Stokes harmonic-sector results

The topology-compiled parity sector passed exact phase/reference parity,
boundary-only projection, missing/wrong/reordered inverse controls, snapshot
separation, restoration, and eight transactions on the same restored
carrier.

Accepted bounded claim:

```text
BOUNDED_PARITY_ADMISSIBLE_STOKES_HARMONIC_SECTOR_DUAL_PRIME_PHASE_SIGNATURE_REDUCTION_WITH_RESTORATION_AND_REUSE
```

Claim ceiling:

```text
BOUNDED_NORMALIZED_TWO_MODE_PARITY_ADMISSIBLE_STOKES_HARMONIC_DUAL_PRIME_LIE_GRADES2_3_4_5_6_SOFTWARE_REFERENCE_ONLY
```

Measured results:

```text
previous Stokes basis cells                 135
parity-admissible basis cells                80
previous logical dual-prime payload       4,320 bytes
reduced logical dual-prime payload        2,560 bytes
correct restoration residual          2.220e-16
maximum eight-use residual             1.110e-15
wrong inverse residual                  1.993
wrong inverse modular mismatch cells       31
```

All exact highest homogeneous sphere-quotient classes at degrees
`2,3,4,5,6` are nonzero. Thus the parity law removes impossible carrier
cells but does not solve the rank obstruction: the highest irreducible
harmonic shell survives at every tested grade. The next phase-owned question
is whether those shells obey a compact exact recurrence or whether their
independent rank necessarily grows.
