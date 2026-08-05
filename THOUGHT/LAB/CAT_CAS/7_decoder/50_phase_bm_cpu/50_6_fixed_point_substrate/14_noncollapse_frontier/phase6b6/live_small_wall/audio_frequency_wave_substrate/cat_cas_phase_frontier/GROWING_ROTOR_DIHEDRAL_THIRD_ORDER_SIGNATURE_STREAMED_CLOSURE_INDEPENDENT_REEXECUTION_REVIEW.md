# M197 independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`
Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

The bounded F103 Rotor-6 package removes the 16 retained shift plans and their
652,048 sparse nonzeros from the accepted path.  It preserves the 2,277-cell
two-triangle refined quotient, final scalar projection, exact same-backing
two-register shear restoration, and unrelated fresh/restored reuse parity.

## Independent reconstruction

The oracle imports neither the production module nor M194.  It independently:

- enumerates exchange-symmetric occupations as particle multisets;
- constructs rotation necklaces and reflection bracelets;
- counts all particle pairs and unordered particle triples;
- proves that the eleven-coordinate signature has exactly the bracelet counts
  `9, 33, 165, 621, 2277` for rotors 2 through 6;
- moves occupation cells directly, builds a verification-only 652,048-nonzero
  bracelet plan, and hashes all 684,624 unaggregated mode-pair-shift records;
- counts affected stencil monomials by scanning all 17 cyclic anchors rather
  than using the production affected-anchor construction; and
- reexecutes the diagonal/scattering word, controls, restoration, and reuse
  through the independently built plan.

The raw transition commitment is
`1774f44283dde49154d3aec11bb0dd22179a061290e44448c795402fa2ca0465`.
The independently reconstructed primary output commitment is
`834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e`.
Both equal the production values.

## Resource and obstruction result

The accepted path retains 25,047 signature integers, 38,709 representative
integers, 2,277 boundary field cells, and the fixed eight-integer triangle
stencil.  It retains no shift basis, shift-plan nonzero, inverse history,
assignment table, or relation table.

Plan storage is exchanged for exact rematerialization.  Each scattering uses
684,624 distinct mode-pair-shift terms, 1,092,960 multiplicity-weighted particle
terms, 24,767,280 affected triangle-monomial evaluations, and 7,669,494 binary
signature comparisons.  The maximum is 44 affected monomials for one move.
Python containers, allocator state, big integers, expression temporaries, and
whole-process peak memory are excluded but not zero.  The verification-only
oracle plan is not part of the accepted path.

The strongest compact classical comparison is the identical topology-streamed
eleven-coordinate recurrence and matches the phase path exactly.  Consequently
this is a retained-plan repair, not evidence of a computational advantage or a
distinct phase-native resource.  The quotient still equals the full bracelet
dimension at Rotor 6.

## Controls and strict ceiling

Missing, wrong-family, and reordered inverse controls leave respectively
2,257, 2,252, and 2,253 nonzero field cells.  Null-carrier rejection passes.
The actual carrier restores with zero field-cell error, preserves both backing
lists, advances to restoration generation 2, and yields reuse boundary 70 for
both restored and fresh carriers without baseline reload.

The verified ceiling is exactly:

`GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_TWO_TRIANGLE_REFINED_SIGNATURE_INPUTS_ROTORS2_TO6_TOPOLOGY_F103_ROTOR6_PRIMARY_DEPTH1_REUSE_DEPTH1_DIRECT_PROCESS_SOFTWARE_ONLY`

This package does not establish CATVM custody, a compact quotient below
bracelet identity, a phase resource unavailable to compact classical software,
computational advantage, a Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

## Next obstruction

The immediate obstruction is no longer retained shift-plan state.  It is the
combination of full 2,277-bracelet carrier dimension, 28S public integer
descriptors, and 24,767,280 exact triangle-monomial evaluations per scattering.
The next mechanism should seek a lower-rank exact update or a phase-native
coupling law that changes this matched recurrence, rather than add another
Rotor-6 fixture or restore the eliminated plan.
