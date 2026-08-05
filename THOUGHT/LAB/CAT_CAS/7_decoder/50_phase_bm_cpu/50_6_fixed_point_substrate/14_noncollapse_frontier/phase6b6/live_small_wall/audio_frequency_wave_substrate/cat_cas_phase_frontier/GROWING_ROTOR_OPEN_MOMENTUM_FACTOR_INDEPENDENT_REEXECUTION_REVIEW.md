# Independent reexecution review: Rotor-6 open-momentum factor closure

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

The declared two-body scattering law factors exactly through a typed
one-body momentum port. Production reflection-pairs the nonzero shifts and
uses one 4,389-cell necklace intermediate at a time before closing to the
2,277-cell bracelet carrier. Every port lease is owner- and momentum-typed,
never projected, zero-released, and reused on the same intermediate backing.

## Exact mechanism

For the one-body shift operator

```text
A_q f(n) = sum_a n_a f(n - e_a + e_(a-q)),
```

direct expansion gives

```text
K_q = A_-q compose A_q - N I.
```

The extra `N I` term is the derivative of the particle inserted by the first
one-body action. The public scattering coefficient depends only on the
unsigned shift distance, so `q` and `-q` have equal weights. Production uses
eight reflection pairs, one leased necklace vector, and two bracelet closure
passes per pair.

## Independent construction

The oracle imports no production or predecessor module. It independently:

- enumerates all 74,613 six-particle occupations;
- obtains 4,389 cyclic necklaces and 2,277 reflection bracelets;
- reconstructs the public source, diagonal, boundary, and topology digest;
- compiles a verification-only 684,624-term direct two-body CSR with 172,838
  nonzeros;
- compiles all sixteen unpaired one-body channels with 325,584 first-pass and
  168,912 closure plan entries; and
- compares the direct and factorized primary states cell for cell.

The unpaired oracle performs 494,496 one-body terms. Independently pairing its
work law gives production's 331,704 terms: eight times 20,349 necklace terms
plus sixteen times 10,557 bracelet terms. Topology, primary state commitment,
primary/reuse boundaries, restoration, fresh/restored reuse, and applicable
controls agree exactly.

## Controls

- Missing, wrong-program, and reordered inverses differ in 2,257, 2,252, and
  2,253 cells.
- Using the wrong reflection pairing differs in 2,254 cells.
- Omitting the required `N I` correction differs in 2,252 cells in the
  independent oracle.
- Wrong momentum type and owner generation are rejected.
- Projection while the 4,389-cell port is live is rejected.
- A null carrier is rejected, and every valid port release clears all cells.

## Resource and baseline ceiling

The accepted phase carrier retains 4,554 source/target field cells, one 4,389
cell open-port vector, and a 2,277-cell output scratch, for a conservative
11,220 named field-cell total at the factorized scattering boundary. It
retains no transition plan and no inverse history. Each scattering performs
331,704 encoded one-body moves, 5,638,968 cyclic code candidates, 5,307,264
rolling code updates, 331,704 topology lookups, and 35,112 port-clear writes.
This replaces M197's 684,624 two-body moves, 24,767,280 triangle-monomial
evaluations, and 7,669,494 binary signature comparisons, but increases the
explicit necklace topology and temporary state. Python container, hash-map,
big-integer, expression-temporary, timing, and whole-process memory costs are
excluded rather than zero.

The strongest matched classical implementation is the identical
reflection-paired factor stream with the same bracelet state, necklace
intermediate, maps, work, restoration, and reuse law. No computational
advantage or distinct phase resource is established.

## Claim ceiling

The result is limited to grid 17, six exchange-symmetric rotors, the declared
global-rotation and reflection sector, F103 root 72, and depth-one primary and
reuse programs in direct-process software. The typed intermediate is not a
machine-enforced CATVM custody boundary. The result does not establish
arbitrary rotor counts, fixed-rank growth, a Small Wall crossing, physical
waveform execution, replacement of physical bits with pi, or unbounded
catalytic computation.
