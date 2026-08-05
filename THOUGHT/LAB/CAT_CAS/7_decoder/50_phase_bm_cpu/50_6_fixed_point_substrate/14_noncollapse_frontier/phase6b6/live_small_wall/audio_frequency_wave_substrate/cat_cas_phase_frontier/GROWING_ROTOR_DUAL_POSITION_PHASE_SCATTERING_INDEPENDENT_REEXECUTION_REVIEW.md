# Independent reexecution review: Rotor-6 position-dual scattering

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

Result: `PASS_NEGATIVE_COMPACTION_RESULT`

The exact F103 order-17 Fourier construction reproduces the established
2,277-cell reflection-paired Rotor-6 scattering result. It retains no dense
`2,277 x 2,277` operator, no 4,389-cell open-momentum port, and no permanent
assignment expansion. The standard exact bosonic lift nevertheless leaves
the rotation quotient while the transform is live and explicitly allocates
all 74,613 exchange-symmetric occupation cells. It is therefore a valid
phase-dual realization and a failed compaction repair.

## Production mechanism

Production compiles the public 17-mode Fourier matrix and its normalized
inverse into elementary row operations over F103. Each operation is lifted to
the degree-six symmetric polynomial. Position-basis scattering is diagonal:

```text
lambda(n) = sum_(q=1..16) w_q (S_q(n) S_-q(n) - 6)
S_q(n) = sum_x n_x zeta^(q x)
```

The transformed state has support only in the 4,389-cell zero-total-coordinate
sector, but the elementary network passes through the complete 74,613-cell
occupation basis. The result closes exactly back to the 2,277 reflection
bracelets.

## Independent construction

The oracle does not import or call the production position-dual module. It:

- reconstructs the forward and inverse Fourier matrices and independently
  Gaussian-eliminates them to 288 and 289 elementary operations;
- obtains the same two network commitments;
- verifies the normalized Fourier inverse cell for cell;
- exhaustively checks
  `S_q S_-q - 6` against the ordered-particle-pair character sum for all
  74,613 occupations and all sixteen nonzero momenta;
- checks both public primary and reuse weight laws with zero mismatch;
- reconstructs the primary and reuse final states through the prior separate
  direct/factor oracle, including a 684,624-raw-term direct CSR with 172,838
  nonzeros; and
- independently derives the lifted-network cell, topology, descriptor, block,
  term, and character-operation counts.

Primary boundary 83, reuse boundary 70, topology commitment, and primary state
commitment agree exactly. Direct and factor outputs differ in zero cells.

The independent oracle intentionally does not execute the production
74,613-cell transform or reproduce its per-network-boundary nonzero counts.
Those nonzero checkpoint counts remain package-local diagnostics. They are not
used to establish the allocation no-go, whose 74,613-cell backing and work law
are independently derived.

## Controls

- Omitting Fourier inverse normalization changes 17 matrix cells.
- Replacing the primitive root with the nonprimitive root 1 changes 272
  inverse-law matrix cells.
- Omitting the `-6` particle correction changes all 74,613 occupation
  eigenvalues for both public programs.
- Treating only the 4,389 zero-total-coordinate cells as the live transform
  basis undercounts the implemented scratch by 70,224 cells.
- Missing, wrong-program, and reordered restoration controls differ in 2,257,
  2,252, and 2,253 final carrier cells.
- Exact rematerialize-and-subtract restoration returns the same target backing
  to zero, advances the restoration generation to two, and gives fresh versus
  restored reuse agreement without baseline reload.

## Resource and matched-baseline result

The phase-dual mechanism has 81,444 active numeric cells at its declared
algorithm boundary: source, target, output, and the full occupation scratch.
Its public occupation topology contributes 1,343,034 logical descriptor
integers, predecessor topology contributes 96,723, and the retained transform
networks contribute 2,308. The resulting named field-and-descriptor count is
1,523,509, before Python containers, allocator overhead, expression
temporaries, timing, or whole-process peaks.

One forward scattering uses:

```text
8,434,176 shear blocks
33,829,728 shear-polynomial terms
40,589,472 diagonal character terms
74,613 explicit occupation cells
```

The strongest compact classical baseline is M199's identical exact
reflection-paired factor stream: 11,220 named field cells and 331,704 one-body
terms per scattering. The identical phase-dual recurrence is also retained as
the representation-matched baseline. Thus the proposed repair is strictly
worse on both explicit state and operation counts; the comparison is not
against an intentionally dense classical representation.

## Claim ceiling

The result is limited to grid 17, six exchange-symmetric rotors, the declared
rotation- and reflection-invariant sector, F103 root 72, depth-one primary and
reuse programs, direct-process software, and this Gaussian-elimination
bosonic Fourier construction.

It does not establish CATVM custody, fixed-rank growth, a distinct phase
resource, computational advantage, a Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, catalytic inference, or
unbounded catalytic computation. The next phase-owned obstruction is to keep
the Fourier-domain scattering inside a compact quotient throughout the
transform, rather than allocating the complete occupation basis.
