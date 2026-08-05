# M192 Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Reconstructed bounded mechanism

The production package tests a multiplicative Jastrow chart over the
exchange-symmetric, global-rotation quotient of the 17-mode rotor family. The
candidate state has one global scale and eight relative pair-distance
coordinates. Its amplitude at a necklace is the corresponding pair-signature
monomial. The public diagonal two-body phase changes only those coordinates,
so diagonal closure is exact by construction.

The independent oracle imports no production module. It generates bosonic
occupations from multisets, independently selects cyclic representatives, and
reconstructs scattering by enumerating ordered particles rather than applying
the production multiplicity loop. It decides chart membership constructively:
for every prime factor of `p-1`, it solves for chart exponents and verifies the
solution against every necklace cell. This is distinct from the production
rank-versus-augmented-rank test.

## Exact chart result

Both implementations agree exactly over `F103` and `F239` for two public
Jastrow input families:

| rotors | necklace cells | occupation histograms | off-diagonal chart result |
| ---: | ---: | ---: | --- |
| 2 | 9 | 153 | closed, because the 9-coordinate chart saturates the 9-cell space |
| 3 | 57 | 969 | escaped, with no zero-cell shortcut |
| 4 | 285 | 4,845 | escaped |
| 5 | 1,197 | 20,349 | escaped |

At three rotors, the scattered primary and alternate states are nonzero in
every cell but fail the exponent equations for every prime factor of `p-1`.
The larger cases also fail, sometimes with zero amplitudes that already lie
outside the multiplicative torus. Exact state commitments, public boundaries,
topology counts, generator nonzero counts, controls, and restoration metrics
match between implementations in all eight field/rotor cases.

The two-rotor closure is not transferable evidence: the chart matrix has full
rank nine in a nine-cell carrier for every tested factor, so it parameterizes
the entire nonzero state torus at that size.

## Restoration, controls, and resources

The accepted fallback is the full-necklace two-register shear `B += K*D(A)`.
The inverse word is rematerialized from public topology and subtracts from the
same target backing. Primary restoration and unrelated reuse are exact in
every field cell, restored reuse matches a fresh carrier boundary, and the
restoration generation reaches two. Missing, wrong, and reordered inverse
controls leave nonzero errors in every case; a null carrier is rejected.

The accepted path retains two full necklace registers and uses two
necklace-sized word temporaries. It retains no forward inverse history and no
assignment or relation table. Public topology and generator plans remain
material resources. The nine-coordinate membership matrices and discrete-log
tables are verification resources, not accepted-path compression.

The strongest compact classical baseline is the identical full-necklace exact
`K*D` shear recurrence. The package therefore rejects this particular stable
nine-parameter nonlinear chart under growing off-diagonal scattering. It does
not prove that no other nonlinear chart, adaptive atlas, tensor factorization,
or exact quotient exists.

## Claim ceiling

`GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS2_TO5_F103_F239_ONE_PRIMARY_AND_ONE_ALTERNATE_PUBLIC_JASTROW_FAMILY_EXACT_K_TIMES_D_GENERATOR_WORD_DIRECT_PROCESS_SOFTWARE_ONLY`

No CATVM custody, distinct phase resource, computational advantage, Small Wall
crossing, physical waveform execution, physical bit replacement, or unbounded
computation is established.
