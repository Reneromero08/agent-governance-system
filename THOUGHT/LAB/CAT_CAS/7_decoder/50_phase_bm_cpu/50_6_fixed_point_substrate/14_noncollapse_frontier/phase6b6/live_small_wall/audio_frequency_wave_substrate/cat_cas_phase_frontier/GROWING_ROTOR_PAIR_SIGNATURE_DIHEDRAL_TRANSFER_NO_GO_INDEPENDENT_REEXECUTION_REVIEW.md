# M195 independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `NO_RESTORATION_CLAIM`

The bounded exact diagnostic finds that the M193/M194 pair-signature fibers
are precisely dihedral bracelet orbits for rotor counts two through five.  At
six rotors, pair signatures merge 2,277 bracelets into 2,253 cells through 24
non-dihedral homometric fibers.  Every one of those 24 fibers fails signed
shift equitability.  Pair signatures alone therefore cannot define the
scattering kernel at this transfer ceiling.

## Independent construction

The production implementation streams weak compositions, canonicalizes
global rotations, groups by unordered cyclic pair counts, and compares exact
ordered mode-pair shift profiles.  The independent oracle does not import the
production source or call its transition.  It independently:

1. enumerates particle multisets with
   `itertools.combinations_with_replacement`;
2. rebuilds occupations and rotation representatives;
3. computes unordered pair signatures from explicit particle positions;
4. reconstructs dihedral orbits by reflecting occupations;
5. computes shift profiles by enumerating ordered particle pairs; and
6. derives the Burnside bracelet counts separately.

All production and oracle case records agree exactly for rotor counts two
through six, including cell laws, fiber-size histograms, orbit comparisons,
equitability counts, and the committed first mismatch.

## Exact result

The occupation, necklace, bracelet, and pair-signature cell counts are:

| Rotors | Occupations | Necklaces | Bracelets | Pair signatures |
|---:|---:|---:|---:|---:|
| 2 | 153 | 9 | 9 | 9 |
| 3 | 969 | 57 | 33 | 33 |
| 4 | 4,845 | 285 | 165 | 165 |
| 5 | 20,349 | 1,197 | 621 | 621 |
| 6 | 74,613 | 4,389 | 2,277 | 2,253 |

For rotor counts two through five, all fibers have size one or two and match
the reflection quotient exactly.  At six rotors, the fiber-size histogram is
157 singleton, 2,072 doubleton, 8 tripleton, and 16 four-member fibers.  The
24 excess identifications beyond reflection are precisely the 24
non-dihedral fibers.

The first nonequitable fiber fails at signed shift one.  Its two exact rows
both sum to 30 ordered particle moves, but differ in 39 destination
coefficients.  The sealed result retains only that count, one representative
difference, and SHA-256 commitments to the full rows; it does not publish an
expanded destination table.

## Resource and baseline boundary

This package is a topology diagnostic, not an accepted carrier execution.
It enumerates all declared occupations and compares exact integer profiles;
it makes no restoration or custody claim.  Its matched classical diagnostic
is the identical homometry and signed-shift-profile comparison.  It does not
establish a computational resource or advantage.

The additional rotor count is applicability-gated: it directly tests whether
the pair-signature closure inferred at M193/M194 transfers beyond their
declared ceiling.  The failure answers that obstruction; further scaling of
the unrefined signature would not.

## Claim ceiling

The result is limited to Grid17, exchange-symmetric, global-rotation
necklaces at rotor counts two through six, all sixteen nonzero signed shifts,
and exact integer topology diagnostics.

Preserved subclaims are the exact dihedral interpretation at rotor counts two
through five, the stated Burnside count for odd Grid17 with rotor count below
17, and an exact rotor-six counterexample to pair-signature-only shift
closure.

Rejected interpretations include failure of every compact invariant,
necessity of full bracelet labels, CATVM custody, restoration, a distinct
phase resource, computational advantage, Small Wall crossing, physical
waveform execution, physical replacement of bits with pi, and unbounded
computation.

## Next obstruction

The smallest relevant repair is a dihedral-invariant higher-order phase
signature that separates the 24 rotor-six homometric collisions and restores
signed-shift equitability without falling back automatically to full bracelet
identity.  It must retain the identical compact classical recurrence as the
matched baseline.
