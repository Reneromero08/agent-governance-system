# F17 C17 common-congruence and Hadamard no-go independent reexecution review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

## Structural result

For a linear quotient of the 17-coefficient space, arbitrary coefficientwise
Hadamard multiplication descends only when the kernel is invariant under all
coordinate projectors. Any nonzero kernel vector therefore yields at least one
coordinate basis vector. Cyclic convolution includes multiplication by the
one-step delta, so a convolution-compatible kernel must also contain the full
17-element shift orbit of that coordinate. The independently computed orbit
rank is 17 for every coordinate. Consequently the only common invariant
kernels are zero and the full space. Every nonzero common quotient therefore
retains all 17 field coordinates per relation.

This is a no-go for linear quotients on which both bilinear operations descend
for arbitrary operands. It is not a no-go for nonlinear encodings,
operation-restricted relations, or a different phase algebra.

## Restricted compatible family

Production solves the exact descent constraints for diagonal multipliers on
the rank-2, rank-4, and rank-8 Hasse-jet quotients. The independent oracle
reconstructs the same constraint with monomial evaluation spaces rather than
the production binomial basis. Both computations find a one-dimensional
multiplier space whose normalized basis is the all-ones vector. Thus only
public constant coefficient multipliers descend in the tested proper jets.
Explicit indicator, linear, and quadratic multipliers send rank-8 kernel
witnesses outside the kernel.

Constant multiplication is scalar scaling, not general relation
intersection. The restricted program therefore does not repair the original
intersection obstruction.

## Independent execution and restoration

Production executes 24 restricted programs: ranks 2, 4, and 8, two public
module orders, and depths 1, 4, 16, and 64. Public constant Hadamard
multipliers are interleaved with invertible convolution kernels, nonlinear
convolution-square shears, and noncommuting linear shears. Only the final
scalar B boundary is projected. Actual reverse operations restore the exact
borrowed jet cells on the same Python backing, and an unrelated alternate
program consumes the carrier restored from a primary program. The direct
backing-identity observation remains package-local.

The independent oracle imports neither production nor NumPy. It executes the
same programs in the complete 34-cell cyclic group algebra, projects only the
final states to the declared jets, and matches 120 fields across all 24 cases.
It restores all 34 semantic-reference cells exactly. Fifty-four additional
checks independently verify convolution, cyclic rotation, and constant
Hadamard descent.

## Controls and resource law

Missing, wrong, and reordered inverses fail restoration for the tested
noncommuting program. Premature projection, null carrier, and wrong-rank
transactions are rejected. A disabled path changes the boundary, and three
nonconstant multiplier mutations violate quotient descent.

At rank 8 the accepted restricted carrier has 16 F17 cells and a conservative
logical working peak of 56 cells with no retained inverse history. The
strongest matched classical implementation is separately executed and is the
identical 16-cell, 56-working-cell truncated-polynomial recurrence. The full
nonzero common quotient for arbitrary convolution and Hadamard operations
requires 17 cells per register, or 34 for the two-register carrier. Dense
17-by-17 tables are absent. Python allocator and whole-process memory were not
measured.

## Claim ceiling

The verified result covers linear F17 quotients of the F17 C17 coefficient
space with arbitrary bilinear convolution and Hadamard operands, plus the
rank-2/4/8 compatible diagonal-multiplier search and the declared bounded
restricted programs. It establishes no CATVM custody, distinct phase
resource, computational advantage, Small Wall crossing, physical waveform or
silicon execution, replacement of physical bits with pi, or unbounded
catalytic computation.
