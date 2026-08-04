# M182 independent reexecution review

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `NO_RESTORATION_CLAIM`

## Verified result

M182 tests whether standard nonlinear multiplicative phase relations repair
M181's state/work obstruction. For every divisor `n > 1` of `h=q-1` and every
character index `j`, production constructs the exponent relation induced by

```text
G(nj) = -chi_(nj)(n) product_k G(j+k*h/n) / product_k G(k*h/n).
```

The sign and character value are elementary public topology constants. Gauss
and Jacobi values are not admitted as free constants. Exact rational row
reduction gives free Hasse--Davenport monomial-generator rank `phi(h)` at all
14 M181 fields. Adding every nontrivial Gauss norm relation
`G(j)G(-j)=chi_j(-1)q` lowers the free rank to `phi(h)/2`, but does not make it
fixed.

The safe-prime cases in the declared family are q=5, 7, 11, 23, and 47. Their
Hasse--Davenport free ranks are 2, 2, 4, 10, and 22. Their norm-augmented ranks
are 1, 1, 2, 5, and 11. Thus the tested standard monomial relation algebra has
growing free rank on this bounded safe-prime sequence. This is not promoted to
a no-go theorem for other nonlinear or nonmonomial algorithms.

## Boundary relevance

The diagnostic does not stop at the rank of the whole Gauss family. For the
actual declared rank-three, nonzero-scale M181 program in each field, it
constructs the Gauss exponent vector of every boundary channel, including the
source coefficient, determinant gamma, fixed quadratic factor, and scale
factor. Those channel products span the complete remaining quotient in every
case: `phi(h)` directions modulo Hasse--Davenport alone and `phi(h)/2`
directions after norm augmentation.

Consequently, the particular boundary family does not live in a smaller
formal monomial subspace hidden inside the tested relation quotient. Removing
the quadratic Hasse--Davenport relations increases free rank in every case.
The false overmerge `G(1)=1` fails numerically in every auxiliary field.

## Jacobi scope

For nontrivial `j`, `k`, and `j+k`, M182 independently verifies

```text
G(j)G(k) = J(j,k)G(j+k).
```

All applicable pairs are checked, from 6 pairs at q=5 through 2,550 at q=53.
A formal Jacobi symbol comes with one defining relation and therefore adds no
constraint to the projected Gauss lattice. Treating its value as a retained
coefficient would instead retain `(q-2)(q-3)` answer-bearing field cells if
all applicable pairs were materialized. Streaming one Jacobi value uses `q`
character terms; doing so once per boundary channel remains theta-q-squared
work. M182 does not reject a different compact Jacobi algorithm.

## Independent reconstruction

The oracle imports neither production nor M180/M181. It independently finds
primitive roots, reconstructs additive and multiplicative phase embeddings,
scans character orbits without production's public log table, and directly
recomputes every declared Gauss and Jacobi identity. Its Hasse--Davenport and
Jacobi value commitments match production.

For relation rank, production uses normalized sparse rational elimination.
The oracle builds sign-reversed rows in reverse character order and uses dense
forward elimination over exact fractions. It independently matches every
rank, omitted-quadratic control, and boundary quotient span.

The production rank matrices, retained Gauss arrays, and public log arrays are
verification diagnostics, not an accepted compact carrier. There is no
catalytic transaction in this package, so no restoration or custody claim is
made.

## Matched baseline and ceiling

The strongest compact classical comparison is the identical
Hasse--Davenport/norm/Jacobi relation algebra with the same ranks and streamed
character-sum costs. The result establishes no state or work advantage.

The claim is limited to the formal integer exponent lattice of the declared
relations, the 14 M181 field/program pairs, and exact direct-process residue
diagnostics. It establishes no universal nonlinear rank lower bound,
subquadratic-work impossibility, compact Jacobi generator, CATVM custody,
distinct phase resource, computational advantage, Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, or
unbounded computation.

## Next obstruction

The standard multiplicative phase relations leave a growing free family that
the actual boundary products fully exercise. The next repair must therefore
change the update law. The selected successor tests an exact reversible
additive Fourier compiler: mixed-radix where possible and a Bluestein/Rader
route for large prime factors. It must measure the whole transform workspace,
roots, convolution buffers, inverse, projection, and reuse against the
identical classical NTT. Success would trade M181's quadratic work for linear
state and subquadratic work, not establish a distinct phase resource.
