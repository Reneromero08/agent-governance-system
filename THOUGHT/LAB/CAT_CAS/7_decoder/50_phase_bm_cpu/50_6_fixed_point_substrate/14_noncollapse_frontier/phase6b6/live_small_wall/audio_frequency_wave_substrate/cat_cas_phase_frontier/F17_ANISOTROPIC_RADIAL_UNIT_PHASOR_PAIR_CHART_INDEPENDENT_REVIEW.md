# M143 independent review

Decision: `PASS` at strict bounded numerical scope.

M143 changes the M140/M142 primitive resident state.  Each normalized radial
coefficient is stored only as two `float64` phase angles whose unit phasors
average to the coefficient.  Quartic phase modules translate both angles
directly; the noncommuting radial Fourier uses the existing matrix-free
16-character contraction and then returns to the canonical two-phasor chart.
The carrier retains 34 phase angles, no exact cyclotomic coefficients, no
dense Fourier kernel, no gate tape, and no inverse history.

The 21 declared cases cover `PRIMARY`, `REUSE`, and `ALTERNATE` at depths
`1,4,16,64,256,1024,4096`.  Worst production error against the identical
17-complex recurrence is `2.242e-11` at the final boundary and `1.658e-12`
in the final radial state, below the predeclared `5e-11` boundary tolerance.
Worst single-transaction physical-state restoration is `5.764e-12`, below
the predeclared `2e-11` phasor tolerance.  The unrelated depth-1537 reuse
boundary differs from fresh execution by `1.646e-12`; 100 consecutive
same-backing depth-64 cycles reach at most `8.189e-12` restoration error.

The accepted inverse applies the actual Fourier involution and inverse phase
translations to the resident angle array.  It does not reset, reload, or
replace that array after the inverse.  Restoration is therefore classified
`NUMERICAL_PHYSICAL_STATE_RESTORATION`, not exact algebraic restoration and
not inverse-plus-post-hoc canonical reset.  The public seed formula is used
only to verify tolerance and is never reloaded into the carrier.  Same-backing
identity and package-local generations are preserved; CATVM custody is not
claimed.

The independent oracle imports no production or predecessor implementation.
It reconstructs all 289 radial Fourier entries in complex long-double
arithmetic from 4,913 public coordinate visits, checks involution to
`3.152e-19`, independently implements the pair chart and public gate formula,
and reexecutes all 21 cases.  It performs 147 declared comparisons, repeats
the unrelated and 100-cycle reuse paths, and detects a final-boundary
mutation.  Its dense matrix is verification-only and is not attributed to the
accepted matrix-free carrier.

Named accepted-path accounting includes 272 resident angle bytes, 672 bytes
of public root/index/shell geometry, a 1,056-byte maximum named update
scratch, and at most 184 public program JSON bytes, totaling 2,184 named
bytes.  This is not a Python or process peak.  Container, allocator, NumPy,
native-library, hashing, and whole-process storage remain excluded.

The strongest compact classical baseline stores the same state as 17
`complex128` values: also 272 resident bytes.  It executes the identical
matrix-free recurrence and avoids the chart's per-Fourier complex
exponentials, magnitudes, arguments, and arccosines.  The phase chart therefore
does not improve the matched software representation or work law.

The chart has a declared nonzero unit-disk domain and this result covers only
the sealed cases through depth 4096.  It establishes neither exact algebraic
semantics, unbounded-depth numerical stability, machine-enforced hidden-state
custody, a distinct phase resource, computational advantage, a Small Wall
crossing, physical waveform execution, replacement of physical bits with pi,
nor unbounded catalytic computation.
