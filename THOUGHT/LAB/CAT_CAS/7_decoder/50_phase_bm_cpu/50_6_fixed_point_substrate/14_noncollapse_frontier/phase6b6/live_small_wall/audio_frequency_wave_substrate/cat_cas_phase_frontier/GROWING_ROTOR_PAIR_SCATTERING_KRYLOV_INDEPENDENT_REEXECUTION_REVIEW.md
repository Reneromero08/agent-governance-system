# M191 Independent Reexecution Review

## Decision

`INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration class: `NUMERICAL_PHYSICAL_STATE_RESTORATION`

## Reconstructed bounded mechanism

The production implementation extends the M190 public diagonal two-body
phase and off-diagonal momentum-conserving pair-scattering law from four
rotors to rotor counts two through five. It streams exchange-symmetric
occupation histograms, retains one representative per global-rotation orbit,
and obtains necklace dimensions `9`, `57`, `285`, and `1197`. The simple
division by seventeen is valid only because every tested rotor count is
strictly between zero and seventeen, so no nonidentity rotation can stabilize
an occupation histogram.

The independent Python oracle imports no production module. It separately
generates `153`, `969`, `4845`, and `20349` occupation histograms, reconstructs
the same necklace quotients, checks their labelled-weight partitions, and
builds sparse integer generators. Exact weighted Hermiticity residual is zero
in every case. SciPy sparse `expm_multiply`, rather than the production
degree-64 Chebyshev recurrence, reproduces every primary boundary within
`5.8e-16`.

## Exact continuation diagnostic

Two independently implemented Berlekamp--Massey routines agree on the scalar
degrees of the declared public `K*D` continuation word:

| rotors | necklace dimension | F103 degree | F137 degree |
| ---: | ---: | ---: | ---: |
| 2 | 9 | 9 | 9 |
| 3 | 57 | 57 | 57 |
| 4 | 285 | 284 | 284 |
| 5 | 1197 | 1182 | 1190 |

The exact modular recurrence omits the numerical generator's nonzero scalar
factor `0.005`. This multiplies the scalar sequence term at power `n` by a
nonzero geometric factor in each tested field and therefore does not change
its Berlekamp--Massey degree.

Full degree for the declared scalar source/probe at two and three rotors does
not prove a lower bound for nonlinear, singular, approximate, or
program-restricted representations. The one-cell deficit at four rotors and
prime-dependent deficits at five rotors do not define a stable transferable
quotient. In particular, this result neither proves a smaller exact complex
carrier nor proves that no other quotient exists.

## Restoration, controls, and resources

Primary restoration errors range from `9.22e-16` to `1.91e-15`; unrelated
reuse restoration remains below `1.87e-15`; fresh-versus-restored reuse
boundaries differ by at most `1.34e-15`. The five-rotor missing, wrong, and
reordered inverse controls leave errors `1.388`, `1.356`, and `0.924`.
Sixteen repeated reuse cycles accumulate at most `8.50e-15` production error.
The original carrier backing is reused and no baseline reload occurs.

For rotor count `R`, the accepted production path retains
`binomial(R+16,R)/17` complex carrier cells, three times that number of
temporary Chebyshev cells, the public necklace topology, and at most one
sparse public scattering plan. Inverse plans are recompiled from public
topology rather than retained from forward history. The largest tested plan
has `68,997` nonzero entries. Topology compilation streams, but does not retain,
`binomial(R+16,R)` occupation histograms. Allocator overhead, native-library
work, and whole-process peaks are excluded, so no timing or memory advantage is
claimed. The oracle's sparse matrices and exact sequences are verification
resources only.

## Claim ceiling and obstruction

The strongest compact classical implementation remains the identical growing
necklace pair-scattering plus diagonal-phase recurrence. M191 therefore shows
that the tested carrier grows from 285 to 1197 complex cells at five rotors
and that the declared scalar continuation does not expose a stable smaller
quotient. It does not establish a distinct phase resource, computational
advantage, Small Wall crossing, CATVM custody, physical waveform execution,
replacement of physical bits with pi, or unbounded computation.

Exact ceiling:

`GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTOR_COUNTS2_3_4_5_PRIMARY_DEPTH2_REUSE_DEPTH1_PUBLIC_K_TIMES_D_WORD_F103_F137_BERLEKAMP_MASSEY_COMPLEX128_CHEBYSHEV64_DIRECT_PROCESS_SOFTWARE_ONLY`
