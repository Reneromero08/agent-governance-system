# One-segment streamed-kernel Bluestein compiler independent reexecution review

## Decision

```text
INDEPENDENTLY_VERIFIED_STRICT_SCOPE
INDEPENDENT_ORACLE_REEXECUTION
EXACT_ALGEBRAIC_RESTORATION
```

The bounded 14-field result is accepted at its declared direct-process exact
residue-software ceiling. The source commit containing this report is the
scientific source head; the subsequent provenance reconciliation records its
exact Git SHA.

## Independent reconstruction

The oracle does not import the successor production module. It imports only
the frozen M184 independent oracle's field, boundary, direct-Gauss, and direct
relation primitives, then separately reconstructs:

- recursive out-of-place source NTTs rather than production's iterative
  in-place transform;
- every public kernel-spectrum coefficient by a direct DFT sum over the
  zero-padded chirp kernel rather than production's support-streaming loop;
- one-segment multiply, boundary projection, exact reverse, and unrelated
  restored-carrier reuse;
- direct Gauss values for all 14 fields;
- the original seven-dimensional relation sum at q=5 and q=7.

All cases agree on convolution width, logical carrier capacity, streamed
kernel-spectrum commitment, primary boundary, unrelated reuse boundary, NTT
calls, butterfly count, streamed spectrum count, and kernel-term count.

## Observed bounded law

For `h=q-1` and the least power of two `M >= 2h-1`, production uses one final
scalar plus one M-cell exact transform segment. The declared carrier therefore
falls from M184's 17--257 cells to 9--129 cells. No Gauss descriptor or public
kernel spectrum is retained, and only the scalar remains after forward
cleanup.

Each forward or inverse compiler performs four NTTs, `2M log2(M)`
butterflies, and streams `2M` kernel-spectrum values. Each value sums the
`2h-1 = 2q-3` nonzero chirp-kernel terms, for exactly
`2M(2q-3)` counted kernel terms per compiler. The accepted work law is
theta-Mq plus M-log-M plus q. Thus removing M184's second segment forfeits
subquadratic transform work. The remaining M-cell carrier still grows
linearly, while M181 retains a fixed ten-cell theta-q-squared stream. At q=5
the one-segment carrier has nine cells; from q=7 onward it has 17--129 and is
larger than M181's declared ten-cell workspace. This is not a new asymptotic
state/work Pareto point.

The scalar survives outside inverse history. The exact same list backing
restores to zero and runs an unrelated program with fresh boundary and
resource-signature parity. No snapshot, generation lease, or machine-enforced
custody is used.

## Controls and accounting

Every declared case passes missing inverse, wrong-program inverse,
frequency-zero omission, forced singular spectrum, and null-carrier controls.
Frequency zero is the only omitted frequency claimed. Singular rejection is
not represented as an atomic rollback.

Package verification materializes a direct q-minus-one Gauss table before
carrier allocation. The separate oracle materializes the public kernel seed
and uses recursive out-of-place transform buffers. Neither is represented as
accepted production carrier state. Production retains no spectrum vector and
streams commitments without an O(M) joined digest buffer. Logical field-cell
accounting excludes Python containers, allocator/native-library memory,
modular-exponentiation bit complexity, and whole-process memory.

## Claim ceiling

The result establishes exact one-transform-segment execution for the 14
declared field/program cases. The strongest matched classical implementation
is the identical one-segment recurrence, with M181 retained as the fixed-state
quadratic-work comparison. It does not establish sublinear state,
subquadratic work, fixed exact bit width, CATVM custody, a distinct phase
resource, computational advantage, Small Wall crossing, physical waveform
execution, replacement of physical bits with pi, or unbounded computation.

## Pre-seal source hashes

```text
2886452fad1f1e8af960960c9d86027c3e32974858f44e80edc4a4d3dcef6730  growing_prime_one_segment_streamed_kernel_bluestein_boundary_compiler.py
ff96553814e11620c605589cd55c992c91453a8035098e63a0dfea3362d775c9  growing_prime_one_segment_streamed_kernel_bluestein_boundary_compiler_independent_oracle.py
26f2291efbf09eecafea8c860858a4f4972c5fa0651602392034f24073e710ed  GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_RESULTS.json
6faf7a6c640e6583e3857a6c400a82ed5bbbc73dcfc824bd47277bc824bf6456  GROWING_PRIME_ONE_SEGMENT_STREAMED_KERNEL_BLUESTEIN_BOUNDARY_COMPILER_INDEPENDENT_ORACLE.json
```
