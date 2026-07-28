# Exact Cyclotomic Cubic Phase Tensor-Train Results

Evidence:

```text
/tmp/cyclotomic-f5-cubic-tt-final.7jbmXC/evidence
```

Accepted bounded claim:

```text
BOUNDED_EXACT_CYCLOTOMIC_CUBIC_PHASE_FOURIER_TENSOR_TRAIN_SEQUENTIAL_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

Across widths `2,4,6` and one, two, and three central crossings, certified
central bond ranks are:

```text
4,14,64
```

The independent `F11/F31` implementation reproduces every full bond vector
and final boundary residue. The accepted path materializes no global
`5^width` wave, statevector, truth table, or assignment expansion.

Peak resident TT cells grow `40,750,9710`; resident logical coefficient
payload grows `320,6000,129678` bytes. At width six, exact factorization
scratch reaches 1,642,143 logical coefficient bytes, coefficient height
reaches 128 numerator and 129 denominator bits, and factorization performs
1,589,231 additions, 1,589,231 multiplications, and 16,640 divisions. These
costs are counted rather than hidden behind rank.

All fixtures apply the actual inverse and restore exactly. An unrelated
second program consumes the actual restored width-four carrier. Missing,
wrong, and noncommuting reordered inverses fail; rank truncation fails
closed; Fourier-disabled execution changes the boundary. Identity and
separable gate ranks are one, cubic gate rank is four, and the bilinear
Clifford sham has rank five but remains classically stabilizer-compact.

The executed snapshot sham matches the forward boundary, transfers a
160-byte baseline at creation and reload, mints no inverse restoration
generation, and then executes reuse.

This advances phase-native composition beyond quadratic/Gaussian closure:
unresolved amplitude and nonseparable cubic phase coupling generate
certified interface rank. It does not establish a distinct computational
resource because the matched exact classical TT is literally the same
representation with identical rank, state, coefficient, and scratch growth.
