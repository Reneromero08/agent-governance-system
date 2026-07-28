# Four-Rotor Matrix-Free Schmidt Closure Results

Evidence:

```text
/tmp/four-rotor-matrix-free-accounted.sTo4ay/evidence
```

Bounded claim candidate:

```text
BOUNDED_MATRIX_FREE_STREAMED_BESSEL_SCHMIDT_CLOSURE_WITHOUT_EXPANDED_MPO_OR_DENSE_INTERFACE_CORE_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The repair rematerializes each public Bessel coupling term for deterministic
`M X` and `M* X` products. A deterministic Fourier probe range is enlarged
until streamed full-column Frobenius accounting certifies the residual.
Only the certified projected matrix is factorized. No expanded MPO bond or
`707,281`-cell dense interface core is constructed.

At matrix-free discarded-L2 tolerance `1e-6`, central ranks are
`11,41,97`; the same-tolerance dense-core reference gives `11,40,96`.
Final boundary disagreement is `9.459e-10`, below the declared `3e-6`.
The actual inverse restores within `1.922e-8`; unrelated actual-carrier reuse
restores within `4.087e-9`, and generations advance `1,2`. Missing, wrong,
reordered, and snapshot controls separate.

Conservative simultaneous-array accounting reduces the bounded peak from
86.091 MB to at most 43.532 MB. It includes QR/SVD factors, projected and
output arrays, nested contractions, shift temporaries, and the old carrier.
The largest single workspace array is 299,628 complex cells, below the
eliminated 707,281-cell core. It does not beat the 11.316 MB dense-equivalent
total: inverse cancellation still requires certified probe rank 492.

The initial 46.522 MB result recursively added a historical maximum when
accounting the final retained tensors. Replacing that term with the selected
currently live workspace gives the stricter 43.532 MB upper bound after
including compact factor copies, squared singular values, and retained
NumPy backing allocations; the
qualification and all qualitative conclusions are unchanged.

This is a real phase-machine resource repair, but not fixed-rank closure,
compactness versus the dense equivalent, a distinct resource, advantage,
Small Wall crossing, unbounded computation, CATVM enforcement for this
carrier, or physical waveform execution. The matched classical matrix-free
TT is identical.

The next repair must canonicalize and prune the actual inverse-restored
carrier, then require fresh-versus-restored reuse rank/resource parity. That
tests whether numerical rank residue is the source of the large inverse
probe space rather than allowing it to move into subsequent transactions.
