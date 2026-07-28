# Four-Rotor Incremental Schmidt Closure Results

Evidence:

```text
/tmp/four-rotor-incremental-schmidt-certified.DHPL2i
```

Bounded claim candidate:

```text
BOUNDED_PROBE_FREE_INCREMENTAL_BESSEL_SCHMIDT_PHASE_CLOSURE_BELOW_DENSE_EQUIVALENT_MEMORY_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The 13 public Bessel factors are composed into the resident two-site phase
relation one at a time. Each low-rank addition updates a compact Schmidt
factorization through explicit combined-basis QR and a small-core SVD. The
`1e-6` per-coupling L2 budget is divided by 13 in worst-case norm, so no
probe rank or adaptive range escalation is used.

Array lifetimes are phased: absorbed left factors are released before right
factor construction, and old bases are released before the corresponding
updated basis is formed. QR uses owned Fortran buffers and queried in-place
ZGEQRF/ZUNGQR workspaces. ZGESVD input, outputs, complex workspace, and real
workspace are counted simultaneously. Explicit core products and update
outputs are also counted. A ZHERK certificate gates every combined basis at
`1e-10`; the observed maximum orthogonality residual is `1.984e-14`.
The finite Bessel tail plus an analytic bound beyond mode 127 is subtracted
before allocating the 13 update budgets and gated per coupling.

At radius 14 and depth three:

```text
central rank history                         11,47,117
maximum incremental rank                            122
probe columns                                          0
accepted-path peak complex-cell equivalents       677010
accepted-path peak payload                        10.832 MB
dense-equivalent wave cells                        707281
dense-equivalent wave payload                     11.316 MB
boundary disagreement                              7.380e-8
inverse restoration error                          2.687e-7
postclosure restoration error                      7.495e-8
maximum per-coupling declared L2 bound              8.968e-7
transaction combined declared L2 bound              8.982e-6
```

No expanded MPO bond or dense interface core is materialized. Missing,
wrong, and reordered inverse controls separate.

An unrelated two-round program consumes the same restored carrier and
advances its restoration generation to two. Its rank history is `11,32`,
inverse restoration error is `1.499e-7`, and accepted peak is 1.688 MB.
A separately created fresh carrier has the same rank history and exact
deterministic resource signature; boundary disagreement is `6.003e-8`.
Both carriers close to ranks `1,1,1` and 116 retained complex cells.

This crosses the nominal dense-wave memory threshold for the declared
bounded problem. It is not a Small Wall crossing or computational advantage:
the best matched classical incremental TT uses the identical algorithm and
resource law. It also does not establish a distinct phase resource,
fixed-rank forward closure, unbounded computation, CATVM enforcement for
this carrier, or physical waveform execution.

The next experiment must determine whether this repaired phase law survives
the enforced machine boundary and matched sham/baseline accounting; it must
not treat the intentionally dense representation as the best classical
method.
