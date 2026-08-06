# Four-Rotor Post-Inverse Canonical Closure Results

Evidence:

```text
/tmp/four-rotor-canonical-final.dQzIFv
```

Bounded claim candidate:

```text
BOUNDED_ACTUAL_POST_INVERSE_TT_CANONICAL_QUOTIENT_CLOSURE_WITH_FRESH_RESTORED_MATRIX_FREE_REUSE_RANK_AND_RESOURCE_PARITY
```

The actual strict inverse-restored carrier, not a snapshot or saved baseline,
undergoes one standard TT-rounding sweep. A left QR sweep is followed by a
right-to-left SVD sweep with the declared `1e-7` total L2 budget divided
across the three cuts. The closure function has no baseline-state argument
and preserves the borrowed carrier object.

The strict depth-three carrier changes as follows:

```text
bond ranks                    29,166,29 -> 1,1,1
carrier complex cells             280894 -> 116
inverse error before closure           5.235e-8
physical closure delta                  5.108e-8
restoration error after closure         1.147e-8
```

The closure's conservative simultaneous payload is 11.636 MB and its largest
workspace array is 139,606 complex cells. The closure is a tolerance-defined
numerical quotient after the actual inverse, not an exact inverse operation
and not exact tensor-entry restoration.

An unrelated two-round matrix-free program consumes the same generation-one
carrier. Against a separately created fresh diagnostic carrier it has:

```text
central rank history                         9,27 == 9,27
maximum probe rank                            256 == 256
Frobenius probe columns                      1276 == 1276
maximum live complex cells                 767506 == 767506
maximum workspace complex cells            739292 == 739292
maximum retained rank                           27 == 27
final bond ranks                              1,1,1 == 1,1,1
boundary disagreement                              6.681e-12
```

The actual carrier advances to restoration generation two, retains 116
complex cells, and does not accumulate canonical rank across the second
transaction. Retained inverse history is zero.

The repaired matrix-free resource accounting uses the selected live
workspace rather than recursively adding a historical maximum. Its prior
claim still passes, and the conservative primary peak is corrected from
46.522 MB to 43.532 MB after counting compact owned factors and unique
retained NumPy backing allocations.

This establishes bounded canonical compact reuse of the actual numerical
carrier. It does not establish fixed-rank forward closure, exact restoration,
a distinct phase resource, advantage, a Small Wall crossing, unbounded
computation, CATVM enforcement for this carrier, or physical waveform
execution. The matched classical TT rounding and matrix-free execution are
identical.

The remaining obstruction is primary inverse cancellation: certified
matrix-free closure still reaches probe rank 492 and 43.532 MB, above the
11.316 MB dense-equivalent wave. The next phase-machine repair must remove
that inverse probe expansion rather than add another rotor fixture.
