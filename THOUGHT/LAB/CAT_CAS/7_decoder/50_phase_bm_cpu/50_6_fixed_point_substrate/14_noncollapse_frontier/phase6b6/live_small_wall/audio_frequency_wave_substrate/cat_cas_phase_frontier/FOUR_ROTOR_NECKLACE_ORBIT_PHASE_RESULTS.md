# Exchange-Symmetric Necklace-Orbit Phase Carrier

## Result

Accepted bounded claim:

```text
BOUNDED_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_NECKLACE_PHASE_CARRIER_CHANGES_FIXED_GRID_ROTOR_GROWTH_FROM_EXPONENTIAL_TO_POLYNOMIAL_WITH_STREAMED_EXACT_CYCLOTOMIC_FREE_CLOSURE_ACTUAL_RESTORATION_AND_REUSE
```

Claim ceiling:

```text
EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_TESTED_NONZERO_CHIRP_SCHEDULE_COMPLEX128_EXACT_CYCLOTOMIC_TRANSITION_COUNTS_SOFTWARE_ONLY
```

Evidence:

```text
/tmp/four-rotor-necklace-orbit-final.RoJ1jK
```

## Phase-owned repair

The previous 4,913-cell carrier quotients global rotation but retains three
labelled relative-angle coordinates. This successor declares a narrower
exchange-symmetric, rotation-invariant non-affine program family and stores
one unresolved cyclotomic amplitude per cyclic necklace of occupation
histograms.

For grid 17 and rotor count `R < 17`, every nonconstant occupation histogram
has a free 17-element rotation orbit, so

```text
D_R = binomial(R + 16, 16) / 17.
```

At `R=4`, 4,845 histograms close to 285 necklace amplitudes rather than 4,913
labelled rotation-quotient amplitudes, a further `17.239x` carrier reduction.
The analytic `R=5` dimension is 1,197 rather than 83,521. At fixed grid size,
the necklace count has `O(R^16)` growth, including stabilizer-aware Burnside
corrections beyond the simple `R < 17` formula.

The native program alternates:

```text
nonseparable collision phase:
    zeta17 ^ (kappa * sum_j binomial(n_j, 2))

circulant quadratic free phase:
    C[y,x] = zeta17 ^ (q * (y-x)^2) / sqrt(17)
```

Both commute with particle permutation and global rotation. The engine
streams each induced free coefficient as an exact 17-component cyclotomic
integer count from a four-rotor permanent. It retains neither a `285^2`
operator nor a labelled wave nor a stored assignment list.

## Exactness and lifecycle

An independent one-step 83,521-cell labelled-wave verifier agrees with the
native carrier within `1.346e-15` and is never used for restoration. The
depth-eight weighted norm error is `2.220e-16`. Correct reverse dependency
order and adjoint chirps restore the actual borrowed carrier within
`7.457e-15`.

An unrelated depth-two program consumes that actual restored carrier,
reaches restoration generation two, restores within `8.247e-15`, and agrees
with fresh execution at the boundary within `8.882e-16`. Missing, wrong, and
applicable reordered inverse controls separate by `1.404`, `1.346`, and
`1.075`.

ASan/UBSan replay passes at
`/tmp/four-rotor-necklace-repaired-san.c9MU6k`.

## Resource accounting and obstruction

The accepted depth-eight path reports:

```text
carrier payload                                      4,560 bytes
retained public topology                            10,532 bytes
conservative plan-compilation explicit payload     10,619 bytes
output scratch                                       4,560 bytes
transition scratch                                     177 bytes
maximum explicit engine payload                     19,829 bytes
maximum explicit wrapper payload                    24,501 bytes
retained inverse history                                 0 bytes
retained transition operator                             0 bytes
stored assignment list                                   0 bytes
```

The bounded verifier peak is 2,701,733 explicit bytes. The memory repair does
not remove work: the primary streams 1,299,600 transition coefficients and
enumerates 530,236,800 permanent assignment terms in about 4.8 seconds on the
recorded warm run.

The best matched classical orbit simulator is identical. This establishes an
exact symmetry-owned polynomial carrier law for a genuinely wave-amplitude
state, but not a distinct phase resource, computational advantage, Small Wall
crossing, arbitrary rotor interactions, the original open-chain program
family, or unbounded computation.

The new obstruction is:

```text
STREAMED_NECKLACE_FREE_CLOSURE_QUADRATIC_TRANSITION_WORK_AND_MATCHED_CLASSICAL_ORBIT_IDENTITY
```

The next phase mechanism must reduce exact free-closure work without retaining
the dense orbit operator, or introduce a phase-native resource not inherited
by the identical classical orbit recurrence.
