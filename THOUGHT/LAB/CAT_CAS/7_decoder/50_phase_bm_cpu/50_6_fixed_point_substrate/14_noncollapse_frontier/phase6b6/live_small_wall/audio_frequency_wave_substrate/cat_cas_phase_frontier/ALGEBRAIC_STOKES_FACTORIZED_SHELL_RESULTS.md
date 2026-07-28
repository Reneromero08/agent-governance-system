# Fixed-rank factorized highest-shell results

Accepted claim:

```text
FIXED_RANK_FACTORIZED_HIGHEST_STOKES_HARMONIC_SHELL_PHASE_RECURRENCE_WITH_RESTORATION_AND_REUSE
```

Ceiling:

```text
EXACT_REPEATED_SINGLE_AXIS_QUADRATIC_STOKES_KERR_HIGHEST_HOMOGENEOUS_SHELL_FACTOR_L_POWER_N_TIMES_Q4_DUAL_PRIME_SOFTWARE_DEPTHS1_TO_2048
```

Results:

```text
maximum executed depth                         2048
highest harmonic degree                       2050
expanded highest-shell dimension              4101
resident phase coordinates                       4
complete dual-prime character phase cells      144
logical fixed phase payload                  2,304 bytes
retained inverse history                         0 bytes
maximum root error                               0
maximum restoration residual                     0
same-carrier transactions                       16
matched classical dual-prime state               8 bytes
```

Missing and wrong inverses leave 136 and 52 nonidentity character cells,
respectively. Snapshot returns no restoration receipt. Reordered inverse is
inapplicable because every repeated fixed-axis transition is identical.
The 96-byte public descriptor is logical packed storage; actual Python
allocation is unmeasured. Reuse uses a second public factorized Q4 seed.

The all-depth factorization removes highest-shell rank growth without moving
an expanded coefficient vector elsewhere. The result also exposes the next
obstruction cleanly: lower harmonic shells are not covered, and the same Q4
recurrence has an immediately equivalent, smaller classical residue machine.
