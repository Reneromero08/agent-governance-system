# Continuous Kicked-Phase Fourier Results

Evidence:

```text
/tmp/continuous-kicked-phase-repaired.I8PlYF/evidence
```

Bounded claim candidate:

```text
BOUNDED_NUMERICAL_CONTINUOUS_IRRATIONAL_KICKED_PHASE_COHERENT_FOURIER_LOCALIZATION_CONTRAST_WITH_EFFECTIVE_BANDWIDTH_PLATEAU_ACTUAL_RESTORATION_AND_REUSE
```

## Phase-coherent contrast

The continuous sampled wave applies

```text
exp(-i sqrt(2) cos(theta))
-> Fourier
-> exp(-i sqrt(3) n^2 / 2)
-> inverse Fourier
```

without decoding intermediate amplitudes. At tail-energy tolerance `1e-12`,
the fixed periodic Floquet law has Fourier radii `26,26,26,26,26,24` at
depths `64,128,256,512,1024,2048`. The same-strength deterministic
17-step phase schedule reaches radii
`50,77,131,239,452,882` over the same depths.

Periodic grids `256,512,1024,2048` agree in final boundary within
`3.827e-13`. Scrambled grids `4096,8192` agree within `3.333e-13` and have
identical final epsilon-sweep radii. A 63-bit-mantissa replay agrees with the
float64 periodic boundary within `2.972e-13`; radii at tail tolerances
`1e-10,1e-12,1e-14` are identically `23,24,27`.

The primary depth-2048 carrier restores within `1.412e-14`. An unrelated
31-step program consumes the actual restored carrier, and eight more reuse
cycles stay within `6.315e-16`. Inverse topology is rematerialized from
public step count; retained inverse history is zero. Missing, wrong,
reordered, and Fourier-disabled controls separate.

## Matched compact classical method

The independent momentum-space Bessel recurrence selects the first public
guard radius meeting the declared `2e-12` whole-state error. It needs modes
`-48..48` and Bessel kernel `-16..16`, or 97 resident complex coefficients,
and matches the 2,048-grid FFT state within `7.298e-13`. Its cumulative
kernel-tail bound is `2.190e-14`.

The phase FFT path uses 32,768 resident payload bytes, 32,768 bytes of public
grid topology, a 32,768-byte restoration-verification copy, and at least
65,536 peak live step-temporary payload bytes. Its four allocated NumPy
arrays per step are counted as payload allocation volume rather than
misreported as cached compiled masks. The Bessel baseline uses 1,552 resident
bytes, 2,080 compiled bytes, and 1,552 scratch bytes. At depth 256 its warm
median was `0.763 ms`, versus `33.153 ms` for the dense FFT path.

The observed plateau is therefore a genuine bounded phase-coherence
phenomenon, but it is immediately available to a smaller classical complex
recurrence. It does not establish exact finite Fourier support, asymptotic
dynamical localization, asymptotic delocalization of the 17-step control, a
distinct phase resource, advantage, Small Wall crossing, unbounded
computation, physical execution, or CATVM enforcement for this carrier.

## Successor

The fixed one-rotor vector is now the obstruction. The next phase-owned
experiment is the smallest nonseparable relational lift: four
nearest-neighbor kicked rotors represented as a Fourier tensor train, with a
`2|2` cut. It must keep local Fourier radius compact while testing whether
unresolved central Schmidt rank grows, restore through the actual inverse,
reuse the same carrier, and compare against the identical best classical
TT/MPS representation without materializing the dense four-dimensional
wave.
