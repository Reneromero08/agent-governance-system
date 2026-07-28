# Four-Rotor Cyclic Phase-Law Results

Evidence:

```text
/tmp/four-rotor-cyclic-phase-law-repaired.HgR1j4
```

Bounded claim candidate:

```text
BOUNDED_DENSE_FINITE_TORUS_CYCLIC_PHASE_UPDATE_LAW_WITH_DEPTH_INDEPENDENT_EXPLICIT_NUMPY_ARRAY_PAYLOAD_ACTUAL_RESTORATION_AND_REUSE
```

The phase machine now has a finite-torus update law that does not reconstruct
Schmidt Grams or inverse sector right-hand sides. A `17^4` resident complex
wave receives onsite and nearest-neighbor `U(1)` phase multipliers directly
in angle coordinates. Free evolution is an orthonormal Fourier transform,
four separable momentum-phase multiplications, and the inverse transform.
The inverse derives the public steps in reverse order and conjugates the same
phase laws. It retains no inverse history.

SciPy PocketFFT preserves the resident complex backing allocation through
every transform in this environment. The accounted NumPy array payload is
constant over depths `1,2,4,8,16,32,64`:

```text
resident carrier                              1,336,336 bytes
retained public plan                                544 bytes
maximum phase-factor scratch                         289 cells
maximum accounted engine arrays                2,009,672 bytes
wrapper peak including verification baseline   3,346,008 bytes
retained inverse history                              0 bytes
```

Restoration error grows from `6.293e-16` at depth one to `1.698e-14`
at depth 64 without changing that signature. The accepted depth-32 primary
restores at `7.671e-15`. An unrelated depth-11 program consumes the same
restored backing allocation, advances restoration generation to two, and
restores at `2.244e-15`. Fresh/restored reuse boundary disagreement is
`6.661e-15`. Missing, wrong, and noncommuting reordered inverse controls
separate.

The verification baseline is counted, does not alias the carrier, and is
never assigned or reloaded. No dense operator, decoded intermediate, Gram,
sector RHS, or inverse history is materialized. The explicit payload counter
includes owned phase-construction buffers, projection
probability/marginal/kernel arrays, plan compilation, and the verification
baseline. No bound is claimed for PocketFFT's internal native workspace.

This dense cyclic path avoids the measured sector-rematerialization work and
establishes a depth-independent explicit-array law for a fixed four-rotor,
fixed-grid machine. It does not remove that obstruction from the compact TT
carrier or establish compact growth in rotor count or grid width. The matched
direct classical cyclic FFT is identical and agrees exactly, so no distinct
phase resource, compact-TT advantage, computational advantage, Small Wall
crossing, unbounded computation, or physical waveform execution is
established.

The next obstruction is width: the resident carrier has `17^4` complex
cells. The next experiment must test whether the same cyclic phase law admits
a compact unresolved phase factorization across increasing depth without
reintroducing Bessel/Gram rematerialization or moving growth into an
equivalent classical recurrence.
