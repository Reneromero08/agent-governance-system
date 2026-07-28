# Four-Rotor Kicked-Phase TT Results

Evidence:

```text
/tmp/four-rotor-kicked-phase-tt-final.MhLb0S/evidence
```

Bounded claim candidate:

```text
BOUNDED_FOUR_ROTOR_NONSEPARABLE_CONTINUOUS_KICKED_PHASE_FOURIER_TT_CENTRAL_INTERFACE_RANK_GROWTH_WITH_ACTUAL_RESTORATION_AND_REUSE
```

Four continuous rotors use local phase kick `K=sqrt(2)`, irrational free
phase `tau=sqrt(3)`, and nearest-neighbor nonseparable coupling `g=0.35`.
The coupling is a 13-term Bessel-factorized MPO. Before every coupling, the
TT is canonically gauged around the target bond, so the recorded central
values are physical `2|2` Schmidt ranks at discarded-L2 tolerance `1e-11`:

```text
depth                         1    2    3
central Schmidt rank         13  100  246
```

The separable `g=0` control remains rank one. Maximum local Fourier radius
is 12. Mode guards 14 and 16 agree in the declared boundary within
`8.895e-13`. Qualitative monotone growth is guard-robust, but the numerical
depth-three rank is guard-dependent (`242,246,247`). These are
tolerance-truncated ranks recorded immediately at central closure, not exact
algebraic final-round invariants.

The actual inverse restores within `5.236e-8` under the declared `1e-7`
tolerance. An unrelated two-round program consumes that actual restored
carrier and restores within `1.251e-8`; generations advance `1,2`. Retained
inverse history is zero.

Restoration equality is physical-state equality at the declared tolerance,
not canonical TT-rank equality. The restored carrier retains small
high-rank numerical residue: its reuse ranks differ from a snapshot-fresh
reuse. Stable compact reuse is therefore not established.

This does not provide compression. The unmaterialized `29^4` global wave
would contain 707,281 cells (11.316 MB). Canonical interface factorization
materializes an equal-sized core, factorized inverse closure reaches live
bond 3,198, and peak resident TT/MPO payload is 5,380,718 complex cells
(86.091 MB). The phase TT and best matched classical TT are identical.

This establishes bounded nonseparable phase-interface rank growth, not
fixed-rank closure, a distinct resource, advantage, Small Wall crossing,
unbounded computation, CATVM enforcement for this carrier, or physical
waveform execution.

The selected repair is matrix-free streamed coupling closure. Public Bessel
terms must become on-demand Schmidt matvecs with exact Frobenius-tail
accounting, eliminating both the expanded live MPO bond and dense interface
core without hiding them in history, snapshots, or a classical side channel.
