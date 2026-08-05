# Controlled local interference-feedback phase coupling independent review

## Decision

```text
INDEPENDENTLY_VERIFIED_STRICT_SCOPE
INDEPENDENT_ORACLE_REEXECUTION
NUMERICAL_PHYSICAL_STATE_RESTORATION (IN-PLACE VIRTUAL PHASE STATE)
SNAPSHOT_RELOAD (SHAM)
NO_RESTORATION_CLAIM (DIRECT BASELINE)
```

The bounded four-case virtual-software diagnostic is accepted at its declared
scope. “Physical state” is the registry restoration class for the borrowed
complex phase coordinates. No physical waveform, audio, oscillator, or
silicon execution occurred.

## Changed phase update law

M187 removes M186's free global unitary DFT. Each public layer has two
disjoint nearest-neighbour unitary-coupler sublayers on a periodic brickwork
topology followed by local intensity-dependent phase feedback. The feedback
is nonlinear in the resident complex coordinates and does not commute with
the couplers. The exact public inverse order is feedback inverse, odd-coupler
inverse, then even-coupler inverse.

The declared cases are width/depth 8/8, 16/16, 32/32, and 64/64. A primary
program and an unrelated same-width program execute on the same restored
backing. Only the final complex scalar is returned. The result remains local
direct-process software and does not enforce custody against a controller
that can call the projection routine.

## Independent reconstruction

The standalone oracle imports neither production nor NumPy. It compiles an
explicit public word of pair and feedback operations, evaluates that word on
Python complex lists, reverses the word operation by operation, and separately
reconstructs all boundaries and controls. Production and oracle final
boundaries differ by at most `2.24e-16`. Oracle forward norm error is at most
`5.56e-16`.

Missing inverse, wrong-feedback inverse, inverse reordering, dephasing, zero
feedback, swapped coupler order, and null carrier all discriminate in
production. The independent oracle reproduces every applicable mathematical
control except the production-only null-object API rejection.

## Restoration and reuse

The in-place path reverses the actual nonlinear and linear operations on the
same `numpy.ndarray` backing. Across all four cases:

- primary restoration error is at most `6.20e-16`;
- unrelated-reuse restoration error is at most `6.91e-16`;
- 128 alternating reuse cycles accumulate at most `2.54e-14` error;
- every unrelated reused boundary agrees with fresh execution within the
  predeclared `2e-11` boundary tolerance;
- no baseline reload is used by the in-place path.

The predeclared state tolerance is `2e-10`. The restoration classification is
`NUMERICAL_PHYSICAL_STATE_RESTORATION` only for the virtual complex128
coordinates. The sham creates one n-complex snapshot and reloads it after both
forward programs, so its classification remains `SNAPSHOT_RELOAD`.

## Complete bounded resource law

For width n and depth d, one forward program executes exactly `n*d` local
two-cell unitary couplers and `n*d` local feedback operations. The accepted
primary-plus-unrelated-reuse lifecycle, including both actual inverses,
executes `4*n*d` of each. Its declared live logical state is n resident
complex128 modes plus two pair temporaries, with n simultaneous mode-bandwidth
units and 128 bits per mode. Its abstract locally parallel depth is `12*d`
for both complete transactions; the n couplers and n feedback actions per
forward layer are not treated as free. Projection streams its public weights
and source construction fills the carrier directly, so neither accepted step
retains or materializes an n-cell public vector.

The direct/fresh path uses n+2 logical complex cells and two forward programs.
The snapshot sham uses 2n+2 cells and accounts for one snapshot creation plus
two reloads, or `3*16*n` bytes. Its separate fresh-reference verification peak
is 3n+2 complex cells. The in-place accepted transaction uses n+2 cells; its
restoration baseline, resident control copy, and one sequential mutated control
copy raise the separately reported verification/control peak to 4n+2 cells.
Three streamed commitment serializations account for `3*16*n` additional
bytes. Request and final-response bytes are identical across all paths.
Verification/control couplers, feedbacks, and projection terms are reported
separately from the accepted lifecycle.

Nine-repetition warm medians range from approximately 0.19 to 8.42 ms for
fresh direct execution, 0.27 to 12.54 ms for snapshot sham, 0.37 to 16.63 ms
for in-place phase execution, and 0.30 to 16.83 ms for the identical compact
classical full-state recurrence. These environment-specific timings are not
claim authority. Python containers, NumPy allocator details, arithmetic
expression temporaries, and whole-process peaks are outside the logical-cell
count.

## Claim ceiling

The mechanism changes the update law and establishes causal nonlinear phase
feedback, noncommuting local interference, final-only projection, numerical
same-backing inverse restoration, and unrelated reuse without a global DFT.
It does not establish a distinct resource: the strongest compact classical
implementation is the identical n-complex local-coupler and feedback
recurrence with the same operation and state laws.

No CATVM custody, general nonlinear lower bound, computational advantage,
Small Wall crossing, physical waveform execution, replacement of physical
bits with pi, or unbounded catalytic computation is established.
