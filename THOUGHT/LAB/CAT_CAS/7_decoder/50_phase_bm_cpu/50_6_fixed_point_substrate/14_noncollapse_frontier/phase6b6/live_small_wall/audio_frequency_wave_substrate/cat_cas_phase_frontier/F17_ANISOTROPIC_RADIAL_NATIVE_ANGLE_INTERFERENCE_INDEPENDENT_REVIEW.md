# M144 independent review

Decision: `PASS` at strict bounded numerical scope after repair.

M144 changes the M143 radial-Fourier update law.  It consumes the 34 resident
`float64` phase angles directly, pairs the public `p` and `-p` character
terms into a real streamed kernel, accumulates one output at a time in two
Cartesian scalars, and returns each output to the canonical two-phasor chart.
The accepted update retains neither a 17-complex decoded state nor a dense
`17 x 17` kernel, gate tape, or inverse history.

The 21 declared cases cover `PRIMARY`, `REUSE`, and `ALTERNATE` at depths
`1,4,16,64,256,1024,4096`.  Worst production error is `2.607e-11` against the
matrix-free compact baseline boundary and `2.012e-11` against the streamed
complex baseline boundary, below the predeclared `5e-11` boundary tolerance.
Worst transaction restoration is `2.102e-12`.  The unrelated depth-1537 reuse
boundary differs from fresh execution by `8.638e-13`; 100 consecutive
same-backing depth-64 cycles reach at most `3.752e-12` restoration error.

The accepted inverse applies the actual real-kernel involution and inverse
phase translations to the resident angle array.  It does not reset, reload,
or replace that array after the inverse.  Restoration is classified
`NUMERICAL_PHYSICAL_STATE_RESTORATION`, not exact algebraic restoration and
not inverse-plus-post-hoc canonical reset.  Same-backing identity, generation
advance, no snapshot, and no retained history are sealed for the transaction,
unrelated reuse, and 100-cycle reuse paths.  CATVM custody is not claimed.

The independent oracle imports no production or predecessor implementation.
It reconstructs the real radial kernel in long-double arithmetic from 2,312
public pair visits, checks zero public character-pairing violations and
involution to `3.931e-19`, and reexecutes all 21 cases.  It performs 474
declared field and boundary comparisons, independently repeats unrelated and
100-cycle reuse, gates the production restoration and reuse records, and
detects a kernel mutation.  The unweighted real kernel has measured asymmetry
`1.0` and is explicitly not claimed symmetric.

Named accepted-path accounting includes 272 resident angle bytes, 64 retained
inverse-parameter bytes, 136 shell-count bytes, a 368-byte maximum update
scratch, 80 projection bytes, and at most 178 public program JSON bytes,
totalling 1,018 named live bytes including the program.  The commitment hashes
a zero-copy memory view: its input copy is zero bytes, public hexadecimal
digest is 64 bytes, and logical SHA state/block allowance is 96 bytes.  The
810-byte named commitment live total excludes Python, allocator, NumPy,
hashlib/native-library internals, and whole-process storage.

Both matched compact classical frontiers are executed in every case.  The
work-minimizing matrix-free 17-complex recurrence reaches 1,770 named live
bytes including program and uses 544 complex character products per Fourier.
The equal-memory streamed real-kernel 17-complex recurrence reaches only 986
named live bytes including program and performs the same 2,312 kernel cosines
while avoiding the native path's 1,156 input-phasor trigonometric calls and 51
chart calls per Fourier.  M144 therefore removes the full-state decode but
does not improve the strongest matched compact classical storage or work law.

Initial review found three evidence defects: the commitment's temporary input
copy was omitted, the classical frontier accounting was incomplete, and the
oracle did not gate production restoration/reuse fields.  The review also
found inaccurate kernel-symmetry prose.  The repaired package uses a zero-copy
commitment, executes and reports both classical frontiers, gates the complete
production restoration/reuse record, and records the measured asymmetry.  A
fresh no-write replay reproduced both sealed JSON files byte-for-byte.  No
concrete defect remained in the repaired requested scope.

This result covers only the sealed numerical cases through depth 4096.  It
establishes neither exact algebraic semantics, unbounded-depth numerical
stability, machine-enforced hidden-state custody, a distinct phase resource,
computational advantage, a Small Wall crossing, physical waveform execution,
replacement of physical bits with pi, nor unbounded catalytic computation.
