# Factorized phase-native computer

## Computing state

Register \(r\) is not stored as a scalar inside the native engine. It is the
relation between two spectral components of borrowed waveform row \(r\):

\[
z_r =
\frac{S_r\overline{R_r}}{|S_r\overline{R_r}|}
\in S^1.
\]

For radix \(q\), symbol \(x\) is represented by
\(z_r = \exp(2\pi i x/q)\). The native state therefore grows linearly with
register count and does not materialize the complete Cartesian state space.

## Native instruction set

- `ROT(t, c)`: \(z_t \leftarrow z_t e^{2\pi i c/q}\).
- `ADD(s, t)`: \(z_t \leftarrow z_t z_s\).
- `CCX(a, b, t)`: two phase-reference nulls interfere. The resulting
  Hermitian drive rotates \(z_t\) by pi only when both binary controls are
  antipodal to their references. No control symbol is decoded.
- `SWAP(a, b)`: the two complete waveform rows exchange places. This is
  phase transport and makes operator order physically relevant.

Programs and inputs have separate identities. A small compiler expands named
suboperations into this one instruction set. Intermediate state remains a bank
of complex phase relations for the entire native execution.

## Programs used during development

The same engine executes:

- affine modular arithmetic over \(\mathbb Z_5\);
- a two-bit adder whose carry remains an intermediate phase register;
- a binary phase-conditioned multiplexer;
- reverse-and-rotate sequence transformation over \(\mathbb Z_5\);
- finite-state accumulation over \(\mathbb Z_3\).

The conventional reference functions live only in the development qualifier.
They never enter the native engine.

## Catalytic lifecycle

The engine borrows a deterministic complex waveform bank, seeds the two
spectral register components, loads input as phase, executes the program, and
extracts output only at the boundary. It then traverses every native operation
in reverse, removes input phase, divides the seed operator, and returns the
actual borrowed waveform. The restored bank is reused directly in a second
execution.

The boundary result is copied outside the reversible history before
restoration. No fresh carrier is substituted.

## Current claim boundary

This is a bounded software reference. It tests whether phase relations support
a programmable, composable computer architecture. It makes no physical
carrier, performance advantage, energy, or universal-computation claim.
