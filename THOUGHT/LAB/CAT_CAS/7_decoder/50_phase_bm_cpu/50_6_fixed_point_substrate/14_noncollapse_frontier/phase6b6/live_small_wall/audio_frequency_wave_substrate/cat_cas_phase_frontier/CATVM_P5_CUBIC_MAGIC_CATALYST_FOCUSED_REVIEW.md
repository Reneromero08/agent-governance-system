# M248 focused independent review

Decision: `PASS_STRICT_SCOPE`

Classification: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `SEPARATE_REFERENCE_PARITY`

Restoration class: `EXACT_ALGEBRAIC_RESTORATION`

Scientific source head: `4c540e0f23d24a4e8b4546cdd971dd59a695e9f1`

The focused read-only review reconstructed the exact interaction

```text
T: c -> c+s
R_a(s,c) = zeta^(3 a s c^2 - 3 a s^2 c)
```

and confirmed, for every `a=1,2,3,4` and `s=0,1,2,3,4`, that

```text
R_a T (|s> |M_a>) = zeta^(-a s^3) |s> |M_a>.
```

The accepted service uses the actual five-cell `Q(zeta_5)` catalyst backing,
one 25-cell joint scratch backing, and two five-cell hidden phase-signature
backings.  Each joint interaction refactors exactly, reverses its scratch to
zero, and leaves the same catalyst backing available for the next coherent
syndrome.  After final-only data contraction the inverse rematerializes both
interactions, clears both phase signatures, verifies canonical state, advances
the lease generation, and only then releases the response.  Generation-two
descriptor-distinct reuse and a fresh-carrier comparison agree exactly.

The first review found two narrow independent-evidence gaps.  The standalone
oracle had asserted the dephased sham without executing the mixed-state
channel, and it had not separately reconstructed the target strength `-a`
Wigner `l1`.  The repaired oracle now traces all 20 off-diagonal syndrome
matrix elements through the exact dephased `I/5` catalyst channel and obtains
zero for each.  It also independently reconstructs the catalyst and target
Wigner `l1 = 1 + 2 sqrt(5)/5` with five negative cells for both strengths.

The abstract Unix-socket service passes disconnect, partial-forward, and
post-projection restoration; stale/wrong custody and projection/snapshot/debug
commands are rejected.  Catalyst, joint-scratch, and phase-signature contents
do not cross the socket.  Source hashes in the seals match the reviewed
service, controller, standalone oracle, qualifier, and exact-field dependency.

The strict result is a bounded exact software catalyst identity, not free
magic.  The required correction retains two bivariate cubic terms, while the
matched direct implementation uses one univariate cubic phase and the same
streamed final-boundary recurrence without catalyst restoration or CATVM
traffic.  No joint-correction magic monotone or optimal synthesis is claimed.
The package establishes no distinct phase resource, computational advantage,
Small Wall crossing, physical waveform execution, physical-bit replacement,
inference, or unbounded catalytic computation.
