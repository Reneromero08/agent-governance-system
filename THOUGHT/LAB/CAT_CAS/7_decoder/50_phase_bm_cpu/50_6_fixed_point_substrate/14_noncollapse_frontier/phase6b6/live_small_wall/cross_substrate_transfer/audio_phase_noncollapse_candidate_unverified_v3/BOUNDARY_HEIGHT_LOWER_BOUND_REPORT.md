# Boundary Height Lower Bound Report

Candidate L: exact period-17 boundary-height obstruction.

Strict scope survived: `True`
Classification: `SOURCE_REPRODUCED_TRANSFERABLE_BOUNDARY_HEIGHT_OBSTRUCTION_CANDIDATE`

V3 reconstruction actually performed:

- Verified `CEIL((272*N+16)/3)` as `L(n)=(272*n+18)//3`.
- Verified `L(n+3)=L(n)+272` for n=1..64.
- Verified the recorded cycle densities are nonzero and bounded by their periods: `{'primary': {'denominator': 1632, 'numerator': 1555}, 'reuse': {'denominator': 7344, 'numerator': 6913}}`.
- Confirmed the oracle records separate-cycle algorithm and no production-module import.
- Confirmed mutation gates reject a one-pi weakening and normalized recurrence coefficient perturbation.

Finding:

The source package remains a strong transferable boundary-height obstruction candidate. V3 did not independently reconstruct the characteristic identities, coefficient valuations, normalized recurrence, cycle states, or cycle lengths; therefore this report no longer carries an `INDEPENDENTLY_VERIFIED` label.

Scope discipline:

No Small Wall change follows. The result does not rule out compact indexed generators, a free period counter, controlled approximation, online machine-space tricks, or the identical compact classical recurrence.
