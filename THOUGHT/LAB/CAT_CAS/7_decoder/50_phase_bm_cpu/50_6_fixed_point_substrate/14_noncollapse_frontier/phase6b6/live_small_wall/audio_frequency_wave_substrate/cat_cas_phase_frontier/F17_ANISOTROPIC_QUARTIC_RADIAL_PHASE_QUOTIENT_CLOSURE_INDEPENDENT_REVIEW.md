# M139 independent review

Decision: `PASS` at strict bounded scope.

The reviewed mechanism is limited to radial functions on the anisotropic
plane `F17^2` with norm `Q(x,y)=x^2-3y^2`, the declared quartic norm-phase
gates, and the normalized anisotropic phase Fourier transform.  The review
confirmed the 17-shell quotient, exact Fourier involution, noncommuting gate
order, final-only scalar projection, exact inverse restoration, same-backing
unrelated reuse, and the independently reconstructed 289-coordinate controls.

The review required and then confirmed repairs for:

- removal of retained public coordinate and shell tables from the accepted
  compiler path;
- separation and counting of the exhaustive quotient verifier;
- comparison of all 17 final shell values and final-state commitments, not
  boundary scalars alone;
- projection, commitment, compile, and dense-control resource accounting;
- separation of per-algebra and whole-package live kernel costs; and
- replacement of an unsupported strongest-classical label with an executed
  matched compact recurrence and an explicit no-lower-bound caveat.

Final counted public geometry costs are 5,202 compiler coordinate visits per
algebra, 4,913 involution multiply-accumulates per algebra, and an exhaustive
verification split of 289 target plus 83,521 source visits.  The dense
verification path uses 867 coordinate-buffer field cells and reaches 1,734
live exact field cells when all three retained compiled kernels are included,
excluding arithmetic expression temporaries.

The independent oracle imports no production module.  It reconstructs the
geometry, schedules, exact forward/inverse recurrence, final commitments, and
selected dense separable transforms.  It independently matches all seven
exact boundaries, final commitments, resident-payload maxima, all 48
structural boundaries, and all selected dense shell states.

This review does not establish CATVM custody, general nonlinear relation
quotients, a distinct phase resource, computational advantage, a Small Wall
crossing, physical execution, replacement of physical bits with pi, or
unbounded catalytic computation.
