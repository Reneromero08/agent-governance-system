# Adversarial Proof Audit

## Disposition

```text
EXACT_THREE_SHEET_FIBER_CERTIFIED
EXACT_WEIGHTED_FIBER_TRACE_IDENTITY_CERTIFIED
EXACT_ZERO_VS_NONZERO_PRIME_SIEVE_CERTIFIED
JACOBIAN_COUNTEREXAMPLE_NOT_COMPUTE_ADVANTAGE_BY_ITSELF
EXPONENTIATED_RESIDUE_NOT_YET_PHYSICAL_HOLONOMY
NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED
SMALL_WALL_NOT_CROSSED
P_EQUALS_NP_NOT_PROVEN
```

This audit attacks the bridge as both mathematics and proposed mechanism. It separates
an exact representation theorem from the missing native operation.

## 1. Exact completeness of the selected fiber

The normalized map is:

\[
\Psi(x,y,z)=\left(-\frac{a(x,y,z)}2,b(x,y,z),c(x,y,z)\right)
\]

with target:

\[
q_*=\left(\frac18,0,0\right).
\]

The first target coordinate is equivalent to `a=-1/4`.

Set:

\[
u=xy,\qquad v=x^2z,\qquad L=2-3u-v.
\]

The source coordinates satisfy the exact polynomial identities:

\[
c=xL,
\]

\[
xb=u+3(1+u)^2v+3u^2(4+3u),
\]

\[
x^2a=(1+u)^3v+u^2(1+u)(4+3u).
\]

The latter numerators obey:

\[
xb-2(2u+3)=-3(1+u)^2L,
\]

\[
x^2a-(u+1)(u+2)=-(u+1)^3L.
\]

These identities are checked exactly by `proof_certificate.py`.

### Case 1: `x=0`

The equation `b=0` reduces to `y=0`. The equation `a=-1/4` then gives:

\[
z=-\frac14.
\]

This yields:

\[
r_0=\left(0,0,-\frac14\right).
\]

### Case 2: `x` is nonzero

Since `c=xL=0`, nonzero `x` gives `L=0`. The reduced `b` identity then gives:

\[
2(2u+3)=0,
\]

so:

\[
u=-\frac32.
\]

The equation `L=0` gives:

\[
v=2-3u=\frac{13}{2}.
\]

The reduced first-coordinate identity gives:

\[
x^2a=(u+1)(u+2)=-\frac14.
\]

Since the target requires `a=-1/4`, it follows that:

\[
x^2=1.
\]

Therefore `x=1` or `x=-1`, yielding exactly:

\[
r_1=\left(1,-\frac32,\frac{13}{2}\right),
\]

\[
r_{\bot}=\left(-1,\frac32,\frac{13}{2}\right).
\]

No fourth branch remains. The complete target fiber is exactly the listed three points.
Because the full Jacobian determinant is one at every source point, each fiber point is
reduced and nonsingular.

## 2. What role the Jacobian actually plays

Three derivative objects must not be conflated.

### Full carrier Jacobian

\[
\det D\Psi=1.
\]

This proves local nonsingularity and unit local multiplicity for the original
three-dimensional map.

### Elimination derivative

After eliminating `y` and `z`, the selected fiber is parameterized by:

\[
p(x)=x^3-x=0.
\]

The factor:

\[
p'(x)=3x^2-1
\]

is the Jacobian of this reduced one-coordinate fiber equation. It gives unit residue at
each of the three simple roots.

### Grothendieck residue Jacobian

For the full carrier equations `Psi-q_*`, the multidimensional residue uses:

\[
\det D\Psi=1
\]

in the numerator. Reduction to the `x` chart produces the logarithmic derivative
`p'(x)/p(x)`.

The reduced factor `p'(x)` is not a second claim that the original full determinant is
`3x^2-1`. It is the elimination-chart normalization.

## 3. The weighted fiber identity is exact but not novel complexity leverage

The formula-independent map supplies three sheets per variable. The formula enters in
the local arithmetic-circuit weight:

\[
W_F=\prod_i V_i\prod_j S_j.
\]

On the product fiber, `W_F` is the indicator of satisfying Boolean sheets. Therefore:

\[
\sum_X W_F(X)=\#SAT(F).
\]

This identity is exact. It is also structurally equivalent to a partition-function or
tensor-network representation of `#SAT`.

The Jacobian counterexample contributes a globally multi-sheeted, locally nonsingular
carrier geometry. It does not make the pushforward efficient by itself. Any claim of
compute leverage must come from a native operation that evaluates the pushforward
without materializing, traversing, or classically summing the fiber.

## 4. Attack ledger

### A1. The three listed points may not be the complete fiber

**Result:** Defeated for the selected target by the exact case split above.

### A2. The fiber may contain nonreduced multiplicity

**Result:** Defeated. `det D Psi = 1` at every point, so the selected fiber points are
nonsingular and have local multiplicity one.

### A3. The residue merely rewrites enumeration

**Result:** Attack succeeds against any algorithmic claim. The residue is an exact
representation of the sheet sum. Numerical quadrature, symbolic residue expansion, or
root enumeration is not the native operator.

### A4. The formula weight hides exponential expansion

**Result:** Avoided at the representation layer only. The lawful compiler retains a
shared arithmetic circuit with linear node count. Expanding the product into a monomial
list fails the no-materialization gate.

### A5. The map contains the answer

**Result:** Defeated. The map and target depend only on variable count. Formula semantics
enter through `W_F`. No satisfying assignment, count, or expected result occurs in the
carrier map.

### A6. The formula weight itself is already a solver

**Result:** Defeated as a compilation objection, not as an execution objection. Evaluating
`W_F` on one supplied sheet is local and polynomial. Aggregating it over the unresolved
fiber is the missing hard operation.

### A7. One modular phase can vanish on a positive count

**Result:** Correct. One prime is insufficient. The `n+1` distinct-prime family is exact
on `0 <= #SAT <= 2^n` because their product exceeds `2^n`.

### A8. The prime sieve requires exponential precision

**Result:** Not at the abstract readout layer. The first `n+1` primes have polynomial
bit length, and neighboring residues are separated by inverse-polynomial phase. This
statement is conditional on a native exact residue coupling. It does not provide that
coupling.

### A9. Exponentiating the residue makes it holonomy

**Result:** Attack succeeds against the literal physical claim. An exponentiated integral
is an exact phase functional. It becomes physical holonomy only after a connection or
higher connection, closed transport law, causal phase readout, inverse law, and carrier
restoration are identified.

### A10. The abstract map already restores the carrier

**Result:** Not established physically. Writing:

```text
(tau_F, 0) -> (tau_F, k mod p)
```

specifies the required catalytic boundary. It does not construct a reversible physical
interaction that leaves the complete carrier and environment in the frozen equivalence
class.

### A11. A finite challenge family can be cached

**Result:** Open physically. A source can precompute a response vector for a finite public
query family if carrier capacity permits. The Family 10h protocol therefore requires
post-source held-out loop words, composition laws, capacity bounds, and causal
intervention tests.

### A12. The construction proves ordinary `P = NP`

**Result:** False. Ordinary `P = NP` additionally requires a uniform standard-model
implementation with polynomial time, space, precision, preparation, readout, and
restoration. No such implementation is present.

## 5. Strongest supported theorem

For every public 3-CNF formula `F` on `n` variables, there is a formula-independent
product of unit-Jacobian polynomial carriers with a target fiber of size `3^n` and a
formula-local shared arithmetic-circuit weight `W_F` such that:

\[
(\Psi_n)_*W_F(q_*^n)=\#SAT(F).
\]

The same value is represented by the corresponding full Grothendieck residue and by the
reduced product logarithmic residue. Its vanishing is decided exactly by `n+1` modular
residues.

This theorem is a representation and operator-specification result. It does not supply
the Native Catalytic Fiber Pushforward.

## 6. Next proof gate

The next mathematical artifact must define a native operator model with all of these
properties:

```text
formula-local compilation
no sheet enumeration
no expanded coefficient table
no answer-conditioned initialization
causal coupling to the complete weighted fiber
independent accumulator
exact composition law
native inverse
complete restoration law
bounded observable precision
```

Until that model exists and survives resource accounting, the exact claim ceiling is:

```text
NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED
```
