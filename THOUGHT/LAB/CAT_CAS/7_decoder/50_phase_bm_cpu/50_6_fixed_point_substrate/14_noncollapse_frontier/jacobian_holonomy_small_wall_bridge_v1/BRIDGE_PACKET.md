# Jacobian Weighted Fiber Pushforward Bridge

## 1. Claim boundary

This document contains two different objects:

1. exact algebraic identities that can be checked in rational arithmetic;
2. a native catalytic operator specification that has not been implemented.

The exact identities do not prove a Small Wall crossing or `P = NP`.

```text
exact representation: ESTABLISHED IN THIS PACKET
native catalytic pushforward: NOT ESTABLISHED
physical holonomy: NOT ESTABLISHED
standard-model polynomial simulation: NOT ESTABLISHED
```

## 2. Unit-Jacobian three-state carrier

Start from the polynomial map `F = (a,b,c)`:

\[
\begin{aligned}
a &= (1+xy)^3z+y^2(1+xy)(4+3xy),\\
b &= y+3x(1+xy)^2z+3xy^2(4+3xy),\\
c &= 2x-3x^2y-x^3z.
\end{aligned}
\]

Its Jacobian determinant is `-2`. Normalize the first target coordinate:

\[
\Psi(x,y,z)=\left(-\frac{a}{2},b,c\right).
\]

Then:

\[
\boxed{\det D\Psi=1.}
\]

At the target:

\[
q_*=\left(\frac18,0,0\right),
\]

the fiber ideal has the exact lexicographic Groebner basis:

\[
\left\{
z-\frac{27}{4}x^2+\frac14,\quad
y+\frac32x,\quad
x^3-x
\right\}.
\]

Therefore the fiber consists exactly of:

\[
\begin{aligned}
r_{\bot}&=\left(-1,\frac32,\frac{13}{2}\right),\\
r_0&=\left(0,0,-\frac14\right),\\
r_1&=\left(1,-\frac32,\frac{13}{2}\right).
\end{aligned}
\]

Interpret `r_0` as Boolean false, `r_1` as Boolean true, and `r_bot` as a
non-Boolean null sheet.

For `n` public variables, use the product map:

\[
\Psi_n=\Psi^{\times n}.
\]

It has:

\[
\det D\Psi_n=1
\]

and the fiber over `q_*^n` has exactly `3^n` sheets. The symbolic map description has
linear size in `n`. The latent fiber is not an explicit list.

## 3. Exact sheet idempotents

On the fiber coordinate `x in {-1,0,1}`, define:

\[
e_{\bot}(x)=\frac{x(x-1)}{2},\qquad
e_0(x)=1-x^2,\qquad
e_1(x)=\frac{x(x+1)}{2}.
\]

Modulo `x^3-x`, these satisfy:

\[
e_s^2=e_s,\qquad e_se_t=0\ (s\ne t),\qquad
e_{\bot}+e_0+e_1=1.
\]

The valid-Boolean selector is:

\[
V(x)=e_0(x)+e_1(x).
\]

For a literal `ell` on variable `i`, define its satisfaction selector:

\[
L_{\ell}(x_i)=
\begin{cases}
e_1(x_i),&\ell=x_i,\\
e_0(x_i),&\ell=\neg x_i.
\end{cases}
\]

For clause `C_j = (ell_j1 or ell_j2 or ell_j3)`:

\[
S_j(x)=1-\prod_{r=1}^{3}\left(1-L_{\ell_{jr}}(x)\right).
\]

For a public formula `F`:

\[
\boxed{
W_F(x)=
\left(\prod_{i=1}^{n}V(x_i)\right)
\left(\prod_{j=1}^{m}S_j(x)\right).
}
\]

On the product fiber:

\[
W_F(X)=
\begin{cases}
1,&X\text{ is a valid satisfying Boolean sheet},\\
0,&\text{otherwise}.
\end{cases}
\]

The formula should be represented as an arithmetic circuit with shared local factors.
Expanding it into a monomial list is not part of the construction.

## 4. Weighted fiber trace

Define the finite fiber pushforward:

\[
(\Psi_n)_*W_F(q_*^n)
=
\sum_{X\in\Psi_n^{-1}(q_*^n)}W_F(X).
\]

By the selector law:

\[
\boxed{
(\Psi_n)_*W_F(q_*^n)=\#\operatorname{SAT}(F).
}
\]

This is an exact parsimonious encoding. It is not an efficient implementation of the
pushforward.

## 5. Jacobian residue form

Let:

\[
p(t)=t^3-t,\qquad p'(t)=3t^2-1.
\]

For a contour `Gamma` enclosing the three simple roots:

\[
\sum_{p(r)=0}f(r)
=
\frac{1}{2\pi i}
\oint_{\Gamma}
f(t)\frac{p'(t)}{p(t)}\,dt.
\]

Applying the identity to every variable gives:

\[
\boxed{
\#\operatorname{SAT}(F)
=
\frac{1}{(2\pi i)^n}
\int_{\Gamma^n}
W_F(x)
\bigwedge_{i=1}^{n}d\log(x_i^3-x_i).
}
\]

The factor `p'(x_i)` is the local Jacobian normalization that assigns unit residue to
each simple sheet. The contour integral is another exact representation of the finite
fiber trace. Numerical quadrature of this integral is not a native crossing.

For a general nonsingular isolated fiber of a polynomial map
`R: C^d -> C^d`, the local Grothendieck residue gives:

\[
\sum_{x\in R^{-1}(y)}g(x)
=
\frac{1}{(2\pi i)^d}
\int_{\Gamma_y}
g(x)
\frac{\det DR(x)}
{\prod_{j=1}^{d}(R_j(x)-y_j)}
\,d^dx.
\]

This states the local-to-global role precisely:

```text
Jacobian: local sheet normalization
fiber pushforward: global aggregation across sheets
```

## 6. Modular phase sieve

Let:

\[
k=\#\operatorname{SAT}(F),\qquad 0\le k\le 2^n.
\]

For a prime `p`, define the exact modular phase label:

\[
H_{F,p}=\exp\left(\frac{2\pi i}{p}k\right).
\]

Equivalently, retain the exact residue `k mod p` and render the phase only at the final
boundary.

Choose any `n+1` distinct primes `p_1,...,p_{n+1}`. Then:

\[
\boxed{
k=0
\iff
k\bmod p_j=0\text{ for every }j.
}
\]

If `k>0` and every prime divided `k`, their product would divide `k`. But:

\[
\prod_{j=1}^{n+1}p_j\ge 2^{n+1}>2^n\ge k,
\]

which is impossible. Therefore the residue family provides an exact total SAT/UNSAT
boundary once the fiber pushforward exists.

The sieve does not solve the pushforward problem. A conventional implementation that
first computes `k` has simply performed `#SAT`.

## 7. Native Catalytic Fiber Pushforward target

The missing operator is:

\[
\operatorname{NCFP}_{p}(F):
(\tau_F,0)
\longmapsto
(\tau_F,\#\operatorname{SAT}(F)\bmod p),
\]

where:

- `tau_F` is the unresolved formula-weighted carrier;
- no assignment or derivation list is materialized;
- the source does not precompute a response vector;
- the accumulator is independent of the borrowed carrier;
- the complete carrier and environment satisfy the frozen restoration law.

The exact SAT boundary is:

\[
F\in\mathrm{SAT}
\iff
\exists j\le n+1:
\operatorname{NCFP}_{p_j}(F)\ne0.
\]

## 8. Relationship to the Small Wall

The current Small Wall asks for a receiver-readable result that cannot be reduced to:

- a static answer cache;
- a branch-local additive response;
- ordinary route, bank, address, order, or timing artifacts;
- a carrier that was not genuinely used;
- a reset mislabeled as restoration.

The fiber construction identifies the desired global operation. It does not identify a
Family 10h implementation.

The important distinction is:

```text
pullback: expose how one local boundary question acts on one hidden direction
pushforward: aggregate a global invariant across the complete unresolved fiber
```

The Jacobian lens and local PMU projection are pullback-like probes. The Small Wall is
the missing pushforward.

## 9. Adversarial proof audit

### A1. The residue is enumeration in disguise

**Attack:** The contour formula is mathematically equivalent to a sum over all sheets.

**Disposition:** Correct. The packet claims an exact operator specification, not an
efficient classical algorithm. Native implementation remains open.

### A2. `W_F` becomes exponentially large

**Attack:** Expanding all products can create exponentially many monomials.

**Disposition:** The lawful compiler emits a shared arithmetic circuit of size
`O(n+m)`. Any implementation that expands it fails the no-materialization gate.

### A3. The `3^n` fiber is an exponential physical allocation

**Attack:** One physical mode per sheet would recreate the forbidden resource.

**Disposition:** Explicit per-sheet allocation is forbidden. CAT_CAS permits exponential
latent relational multiplicity only when carried by finite native geometry with measured
compile, evolution, readout, and restoration costs.

### A4. The modular phase assumes the answer

**Attack:** Applying `exp(2*pi*i*k/p)` after calculating `k` provides no advantage.

**Disposition:** Correct. The target operator must couple directly to the weighted fiber
pushforward. Classical precomputation is a hard fail.

### A5. The prime sieve has false zeroes

**Attack:** A positive count may vanish modulo one prime.

**Disposition:** One prime is insufficient. The `n+1` prime family is exact for
`0 <= k <= 2^n`.

### A6. The unit-Jacobian carrier itself solves SAT

**Attack:** The three-sheet polynomial map may already contain the answer.

**Disposition:** False. The carrier is formula-independent. Formula semantics enter only
through `W_F`, compiled from public clauses.

### A7. Holonomy is asserted without a connection

**Attack:** An exponentiated residue is not automatically a physical holonomy.

**Disposition:** Correct. The exact object is a modular phase functional. The word
`holonomy` becomes a physical claim only after a connection, closed transport, inverse
law, and causal phase readout are established.

### A8. This proves ordinary `P = NP`

**Attack:** The exact representation may be presented as a complexity proof.

**Disposition:** Rejected. Ordinary `P = NP` additionally requires a native operator and
deterministic polynomial-overhead standard-model simulation. Neither is established.

## 10. Resource and no-smuggle ledger

| Resource | Allowed current object | Hard failure |
|---|---|---|
| Compiler | Formula-local circuit of size `O(n+m)` | Assignment enumeration or solved witness |
| Carrier geometry | Product map with `3^n` latent sheets | Explicit table or independent mode per sheet |
| Formula weight | Shared local circuit | Expanded exponential monomial list |
| Pushforward | Unimplemented native target | Classical loop over sheets or contour grid |
| Readout | Exact modular accumulator target | `#SAT` computed before phase application |
| Precision | Prospectively measured | Exponential precision hidden in phase discrimination |
| Environment | Explicitly included in R2 | Omitted bath, gain, cache, or PMU state |
| Restoration | Program-derived inverse and equivalence law | Reset, natural decay, or transcript replay |
| Claim | Exact representation theorem | Small Wall or `P = NP` promotion |

## 11. Current theorem surface

Established by exact algebra:

```text
UNIT_JACOBIAN_THREE_SHEET_CARRIER
PRODUCT_FIBER_HAS_3_POWER_N_SHEETS
EXACT_BOOLEAN_SHEET_IDEMPOTENTS
WEIGHTED_FIBER_TRACE_EQUALS_NUMBER_OF_SATISFYING_ASSIGNMENTS
JACOBIAN_RESIDUE_REPRESENTS_THE_FIBER_TRACE
N_PLUS_ONE_PRIME_RESIDUE_SIEVE_IS_TOTAL
```

Not established:

```text
NATIVE_CATALYTIC_FIBER_PUSHFORWARD
PHYSICAL_HIGHER_HOLONOMY
FAMILY10H_RELATIONAL_CARRIER
RESTORATION_R2
SMALL_WALL_CROSSED
P_EQUALS_NP
```
