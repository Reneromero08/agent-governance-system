# Nonlinear symplectic Lie-relation phase format

## Question

This experiment asks whether two existing Kerr/`SU(2)` wave modules close
through a typed shared wave port inside a fixed finite polynomial relation
signature. It does not enlarge the point-wave fixture.

For canonical coordinates `x=(q1,p1,q2,p2)`, the public modules use

```text
K0 = (q1^2+p1^2)^2 + (q2^2+p2^2)^2
K1 = K0 composed with U^-1
U  = [[3/5,4/5],[-4/5,3/5]]
```

Factoring the public linear map exposes the canonical-index contractions in
the Lie algebra required by any polynomial Hamiltonian composition. The
carrier computes

```text
L0 = K0
L(r+1) = {K1,Lr}
```

with the canonical Poisson bracket. Since
`degree({f,g})=degree(f)+degree(g)-2`, the predeclared grades have degrees
`4,6,8,10,12`.

## Native phase representation

Every coefficient is resident as a root of unity in both `F17` and `F19`.
Field addition is phase multiplication. Field multiplication uses the
root-of-unity Fourier interpolant

```text
P(X,Y) = (1/p) sum(r,s) zeta^(-r*s) X^r Y^s.
```

It maps `X=zeta^a,Y=zeta^b` to `zeta^(a*b)` without decoding either
coefficient. Monomial keys are public signature topology, not wave values or
assignments. The independent oracle uses exact rational sparse polynomials.

All five resident grade blocks are projected only as the final graded
boundary. The forward Poisson accumulations are then applied with conjugate
phase factors in reverse dependency order. No inverse kernel is retained.
The restored carrier runs a different rational mixer and seven further
alternating transactions.

## Adjudication

The exact rational nonzero counts are

```text
degree                 4    6    8    10    12
nonzero terms          6   32   85   126   231
full basis cells      35   84  165   286   455
```

The dual-prime phase hashes match the rational oracle at every grade. This
establishes bounded growth in the tested polynomial Lie-signature class. It
does not prove unbounded growth, exclude another exact signature algebra, or
show that the full BCH series has been materialized.

The direct four-complex-cell wave recurrence remains a matched 64-byte
point-evaluation baseline. The relation carrier uses 2,050 active phase
cells (32,800 logical packed payload bytes) plus the same-size stored cell
baseline. The allocated C carrier is 154,800 bytes before compiler and stack
temporaries. This is no advantage.

## Controls and ceiling

Zero Kerr and an identity mixer annihilate all higher grades. Swapping module
order changes the first bracket. Missing, wrong, and dependency-reordered
inverse controls leave continuous residuals and nonidentity modular cells.
Snapshot reload has no successful restoration receipt. Shared-port
projection and a null carrier are denied.

The claim is bounded to rational two-mode Kerr generators, the stated
`SU(2)` mixers, dual-prime grades through degree 12, and a software
root-of-unity realization. Poisson contraction here contracts canonical
derivative indices; it does not eliminate the coordinate interface of a
general open relation. It does not establish fixed-rank closure, a
distinct phase resource, computational advantage, Small Wall crossing,
physical waveform execution, or unlimited computation.
