# Stokes-sphere reduced nonlinear relation format

## Phase-owned repair

The four-real-coordinate Kerr signature contains a global optical phase and
the conserved total norm. For a normalized two-mode wave, the exact Stokes
map removes global phase and places the state on

```text
x^2 + y^2 + z^2 = 1.
```

The Kerr Hamiltonian becomes a quadratic axis twist. For the primary mixer,

```text
H0 = z^2
H1 = (24*x - 7*z)^2 / 625.
```

The phase machine uses the `su(2)` Lie-Poisson bracket

```text
{f,g} = S dot (gradient(f) cross gradient(g))
```

and reduces every product exactly by
`z^2 = 1-x^2-y^2`. Canonical monomials therefore have `z` exponent zero or
one. This is an exact relation-preserving quotient, not numerical pruning.

## Carrier law

Coefficients remain dual-prime root phases, with the same Fourier
field-product primitive as the unreduced experiment. Poisson contraction
contracts canonical derivative indices; it does not eliminate a general
coordinate interface. Five resident quotient blocks cover degree limits
`2,3,4,5,6`. Only the final graded boundary is
decoded. Reverse Lie-Poisson accumulation clears the actual blocks, and the
same carrier runs the sign-changed mixer plus seven alternating reuse
transactions.

An independent exact-rational Groebner reduction checks every `F17` and
`F19` coefficient hash. No wave values, hidden port coordinates, truth
tables, path expansions, or intermediate coefficient vectors are emitted.

## Result

```text
degree limit                 2    3    4    5    6
rational nonzero terms       3    4    9    9   16
quotient basis cells         9   16   25   36   49
```

Across all grades, the exact quotient reduces 1,025 four-dimensional basis
cells to 135 Stokes cells. The dual-prime carrier falls from 2,050 to 270
phase cells, or from 32,800 to 4,320 logical packed payload bytes.
Restoration and reuse remain within the predeclared tolerance.

The repair removes redundant coordinates but does not close at fixed rank.
The remaining degree and spherical-harmonic rank growth is genuine within
this tested quotient. The matched point-evaluation recurrence is still only
64 bytes.

## Controls and ceiling

Zero Kerr and an identity mixer remove all higher grades. Swapped order,
missing inverse, wrong inverse, and dependency-reordered inverse are
discriminated by both continuous residual and nonidentity modular cells.
Snapshot is separate from inverse restoration. Shared-port projection and
null carrier requests are denied.

The result is bounded to the normalized two-mode Stokes sphere, the stated
rational mixers, grades through degree six, dual-prime coefficient shadows,
and software execution. The 4,320-byte number is logical packed phase
payload; CPython object allocation and temporaries are not measured. It
establishes an exact rank reduction, not
fixed-rank closure, unbounded growth, a distinct phase resource, advantage,
Small Wall crossing, physical execution, or unlimited computation.
