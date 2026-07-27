# Fixed-Schema Quadratic ANF Relation Chain

## Status

This mutable checkpoint tests a composition algebra strictly broader than
GF(2) affine relations:

```text
ALGEBRAIC_FIXED_SCHEMA_QUADRATIC_ANF_TWO_HIDDEN_PORT_PHASE_COMPOSITION_ESTABLISHED
```

within:

```text
BOUNDED_BOOLEAN_GF2_MONIC_QAND_CHAIN_DEGREE4_FIVE_COEFFICIENT_BOUNDARY_SOFTWARE_REFERENCE_ONLY
```

It is a bounded software reference mechanism. It does not establish general
Boolean ANF elimination, bounded degree growth, advantage, a Small Wall
crossing, physical waveform execution, or unlimited catalytic computation.

## Public format

```text
CATCAS_QUADRATIC_ANF_CHAIN 1
TYPE BOOLEAN_ANF_GF2
F <monic-u> <constant> <ab>
G <monic-v> <constant> <uc>
J <monic-d> <constant> <ve>
END
```

All coefficients are canonical bits and every monic coefficient must be one.
The port/support topology is fixed by the schema, not selected by a fixture.
Fixtures contain input coefficients only; they do not contain selectors,
expected outputs, witnesses, candidates, membership masks, or answer hashes.

The three public relation factors are:

```text
F(a,b;u) = u + alpha + beta*a*b
G(u,c;v) = v + gamma + delta*u*c
J(v,e;d) = d + eta + theta*v*e
```

They are zero-set relations in the Boolean ANF quotient
`GF(2)[a,b,c,d,e,u,v]/(x^2+x)`.

## Exact two-hidden-port closure

Because `F` and `G` are monic definitions, the internal ports have unique
values and close by symbolic substitution:

```text
H(a,b,c;v)
  = v + gamma + delta*alpha*c + delta*beta*a*b*c

Z(a,b,c,e;d)
  = d + eta
      + theta*gamma*e
      + theta*delta*alpha*c*e
      + theta*delta*beta*a*b*c*e
```

The complete fixed-family boundary signature has the five public supports:

```text
[d, 1, e, c*e, a*b*c*e]
```

This is a complete ANF equation for the bounded family, not a selected sparse
probe into an unavailable output relation. It uses five coefficient cells
instead of the 32 membership entries for an arbitrary relation over the same
five Boolean variables. No general compression ratio is claimed.

The primary process closes:

```text
u + a*b = 0
v + u*c = 0
d + v*e = 0
```

to:

```text
d + a*b*c*e = 0
```

Its fourth mixed Boolean derivative is one. Every affine Boolean function has
all second and higher mixed derivatives zero, so the graph relation is
strictly non-affine. The unrelated reuse process also has a nonzero degree-four
coefficient. An affine sham follows the identical plan and native operation
counts but closes to `d+e=0` with zero fourth derivative.

A separate degree-two counterexample closes to `d+c*e=0`. It has zero
degree-four coefficient but a nonzero `c*e` ANF coefficient, so it is also
non-affine under the same plan. The reference classifies this fixed schema as
affine only when both the `c*e` and `a*b*c*e` coefficients are zero. This
prevents the degree-four certificate from being mistaken for a complete
non-affinity test.

## Phase-resident coefficient law

Every coefficient bit is stored as an F3 root-of-unity phase symbol. The
branch-native fixed Fourier polynomial computes Boolean-subset products
without decoding either input symbol. The support topology fixes every loop
bound, source address, output address, and operation:

```text
H coefficients = 4 resident phase products
Z coefficients = 5 resident phase products
```

The sealed plan hash is:

```text
f8198cf1e338bbb5
```

It is identical for the primary, unrelated non-affine reuse, and affine sham
fixtures. Each correct transaction performs 18 phase products across forward
and inverse execution, 97 carrier reads, 46 phase-cell updates, five final
boundary decodes, two final-boundary copies, zero intermediate decodes, and
zero intermediate copies.

The separate reference executable may stream the bounded 32 external rows and
four hidden probes per row after native execution. It stores no extensional
relation and is not linked into the native binary. The accepted native path
has no Boolean-value loop, membership relation, hidden-port probe, scalar
substitution, or expected boundary.

## Carrier and restoration law

The carrier has:

```text
public F/G/J coefficient cells       9
unresolved resident H cells          4
resident final Z cells               5
public final-boundary cells           5
total phase cells                    23
baseline-plus-working bytes         736
comparison snapshot bytes           736
```

The accepted transaction is:

```text
encode actual F/G/J
-> derive resident H from actual F/G
-> derive resident Z directly from actual H/J
-> copy and decode only final Z
-> remove actual boundary copy
-> inverse Z while actual H/J remain resident
-> inverse H while actual F/G remain resident
-> inverse public encodings
-> verify restoration
-> run an unrelated program on the same restored carrier
```

The latched final boundary survives outside the reversed history. One carrier
completes the primary transaction, unrelated reuse, and 256 alternating reuse
sentinels. The complex restoration tolerance is predeclared as `2e-12`; the
accepted run restores exactly at the reported precision.

A separate snapshot control reloads the comparison state and performs no
inverse. It is a weaker transactional baseline, not the accepted restoration
path.

## Controls and claim boundary

Qualification requires:

```text
wrong Z inverse fails restoration
missing Z inverse fails restoration
H-before-Z reordered inverse fails restoration
altered resident-H source changes the boundary and reverses cleanly
quadratic-term cut changes the non-affine boundary and reverses cleanly
affine sham has the same plan and operation counts
degree-two non-affine counterexample has the same plan and operation counts
intermediate projection rejects
null carrier rejects
nonmonic and malformed input reject
actual restored-carrier reuse passes 258 transactions
independent symbolic/extensional parity passes
deterministic replay passes
strict compiler, analyzer, ASan, UBSan, and leak checks pass
native/reference linkage separation passes
output-key allowlist and traced no-smuggle gates pass
source, fixture, binary, trace, and result hashes verify
focused independent review
```

The result does not establish:

```text
arbitrary non-affine relation closure
general Boolean ANF elimination
bounded degree or term growth
many-to-many non-affine boundary relations
CATVM enforcement for the quadratic ANF backend
computational or total-memory advantage
Small Wall crossing
physical waveform or silicon execution
unlimited catalytic computation
```

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/qanf-qual.XXXXXX)
bash qualify_quadratic_anf_chain_phase.sh "$evidence_dir"
sha256sum -c "$evidence_dir/SHA256SUMS"
```
