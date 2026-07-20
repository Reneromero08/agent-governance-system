# Integrated Catalytic Waveform-Ising Computation

**Status:** `REPAIRED_QUALIFICATION_PENDING_RENEWED_FOCUSED_REVIEW`
**Package:** `audio_integrated_catalytic_computation_v1`
**Parents:** `audio_catalytic_wave_loop_v1`; `audio_recursive_catalytic_ising_v1`
**Claim ceiling:** `BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY`
**Physical authority:** none

## Result under test

The bounded reference executes:

```text
borrow one deterministic complex carrier
-> expand it into five carrier-bearing waveform sites
-> encode J and h through recursive-geometry complex interaction channels
-> evolve 5 x 256 samplewise complex states
-> project once after native evolution into an external result latch
-> reverse every recorded phase, mask, and shift operation
-> restore the borrowed carrier
-> reuse the restored carrier with a second field
-> extract a different optimum
-> restore the carrier again
```

The implementation is ordinary deterministic Python/NumPy software. Neither exact
oracle results nor decoded spins are reachable from native evolution.

## Carrier-causal mechanism

For site `i`, the initial state and reference frame are:

```text
x_i[n] = c[n] * B_i[n] * exp(i * theta_i)
F_i[n] = c[n] * B_i[n]
```

The recursive-tree pair channel is:

```text
G_ij[n] = exp(i * 1.8 * sin(arg(B_i[n] * conjugate(B_j[n]))))
```

Each tested geometry receives its own carrier-weighted mean-only complex calibration.
That calibration normalizes the channel mean but does not remove its pointwise phase
distribution. The canonical, flat, and parent-child-scrambled geometries are therefore
tested self-consistently rather than against a calibration mismatch.

At each native step, `J` and `h` enter pointwise complex interference. Spatial neighbor
coupling and within-site coherence coupling act on the full samplewise orientation
field. The resulting 5 x 256 phase-update field is applied before noncommuting
sample-dependent masks and circular shifts. The final scalar projection is outside the
recurrence.

The carrier is causal in two recorded senses: replacing its complex content changes the
native history by L2 `4.5067370631`, and the canonical operator history has maximum
rank-one residual `0.355170215528`. A five-scalar continuous-phase reduction therefore
does not reproduce the recorded samplewise trajectory.

## Frozen numerical contract

```text
sites                              5
samples per site                   256 complex128
native steps                       1000
time step                          0.03
geometry gate depth                1.8
spatial coupling                   0.6
global coherence coupling          4.0
final lock strength                1.2
restoration max error              2e-12
wrong-restoration minimum error    1e-3
minimum carrier displacement L2    1.0
final lock residual maximum        0.15 rad
query coherence minimum            0.90
operator-history change minimum    1e-3
samplewise-dynamics minimum         1e-3
boundary energy tolerance          1e-12
```

## Primary and reuse results

```text
primary field                      [-2, +1, -2, -2, -2]
primary projected spins            [-1, -1, -1, -1, +1]
primary energy                     -15.0
primary minimum coherence          0.950844481428
primary lock residual              0.0937407618198 rad
primary oracle agreement           true, unique
primary displacement L2            42.0391253493
primary restoration max error      1.17212686392e-14

reuse field                        [+1, -1, +0.5, +0.5, -1]
reuse projected spins              [+1, +1, +1, +1, -1]
reuse energy                       -14.0
reuse minimum coherence            0.937118450077
reuse lock residual                0.0846321696061 rad
reuse oracle agreement             true, unique
reuse input difference from
  primary restored carrier         0.0
reuse displacement L2              45.3020641203
reuse restoration max error        1.29393385716e-14
```

## Decisive controls

```text
remove waveform transform          invalid; minimum coherence 0.719090373638
self-calibrated flat geometry       invalid; raw shadow [-1,+1,-1,-1,+1]
self-calibrated scrambled geometry  invalid; raw shadow [-1,+1,-1,-1,+1]
canonical raw shadow               [-1,-1,-1,-1,+1]
remove one phase operator          invalid; history L2 change 15.2291217827
remove antipodal lock              invalid; residual 1.22794101834 rad
wrong query                        invalid; minimum coherence 0.739203384209
replace carrier by uniform content history L2 change 4.5067370631
zero carrier                       rejected before evolution
scalar J@s recurrence              structurally detected and rejected
wrong inverse order                restoration error 0.589772264895
omit one inverse step              restoration error 1.79934549508
omit restoration                   restoration error 1.77134422465
```

The exact oracle is invoked only after the boundary and is used only for adjudication.

## Evidence identity

```text
source bytes                        46388
source SHA-256                      50b6db77e2602e18356636ddb892f6d51aedb0573c6b2418afc8e5cc174991cc
fixture count                       5
fixture bytes                       9549
fixture root SHA-256                191dec29841f3cde6328d08c895014c07d98053cf2fdfa2a74b352f60e534968
fixture manifest SHA-256            dbcb27062f5e92b079a29edca8472a70b1fb728bdb06834d432c4b1215a700b2
reference results SHA-256           00c9b1703fda138349c09c2b34b27ef24b2337c99e2c003704c5948b81fc4613
reference tests                     31 PASS / 0 FAIL
```

Fresh-process verification recomputes the carrier, samplewise native mechanism,
latches, post-boundary adjudication, controls, restoration, reuse, manifest, and result
bytes.

## Review disposition and scientific boundary

The first focused review rejected the earlier rank-one formulation because it reduced
to five scalar phases and because geometry replacement was not self-calibrated. This
revision replaces that mechanism with recorded samplewise dynamics and requires flat
and scrambled geometries to fail after their own mean-only calibration. The repaired
candidate must receive a new exact-root focused review before the decision is final.

Passing the renewed review establishes only a bounded software carrier-causal reference
with restoration and reuse. It does not establish a general solver, computational
advantage, physical catalysis, physical persistence or restoration, hardware bit
replacement, or a Wall crossing.
