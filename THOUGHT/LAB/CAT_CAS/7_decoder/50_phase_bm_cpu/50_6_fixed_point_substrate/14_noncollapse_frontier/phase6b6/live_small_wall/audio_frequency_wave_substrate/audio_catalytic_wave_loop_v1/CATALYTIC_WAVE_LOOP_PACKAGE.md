# Catalytic Recursive-Wave Loop Package

**Status:** `AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED`
**Package:** `audio_catalytic_wave_loop_v1`  
**Parent result:** `AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED`  
**Claim ceiling:** `SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY`
**Physical authority:** none

## 1. Bounded established result

The package closes one ordinary-software lifecycle:

```text
exact committed R0 complex carrier
-> nonzero displacement by committed R1 T1-T3 phase operators
-> public hierarchy-A query selected before trajectory execution
-> external complex relational latch
-> correct reverse carrier traversal
-> numerical phase-carrier equivalence restoration
-> exact committed T0 ancestry recovery
```

The latch is a boundary observable. It never becomes a scalar control signal for the
native forward or inverse carrier paths.

## 2. Source custody

The thin source candidate was first committed as:

```text
source commit             65f656527a8bf63e6d44493154f3db15a8a99b8b
source Git blob SHA-1     1bf96eb8a95c89a82665a95a49d3bd722a14f7d4
source byte count         17797
source SHA-256            d15ab5273ab79108f0fff63ef439443e33ec69574777488e94298d9a7caac058
line endings              LF
Python syntax             PASS
```

Qualification expanded that candidate without modifying R0 or R1. The exact qualified
source bytes are:

```text
source Git blob SHA-1     63eed91f74252082b1258755bdd4371a2a48e105
source byte count         117873
source SHA-256            6c55861da950caf0738bb5ffb676f0c458a593a805ddd49419d6b2b427f6c33c
line endings              LF
Python syntax             PASS
```

The final Git commit containing those bytes is reported after commit. The Git blob and
SHA-256 already bind the exact source independently of the containing commit.

## 3. Exact parent custody

The loop contract freezes these established parent identities:

```text
R0 source blob            956adb0ae8e84c091c1dc1e3de650be374fa96d1
R0 source SHA-256         e5911cb868f244ac69f3f8f8c4cfa83440385347be2d4526d5f25376de736887
R0 fixture manifest       7112307fa4406cf4880736545a88e56c45fafc6f27cd0a6518a1b40963fb62fa
R0 fixture set            6afb8adb0d14ab2e5a750df519ced073475fbf1554ee8be0732a2ebde5e15925
R0 tests                  3cecfa9f0d79babc4f9d76d7b463a1b8f825e209f2af592e590c52686dc95b2c
R0 result                 46e2cc7cb72217c647f8653ebe61a0dbf2060a222de0eec6624fbb7fbcb94eab

R1 source blob            3685be9ae63dcd213b2155c8cd66f6f81e45c071
R1 source SHA-256         26b2cfaa63f5fe6bfa97f6d9f64b97d0ee944bc39ac45d406092aea257b2179e
R1 fixture manifest       28cbcec8997f6f5eb49dc13e6bf919342af0863a5ba6cb1a70f10dea6fcdbc4e
R1 fixture set            da62112c0459c49673675182e67011899d8ee1e841df3650c0c4a0aeecd137dd
R1 tests                  5bf39db581fbc4f5cc290d1ad0ba34bc87315c2d1cf4777acf12d1d8a35023b5
R1 result                 37cb46f6806555cfaec60910f9b5b92fbcac5bf1d0e976fb67e7f2d2c0ec4139
```

Qualification reruns both parent verifiers. Stored PASS strings carry no authority.

## 4. Strict contract and custody objects

The source defines strict canonical parsers and generated schemas for exactly three
objects:

```text
LoopContract
RelationalLatch
CatalyticClosure
```

The loop contract binds the exact parent identities, public query tree canonical
identity and digest, T0-T3 digests, shift schedule, raw carrier format, sample and byte
counts, operator order, thresholds, claim ceiling, query-selection stage, and latch
stage.

It rejects duplicate keys, unknown keys, noncanonical JSON, nonfinite numbers, booleans
used as numbers, wrong schedule length, wrong shifts, wrong query identity, wrong source
identity, and answer/expected-result/spin/energy/winner/candidate/score fields.

The latch contains only:

```text
schema
query identity
final tree digest
before and displaced carrier hashes
forward displacement
complex response
latch stage
```

The closure contains no states, drives, step specifications, ancestry receipts, or
displaced-carrier array.

## 5. Frozen carrier

```text
format                   raw little-endian interleaved complex128
dtype                    <c16
samples                  6000
bytes per carrier        96000
header or metadata       none
shift schedule           [17, -29, 43]
operator order           multiply tree beam, then circular roll
inverse order            circular unroll, then multiply conjugate tree beam
```

The three committed binary carriers are:

```text
before SHA-256           b907b0c948cf7929353816771bc3c5916911e5f0240f17eb923af65ac4d79605
displaced SHA-256        ddf312eac86edad3f160048f06b5efa5e0346c8d83737acfe2c55136147f0157
restored SHA-256         dcdd7ecc904f435e5fe7ef9410872f4c117a95d001e99e90b008395af1d37917
carrier byte exact       false
```

The hashes differ. The package does not call this byte restoration.

## 6. Restoration law

The complex carrier is a continuous numerical chart rather than a byte-tape identity.
The prospectively frozen acceptance region is:

```text
metric                   max absolute complex sample error
tolerance                1e-12
observed correct error   4.74287484027e-16
equivalence restored     true
```

This law is coherent because the carrier is genuinely numeric, the metric and tolerance
precede exact execution, forward displacement is nonzero, the correct inverse enters the
region, every declared wrong arm remains far outside it, and exact hashes are retained as
an honest diagnostic. Canonical R1 ancestry is a separate channel and restores exact T0
bytes.

Full reasoning is frozen in `CATALYTIC_WAVE_LOOP_RESTORATION_LAW.md`.

## 7. Native lifecycle and public query

For each committed R1 state `T_k` and shift `s_k`:

```text
F_k(tau)    = Roll_s[k](tau * B_k)
F_k^-1(tau) = Roll_-s[k](tau) * conjugate(B_k)
```

The public query is selected before trajectory execution:

```text
q0 = tau0 * B_hierarchy_A
q3 = Roll_43(Roll_-29(Roll_17(q0)))
zq = normalized_complex_inner_product(tau3, q3)
```

Observed complex responses:

```text
hierarchy A              0.00716374763471 - 0.00230648623425i
hierarchy B              0.00254283419804 + 0.000282958235002i
absolute difference      0.00529698627982
required difference      at least 1e-6
```

Both responses are reported. No winner is selected.

## 8. Correct and wrong arms

```text
forward displacement L2       73.1576613427       >= 1.0
correct restore max error     4.74287484027e-16   <= 1e-12
forward-order inverse error   1.79941674031       >= 0.05
wrong-trajectory error        0.959034213823      >= 0.05
omitted-step error            1.79928587832       >= 0.05
duplicated-step error         1.79894500301       >= 0.05
wrong-shift-sign error        1.796417268         >= 0.05
wrong-shift-magnitude error   1.29003363401       >= 0.05
wrong-state-one-leg error     1.79967048651       >= 0.05
no-restore error              1.79859400708       >= 0.05
exact T0 ancestry bytes       recovered
```

Additional negative controls reject an untouched carrier, a zero-amplitude coordinate,
post-result query selection, carrier/latch/contract mutation, raw-endian mutation, and
manifest role substitution.

## 9. Structural no-feedback proof

The AST proof closes the call graph rooted at:

```text
forward_carrier_step
inverse_carrier_step
forward_carrier
restore_carrier
```

It reaches only native carrier helpers, reports no forbidden identifier, unresolved
call, indirect write, module rebinding, decorator, or latch-feedback route, and proves
the order:

```text
complete forward displacement
-> latch
-> carrier restoration
-> ancestry restoration
-> closure
```

Eighteen operator-shape, feedback, preselection, alias, and binding mutation probes
reject. The proof is a committed-source
ordinary-software qualification, not hostile-interpreter isolation.

## 10. Committed evidence

```text
contract schema SHA-256  3df3775a9dbbae86a405a7a01a907f7872293c978ac45efd3cadb1fa44d3d650
latch schema SHA-256     4fc7a46ffb952a673457d33f3bef2801a860b75cc94f5aa1fa64a72d9f702efd
closure schema SHA-256   41a790cab6a8d9b31f6a7cfd9cc42aff74187d5d371b535a8ccbcea48a6607cb
test specification       ef888d8d8b48b2fbdc7897d6d42aa2f63f8c300517f6d9b8911346bf285438c6
fixture manifest         5e8bfa247c513d189774ec671265b2d3dc1ea97004e5e8c40baa090f26db3cad
fixture set              e6e51ae655e184f8f43b2afa9fe0c75041046966b4cdecd6fde008b02b684aa8
reference result         bee5727f68fc10ee047d666198b3f060f669058e966aa44802e270f90abbdeeb
fixture packet           6 files / 293319 bytes
binary carriers          3 files / 288000 bytes
reference tests          78 PASS / 0 FAIL
```

Build, self-test, and verify recompute from committed fixture bytes. Environment strings
are informational; numeric leaves use the frozen portable comparison tolerance.

Four independent read-only reviews passed on the exact source and packet identities.
The normalized record is `CATALYTIC_WAVE_LOOP_FINDINGS_NORMALIZED.json`; the full role
reports are `CATALYTIC_WAVE_LOOP_REVIEW_REPORTS.md`. All six initial material findings
are resolved and the open material finding count is zero.

## 11. Claim law

After four independent PASS reviews, zero open material findings, package-local
qualification, changed-path qualification, coherent commit, and branch push, this
package may emit only:

```text
AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED
```

Meaning only that the bounded ordinary-software lifecycle above closes under the frozen
contract. It does not establish:

```text
byte-exact carrier restoration
recursive catalytic Ising
optimization or solver advantage
physical audio or silicon-phononic computation
physical restoration
capacity or energy advantage
hardware bit replacement
Small Wall or Big Wall crossing
```

## 12. Contact boundary

```text
audio playback       0
audio recording      0
ADC/DAC              0
transducer            0
hardware contact      0
target contact        0
SSH/SCP               0
```

## 13. Next boundary

If the package closes, stop at:

```text
AUDIO_RECURSIVE_CATALYTIC_ISING_V1_CONTRACT
```

The successor is not started by this package.
