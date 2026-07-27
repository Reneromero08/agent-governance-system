# CATVM Width-Two Open Intermediate Composition

## Status

This mutable checkpoint establishes:

```text
CATVM_WIDTH2_OPEN_INTERMEDIATE_CONTRACT2_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND
```

for the fixed tested two-`CONTRACT2` transaction behind an enforced Linux
userspace machine boundary.

It encloses the previously qualified direct width-two relational engine. It
does not replace or weaken that engine, and it does not establish general
CATVM hardware or compact separator closure.

## Logical machine boundary

The carrier-owning service is a separate same-UID process reached only through
an `AF_UNIX/SOCK_SEQPACKET` protocol. Before allocating the carrier it:

```text
disables core dumps
sets PR_SET_DUMPABLE=0
clears PR_SET_PTRACER
sets PR_SET_NO_NEW_PRIVS
```

The complete machine state is one private, `mlock`ed,
`MADV_DONTDUMP`/`MADV_DONTFORK` mapping. After accepting its one client, the
service installs a seccomp allowlist. The controller cannot access the phase
core and is not linked to the `CONTRACT2` kernel or phase decode.

The fixed protocol is:

```text
HELLO
SEAL <48 public F3 coefficients>
F
G
PROJECT Z
RESTORE
SHUTDOWN
```

`PROJECT Y` returns only:

```text
E_INTERMEDIATE_PROJECTION_DENIED
```

Unknown read, dump, debug, snapshot, and state-detail commands, embedded NUL
requests, and oversized packets fail through fixed protocol responses.

## Resident transaction

The protected carrier layout is:

```text
F public encoding       cells  0..15
G public encoding       cells 16..31
K public encoding       cells 32..47
actual resident H       cells 48..63
actual resident Z       cells 64..79
public boundary         cells 80..95
```

The first command applies:

```text
H = CONTRACT2(F,G)
```

and retains `H` only in the actual 16 carrier cells. The second applies:

```text
Z = CONTRACT2(actual H,K)
```

The controller cannot select addresses or request an internal coefficient.
It receives only a typed custody receipt after `F` and only the 16 final
boundary coefficients after `PROJECT Z`.

The controller does not contain the phase kernel, decode, a classical
coefficient oracle, expected final vectors, or an answer-bearing table. The
separate qualifier compares the emitted final vectors with the already
qualified direct branch-native engine after the transaction.

## Protected CONTRACT2 workspace

All 240 complex workspace values are fields of the locked machine mapping:

```text
left and right relations                32
two squared relations                   32
lifted six-variable intersection        64
first norm output                       32
final norm output                       16
two maximum restriction buffers         64
total                                  240
```

Every `CONTRACT2` call starts with an exact zero-workspace check, computes
through branch-native phase symbol algebra, applies its output factor to the
target carrier cells, and securely wipes all 240 values. Restoration evidence
gates that workspace as exactly clear.

## Actual inverse and canonical equality

After projecting the result, the in-place path:

1. removes the public boundary using the actual resident `Z`;
2. recomputes `CONTRACT2(actual H,K)` and applies its inverse to `Z`;
3. removes K;
4. recomputes `CONTRACT2(actual F,G)` and applies its inverse to `H`;
5. removes G and F; and
6. clears program, open-port, resident-message, pending-operation, and
   morphism-stack state.

Canonical restoration checks:

```text
all 96 carrier cells against baseline at 2e-12
carrier unit-modulus integrity
baseline digest
topology digest
compiled-morphism digest
immutable lease identity
carrier creation count
restoration generation +1
empty morphism stack
zero resident-internal-message count
closed boundary
zero program
zero pending operation
zero 240-value workspace
snapshot absence or cleared validity
cleared protocol receive buffer
empty backend receive queue
```

Host allocator addresses, process identifiers, and unrelated wall time are
not part of equality.

The result survives in the controller's ordinary final-boundary response
while the service restores the carrier.

## Demonstrated result

The service reproduces the direct width-two engine:

```text
primary [1,0,2,1,2,1,2,1,0,0,1,1,1,1,1,1]
reuse   [1,0,2,1,0,0,1,1,2,1,2,1,1,1,1,1]
```

The accepted run uses one carrier allocation for 258 transactions:

```text
primary transaction                        1
unrelated reuse transaction                1
alternating reuse transactions           256
maximum restoration error    8.61764809305e-16
predeclared tolerance                     2e-12
```

The six same-UID process-access attacks run while `H` is resident:

```text
/proc/<pid>/mem
/proc/<pid>/maps
/proc/<pid>/fd/0
process_vm_readv
ptrace
pidfd_getfd
```

All are denied.

## Controls

Qualification requires:

```text
wrong parent inverse
missing parent inverse
noncommuting child-before-parent inverse
attempted H projection
null carrier
snapshot-backed comparison
different-program actual restored-carrier reuse
256-cycle alternating reuse drift sentinel
strict protocol adversaries
same-UID process inspection attacks
service stdout/stderr byte-zero checks
controller binary phase-symbol exclusion
exact output-key allowlists
direct branch-native boundary parity
fresh-process deterministic replay
strict GCC and static analysis
ASan, UBSan, and leak detection
regression of the original four-cell CATVM proof
source and result hashes
focused independent review
```

Wrong, missing, and reordered inverses each leave restoration error
`1.73205080757`. Reordering is applicable for the selected noncommuting pair.
All three controls retain exact invariant metadata and a cleared workspace
while correctly failing carrier restoration.

The snapshot baseline maps an additional 4,096 locked bytes, writes and
reloads 1,536 carrier bytes per transaction, performs no actual inverse, and
supports only:

```text
CATVM_WIDTH2_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED
```

## Resource accounting

Per accepted transaction:

```text
carrier cells                               96
baseline + working complex values          192
locked CONTRACT2 workspace values           240
public program coefficients                  48
compiled morphism descriptor bytes           16
resident H cells                             16
boundary decodes                             16
forward CONTRACT2 calls                       2
inverse CONTRACT2 calls                       2
symbol products                           7,168
coefficient accumulation additions        7,168
restriction/intersection additions          448
phase-cell updates                           192
inverse-factor recomputations                  2
restoration-cell checks                       96
retained inverse factors                       0
```

The in-place service locks 8,192 bytes. The snapshot service locks 12,288
bytes. The 258-transaction accepted evidence accounts for:

```text
CONTRACT2 calls                         1,032
symbol products                     1,849,344
phase-cell updates                      49,536
request / response packets        1,557 / 1,557
request / response bytes        33,594 / 354,483
```

One uncontrolled warm run measured:

```text
direct-process phase       795,698 ns / transaction
isolated inert boundary     55,562 ns / transaction
snapshot CATVM             477,140 ns / transaction
in-place inverse CATVM     868,223 ns / transaction
```

These timings establish cost accounting, not an advantage. Snapshot avoids
the inverse work and is not semantically equivalent to the accepted path.

## Claim boundary

The accepted claim is limited to:

```text
one fixed two-CONTRACT2 topology
two shared Boolean ports per contraction
one 16-cell unresolved intermediate
one same-UID Linux userspace controller/service boundary
software phase simulation
```

It does not establish compact separator storage, factor-preserving full
relation closure, arbitrary interface width or topology, arbitrary CATVM
programs, multi-client isolation, root/kernel or microarchitectural secrecy,
computational advantage, physical waveform or silicon execution, Small Wall
crossing, or unlimited catalytic computation.

## Reproduction

```bash
evidence_dir=$(mktemp -d /tmp/catvm-wide2-qual.XXXXXX)
./qualify_catvm_wide2_phase.sh . "$evidence_dir" \
    >"$evidence_dir.summary.json"
```
