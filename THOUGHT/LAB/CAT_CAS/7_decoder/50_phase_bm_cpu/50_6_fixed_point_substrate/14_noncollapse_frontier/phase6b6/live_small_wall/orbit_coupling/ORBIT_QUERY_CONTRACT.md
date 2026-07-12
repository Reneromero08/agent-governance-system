# OrbitState Public Query Contract

Date: 2026-07-12

## Claim Ceiling

This contract can support at most:

```text
ORBITSTATE_PHYSICAL_QUERY_COUPLING_CANDIDATE
```

It cannot emit `SMALL_WALL_CROSSED`.

## Public Mathematical Law

Public modulus:

```text
N = 256
```

Private source members for the primary fold pair:

```text
d = 23
fold(d) = N - d = 233
```

The source owns the private object:

```c
typedef struct {
    uint32_t modulus;
    uint32_t member;
} OrbitState;
```

The public quadrature phases are frozen:

```text
0
pi/2
pi
3pi/2
```

The pre-projection source response is:

```text
r_theta(d) = cos(2*pi*d/N - theta)
```

The post-projection control may compute only:

```text
x(d) = cos(2*pi*d/N)
```

and may then apply public phase weights using `x` alone. It must not read the private
member again.

The frozen decoder is:

```text
Z = (2/K) * sum_k response_k * exp(i*theta_k)
```

The primary fold-odd coordinate is `Im Z` for the configured sign convention.

## Physical Encoding Law

The source quantizes:

```text
q_theta = fixed_quantize(r_theta(d))
```

and maps it into two public physical banks:

```text
positive_work = M + q_theta
negative_work = M - q_theta
positive_work + negative_work = 2M
```

Frozen parameters:

```text
M = 2048 work units
quantization scale = 1024 work units
operator = byte-preserving same-value store
home/source core = 4
receiver/sentinel core = 5
bank lines = 4096
line bytes = 64
```

The maximum measured-bank source work is `M + scale = 3072`, which is below the
4096-line bank size. This prevents the encoder from collapsing to a full-bank sweep.

Each captured phase record uses fresh anonymous zero-page carrier banks. The runtime
must not initialize or restore the measured banks with a full-bank write sweep inside
the capture matrix. Byte restoration is checked by comparing final measured-bank bytes
with the computed zero baseline after measurement.

The private member may affect only the declared balanced work allocation. It must not
affect filenames, retries, process topology, logging shape, output schema, allocation
size, public phase order, or the number of total source work units.

The `source_off` control preserves process lifecycle and total source operation count
using an experiment-owned dummy bank that is never measured by the receiver. The
measured positive and negative banks receive zero source-side work in that control.

## Receiver Boundary

The receiver and feature extractor may see:

- opaque run IDs;
- public phase ordinal, which may be a label permutation;
- public decoder phase;
- fixed operator and geometry identifiers;
- positive-bank and negative-bank physical responses;
- restoration and process-custody status.

Before feature freezing, the receiver and feature extractor must not see:

- branch identity;
- orientation label;
- plus/minus label;
- target identity;
- member value;
- member-is-lower-half predicate;
- condition label;
- quantized private response;
- per-phase positive/negative/dummy source work allocation;
- public-query phase schedule or public-label swaps that would identify controls.

The feature extractor must hash `FEATURES_FROZEN.json` and write
`FEATURE_FREEZE_RECEIPT.json` before reading `UNBLINDING_MAP.json`. The
unblinding gate must rehash the current feature file and compare it with the
stored feature hash, current manifest hash, and current raw-capture hash before
loading private maps.

The target imports only the public extractor before feature freeze. The private
model/adjudicator is loaded only after the freeze receipt exists and the current
feature, manifest, and raw hashes match the receipt.

For each opaque run ID, the receiver must bind the raw `phase_ordinal` and
`decoder_phase_index` exactly to the corresponding manifest record before decoding.

## Capture Artifacts

The live run must create:

```text
CAPTURE_MANIFEST.json
UNBLINDING_MAP.json
RAW_CAPTURE.jsonl
FEATURES_FROZEN.json
FEATURES_FROZEN.sha256
FEATURE_FREEZE_RECEIPT.json
ADJUDICATION.json
FINAL_RESULT.json
```

`CAPTURE_MANIFEST.json` contains opaque group IDs and public execution parameters.
`UNBLINDING_MAP.json` contains private members, condition identities, expected
work allocation, and the source-owned work receipt. It is read only after the
feature freeze receipt exists and matches the current frozen feature file.

## Controls Frozen Before Live Execution

The first-light matrix contains:

- pre-projection member `d`;
- pre-projection member `N-d`;
- source off;
- query off;
- post-projection;
- declaration sham;
- query scramble using public source schedule `0, pi, 0, pi`;
- equal-orbit odd-zero control using member `0`;
- physical bank swap;
- public label swap;
- two fresh-process replicates of the full matrix.

The primary candidate law is:

- `Im Z(d)` and `Im Z(N-d)` have opposite sign;
- magnitudes are balanced within the frozen tolerance;
- `min(abs(Im Z(d)), abs(Im Z(N-d)))` exceeds every null-control ceiling;
- `Re Z(d)` and `Re Z(N-d)` match within the frozen tolerance;
- post-projection, source-off, query-off, declaration-sham, query-scramble, and
  equal-orbit controls are fold-odd null under the frozen ceiling;
- bank swap follows `Z_bank_swap ~= -Z_d`;
- public label swap permutes public phase ordinals while preserving decoder phase
  indices, and follows `Z_label_swap ~= Z_d`;
- restoration passes for every opaque group;
- both fresh-process replicates satisfy the same sign law;
- both fresh-process replicates contain exactly one feature group for each
  frozen condition;
- the source-owned work receipt exactly matches the frozen expected allocation
  for every condition and phase;
- the adjudicator independently recomputes expected allocation, including
  effective `q_theta`, from the unblinded condition, member, mode, phase, and
  bank-swap law before comparing producer expected rows and actual source receipts;
- measured-bank source work is always below the 4096-line bank size, with
  `source_off` using only the unmeasured dummy bank and `query_off` using
  balanced `M/M` measured-bank work.
- temperature is below the veto threshold before the run, before each replicate,
  after each replicate, and before final target success is emitted.

Frozen tolerances:

```text
null multiplier = 3.0
relative balance tolerance = 0.35
relative real-coordinate tolerance = 0.35
absolute floor = 1.0 measured unit
```
