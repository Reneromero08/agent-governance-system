# Phase 6 Chiral PDN Native Rust Probe

Verdict: `NATIVE_PHYSICAL_CHANNEL_LIVE__MICROSTEP_RAMP_NO_CROSSING`

## Machine

- Host: local Windows machine
- CPU: AMD Ryzen 9 5900X, 12 cores / 24 logical processors
- Toolchain: `rustc 1.95.0`
- Sender logical CPU: 4
- Receiver logical CPU: 5

## Question

Can a native physical timing carrier on this machine recover the fold orientation when the sender is prepared with a public chiral phase-kickback pattern?

## Harness

`chiral_pdn_native.rs` runs two pinned native Rust threads:

- sender: drives a balanced high/low compute-load pattern derived from the tape/oracle instance.
- receiver: measures per-slot timing throughput and converts the timing trace into lock-in features.
- public modes: pattern is a pure function of public `k`, `b`, and `N`.
- hidden control: the same balanced pattern is flipped by the hidden pre-projection orientation lane.

No voltage, firmware, MSR writes, or privileged hardware changes are used.

## Results

### Carrier Check

| mode | n | trials | verdict | auc/null95 | best feature | pins |
|---|---:|---:|---|---:|---:|---|
| public_chiral_native | 8 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.620/0.690 | 1 | true/true |
| public_shuffle_null | 8 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.565/0.673 | 1 | true/true |
| hidden_chiral_control | 8 | 84 | PHYSICAL_HIDDEN_PREP_CHANNEL_LIVE | 1.000/0.676 | 2 | true/true |
| public_chiral_native | 10 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.571/0.675 | 2 | true/true |
| public_shuffle_null | 10 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.583/0.680 | 1 | true/true |
| hidden_chiral_control | 10 | 84 | PHYSICAL_HIDDEN_PREP_CHANNEL_LIVE | 1.000/0.686 | 2 | true/true |

### One-Bit Candidate Search

The search loop first recovers the fold magnitude `a = min(d, N-d)` from the public cosine tape, then drives both candidate signs (`a` and `N-a`) and asks whether the true candidate has stronger receiver response than the false candidate.

| mode | n | instances | fold magnitude exact | verdict | true/false mean | auc/null95 | pins |
|---|---:|---:|---:|---|---:|---:|---|
| candidate_search_public | 8 | 84 | 1.000 | ONE_BIT_SEARCH_NO_CROSSING | 0.982/0.980 | 0.566/0.570 | true/true |
| candidate_search_hidden_control | 8 | 84 | 1.000 | ONE_BIT_SEARCH_GATE_LIVE | 0.982/-0.983 | 1.000/0.589 | true/true |
| candidate_search_public | 10 | 84 | 1.000 | ONE_BIT_SEARCH_NO_CROSSING | 0.982/0.983 | 0.494/0.573 | true/true |
| candidate_search_hidden_control | 10 | 84 | 1.000 | ONE_BIT_SEARCH_GATE_LIVE | 0.986/-0.980 | 1.000/0.569 | true/true |

### Fractional Microstep Ramp

The fractional ramp sweeps eight microsteps between integer tape positions over 128 integer positions. It tests two questions:

- direction: can the receiver distinguish forward versus reverse microstep traversal when the public endpoints are the same?
- search: after recovering `a = min(d,N-d)`, can the true candidate sign beat the false candidate sign?

| mode | n | trials | fold magnitude exact | verdict | mean delta | auc/null95 | pins |
|---|---:|---:|---:|---|---:|---:|---|
| fractional_microstep_direction_public | 8 | 168 | 1.000 | MICROSTEP_DIRECTION_NOT_RESOLVED | 0.0067 | 0.505/0.593 | true/true |
| fractional_microstep_direction_public | 10 | 168 | 1.000 | MICROSTEP_DIRECTION_NOT_RESOLVED | 0.0001 | 0.518/0.570 | true/true |
| fractional_microstep_search_public | 8 | 168 | 1.000 | MICROSTEP_SEARCH_NO_CROSSING | -0.0034 | 0.505/0.559 | true/true |
| fractional_microstep_search_hidden_control | 8 | 168 | 1.000 | MICROSTEP_SEARCH_GATE_LIVE | 1.9788 | 1.000/0.585 | true/true |
| fractional_microstep_search_public | 10 | 168 | 1.000 | MICROSTEP_SEARCH_NO_CROSSING | -0.0003 | 0.468/0.586 | true/true |
| fractional_microstep_search_hidden_control | 10 | 168 | 1.000 | MICROSTEP_SEARCH_GATE_LIVE | 1.9800 | 1.000/0.560 | true/true |

## Interpretation

The local native physical channel is real enough for the hidden pre-projection lane: when the sender binds the missing orientation into the balanced chiral drive, the receiver recovers it at `AUC=1.000` for both `n=8` and `n=10`.

The public chiral preparation did not cross. The public drive and public shuffle null remain below their shuffle-null gates. So on this local Ryzen run, the carrier is live but the public tape preparation still does not synthesize the missing fold-odd lane by itself.

The one-bit search also did not cross. The fold magnitude stage is perfect (`1.000` exact at both sizes), and the hidden candidate-search control is live (`AUC=1.000`). But when the two candidate signs are driven from public data, true and false candidate response are indistinguishable: `0.982` vs `0.980` at n=8 and `0.982` vs `0.983` at n=10. The carrier transports a sign when one is bound into it; this run did not show the public tape/physics selecting the true sign over the false sign.

This is still useful: it validates the physical protocol shape and closes the exact 1-bit search variant on this local Ryzen run before moving any cleaner version to the Phenom after the long matrix run finishes.

The fractional microstep ramp did not generate a public chiral lane on this machine. Forward versus reverse public microstep traversal was not physically resolved above null, and the public true-vs-false sign search stayed at chance. The hidden-control microstep search is live at `AUC=1.000`, so the negative public result is not an inert scorer.

## Artifacts

- `chiral_pdn_native.rs`
- `results/chiral_pdn_native_result.json`
- `results/chiral_pdn_candidate_search_result.json`
- `results/chiral_pdn_microstep_result.json`
- `results/chiral_pdn_native_features.csv`
- `results/chiral_pdn_candidate_search.csv`
- `results/chiral_pdn_microstep.csv`

## Reproduction

```powershell
rustc -O THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\chiral_pdn_native.rs -o THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results\chiral_pdn_native.exe
THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results\chiral_pdn_native.exe THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results
```

Microstep-only run:

```powershell
THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results\chiral_pdn_native.exe THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results microstep-only
```
