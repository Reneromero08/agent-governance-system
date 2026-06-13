# Phase 6 Chiral PDN Native Rust Probe

Verdict: `NATIVE_PHYSICAL_CHANNEL_LIVE__PUBLIC_CHIRAL_NO_CROSSING`

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

| mode | n | trials | verdict | auc/null95 | best feature | pins |
|---|---:|---:|---|---:|---:|---|
| public_chiral_native | 8 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.542/0.686 | 1 | true/true |
| public_shuffle_null | 8 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.584/0.684 | 2 | true/true |
| hidden_chiral_control | 8 | 84 | PHYSICAL_HIDDEN_PREP_CHANNEL_LIVE | 1.000/0.688 | 2 | true/true |
| public_chiral_native | 10 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.549/0.694 | 2 | true/true |
| public_shuffle_null | 10 | 84 | PHYSICAL_PUBLIC_CHIRAL_NO_CROSSING | 0.557/0.680 | 1 | true/true |
| hidden_chiral_control | 10 | 84 | PHYSICAL_HIDDEN_PREP_CHANNEL_LIVE | 1.000/0.693 | 2 | true/true |

## Interpretation

The local native physical channel is real enough for the hidden pre-projection lane: when the sender binds the missing orientation into the balanced chiral drive, the receiver recovers it at `AUC=1.000` for both `n=8` and `n=10`.

The public chiral preparation did not cross. The public drive and public shuffle null remain below their shuffle-null gates. So on this local Ryzen run, the carrier is live but the public tape preparation still does not synthesize the missing fold-odd lane by itself.

This is still useful: it validates the physical protocol shape before moving the same idea to the Phenom after the long 300-trial matrix run finishes.

## Artifacts

- `chiral_pdn_native.rs`
- `results/chiral_pdn_native_result.json`
- `results/chiral_pdn_native_features.csv`

## Reproduction

```powershell
rustc -O THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\chiral_pdn_native.rs -o THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results\chiral_pdn_native.exe
THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results\chiral_pdn_native.exe THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\chiral_phase_kickback\rust_baremetal\results
```
