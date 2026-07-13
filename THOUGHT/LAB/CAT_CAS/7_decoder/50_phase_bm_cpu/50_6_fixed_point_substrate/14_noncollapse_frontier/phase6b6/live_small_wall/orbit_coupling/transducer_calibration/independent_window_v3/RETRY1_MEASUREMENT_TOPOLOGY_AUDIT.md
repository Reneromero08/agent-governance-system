# Retry-One Measurement Topology Audit

## Scope

This audit binds the retained retry-one evidence root without modifying it:

`runs/balanced_transducer_confirmation_v2_1/`

The retained retry-one classification remains
`V1_PARTIAL_V2_TRANSFER_CANDIDATE`. This V3 package does not pool retry-one
measurement rows into V3 evidence, does not reinterpret retry-one as confirmed,
and does not modify any retained run artifact.

## Retry-One Topology Finding

Retry-one had strong q-ladder transfer structure, but its topology used the
prior one-encode/two-measure window shape. The V3 protocol repairs that
topology by requiring independent baseline, rebaseline, full source encoding,
single logical-bank measurement, restoration, and sentinel checks for each
positive and negative component. V3 derives `F(q)` only from raw component
windows and rejects any old runtime-derived delta fields.

## Read-Only Evidence Summary

Observed retry-one controller status:

- `CONFIRMATION_V2_CONTROLLER_TARGET_COMPLETE`

Observed retry-one final target status:

- `CONFIRMATION_V2_TARGET_COMPLETE`

Preserved adjudication status:

- `V1_PARTIAL_V2_TRANSFER_CANDIDATE`

Controller-reported evidence hashes:

- raw capture: `105a919bd18b22e038a676ce9c3b985bd5d82340282560035ce593cc1c19b631`
- restoration sentinels: `811d7379f43b193623a670e30c922a3d627e5317bf4f2c56bafbbec641022167`
- features: `40573a1f87be5b96fe0351c44e3d36c2ffec49335a3a9b00402dde9332b4e4b4`
- adjudication: `a29853a1bbeb1164122faef226d7374576228961133452eb9d9b06188a599bff`
- source bundle: `a670684fea5ed1e383e84923021fbb0b633557ade932e9501f3337bf6ae50517`
- live runtime binary: `0a0e9df6c7a87b478b01eac8c834f6d57123c5e49e6a16e5c665438339c6b0db`
- offline validation binary: `d187e5851282b85f2b9e05cc499d0e822b732e3beffe12ad78ba023209606876`

Read-only extracted retry-one scientific shape:

- aggregate worst nonzero pointer error: `0.15875370919881307`
- replicate 0 worst nonzero pointer error: `0.1328125`
- replicate 1 worst nonzero pointer error: `0.15875370919881307`
- aggregate q0 ceiling: `85.5`
- max held-out q0 logical mapping absolute value: `101.0`
- max held-out q0 logical pair residual: `102.0`
- max held-out q0 physical pair sum residual: `102.0`
- physical pair sum bound: `171.0`

Held-out q0 pair examples:

- replicate 0 pair 17: mapping0 `1.0`, mapping1 `-101.0`, residual `102.0`
- replicate 0 pair 23: mapping0 `1.0`, mapping1 `10.0`, residual `-9.0`
- replicate 1 pair 3: mapping0 `6.0`, mapping1 `-35.0`, residual `41.0`
- replicate 1 pair 22: mapping0 `-32.0`, mapping1 `33.0`, residual `-65.0`

## V3 Boundary

V3 is not a retry-one re-adjudication. V3 is a new, isolated prospective lane:

`independent_window_v3/`

It preserves retry-one custody and hashes, rejects V1/V2 evidence smuggling,
requires fresh evidence, and allows only the three V3 classes named in the V3
contract.
