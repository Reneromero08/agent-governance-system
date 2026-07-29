# Bounded Two-Port Descriptor CATVM Focused Review

## Scope

This review covers only the bounded public-descriptor successor implemented by:

- `catvm_necklace_two_shared_latent_descriptor_service.cpp`
- `catvm_necklace_two_shared_latent_descriptor_protocol.py`
- `catvm_necklace_two_shared_latent_descriptor_controller.py`
- `TWO_SHARED_LATENT_DESCRIPTOR_PROGRAMS.json`
- `two_shared_latent_descriptor_oracle.py`

The reviewer inspected source and the first complete result independently from
the implementation owner. The independent Python oracle covers public
descriptor validation, stage-cut derivation, canonical topology checksums, and
reverse index order. It does not reimplement the full 285-necklace numerical
recurrence or establish separate compact-classical parity.

## Initial findings

The first review accepted the bounded mechanism but found three evidence
defects:

1. The response field recorded only streamed generator terms but the result
   called it `native_operations`, which was too broad.
2. Fresh/restored comparison checked checksum and boundary but not the exposed
   streamed-term resource signature.
3. The resource record omitted active bound-program custody objects and did
   not name all associated vector/plan allocation exclusions.

## Repairs

The result now calls the exposed value `streamed_generator_terms`. The
controller hard-fails if fresh and restored executions disagree on it.

The x86-64 service pins the relevant layouts with compile-time assertions:

```text
TwoPortModule          32 bytes
PortCustody            24 bytes
BoundTwoPortModule     88 bytes
DescriptorSlot        128 bytes
```

The resource record now includes the 384-byte three-slot registry, the
616-byte observed active bound-program peak, and the 704-byte eight-module
ceiling. It explicitly excludes unmeasured plan/topology vector capacity,
compiled/active vector capacity, allocator, native-library, OS, and total
process-peak memory.

## Recheck decision

```text
RECHECK: PASS
CLASSIFICATION: INDEPENDENTLY_VERIFIED_SOURCE_LOCAL
RESTORATION: NUMERICAL_PHYSICAL_STATE_RESTORATION
```

Source inspection supports restoration-before-response, staged disconnect
rollback, immutable sealed topology checks, full live tuple binding, program
identity custody, no accepted-path baseline reload, sequential restored
carrier reuse, and zero-valued hidden-stage boundaries. The result remains
bounded to three public slots, four-to-eight modules per slot, and three
executed families on the fixed 1,140-complex software carrier.

It does not establish a generic scheduler, arbitrary program algebra,
transferable numerical recurrence parity, a distinct phase resource,
computational advantage, Small Wall crossing, catalytic inference, physical
waveform execution, or replacement of physical bits with pi.
