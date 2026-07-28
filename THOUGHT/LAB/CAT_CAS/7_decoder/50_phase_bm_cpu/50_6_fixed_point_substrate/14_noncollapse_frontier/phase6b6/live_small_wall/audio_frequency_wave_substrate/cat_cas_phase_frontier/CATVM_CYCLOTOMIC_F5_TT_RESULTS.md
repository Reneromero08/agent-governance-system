# CATVM Cyclotomic Cubic TT Results

Evidence:

```text
/tmp/catvm-cyclotomic-f5-tt-repaired.MNFtp2/evidence
```

Accepted bounded claim:

```text
CATVM_ENFORCED_CYCLOTOMIC_CUBIC_TT_HIDDEN_BOND_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_REUSE
```

The controller receives exact primary and unrelated reuse amplitudes matching
the direct reference. The persistent carrier restores exactly after each
transaction, and restoration generations advance `1,2`. Requests and
responses remain fixed at 1,024 and 4,096 bytes. The controller imports only
the protocol module; a source/runtime module scan proves that it loads neither
the service nor the phase engine.

The service emits no stdout/stderr, owns a mode-`0600` socket, verifies peer
credentials, and sets itself non-dumpable. Direct `/proc/<pid>/mem`
inspection is denied. Intermediate projection, null carrier, snapshot on the
in-place service, and in-place execution on the snapshot service all fail
closed.

The accepted process has no saved carrier image. The separate snapshot sham
matches the primary boundary, charges 160 logical bytes each for image
creation, execution load, and restoration reload (480 total), reports
implementation-level Python resident sizes separately, reports
snapshot-loaded, and has restoration generation zero.

This establishes machine-enforced hidden bond/Fourier custody, actual
restoration, and same-carrier unrelated reuse. It does not improve the
underlying `4->14->64` rank obstruction, and the identical exact classical
TT remains the matched representation.
