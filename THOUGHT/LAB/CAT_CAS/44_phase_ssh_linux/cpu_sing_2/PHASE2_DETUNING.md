# PHASE2_DETUNING

## Verdict

DETUNING_SIGNAL_NOT_REPRODUCIBLE

## Protocol

Core4 was fixed at DID3. Core3 was swept through DID0-DID4, two repeats per DID. Each run used the same active probe with shared-line pressure and Core5 reference. P4 definitions were rolled back to stock after the sweep.

Thermal and rollback evidence:

```text
detune2 precheck temp: +48.6 C
post rollback temp: +47.2 C
core3 after P4: 8000013540003440
core4 after P4: 8000013540003440
```

COFVID confirmed DID changes:

```text
DID0 core3 COFVID 180000140042400
DID1 core3 COFVID 180000140042440
DID2 core3 COFVID 180000140042480
DID3 core3 COFVID 1800001400424c0
DID4 core3 COFVID 180000140042500
Core4 fixed COFVID 1800001400424c0
```

## Results

```text
detune_did0 n=2 k=0.6412 p34=0.0802 corr=-0.0013
detune_did1 n=2 k=0.6379 p34=0.0826 corr=0.0119
detune_did2 n=2 k=0.6393 p34=0.0811 corr=0.0187
detune_did3 n=2 k=0.6322 p34=0.0925 corr=-0.0030
detune_did4 n=2 k=0.6352 p34=0.0824 corr=-0.0053
```

Null comparison:

```text
detune real_k_mean 0.6372
detune shuf_k_mean 0.6372
detune real_p34_mean 0.0837
detune shuf_p34_mean 0.0797
detune real_corr34_mean 0.0042
detune shuf_corr34_mean 0.0039
```

## Decision

No reproducible detuning signal with null separation was found. DID control works, but it did not produce an accepted Phase 2 coupling signal.

