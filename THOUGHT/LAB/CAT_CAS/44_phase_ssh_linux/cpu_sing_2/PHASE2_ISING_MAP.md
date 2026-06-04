# PHASE2_ISING_MAP

## Verdict

ISING_ROUTE_NOT_OBSERVED

## Encoding

Candidate spin states were encoded as workload/frequency choices:

- `lcg/lcg`
- `atomic/atomic`
- `branch/branch`
- `mul/mem`
- Core3 DID0-DID4 against Core4 DID3

Candidate energy:

```text
E = -phase34_r
```

This rewards Core3/Core4 phase alignment and rejects static idle as a valid spin solve.

## Result

Best non-idle active workload phase34 r was only about `0.0806`. Best detuning phase34 r was about `0.0925` at DID3, with weak separation and no descent process beyond selecting the maximum after measurement.

```text
active best: branch_shared p34=0.0806
detune best: detune_did3 p34=0.0925
null p34 mean was comparable after shuffling
```

## Decision

No Ising-style energy descent or constraint solving beyond null was demonstrated. This route is rejected for Phase 2 success.

