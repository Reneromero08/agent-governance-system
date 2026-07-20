# Held-Out Catalytic Generalization Focused Review

**Reviewer:** `/root/heldout_catalytic_generalization_review`

**Exact reviewed candidate root:**
`f99e438c0414c952f756f330263457ffd28d7be0251ae44d577ee11d07cb7210`

**Candidate identity:** `11 files / 78,953 bytes`

**Verdict:** `PASS`

**Authorized decision:**
`CATALYTIC_WAVEFORM_ISING_HELD_OUT_GENERALIZATION_PARTIAL`

**Normalized findings:** `P0 0; P1 0; P2 0; open 0`

The candidate root excludes this final review receipt and the structured normalized
findings receipt. It is the SHA-256 of path-sorted records:

```text
relative_path<TAB>byte_count<TAB>file_sha256<LF>
```

The root reproduced exactly before and after the read-only review.

## Custody and ordering

The reviewer confirmed that freeze commit
`c5da993afd20649bbf0413dda23b22e8c9c7bb45` directly follows required starting head
`62e8dab8c8631d112122a6e43cb9dcd7a4985bee` and adds only the freezer, custody JSON, and
freeze note. The deterministic rule reproduced instance SHA-256
`49db989fd525366867cf9c6866ebc7000b531b438b0227d7bb919e0ff3bf2704`.

The first review found one P1 evidence-ordering defect: raw-shadow energy first appeared
after oracle adjudication. The repaired source now computes and seals that energy
immediately after boundary projection. Event 2 binds raw signs
`[-1,+1,-1,+1,-1]` and energy `-12.5` with payload SHA-256
`1f552ef69ce5ac56939adc6bd3b515a51293a7887d887529e1b4f8af3b83b242`.
Oracle opening remains event 9. The P1 is closed.

## Independent mechanism findings

The reviewer manually traced all thirteen native-path functions. `J,h` act directly on
complex fields and inter-site waveforms. No decoded spin, scalar `J@s`, energy, oracle,
optimum, winner, score, cached answer, latch feedback, matrix multiply, or dynamic
evaluation enters native evolution.

The variable-`J` adapter reproduced predecessor primary and reuse displaced waveforms,
complete operator histories, and query frames with six exact zero deltas. Frozen machine
identity remained
`cf95d0cd364af38d47a2f2784aa489ab5a52dc8aea62131c1a8545ff4978203a`.

Independent reproduction confirmed:

```text
raw signs / raw energy            [-1,+1,-1,+1,-1] / -12.5
unique oracle optimum / energy    [-1,+1,-1,+1,-1] / -12.5
oracle gap                        2.0
minimum coherence                 0.515990928533
frozen coherence threshold        0.90
accepted projection               no

held-out displacement L2          42.2835813308
held-out restoration error        1.17416254896e-14
reuse input error                 0.0
reuse result / energy             [+1,+1,+1,+1,-1] / -14.0
reuse restoration error           1.09468215425e-14
```

All geometry, transform, phase-operator, lock, query, carrier-content, samplewise, and
inverse controls retained material history/response deltas and passed. Deterministic
verification remained `13/15`; only the two declared scientific acceptance tests fail.

## Adjudication

`PARTIAL` is required. Catalytic execution, causality, geometry sensitivity, restoration,
reuse, and raw optimum agreement are established for the frozen instance, but the
unchanged boundary rejects the result at its frozen coherence threshold. No threshold or
machine parameter may be moved after observing that result.

The claim remains bounded to ordinary-software reference evidence. No physical, P0,
performance, scaling, hardware-bit-replacement, or Wall claim is authorized.

The review was read-only and made no file, Git, hardware, audio, network, procurement,
fabrication, or physical contact.
