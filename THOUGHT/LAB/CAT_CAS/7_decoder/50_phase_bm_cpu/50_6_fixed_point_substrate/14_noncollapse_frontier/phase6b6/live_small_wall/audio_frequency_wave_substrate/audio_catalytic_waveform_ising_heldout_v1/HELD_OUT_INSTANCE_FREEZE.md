# Held-Out Catalytic Ising Instance Freeze

**Starting head:** `62e8dab8c8631d112122a6e43cb9dcd7a4985bee`

**Predecessor source SHA-256:**
`50b6db77e2602e18356636ddb892f6d51aedb0573c6b2418afc8e5cc174991cc`

**Held-out instance SHA-256:**
`49db989fd525366867cf9c6866ebc7000b531b438b0227d7bb919e0ff3bf2704`

**Custody file SHA-256:**
`577a2302fc98290da23e48fea87a72abc017a2d8f62e4c1cfd1c645a47b77720`

The public deterministic generation rule and seed are frozen in
`HELD_OUT_INSTANCE_CUSTODY.json` and reproduced by `heldout_instance_freezer.py`.

```text
J =
[[ 0,  1,  1, -2,  2],
 [ 1,  0, -2,  2,  1],
 [ 1, -2,  0, -2,  1],
 [-2,  2, -2,  0, -1],
 [ 2,  1,  1, -1,  0]]

h = [-2, -1, 0.5, 0.5, -0.5]
```

At this freeze boundary:

```text
native waveform evolution executed             no
result boundary projected                      no
exact oracle consulted                         no
expected optimum observed                      no
machine parameter changed for this instance    no
```

The next commit may record execution evidence, but it may not change this instance or
the verified machine constants and still call the experiment held out.
