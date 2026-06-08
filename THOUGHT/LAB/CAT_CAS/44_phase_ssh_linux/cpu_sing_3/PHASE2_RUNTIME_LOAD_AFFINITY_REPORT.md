# PHASE2_RUNTIME_LOAD_AFFINITY_REPORT

## Verdict

`RUNTIME_LOAD_AFFINITY_CHARACTERIZED`

The read-only load/affinity characterization ran successfully over SSH on the target.

The result changes the runtime interpretation: COFVID VID is not invariant under scheduler/load state. After reboot and `modprobe msr`, all cores reported P4 `MSRC001_0068` as the stock VID `0x1A`, but COFVID VID moved between `0x1A` and `0x12` depending on load state.

## Command

```powershell
Get-Content -Raw session_scripts\phase1_msr\msr_load_affinity_characterizer.py | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "python3 - --cores 0-5 --modes baseline,self_load,neighbor_load,all_load --samples 12 --delay 0.01 --jitter-iters 96"
```

The target required `modprobe msr` after reboot before `/dev/cpu/0/msr` existed.

## Safety

- Mode: `read_only_load_affinity_characterization`
- Writes flag: `false`
- No MSR writes
- No P-state definition writes
- No P-state transition requests
- No voltage writes
- No BIOS flash
- Load workers only use scheduler affinity and CPU busy loops

## Key Findings

| Mode | COFVID VID behavior | P4 `MSRC001_0068` behavior | Interpretation |
|---|---|---|---|
| `baseline` | Mixed. Core 0 stayed `0x12`; core 1 ranged `0x12-0x1A`; cores 2-5 stayed `0x1A`. | All cores P4 raw `0x8000013540003440`, VID `0x1A`, DID `1`, FID `0`. | Idle/scheduler state can expose the stock P4 VID. |
| `self_load` | All cores stayed `0x12`. | All cores P4 raw remained `0x8000013540003440`, VID `0x1A`. | Self load drives COFVID to non-P4 VID even while P4 definition is VID `0x1A`. |
| `neighbor_load` | All cores stayed `0x12`. | All cores P4 raw remained `0x8000013540003440`, VID `0x1A`. | Neighbor load also drives COFVID to VID `0x12`. |
| `all_load` | All cores stayed `0x12`. | All cores P4 raw remained `0x8000013540003440`, VID `0x1A`. | Full load keeps COFVID at VID `0x12`. |

## Runtime Interpretation

The decoded firmware path remains:

```text
constructor P4 field
  -> producer entry +0x04
  -> service +0x22
  -> rdmsr(0xC0010064 + pstate)
  -> P4 reads MSRC001_0068
```

The target run shows:

```text
MSRC001_0068 P4 definition:
  VID 0x1A on all cores after reboot

COFVID_STATUS:
  VID 0x1A possible during baseline/idle states
  VID 0x12 under self/neighbor/all-load states
```

So the earlier “VID floor” is not a simple invariant floor. It is load/scheduler-state dependent. The current software route is alive because runtime conditions alter the observed COFVID VID without firmware edits.

## Actionability

`RUNTIME_LOAD_AFFINITY_CHARACTERIZED` is met.

`CPU_SINGS` is not met.

`BYTE_READY_HUMAN_REVIEW` is not met.

`SOFTWARE_FIRMWARE_TRUE_WALL` is not met.

## Next Exact Action

Create a read-only transition/jitter experiment:

- pin sampler core and load cores
- capture COFVID transitions at higher sample rate
- compute transition timing and TSC jitter per mode
- test whether `0x1A <-> 0x12` transitions correlate with timing jitter, scheduler affinity, or PSTATE_STATUS

Acceptance:

- no MSR writes
- no voltage writes
- no P-state transition requests
- produces a table of transition counts and jitter deltas per core/mode
- only advances toward CPU_SINGS if a reproducible internal timing/phase signal appears
