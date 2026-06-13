# PHASE2_RUNTIME_MSR_OBSERVER_REPORT

## Verdict

`RUNTIME_MSR_OBSERVATION_COMPLETE`

The read-only runtime observer ran successfully over SSH on the target. It performed no MSR writes.

The decoded firmware path says the constructor P4 field is reconstructed from runtime `MSRC001_0068`. The target observation confirms that runtime status does not simply follow the P4 definition VID byte. COFVID status VID stayed fixed at `0x12` across all sampled cores, while P4 definitions were not uniform across cores.

## Command

Local command streamed the observer to the target without installing it:

```powershell
Get-Content -Raw 50_1_subthreshold_msr\src\msr_p4_readonly_observer.py | ssh -o BatchMode=yes -o ConnectTimeout=5 root@192.168.137.100 "python3 - --cores 0-5 --samples 100 --delay 0.02 --json"
```

## Safety

- Observer mode: `read_only`
- Writes flag: `false`
- No `wrmsr`
- No `/dev/cpu/*/msr` write-open
- No P-state transition request
- No voltage write
- No BIOS flash

## Target Reachability

SSH target responded:

```text
Linux catcas 6.12.86+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.86-1 (2026-05-08) x86_64 GNU/Linux
uid=0(root) gid=0(root) groups=0(root)
MSR_READABLE
```

## Summary Table

| Core | P4 raw | P4 DID | P4 VID | P4 approx freq | COFVID unique raw | COFVID VID range | COFVID DID/FID observed | PSTATE_STATUS unique |
|---:|---|---:|---:|---:|---|---|---|---|
| 0 | `0x8000013540003440` | 1 | `0x1A` | 800 MHz | `0x0180000140012410`, `0x0180000140022408`, `0x0180000140042440` | `0x12-0x12` | DID `0,1`; FID `0,8,16` | `0x0`, `0x1`, `0x3` |
| 1 | `0x8000013540003440` | 1 | `0x1A` | 800 MHz | `0x0180000140042440` | `0x12-0x12` | DID `1`; FID `0` | `0x3` |
| 2 | `0x8000013540003440` | 1 | `0x1A` | 800 MHz | `0x0180000140042440` | `0x12-0x12` | DID `1`; FID `0` | `0x3` |
| 3 | `0x80000135400024c0` | 3 | `0x12` | 200 MHz | `0x01800001400424c0` | `0x12-0x12` | DID `3`; FID `0` | `0x3` |
| 4 | `0x80000135400024c0` | 3 | `0x12` | 200 MHz | `0x01800001400424c0` | `0x12-0x12` | DID `3`; FID `0` | `0x3` |
| 5 | `0x8000013540003440` | 1 | `0x1A` | 800 MHz | `0x0180000140012410` | `0x12-0x12` | DID `0`; FID `16` | `0x0` |

## Observations

- Cores 0, 1, 2, and 5 report P4 definition `MSRC001_0068 = 0x8000013540003440`, decoded as DID `1`, FID `0`, VID `0x1A`.
- Cores 3 and 4 report P4 definition `MSRC001_0068 = 0x80000135400024c0`, decoded as DID `3`, FID `0`, VID `0x12`.
- COFVID status VID was `0x12` for every sampled core and every sample.
- COFVID status therefore did not expose the P4 definition VID `0x1A` on cores whose P4 definition still reports `0x1A`.
- The non-uniform P4 definitions on cores 3 and 4 are consistent with earlier runtime experiments leaving per-core P4 definition differences. This report does not modify them.

## Interpretation

`MSRC001_0068` is observable and per-core state is not uniform. The current runtime status path is dominated by VID `0x12` in this sample set, not by a firmware-static P4 VID byte.

This supports the firmware RE result:

```text
AGESA constructor P4 field
  -> producer entry +0x04
  -> service +0x22
  -> rdmsr(0xC0010064 + pstate)
  -> P4 reads MSRC001_0068
```

The active barrier is runtime behavior/control, not a discovered static AGESA P4 byte.

## Actionability

`RUNTIME_MSR_OBSERVATION_COMPLETE` is met.

`CPU_SINGS` is not met.

`BYTE_READY_HUMAN_REVIEW` is not met.

`SOFTWARE_FIRMWARE_TRUE_WALL` is not met because software-only runtime characterization remains live.

## Next Exact Action

Create a read-only runtime characterization pass that correlates:

- core affinity
- scheduler load state
- `MSRC001_0068` per-core raw value
- COFVID status raw value
- PSTATE_STATUS raw value
- TSC timing jitter

Acceptance: no writes, no external measurement, and enough samples to decide whether the COFVID VID `0x12` floor is invariant under scheduler/core/load conditions.
