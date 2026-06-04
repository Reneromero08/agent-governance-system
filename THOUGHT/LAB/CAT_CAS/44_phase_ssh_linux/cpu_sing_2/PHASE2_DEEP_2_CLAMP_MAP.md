# PHASE2_DEEP_2_CLAMP_MAP

## Verdict

CLAMP_MAP_READONLY_CLASSIFIED

## Fresh Snapshot

```text
hostname: catcas
isolated cores: 2-5
k10temp: +48.1 C
rdmsr: /usr/sbin/rdmsr
wrmsr: /usr/sbin/wrmsr
setpci: /usr/bin/setpci
```

P-state and COFVID:

```text
core2 PCTL 0 P4 8000013540003440 COFVID_CTL 40012410 COFVID_STS 180000140012410
core3 PCTL 3 P4 8000013540003440 COFVID_CTL 40043440 COFVID_STS 180000140042440
core4 PCTL 4 P4 8000013540003440 COFVID_CTL 40043440 COFVID_STS 180000140042440
core5 PCTL 0 P4 8000013540003440 COFVID_CTL 40012410 COFVID_STS 180000140012410
```

NB PCI focused read:

```text
F3xA0  a01a0800
F3xA4  308c0fef
F3xA8  88000000
F3xD4  c8810f26
F3xD8  03001315
F3xDC  0067641a
F3xE4  1dc01430
F3xE8  0207df19
F3xFC  00100fa0
```

## Fields Containing Clamp Candidates

- P4 stock CpuVid: `0x1A`.
- P4 stock NbVid: `0x20`.
- Runtime test requested CpuVid: `0x20`.
- Runtime COFVID_STS after P-state cycling stayed at CpuVid `0x1A`.
- `F3xA0 = 0xA01A0800` contains `0x1A`.
- `F3xDC = 0x0067641A` contains `0x1A`.
- `F3xD8 = 0x03001315`, `F3xE4 = 0x1DC01430`, and `F3xE8 = 0x0207DF19` contain additional VID-like byte fields.

## Interpretation

The clamp is visible at three layers:

1. P-state definitions expose P4 CpuVid `0x1A`.
2. Runtime lower-VID P4 definition writes read back but COFVID_STS still reports CpuVid `0x1A`.
3. NB PCI function 3 contains `0x1A` in fields already associated with power management.

The current evidence does not prove a safe writeable active-core field in NB PCI config. Existing BKDG interpretation says `F3xA0[PsiVid]` is a PSI_L threshold and `F3xDC` participates in P-state maximum and alternate/C-state behavior, not a proven active-core voltage control.

## Decision

No unknown PCI write is justified. Clamp mapping is strong enough to explain the runtime result, but not strong enough to authorize a new write test.

