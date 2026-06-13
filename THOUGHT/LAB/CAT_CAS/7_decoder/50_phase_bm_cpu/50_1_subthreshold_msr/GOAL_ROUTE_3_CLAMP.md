# GOAL_ROUTE_3_CLAMP

## Verdict

CLAMP_LOCALIZED_READONLY

## Read-Only NB Dump

Command:

```sh
ssh root@192.168.137.100 'lspci -s 00:18.0-4; setpci 00:18.0-4 0x00-0xfc read-only dump'
```

Focused evidence:

```text
00:18.0 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Family 10h Processor HyperTransport Configuration [1022:1200]
00:18.1 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Family 10h Processor Address Map [1022:1201]
00:18.2 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Family 10h Processor DRAM Controller [1022:1202]
00:18.3 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Family 10h Processor Miscellaneous Control [1022:1203]
00:18.4 Host bridge [0600]: Advanced Micro Devices, Inc. [AMD] Family 10h Processor Link Control [1022:1204]

00:18.3:
a0 a01a0800
d8 03001315
dc 0067641a
e8 0207df19
fc 00100fa0
```

## Reconciliation

The earlier raw bytes and current `setpci` now agree for F3xA0:

- `setpci -s 00:18.3 a0.l` = `a01a0800`.
- Little-endian byte view = `00 08 1a a0`.
- The visible `0x1A` field is present in the live config value.

No PCI writes were performed.

## Candidate Fields

- `F3xA0 = 0xA01A0800`: contains the visible `0x1A` clamp-floor candidate.
- `F3xDC = 0x0067641A`: contains another low byte `0x1A`; existing evidence ties this area to alternate/P-state VID parameters, but not enough to authorize writes.
- `F3xD8 = 0x03001315`, `F3xE8 = 0x0207DF19`: VID-like fields exist but remain read-only candidates.

## Decision

Clamp localization is strong enough for route triage, but no unknown NB PCI write is authorized. The runtime VID clamp result is authoritative for software MSR behavior.

