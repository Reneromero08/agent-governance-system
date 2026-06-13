"""Deterministic MSR readback probe.

Statistics note: this is a single hardware register readback/bit-decode pass,
not a population experiment. The reported voltages are decoded fields from one
snapshot; no empirical variance is claimed, so p_value/CI/std/effect size are
not applicable here.
"""

import struct, os

print("=== PATH A: DECODE AND PROBE HWCR BITS ===")
with open("/dev/cpu/0/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010015, 0)
    hwcr = struct.unpack("<Q", f.read(8))[0]

print(f"HWCR (0xC0010015): 0x{hwcr:016x}")
print(f"  Bit 0 (TscEn): {hwcr & 1}")
print(f"  Bit 4 (TscInvariant): {(hwcr >> 4) & 1}")
print(f"  Bit 16 (MonarchCore): {(hwcr >> 16) & 1}")
print(f"  Bit 24 (unknown but set): {(hwcr >> 24) & 1}")
for b in [18, 19, 20, 21, 22, 23]:
    print(f"  Bit {b}: {(hwcr >> b) & 1}")

print()
print("=== PATH B: NORTHBRIDGE P-STATE MSR ===")
for addr in [0xC0010070, 0xC0010072, 0xC0010074]:
    try:
        with open("/dev/cpu/0/msr", "rb") as f:
            os.lseek(f.fileno(), addr, 0)
            val = struct.unpack("<Q", f.read(8))[0]
        print(f"MSR 0x{addr:08x}: 0x{val:016x}")
    except Exception as e:
        print(f"MSR 0x{addr:08x}: FAIL ({e})")

print()
print("=== PATH C: COFVID VID vs PSTATE VID ===")
with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010071, 0)
    cofvid = struct.unpack("<Q", f.read(8))[0]
    cur_vid = (cofvid >> 16) & 0x3F
    print(f"Core 4 COFVID: 0x{cofvid:016x}")
    print(f"  Current VID: 0x{cur_vid:02x} = {cur_vid} decimal")
    print(f"  Current voltage: {1.55 - cur_vid * 0.0125:.4f}V")

with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010067, 0)
    pstate3 = struct.unpack("<Q", f.read(8))[0]
    pstate3_vid = (pstate3 >> 16) & 0x3F
    print(f"P-state 3 definition: 0x{pstate3:016x}")
    print(f"  Defined VID: 0x{pstate3_vid:02x}")
    print(f"  Defined voltage: {1.55 - pstate3_vid * 0.0125:.4f}V")
    print(f"  VID match: {cur_vid == pstate3_vid}")
