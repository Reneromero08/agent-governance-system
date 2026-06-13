"""Deterministic VID/NB state inventory.

Statistics note: this is a one-snapshot register inventory and derived voltage
decode. It reports exact fields, not repeated-trial inference; p_value/CI/std/
effect size are not applicable here.
"""

import struct, os, time

print("=== ATTACK 1: P-STATE LIMIT ===")
try:
    fd = os.open("/dev/cpu/4/msr", os.O_RDWR)
    os.lseek(fd, 0xC0010061, 0)
    old = struct.unpack("<Q", os.read(fd, 8))[0]
    print(f"Current limit: 0x{old:016x}")
    print("(Write-locked - skipping)")
    os.close(fd)
except Exception as e:
    print(f"FAIL: {e}")

print()
print("=== ATTACK 2: NORTHBRIDGE VOLTAGE ===")
fd = os.open("/dev/cpu/0/msr", os.O_RDONLY)
os.lseek(fd, 0xC0010070, 0)
nb = struct.unpack("<Q", os.read(fd, 8))[0]
os.close(fd)
print(f"NB COFVID: 0x{nb:016x}")
nbv = (nb >> 9) & 0x7F
print(f"  NB VID: 0x{nbv:02x} = {1.55 - nbv * 0.0125:.4f}V")
for off in [0.05, 0.10, 0.15, 0.20]:
    print(f"  Core min if {off:.2f}V offset: {1.55 - nbv * 0.0125 - off:.4f}V")

print()
print("=== ATTACK 3: CORE 4 ACTUAL STATE ===")
fd = os.open("/dev/cpu/4/msr", os.O_RDONLY)
os.lseek(fd, 0xC0010071, 0)
c4 = struct.unpack("<Q", os.read(fd, 8))[0]
os.close(fd)
dv = (c4 >> 6) & 7
vv = (c4 >> 9) & 0x7F
freq = 100 * 16 / (2 ** dv) if dv > 0 else 1600
print(f"Core 4 COFVID: 0x{c4:016x} DID={dv} VID=0x{vv:02x} = {1.55 - vv * 0.0125:.4f}V ~{freq:.0f}MHz")
print()
print("=== DONE ===")
