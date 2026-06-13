"""Single-shot subthreshold P-state write/readback probe.

Statistics note: this script tests one deterministic MSR write/readback path and
core responsiveness. It reports decoded register values from one attempt; no
empirical variance is claimed, so p_value/CI/std/effect size are not applicable.
"""

import struct, os, time, subprocess

subthreshold = 0x8000013A40002C00

print(f"Writing sub-threshold P-state to Core 4...")
print(f"Value: 0x{subthreshold:016x}")
print(f"Estimated voltage: ~0.825V")
print(f"Estimated frequency: ~400 MHz")
print()

with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    f.write(struct.pack("<Q", subthreshold))
print("MSR written. Reading back...")

with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    verify = struct.unpack("<Q", f.read(8))[0]
print(f"Verify: 0x{verify:016x}")
print(f"Match: {verify == subthreshold}")

time.sleep(2)

try:
    with open("/dev/cpu/4/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010071, 0)
        cofvid = struct.unpack("<Q", f.read(8))[0]
    print(f"Core 4 COFVID: 0x{cofvid:016x} (CORE RESPONSIVE)")
except Exception as e:
    print(f"Core 4 COFVID: READ FAILED - Core may be hung ({e})")

result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())

print()
print("=== SUB-THRESHOLD TEST COMPLETE ===")
print("If Core 4 is still responsive, we proceed to stability testing.")
print("If Core 4 hung, we will restore it with a known-good P-state.")
