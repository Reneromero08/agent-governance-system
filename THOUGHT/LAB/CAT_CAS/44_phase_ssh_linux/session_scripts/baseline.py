import struct, os, time

print("=== PRE-TEST STATUS ===")
for core in range(6):
    with open(f"/dev/cpu/{core}/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010071, 0)
        val = struct.unpack("<Q", f.read(8))[0]
    print(f"Core {core}: 0x{val:016x}")

print()
print("=== CORE 4 CURRENT STATE ===")
with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    pstate_ctl = struct.unpack("<Q", f.read(8))[0]
print(f"PStateCtl: 0x{pstate_ctl:016x}")

import subprocess
result = subprocess.run(["cat", "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"], capture_output=True, text=True)
print(f"Core 4 governor: {result.stdout.strip() if result.returncode == 0 else 'NOT_AVAILABLE'}")

if result.returncode == 0:
    with open("/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor", "w") as f:
        f.write("userspace")
    print("Core 4 governor set to userspace")

result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())

print()
print("=== BASELINE COMPLETE ===")
