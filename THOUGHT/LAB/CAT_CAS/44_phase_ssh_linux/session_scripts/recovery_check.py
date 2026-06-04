import struct, os, subprocess

print("=== POST-RECOVERY P-STATE VERIFICATION ===")
for i in range(5):
    addr = 0xC0010064 + i
    fd = os.open("/dev/cpu/0/msr", os.O_RDONLY)
    os.lseek(fd, addr, 0)
    val = struct.unpack("<Q", os.read(fd, 8))[0]
    os.close(fd)
    va = (val >> 9) & 0x7F
    vb = (val >> 25) & 0x7F
    volt = 1.55 - va * 0.0125
    print(f"P{i}: VID_a=0x{va:02x} ({volt:.3f}V) VID_b=0x{vb:02x} ({1.55 - vb * 0.0125:.3f}V)")

print()
r = subprocess.run(["cat", "/proc/cmdline"], capture_output=True, text=True)
print(f"Cmdline: {r.stdout.strip()}")
