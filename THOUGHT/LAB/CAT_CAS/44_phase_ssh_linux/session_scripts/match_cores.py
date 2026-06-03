import struct, os, time

def set_pstate(core, fid, did, vid=0x12):
    hi = 0x80000135
    lo = 0x40000000 | ((vid & 0x7F) << 9) | ((did & 0x07) << 6) | (fid & 0x3F)
    val = (hi << 32) | lo
    fp = f"/dev/cpu/{core}/msr"
    fd = os.open(fp, os.O_RDWR)
    os.lseek(fd, 0xC0010068, 0)
    os.write(fd, struct.pack("<Q", val))
    os.close(fd)
    fd = os.open(fp, os.O_RDWR)
    os.lseek(fd, 0xC0010062, 0)
    os.write(fd, struct.pack("<Q", 0))
    os.close(fd)
    time.sleep(0.1)
    fd = os.open(fp, os.O_RDWR)
    os.lseek(fd, 0xC0010062, 0)
    os.write(fd, struct.pack("<Q", 4))
    os.close(fd)
    time.sleep(0.1)

print("=== SETTING BOTH CORES TO 200 MHz ===")
for core in [3, 4]:
    set_pstate(core, fid=0, did=3)

for core in [3, 4]:
    fd = os.open(f"/dev/cpu/{core}/msr", os.O_RDONLY)
    os.lseek(fd, 0xC0010071, 0)
    cofvid = struct.unpack("<Q", os.read(fd, 8))[0]
    os.close(fd)
    d = (cofvid >> 6) & 7
    f = cofvid & 0x3F
    freq = 100 * (f + 0x10) / (2 ** d)
    print(f"Core {core}: DID={d} ~{freq:.0f}MHz")
print("Both cores matched at 200 MHz.")
