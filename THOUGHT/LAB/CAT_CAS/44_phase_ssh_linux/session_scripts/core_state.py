import struct, os
for core in range(6):
    fd = os.open(f"/dev/cpu/{core}/msr", os.O_RDONLY)
    os.lseek(fd, 0xC0010071, 0)
    v = struct.unpack("<Q", os.read(fd, 8))[0]
    os.close(fd)
    fid = v & 0x3F
    did = (v >> 6) & 7
    vid = (v >> 9) & 0x7F
    freq = 100 * (fid + 0x10) / (2 ** did) if did > 0 else 100 * (fid + 0x10)
    volt = 1.55 - vid * 0.0125
    print(f"Core {core}: FID=0x{fid:02x} DID={did} ~{freq:.0f}MHz VID=0x{vid:02x} ~{volt:.3f}V")
