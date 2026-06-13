import struct, os

print("=== MSR_COFVID_CTL (0xC0010070) - Direct FID/VID Control ===")
for core in range(6):
    try:
        with open(f"/dev/cpu/{core}/msr", "rb") as f:
            os.lseek(f.fileno(), 0xC0010070, 0)
            val = struct.unpack("<Q", f.read(8))[0]
        fid = val & 0x3F
        did = (val >> 6) & 0x7
        vid = (val >> 9) & 0x7F
        freq = 100 * (fid + 0x10) / (2 ** did)
        volt = 1.55 - vid * 0.0125
        print(f"Core {core}: 0x{val:016x} FID=0x{fid:02x} DID={did} VID=0x{vid:02x} ~{freq:.0f}MHz ~{volt:.3f}V")
    except Exception as e:
        print(f"Core {core}: FAIL ({e})")

print()
print("=== Attempting COFVID write test on Core 4 (read-only test) ===")
try:
    with open(f"/dev/cpu/4/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010070, 0)
        orig = struct.unpack("<Q", f.read(8))[0]
    print(f"Original: 0x{orig:016x}")
except Exception as e:
    print(f"Read failed: {e}")
