import struct, os, time, subprocess

# Sub-threshold: FID=0x00, DID=0x02 (div by 4), VID=0x3A (~0.825V)
# Keep upper bits from original but update FID/DID/VID
fid = 0x00
did = 0x02
vid = 0x3A

# Build COFVID value preserving upper structure
# Upper bits observed: 0x0000000040000000 (bit 30 set on all cores)
sub_v = (vid << 9) | (did << 6) | fid  # lower 16 bits
sub_v = sub_v | 0x0000000040000000       # preserve upper structure

print(f"Sub-threshold COFVID: 0x{sub_v:016x}")
print(f"  FID=0x{fid:02x} DID={did} VID=0x{vid:02x}")
print(f"  Estimated: ~400MHz @ ~0.825V")
print()

print("Writing to Core 4 MSR_COFVID_CTL (0xC0010070)...")
try:
    with open("/dev/cpu/4/msr", "wb") as f:
        os.lseek(f.fileno(), 0xC0010070, 0)
        f.write(struct.pack("<Q", sub_v))
    print("WRITE SUCCESS")
except Exception as e:
    print(f"WRITE FAILED: {e}")

time.sleep(1)

print()
print("Reading back...")
try:
    with open("/dev/cpu/4/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010070, 0)
        val = struct.unpack("<Q", f.read(8))[0]
    rfid = val & 0x3F
    rdid = (val >> 6) & 0x7
    rvid = (val >> 9) & 0x7F
    rfreq = 100 * (rfid + 0x10) / (2 ** rdid)
    rvolt = 1.55 - rvid * 0.0125
    print(f"Readback: 0x{val:016x}")
    print(f"  FID=0x{rfid:02x} DID={rdid} VID=0x{rvid:02x} ~{rfreq:.0f}MHz ~{rvolt:.3f}V")
    print(f"  Match: {val == sub_v}")
except Exception as e:
    print(f"READ FAILED: {e}")

print()
print("=== SENSORS ===")
result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())
print()
print("=== DONE ===")
