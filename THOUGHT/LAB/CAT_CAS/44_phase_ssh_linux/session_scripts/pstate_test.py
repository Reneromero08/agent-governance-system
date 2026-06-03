import struct, os, time

# Try to modify P-state definition (P4, MSR 0xC0010068) to sub-threshold values
# Original P4: 0x8000013540003440 -> FID=0x00 DID=2 VID=0x1A (800MHz @ ~1.0V)
# Sub-threshold: FID=0x00 DID=3 VID=0x3A (div by 8, ~200MHz @ ~0.825V)

fid = 0x00
did = 0x03
vid = 0x3A

new_p4 = (0x80000000 << 32) | (vid << 9) | (did << 6) | fid
# P-state definitions have format: 0x80000[VID][7:0 high][DID][FID][extra]
# Actually need to look at the full structure. Original P4: 0x8000013540003440
# Let me decode: 0x80000135_40003440
# Upper: 0x80000135 (includes some control bits)
# Lower: 0x40003440 -> FID=0x10? no... 
# Let me re-decode from known values
# P4: 0x8000013540003440 
# Low 32 bits: 0x40003440
# FID = bits[5:0] = 0x00
# DID = bits[8:6] = (0x40 >> 6) & 7 = 0x01? no... 
# Let me look at the raw bits differently

print("=== P-STATE DEFINITION DECODE ===")
for pstate in range(5):
    addr = 0xC0010064 + pstate
    with open("/dev/cpu/0/msr", "rb") as f:
        os.lseek(f.fileno(), addr, 0)
        val = struct.unpack("<Q", f.read(8))[0]
    
    lo = val & 0xFFFFFFFF
    hi = (val >> 32) & 0xFFFFFFFF
    
    fid = lo & 0x3F
    did = (lo >> 6) & 0x7
    vid = (lo >> 9) & 0x7F
    
    freq = 100 * (fid + 0x10) / (2 ** did) if did > 0 else 100 * (fid + 0x10)
    volt = 1.55 - vid * 0.0125
    
    print(f"P{pstate}: 0x{val:016x} FID=0x{fid:02x} DID={did} VID=0x{vid:02x} ~{freq:.0f}MHz ~{volt:.3f}V")
    print(f"  Lo=0x{lo:08x} Hi=0x{hi:08x}")

print()

# Now try to write a modified P-state definition
print("=== ATTEMPTING P-STATE DEFINITION WRITE (CORE 0) ===")
sub_p4 = 0x8000013A40001AC0  # approximate sub-threshold with FID=0x00 DID=3 VID=0x3A
try:
    with open("/dev/cpu/0/msr", "wb") as f:
        os.lseek(f.fileno(), 0xC0010068, 0)
        f.write(struct.pack("<Q", sub_p4))
    print(f"WRITE P4 MSR SUCCESS")
    
    with open("/dev/cpu/0/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010068, 0)
        verify = struct.unpack("<Q", f.read(8))[0]
    print(f"Readback: 0x{verify:016x} Match: {verify == sub_p4}")
except Exception as e:
    print(f"WRITE FAILED: {e}")

print()

# Also try P-state transition via PSTATE_CTL
print("=== P-STATE TRANSITION TEST (CORE 4) ===")
for target_pstate in [0, 4]:
    try:
        with open("/dev/cpu/4/msr", "wb") as f:
            os.lseek(f.fileno(), 0xC0010062, 0)
            f.write(struct.pack("<Q", target_pstate))
        time.sleep(0.5)
        with open("/dev/cpu/4/msr", "rb") as f:
            os.lseek(f.fileno(), 0xC0010062, 0)
            cur = struct.unpack("<Q", f.read(8))[0]
        with open("/dev/cpu/4/msr", "rb") as f:
            os.lseek(f.fileno(), 0xC0010071, 0)
            sts = struct.unpack("<Q", f.read(8))[0]
        print(f"Request P{pstate_count}: PStateCtl=0x{cur:016x} COFVID=0x{sts:016x}")
    except Exception as e:
        print(f"Request P{target_pstate}: FAIL ({e})")

print()
print("=== DONE ===")
