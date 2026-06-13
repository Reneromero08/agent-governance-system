"""Single-shot P4 definition transition probe.

Statistics note: this reports exact MSR readback fields for one transition
attempt and deterministic pass/fail state. It is not a sampled experiment;
p_value/CI/std/effect size are not claimed by this script.
"""

import struct, os, time, subprocess

# Sub-threshold P4: FID=0x00 DID=3 VID=0x3A (~200MHz @ ~0.825V)
# Preserve upper word from original P4 (0x80000135)
# New lower: 0x40000000 | (0x3A << 9) | (3 << 6) | 0x00 = 0x400074C0

hi = 0x80000135
lo = 0x40000000 | (0x3A << 9) | (3 << 6) | 0x00
sub_p4 = (hi << 32) | lo

fid = 0
did = 3
vid = 0x3A

print("=== WRITING SUB-THRESHOLD P4 DEFINITION ===")
print(f"Original P4: FID=0x00 DID=1 VID=0x1A (~800MHz @ ~1.225V)")
print(f"Sub-threshold P4: FID=0x00 DID=3 VID=0x3A (~200MHz @ ~0.825V)")
print(f"MSR value: 0x{sub_p4:016x}")
print()

# Write modified P4 to Core 0 (shared P-state table)
with open("/dev/cpu/0/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010068, 0)
    f.write(struct.pack("<Q", sub_p4))
print("P4 definition written. Verifying...")

with open("/dev/cpu/0/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010068, 0)
    verify = struct.unpack("<Q", f.read(8))[0]
print(f"Verify: 0x{verify:016x} Match: {verify == sub_p4}")

print()
print("=== REQUESTING SUB-THRESHOLD P4 ON CORE 4 ===")
with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    f.write(struct.pack("<Q", 4))

time.sleep(2)

print("Reading Core 4 state after transition...")
try:
    with open("/dev/cpu/4/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010071, 0)
        cofvid = struct.unpack("<Q", f.read(8))[0]
    
    cfid = cofvid & 0x3F
    cdid = (cofvid >> 6) & 0x7
    cvid = (cofvid >> 9) & 0x7F
    cfreq = 100 * (cfid + 0x10) / (2 ** cdid) if cdid > 0 else 100 * (cfid + 0x10)
    cvolt = 1.55 - cvid * 0.0125
    print(f"COFVID: 0x{cofvid:016x}")
    print(f"  FID=0x{cfid:02x} DID={cdid} VID=0x{cvid:02x} ~{cfreq:.0f}MHz ~{cvolt:.3f}V")
    
    if cdid == did and cvid == vid:
        print("SUB-THRESHOLD ACTIVE - Core 4 running at target voltage!")
    elif cdid == 1:
        print("Core reverted to original P4 - hardware rejected sub-threshold")
    else:
        print(f"Core at different state: DID={cdid} VID=0x{cvid:02x}")
except Exception as e:
    print(f"Core 4 READ FAILED - may be hung at sub-threshold ({e})")

print()
print("=== SENSORS ===")
result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())

print()
print("=== OTHER CORES STATUS ===")
for core in [0, 1, 2, 3, 5]:
    try:
        with open(f"/dev/cpu/{core}/msr", "rb") as f:
            os.lseek(f.fileno(), 0xC0010071, 0)
            val = struct.unpack("<Q", f.read(8))[0]
        fid_v = val & 0x3F
        did_v = (val >> 6) & 0x7
        vid_v = (val >> 9) & 0x7F
        freq_v = 100 * (fid_v + 0x10) / (2 ** did_v) if did_v > 0 else 100 * (fid_v + 0x10)
        volt_v = 1.55 - vid_v * 0.0125
        print(f"Core {core}: FID=0x{fid_v:02x} DID={did_v} VID=0x{vid_v:02x} ~{freq_v:.0f}MHz ~{volt_v:.3f}V")
    except:
        print(f"Core {core}: READ FAILED")

print()
print("=== DONE ===")
