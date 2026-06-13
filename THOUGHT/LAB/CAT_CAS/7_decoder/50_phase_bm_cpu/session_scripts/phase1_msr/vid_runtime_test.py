import struct, os, time

CORE = 4
PSTATE_DEF_MSR = 0xC0010068
PSTATE_CTL_MSR = 0xC0010062
COFVID_STS_MSR = 0xC0010071

print("=== RUNTIME VID=0x20 TEST ON CORE 4 ===")

fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDONLY)
os.lseek(fd, PSTATE_DEF_MSR, 0)
old_p4 = struct.unpack("<Q", os.read(fd, 8))[0]
os.close(fd)
old_vid_a = (old_p4 >> 9) & 0x7F
old_vid_b = (old_p4 >> 25) & 0x7F
print(f"Current P4 def: 0x{old_p4:016x}")
print(f"  VID_a=0x{old_vid_a:02x} ({1.55 - old_vid_a * 0.0125:.3f}V)")
print(f"  VID_b=0x{old_vid_b:02x} ({1.55 - old_vid_b * 0.0125:.3f}V)")

new_vid = 0x20
new_did = 3
new_fid = 0
hi = 0x80000135
lo = 0x40000000 | ((new_vid & 0x7F) << 9) | ((new_did & 0x07) << 6) | (new_fid & 0x3F)
new_p4 = (hi << 32) | lo

print(f"\nWriting P4 def: 0x{new_p4:016x}")
print(f"  Target: VID=0x{new_vid:02x} ({1.55 - new_vid * 0.0125:.3f}V), DID={new_did} ({100*16/(2**new_did):.0f}MHz)")

fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDWR)
os.lseek(fd, PSTATE_DEF_MSR, 0)
os.write(fd, struct.pack("<Q", new_p4))
os.close(fd)

fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDONLY)
os.lseek(fd, PSTATE_DEF_MSR, 0)
verify = struct.unpack("<Q", os.read(fd, 8))[0]
os.close(fd)
print(f"  Verify: 0x{verify:016x} (match: {verify == new_p4})")

print("\nCycling P-state: P0 -> P4...")
fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDWR)
os.lseek(fd, PSTATE_CTL_MSR, 0)
os.write(fd, struct.pack("<Q", 0))
os.close(fd)
time.sleep(0.2)
fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDWR)
os.lseek(fd, PSTATE_CTL_MSR, 0)
os.write(fd, struct.pack("<Q", 4))
os.close(fd)
time.sleep(0.3)

fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDONLY)
os.lseek(fd, COFVID_STS_MSR, 0)
cofvid = struct.unpack("<Q", os.read(fd, 8))[0]
os.close(fd)
cur_vid = (cofvid >> 9) & 0x7F
cur_did = (cofvid >> 6) & 0x7
cur_fid = cofvid & 0x3F
cur_volt = 1.55 - cur_vid * 0.0125
cur_freq = 100 * (cur_fid + 0x10) / (2 ** cur_did)

print(f"\n=== HARDWARE RESPONSE ===")
print(f"COFVID_STATUS: 0x{cofvid:016x}")
print(f"  Current VID: 0x{cur_vid:02x} ({cur_volt:.3f}V)")
print(f"  Current DID: {cur_did}")
print(f"  Current FID: 0x{cur_fid:02x}")
print(f"  Current freq: {cur_freq:.0f} MHz")

if cur_vid == new_vid:
    print(f"\nSUCCESS: SVI accepted VID=0x{new_vid:02x} ({1.55 - new_vid * 0.0125:.3f}V)!")
    print("No BIOS patch needed.")
elif cur_vid == old_vid_a:
    print(f"\nCLAMPED: SVI rejected VID=0x{new_vid:02x}, kept 0x{cur_vid:02x} ({cur_volt:.3f}V)")
    print("SVI enforced minimum. BIOS patch or hardware mod required.")
else:
    print(f"\nUNEXPECTED: SVI chose VID=0x{cur_vid:02x} ({cur_volt:.3f}V)")

print(f"\nRestoring original P4 definition...")
fd = os.open(f"/dev/cpu/{CORE}/msr", os.O_RDWR)
os.lseek(fd, PSTATE_DEF_MSR, 0)
os.write(fd, struct.pack("<Q", old_p4))
os.close(fd)
print("Done.")
