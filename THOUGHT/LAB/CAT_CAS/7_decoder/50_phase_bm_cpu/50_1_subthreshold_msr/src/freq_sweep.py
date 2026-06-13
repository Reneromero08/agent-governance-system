import struct, os, time

print("=== FORCING P-STATE CYCLE ON CORE 4 ===")

# Write sub-threshold P4 (DID=3) to Core 4
hi = 0x80000135
did = 3
lo = 0x40000000 | (0x24 << 9) | (did << 6) | 0x00
sub_p4 = (hi << 32) | lo
print(f"Writing DID={did} P4: 0x{sub_p4:016x}")

with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010068, 0)
    f.write(struct.pack("<Q", sub_p4))

# Read current state
with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    cur_pstate = struct.unpack("<Q", f.read(8))[0]
print(f"Current P-state: {cur_pstate}")

# Cycle: transition to P0 (highest), then back to P4
print("\nCycling to P0...")
with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    f.write(struct.pack("<Q", 0))
time.sleep(0.3)

# Read COFVID at P0
with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010071, 0)
    v = struct.unpack("<Q", f.read(8))[0]
print(f"At P0: DID={(v>>6)&7} VID=0x{(v>>9)&0x7F:02x}")

print("\nCycling back to P4...")
with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010062, 0)
    f.write(struct.pack("<Q", 4))
time.sleep(0.5)

# Now check if DID=3 took effect
with open("/dev/cpu/4/msr", "rb") as f:
    os.lseek(f.fileno(), 0xC0010071, 0)
    cofvid = struct.unpack("<Q", f.read(8))[0]
cfid = cofvid & 0x3F; cdid = (cofvid >> 6) & 0x7; cvid = (cofvid >> 9) & 0x7F
cfreq = 100 * (cfid + 0x10) / (2 ** cdid) if cdid > 0 else 100 * (cfid + 0x10)
cvolt = 1.55 - cvid * 0.0125
print(f"\nAfter cycle to P4: DID={cdid} (target=3) ~{cfreq:.0f}MHz ~{cvolt:.3f}V")
if cdid == did:
    print("SUCCESS - DID=3 took effect!")
else:
    print(f"FAILED - DID still {cdid}, not {did}")

# Restore original P4
with open("/dev/cpu/4/msr", "wb") as f:
    os.lseek(f.fileno(), 0xC0010068, 0)
    f.write(struct.pack("<Q", 0x8000013540003440))
print("\nCore 4 P4 restored")

# All cores
print()
for core in range(6):
    with open(f"/dev/cpu/{core}/msr", "rb") as f:
        os.lseek(f.fileno(), 0xC0010071, 0)
        v = struct.unpack("<Q", f.read(8))[0]
    dv = (v >> 6) & 0x7; fr = 100 * ((v & 0x3F) + 0x10) / (2 ** dv) if dv > 0 else 100 * ((v & 0x3F) + 0x10)
    print(f"Core {core}: DID={dv} ~{fr:.0f}MHz")
print("\n=== DONE ===")
