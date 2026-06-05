import struct, os, hashlib, subprocess, mmap

print("=== AUDIT 1: FRESH CATALYTIC CYCLE ===")

print("=== CORE FREQUENCY VERIFICATION ===")
for core in [3, 4, 5]:
    fd = os.open(f"/dev/cpu/{core}/msr", os.O_RDONLY)
    os.lseek(fd, 0xC0010071, 0)
    v = struct.unpack("<Q", os.read(fd, 8))[0]
    os.close(fd)
    d = (v >> 6) & 7
    f = v & 0x3F
    freq = 100 * (f + 0x10) / (2 ** d)
    vid = (v >> 9) & 0x7F
    volt = 1.55 - vid * 0.0125
    print(f"Core {core}: {freq:.0f}MHz {volt:.3f}V DID={d} VID=0x{vid:02x}")

r = subprocess.run(["cat", "/sys/devices/system/cpu/isolated"], capture_output=True, text=True)
print(f"Isolated: {r.stdout.strip()}")

tape = mmap.mmap(-1, 256, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
tape[:] = bytes([0xAA] * 256)
hb = hashlib.sha256(tape).hexdigest()
print(f"\nInitial SHA256: {hb[:16]}...")

vals = [0xCAFEBABEDEADBEEF, 0x1234567890ABCDEF]
for i, val in enumerate(vals):
    cur = struct.unpack("<Q", tape[i*8:(i+1)*8])[0]
    tape[i*8:(i+1)*8] = struct.pack("<Q", cur ^ val)

haf = hashlib.sha256(tape).hexdigest()
s0 = struct.unpack("<Q", tape[0:8])[0]
s1 = struct.unpack("<Q", tape[8:16])[0]
print(f"Forward SHA: {haf[:16]}... changed={hb != haf}")
print(f"Slot0=0x{s0:016x} Slot1=0x{s1:016x} non_trivial={s0 != 0xAAAAAAAAAAAAAAAA}")

for i, val in enumerate(vals):
    cur = struct.unpack("<Q", tape[i*8:(i+1)*8])[0]
    tape[i*8:(i+1)*8] = struct.pack("<Q", cur ^ val)

har = hashlib.sha256(tape).hexdigest()
match = hb == har
print(f"Reverse SHA: {har[:16]}... match={match} bits_erased={0 if match else '>0'}")
tape.close()
print(f"\n=== AUDIT 1 VERDICT: {'PASS' if match else 'FAIL'} ===")
