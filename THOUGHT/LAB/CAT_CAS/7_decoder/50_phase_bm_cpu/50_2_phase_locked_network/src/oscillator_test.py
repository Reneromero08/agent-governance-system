import struct, os, time, subprocess, ctypes

print("=== CONFIGURING TWO-OSCILLATOR NETWORK ===")

def write_pstate_def(core, fid, did, vid=0x12):
    hi = 0x80000135
    lo = 0x40000000 | ((vid & 0x7F) << 9) | ((did & 0x07) << 6) | (fid & 0x3F)
    val = (hi << 32) | lo
    fps = f"/dev/cpu/{core}/msr"
    fd = os.open(fps, os.O_RDWR)
    os.lseek(fd, 0xC0010068, 0)
    os.write(fd, struct.pack("<Q", val))
    os.close(fd)
    fd = os.open(fps, os.O_RDWR)
    os.lseek(fd, 0xC0010062, 0)
    os.write(fd, struct.pack("<Q", 0))
    os.close(fd)
    time.sleep(0.1)
    fd = os.open(fps, os.O_RDWR)
    os.lseek(fd, 0xC0010062, 0)
    os.write(fd, struct.pack("<Q", 4))
    os.close(fd)

write_pstate_def(4, fid=0, did=3)
write_pstate_def(5, fid=0, did=3)

print("Both cores set to DID=3 (~200MHz)")
print()

for core in [4, 5]:
    fd = os.open(f"/dev/cpu/{core}/msr", os.O_RDONLY)
    os.lseek(fd, 0xC0010071, 0)
    cofvid = struct.unpack("<Q", os.read(fd, 8))[0]
    os.close(fd)
    c_did = (cofvid >> 6) & 7
    c_fid = cofvid & 0x3F
    freq = 100 * (c_fid + 0x10) / (2 ** c_did)
    c_vid = (cofvid >> 9) & 0x7F
    volts = 1.55 - c_vid * 0.0125
    print(f"Core {core}: FID={c_fid} DID={c_did} -> {freq:.0f}MHz VID=0x{c_vid:02x} -> {volts:.3f}V")

print()
print("=== TSC SAMPLING ON CORE 0 ===")

# Read TSC via rdtsc intrinsic - use ctypes to call the instruction
# Actually, time_ns() on modern Linux uses rdtsc as clocksource
N = 500000
samples = []
t0 = time.time()
for i in range(N):
    samples.append(time.time_ns())
t1 = time.time()
rate = N / (t1 - t0)
print(f"Sampled {N} timestamps in {t1-t0:.1f}s ({rate/1e6:.1f} Msamples/s)")

# Compute deltas
deltas = []
for i in range(1, len(samples)):
    deltas.append(samples[i] - samples[i-1])
avg = sum(deltas) / len(deltas)
print(f"Average delta: {avg:.1f} ns")

# Simple autocorrelation-based beat detection
# No numpy - use pure Python math
print()
print("=== BEAT DETECTION (simple autocorrelation) ===")

# Find max deviation from mean
max_dev = max(abs(d - avg) for d in deltas)
min_d = min(deltas)
max_d = max(deltas)
print(f"Delta range: [{min_d}, {max_d}] ns, avg={avg:.1f}, max_dev={max_dev:.1f}")

# Check for periodic patterns in the deltas
# If two oscillators at nearly same freq couple, we'll see periodic jitter
window = min(10000, len(deltas))
chunks = []
for i in range(0, min(50000, len(deltas)), window):
    chunk = deltas[i:i+window]
    chunks.append(sum(chunk) / len(chunk))
if len(chunks) > 1:
    chunk_diffs = [chunks[i] - chunks[i-1] for i in range(1, len(chunks))]
    print(f"Chunk means (window={window}): {[f'{c:.1f}' for c in chunks[:10]]}")

result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())

print()
print("=== EXPERIMENT COMPLETE ===")
