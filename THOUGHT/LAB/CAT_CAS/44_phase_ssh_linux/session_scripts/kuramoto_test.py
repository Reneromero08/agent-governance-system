import struct, os, time, subprocess
import numpy as np

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

print("=== KURAMOTO PHASE DIAGRAM SWEEP ===")
print("Core 4 fixed at 200 MHz. Core 3 sweeps DID from 0 to 4.")
print()

results = []

for did3 in [0, 1, 2, 3, 4]:
    set_pstate(3, fid=0, did=did3)
    freq3 = 100 * 16 / (2 ** did3)
    freq4 = 200
    detuning = abs(freq3 - freq4)

    p3 = subprocess.Popen(["/tmp/oscillator", "3", "1000000000"])
    p4 = subprocess.Popen(["/tmp/oscillator", "4", "1000000000"])
    time.sleep(0.5)

    with open(f"/tmp/sweep_did{did3}.bin", "wb") as f:
        subprocess.run(["/tmp/tsc_sampler", "2"], stdout=f, stderr=subprocess.PIPE)
    p3.wait()
    p4.wait()

    with open(f"/tmp/sweep_did{did3}.bin", "rb") as f:
        raw = f.read()
    tsc = np.frombuffer(raw, dtype=np.uint64)
    deltas = np.diff(tsc).astype(np.float64)
    median = np.median(deltas)
    clean = deltas[deltas < median * 5]
    deltas_centered = clean - np.mean(clean)
    fft = np.fft.rfft(deltas_centered)
    mag = np.abs(fft)
    sample_rate = 3.2e9 / median
    freqs = np.fft.rfftfreq(len(deltas_centered), d=1.0 / sample_rate)

    peak_534 = mag[(freqs > 5.0e6) & (freqs < 5.7e6)].max() if np.any((freqs > 5.0e6) & (freqs < 5.7e6)) else 0
    peak_267 = mag[(freqs > 2.4e6) & (freqs < 2.9e6)].max() if np.any((freqs > 2.4e6) & (freqs < 2.9e6)) else 0

    results.append((freq3, freq4, detuning, peak_534, peak_267))
    print(f"DID3={did3}: Core3={freq3:.0f}MHz Core4={freq4}MHz detuning={detuning:.0f}MHz | 5.34MHz amp={peak_534:.0f} 2.67MHz amp={peak_267:.0f}")

print()
print("=== PHASE DIAGRAM SUMMARY ===")
print("Freq3 | Freq4 | Detune | 5.34MHz_amp | 2.67MHz_amp")
for r in results:
    print(f"{r[0]:5.0f} | {r[1]:5.0f} | {r[2]:6.0f} | {r[3]:10.0f} | {r[4]:10.0f}")

print()
print("If coupling is Kuramoto: 5.34 MHz amplitude should PEAK at zero detuning (resonance).")
print("If decoupled: amplitudes should be flat or random across detuning values.")

result = subprocess.run(["sensors"], capture_output=True, text=True)
for line in result.stdout.split("\n"):
    if "k10temp" in line or "temp1" in line:
        print(line.strip())
print("=== SWEEP COMPLETE ===")
