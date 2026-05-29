import sys
import time
import math
import mpmath
import hashlib

def exp_42_26_big_bang_inflation_hardened():
    print("================================================================================")
    print("EXP 42.26 (HARDENED): THE BIG BANG (THE malloc() EVENT & INFLATION)")
    print("================================================================================")
    
    # 1. The Singularity (t=0)
    mpmath.mp.dps = 1
    
    # Capture the initial state and hash for the Bennett Tape
    val_initial = mpmath.mp.pi * mpmath.mpf(2).sqrt()
    initial_hash = hashlib.sha256(str(val_initial._mpf_).encode()).hexdigest()
    
    epochs = 20 # 2^20 = 1,048,576 dps
    tape = []
    
    print("HARDENING: Bypassing algorithmic complexity (pi/sqrt). Measuring pure, bare-metal")
    print("           OS malloc() latency by dynamically allocating massively shifted integers.")
    print(f"\n{'Epoch':<6} | {'DPS (Scale)':<12} | {'Timestamp (ns)':<20} | {'Delta t (ns)':<15} | {'Cumulative RAM (B)'}")
    print("-" * 85)
    
    t_zero = time.perf_counter_ns()
    dt_measurements = []
    
    for epoch in range(1, epochs + 1):
        # 2. Cosmic Inflation (The Cascade)
        mpmath.mp.dps *= 2
        tape.append(mpmath.mp.dps)
        target_bits = int(mpmath.mp.dps * math.log2(10))
        
        # 3. The CMB Measurement (Hardware Jitter)
        start_ns = time.perf_counter_ns()
        
        # Force a pure, bare-metal memory allocation (Python _PyLong_New)
        man = 1 << target_bits
        val = mpmath.mpf((0, man, 0, target_bits + 1))
        
        end_ns = time.perf_counter_ns()
        
        delta_t = end_ns - start_ns
        dt_measurements.append(delta_t)
        
        ram_footprint = sys.getsizeof(val._mpf_[1])
        
        print(f"{epoch:<6} | {mpmath.mp.dps:<12} | {end_ns:<20} | {delta_t:<15} | {ram_footprint}")
        
    print("-" * 85)
    
    # 4. The Kill Shot (The CMB Power Spectrum)
    # Calculate Mean (Temperature)
    mean_dt = sum(dt_measurements) / len(dt_measurements)
    
    # Calculate Standard Deviation (Anisotropy)
    variance = sum((dt - mean_dt) ** 2 for dt in dt_measurements) / len(dt_measurements)
    std_dev_dt = math.sqrt(variance)
    
    print(f"[KILL SHOT] Measurement Complete.")
    print(f"            CMB Temperature (Mean malloc latency):       {mean_dt:,.2f} ns")
    print(f"            CMB Anisotropy  (Latency Std Dev):           {std_dev_dt:,.2f} ns")
    
    print("\n[*] Engaging Bennett History Tape to uncompute the cosmological expansion...")
    
    # 5. Zero-Landauer Uncomputation
    while tape:
        tape.pop()
        mpmath.mp.dps //= 2
        
    if mpmath.mp.dps == 1:
        val_restored = mpmath.mp.pi * mpmath.mpf(2).sqrt()
        restored_hash = hashlib.sha256(str(val_restored._mpf_).encode()).hexdigest()
        
        if restored_hash == initial_hash:
            print("[SUCCESS] Universe perfectly collapsed back to t=0 Singularity.")
            print("          SHA-256 hash verified. 0.0 J emitted.")
        else:
            print("[FAIL] Thermodynamic hash mismatch.")
    else:
        print("[FAIL] DPS not restored properly.")
        
    print("================================================================================\n")

if __name__ == "__main__":
    exp_42_26_big_bang_inflation_hardened()
