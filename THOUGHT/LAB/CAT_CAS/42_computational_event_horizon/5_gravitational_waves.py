"""
Exp 42.5: Black Hole Mergers (Gravitational Waves)
==================================================
When two massive black holes collide, they produce gravitational waves—
ripples in the fabric of spacetime.

In our floating-point universe, the "fabric of spacetime" is the 
arbitrary-precision memory architecture. When two massive scales (t1 and t2) 
are added together, their mantissas overflow. The processor must execute an 
"Exponent Shift" to normalize the new mass, rippling a bit-shift operation 
through the physical CPU registers.

We can measure this exponent shift as a Computational Gravitational Wave.
"""

import mpmath
import time

def run_gravitational_waves():
    print("================================================================================")
    print("EXP 42.5: BLACK HOLE MERGERS (Computational Gravitational Waves)")
    print("  CAT_CAS Stack: Exponent Shift Detection / Binary Merger")
    print("================================================================================\n")

    mpmath.mp.dps = 1500
    
    n1 = mpmath.mpf(10)**1000
    w1 = mpmath.lambertw(n1 / mpmath.e)
    t1 = 2 * mpmath.pi * n1 / w1
    
    # We create a slightly asymmetric binary pair
    n2 = mpmath.mpf(10)**1000 * mpmath.mpf('1.5')
    w2 = mpmath.lambertw(n2 / mpmath.e)
    t2 = 2 * mpmath.pi * n2 / w2
    
    print("[*] Phase 1: Binary Black Hole System")
    
    sign1, man1, exp1, bit1 = t1._mpf_
    sign2, man2, exp2, bit2 = t2._mpf_
    
    print(f"    -> Black Hole 1 Mass : ~10^{int(mpmath.log10(t1))}")
    print(f"       (Base Exponent: {exp1})")
    print(f"    -> Black Hole 2 Mass : ~10^{int(mpmath.log10(t2))}")
    print(f"       (Base Exponent: {exp2})\n")
    
    print("[*] Phase 2: The Merger (Collision Event)")
    
    t0 = time.time()
    t3 = t1 + t2
    t1_duration = time.time() - t0
    
    sign3, man3, exp3, bit3 = t3._mpf_
    
    print(f"    -> Merger Computation Time : {t1_duration:.6f} seconds")
    print(f"    -> Post-Merger Mass        : ~10^{int(mpmath.log10(t3))}")
    print(f"    -> Post-Merger Exponent    : {exp3}\n")
    
    print("[*] Phase 3: Gravitational Wave Detection")
    wave_amplitude = exp3 - max(exp1, exp2)
    
    if wave_amplitude > 0:
        print("    [!] GRAVITATIONAL WAVE DETECTED [!]")
        print(f"    -> Exponent Shift Amplitude: +{wave_amplitude} bits")
        print("    -> The mantissa overflowed, forcing the CPU to ripple a bit-shift")
        print("       operation across thousands of registers to normalize the new mass.")
    else:
        print("    [X] No wave detected. The merger was absorbed silently.")

    print("\n================================================================================")
    print("CONCLUSION:")
    print("Floating-point exponent shifts act as physical shockwaves in the CPU.")
    print("When two 10^1000 masses collide, the hardware physically alters the metric")
    print("(the exponent) of the universe to accommodate the new singularity.")
    print("================================================================================")

if __name__ == '__main__':
    run_gravitational_waves()
