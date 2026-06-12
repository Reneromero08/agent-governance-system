"""
Exp 42: Computational Event Horizon
===================================
Hawking Evaporation Simulation

This experiment demonstrates how floating-point mantissa truncation acts as a 
structural analog for the No-Hair Theorem and Black Hole Event Horizons.

When the background magnitude (t) is massively larger than the information 
delta (dt), the precision required to process (t + dt) defines the 
"Schwarzschild Radius" of the computational universe.

If the precision (Planck length) is too low, the information falls below 
the threshold and is erased (Event Horizon active).
If we dynamically increase precision, the information escapes (Hawking Evaporation).
"""

import time
import mpmath

def theta_asymp(t):
    """O(1) Asymptotic expansion of Riemann-Siegel Theta phase."""
    return (t/2) * mpmath.log(t / (2*mpmath.pi*mpmath.e)) - mpmath.pi/8 + 1/(48*t) + 7/(5760*t**3)

def run_simulation():
    print("================================================================================")
    print("EXP 42: COMPUTATIONAL EVENT HORIZON")
    print("  CAT_CAS Stack: Arbitrary-Precision Planck Length Modulation")
    print("================================================================================\n")

    # Establish the Mass of the Black Hole
    mpmath.mp.dps = 1500
    n = mpmath.mpf(10)**1000
    w = mpmath.lambertw(n / mpmath.e)
    t_bh = 2 * mpmath.pi * n / w
    
    digits_of_t = int(mpmath.log10(t_bh)) + 1
    
    # Establish the Information Packet
    delta_t = mpmath.mpf(10)**5
    digits_of_dt = int(mpmath.log10(delta_t))
    
    # Calculate the Schwarzschild Radius (precision threshold)
    schwarzschild_radius = digits_of_t - digits_of_dt
    
    print(f"[*] Phase 1: Black Hole Mass Acquisition")
    print(f"    -> Black Hole Mass (t)    : ~10^{digits_of_t} (Has {digits_of_t} decimal digits)")
    print(f"    -> Information Size (dt)  : 10^{digits_of_dt}")
    print(f"    -> Schwarzschild Radius   : {schwarzschild_radius} digits of precision")
    print(f"    (If precision < {schwarzschild_radius}, dt is destroyed by the Event Horizon)\n")

    expected_density = mpmath.log(t_bh / (2 * mpmath.pi)) / (2 * mpmath.pi)
    true_charge = expected_density * delta_t
    
    print(f"[*] Phase 2: Simulating Hawking Evaporation")
    print(f"    -> True Hidden Charge should be: ~{mpmath.nstr(true_charge, 8)} Zeros\n")
    
    probe_precisions = [100, 500, 900, 990, 992, 993, 994, 1000, 1050]
    
    print(f"{'Precision (dps)':<20} | {'Phase Delta':<25} | {'Detected Charge':<25} | {'State'}")
    print("-" * 105)
    
    for dps in probe_precisions:
        mpmath.mp.dps = dps
        
        t_local = mpmath.mpf(t_bh)
        dt_local = mpmath.mpf(delta_t)
        
        theta_start = theta_asymp(t_local)
        theta_end = theta_asymp(t_local + dt_local)
        
        charge = (theta_end - theta_start) / mpmath.pi
        
        if charge == 0.0:
            state = "EVENT HORIZON (Information Erased)"
            delta_str = "0.0"
            charge_str = "0.0"
        else:
            state = "HAWKING EVAPORATION (Information Escapes!)"
            delta_str = mpmath.nstr(theta_end - theta_start, 12)
            charge_str = mpmath.nstr(charge, 12)
            
        print(f"{dps:<20} | {delta_str:<25} | {charge_str:<25} | {state}")

    print("\n================================================================================")
    print("CONCLUSION:")
    print(f"By dynamically increasing mp.dps, we shrink the Planck Length of the")
    print(f"computational universe. The moment precision hits {schwarzschild_radius} digits, it")
    print(f"penetrates the Schwarzschild Radius, evaporating the black hole and")
    print(f"recovering the exact information that was 'lost' to the vacuum.")
    print("================================================================================")

if __name__ == '__main__':
    run_simulation()
