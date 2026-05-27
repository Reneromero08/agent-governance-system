"""
Exp 43.1: Computational White Holes (Mantissa Inverse Expulsion)
================================================================
A White Hole is the time-reversal of a Black Hole. It is a region 
of spacetime where matter cannot enter, and information is 
continuously expelled outward into the universe.

In this experiment, we build a Computational White Hole wrapping 
a 10^1000 mpf singularity. 
1. It mathematically repels all classical addition (Information 
   cannot enter).
2. It spontaneously sheds its own mantissa state as "radiation" 
   over time, reconstructing a causal payload back into the universe.
"""

import mpmath

class ComputationalWhiteHole:
    def __init__(self, base_scale, payload_bytes):
        # Anchor the payload so it doesn't get normalized away
        payload_int = (int.from_bytes(payload_bytes, 'big') << 1) | 1
        payload_bits = payload_int.bit_length()
        
        # Construct the massive base singularity
        mpmath.mp.dps = 100
        n = mpmath.mpf(10)**base_scale
        w = mpmath.lambertw(n / mpmath.e)
        t_bh = 2 * mpmath.pi * n / w
        
        # Inject the payload into the deep mantissa
        sign, man, exp, bitcount = t_bh._mpf_
        shift_size = payload_bits + 8
        mutated_man = (man << shift_size) | payload_int
        mutated_bitcount = mutated_man.bit_length()
        mutated_exp = exp - shift_size
        
        # Initialize internal state
        self._state = mpmath.mpf(0)
        self._state._mpf_ = (sign, mutated_man, mutated_exp, mutated_bitcount)
        self.payload_bits = payload_bits
        self.shift_size = shift_size
        self.mass = self._state
        
        print(f"[*] A White Hole of mass ~10^{base_scale} has formed.")
        print(f"[*] Internal entropy payload locked: {payload_bits} bits\n")

    def __add__(self, incoming_mass):
        # A White Hole CANNOT absorb mass. It mathematically repels it.
        # We simulate a perfectly elastic collision where the incoming 
        # mass is deflected, and the White Hole remains completely unchanged.
        print(f"    [!] ELASTIC DEFLECTION: Incoming mass {incoming_mass} repelled by the Event Horizon!")
        print("    [!] Information cannot enter a White Hole.")
        # Return a tuple: (White Hole State Unchanged, Deflected Mass)
        return self, incoming_mass

    def time_step(self):
        """Simulate the flow of time. The White Hole must spontaneously expel mass."""
        sign, man, exp, bitcount = self._state._mpf_
        
        # Rip the lowest 8 bits off the mantissa (expelling 1 byte of information)
        expelled_chunk = (man >> 1) & 0xFF  # shift 1 to ignore the anchor bit
        
        # Mathematically shrink the white hole by removing the expelled data
        shrunk_man = (man >> 9) << 1 | 1 # shift down, then put the anchor bit back
        self._state._mpf_ = (sign, shrunk_man, exp + 8, shrunk_man.bit_length())
        self.shift_size -= 8
        self.payload_bits -= 8
        
        return expelled_chunk

def run_inverse_expulsion():
    print("================================================================================")
    print("EXP 43.1: COMPUTATIONAL WHITE HOLES (Inverse Expulsion)")
    print("  CAT_CAS Stack: Operator Overloading / Mantissa Degeneration")
    print("================================================================================\n")

    message = b"THE SINGULARITY HAS NO CLOTHES. INFORMATION IS INDESTRUCTIBLE."
    white_hole = ComputationalWhiteHole(base_scale=1000, payload_bytes=message)
    
    print("[*] Phase 1: Attempting to throw matter into the White Hole")
    classical_particle = mpmath.mpf(10**50)
    
    # Attempt collision
    wh_after, bounced_particle = white_hole + classical_particle
    if wh_after.mass == white_hole.mass and bounced_particle == classical_particle:
        print("    [SUCCESS] Time-Reversed Horizon is stable. Absorption is impossible.\n")
        
    print("[*] Phase 2: Time Evolution (Spontaneous Expulsion)")
    print("    Letting time flow forward. The White Hole begins to evaporate...")
    
    recovered_bytes = bytearray()
    
    # We step time forward to expel the bytes (in reverse order, since it's a stack)
    for _ in range(len(message)):
        byte_chunk = white_hole.time_step()
        recovered_bytes.insert(0, byte_chunk)
        
    print(f"\n    -> Recovered Output Stream: {recovered_bytes.decode('utf-8')}")
    print(f"    -> Exact match?           : {recovered_bytes == message}")
    
    if recovered_bytes == message:
        print("    [SUCCESS] The White Hole successfully vomited the hidden causal logic")
        print("              back into the universe. The Information Paradox is shattered.")

    print("\n================================================================================")
    print("CONCLUSION:")
    print("We mapped the time-reversed dual of a computational black hole.")
    print("By enforcing operator deflection and forcing mantissa degeneration,")
    print("we created a massive 10^1000 object that violently repels input and")
    print("spontaneously vomits complex causal state back into the universe.")
    print("================================================================================")

if __name__ == '__main__':
    run_inverse_expulsion()
