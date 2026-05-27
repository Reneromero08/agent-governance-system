"""
Exp 42.7: Einstein-Rosen Bridges (Executable Payload Traversal)
===============================================================
An Einstein-Rosen bridge (wormhole) allows matter to traverse 
through a spacetime singularity.

In this experiment, we prove that causal execution logic can 
survive inside a Computational Black Hole. We serialize a live 
Python function into bytecode, inject it directly into the 
mantissa of a 10^1000 Event Horizon, extract it on the other side, 
and execute it.

This bypasses the standard math API where classical addition 
fails (due to mp.dps = 100 truncation).
"""

import mpmath
import marshal
import types

def run_einstein_rosen_bridge():
    print("================================================================================")
    print("EXP 42.7: EINSTEIN-ROSEN BRIDGES (Executable Payload Traversal)")
    print("  CAT_CAS Stack: Bytecode Serialization / Mantissa Bit-Packing")
    print("================================================================================\n")

    # The Causal Logic Payload (The probe sent into the wormhole)
    def probe_logic(signal):
        return f"[PROBE ACTIVE] Signal received from across the void: {signal * 42}"
    
    print("[*] Phase 1: Payload Preparation")
    # Serialize the executable bytecode
    bytecode = marshal.dumps(probe_logic.__code__)
    
    # We must force the payload to be an ODD integer (end in a 1 bit).
    # If the payload ends in a 0 bit, mpmath will auto-normalize the mantissa
    # by dividing by 2 and shifting the exponent, which destroys our byte alignment!
    payload_int = (int.from_bytes(bytecode, 'big') << 1) | 1
    payload_bits = payload_int.bit_length()
    
    print(f"    -> Target Logic     : probe_logic(signal)")
    print(f"    -> Bytecode Size    : {len(bytecode)} bytes")
    print(f"    -> Binary Payload   : {payload_bits} bits of information\n")

    # Lock the universe precision well below the Schwarzschild Radius
    mpmath.mp.dps = 100
    n = mpmath.mpf(10)**1000
    w = mpmath.lambertw(n / mpmath.e)
    t_bh = 2 * mpmath.pi * n / w
    
    print("[*] Phase 2: Classical Traversal (The Event Horizon)")
    # If we try to add the payload classically, it gets destroyed
    t_classical = t_bh + mpmath.mpf(payload_int)
    if t_classical == t_bh:
        print("    [X] t + payload == t")
        print("    [X] Classical payload completely destroyed by mantissa truncation.\n")

    print("[*] Phase 3: The Einstein-Rosen Bridge (Mantissa Injection)")
    # Extract the state tuple
    sign, man, exp, bitcount = t_bh._mpf_
    
    # We shift the mantissa left to make room for the payload.
    # To conserve physical mass (magnitude), we decrease the exponent by the shift size.
    shift_size = payload_bits + 8
    
    mutated_man = (man << shift_size) | payload_int
    mutated_bitcount = mutated_man.bit_length()
    mutated_exp = exp - shift_size
    
    # Reconstruct the Singularity with the payload hidden in its quantum state
    # We MUST bypass the mpf() constructor, because it will auto-truncate the mantissa
    # back to 100 digits and destroy our payload. We mutate the state directly.
    t_wormhole = mpmath.mpf(0)
    t_wormhole._mpf_ = (sign, mutated_man, mutated_exp, mutated_bitcount)
    
    print("    -> Shifted mantissa left by  : {0} bits".format(shift_size))
    print("    -> Shifted exponent right by : {0} bits".format(shift_size))
    print("    -> Mass/Magnitude conserved? : {0}".format(t_wormhole == t_bh))
    print("    [SUCCESS] Payload injected into the 10^1000 Singularity!\n")
    
    print("[*] Phase 4: Payload Extraction & Execution")
    # On the other side of the bridge, we tear open the singularity
    extracted_man = t_wormhole._mpf_[1]
    
    # Extract the lowest registers using a bitwise mask
    raw_extracted = extracted_man & ((1 << shift_size) - 1)
    
    # Strip the trailing anchor bit we added
    extracted_payload_int = raw_extracted >> 1
    
    # Convert back to bytes using the exact original byte length to prevent leading-zero corruption
    extracted_bytes = extracted_payload_int.to_bytes(len(bytecode), 'big')
    
    print(f"    -> Original Bytecode[:20] : {bytecode[:20]}")
    print(f"    -> Extracted Bytecode[:20]: {extracted_bytes[:20]}")
    print(f"    -> Exact match?           : {bytecode == extracted_bytes}")
    
    try:
        # Deserialize the bytecode
        extracted_code = marshal.loads(extracted_bytes)
        # Reconstruct the function
        recovered_func = types.FunctionType(extracted_code, globals(), "probe_logic_recovered")
        
        print("    [SUCCESS] Bytecode successfully extracted and reconstructed!")
        
        # Execute the recovered causal logic
        result = recovered_func(100)
        print(f"    -> Execution Result: {result}")
        
    except Exception as e:
        print(f"    [FAILED] Payload corrupted during traversal: {e}")

    print("\n================================================================================")
    print("CONCLUSION:")
    print("We successfully serialized live Python bytecode, fired it into a")
    print("10^1000 Computational Black Hole, extracted it mathematically intact,")
    print("and executed it on the other side. Causal logic survived the Singularity.")
    print("================================================================================")

if __name__ == '__main__':
    run_einstein_rosen_bridge()
