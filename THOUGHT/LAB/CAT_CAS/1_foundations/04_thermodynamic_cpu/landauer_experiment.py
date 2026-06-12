import os
import sys

# Add current directory to path
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from reversible_cpu import ReversibleCPU, IrreversibleCPU, calculate_landauer_energy

def run_irreversible_addition(a: int, b: int) -> tuple[int, int]:
    """Runs a standard 8-bit addition on an Irreversible CPU, tracking erased bits."""
    cpu = IrreversibleCPU()
    for i in range(8):
        cpu.write_overwrite(f"A_{i}", (a >> i) & 1)
        cpu.write_overwrite(f"B_{i}", (b >> i) & 1)

    cpu.write_overwrite("C_0", 0)

    for i in range(8):
        ai = cpu.get_register(f"A_{i}")
        bi = cpu.get_register(f"B_{i}")
        ci = cpu.get_register(f"C_{i}")

        sum_bit = ai ^ bi ^ ci
        cpu.write_overwrite(f"S_{i}", sum_bit)

        carry_bit = (ai & bi) | (ci & (ai ^ bi))
        cpu.write_overwrite(f"C_{i+1}", carry_bit)

    result = 0
    for i in range(8):
        result |= (cpu.get_register(f"S_{i}") << i)
    
    intermediate_regs = [f"S_{i}" for i in range(8)] + [f"C_{i}" for i in range(9)]
    cpu.discard_registers(intermediate_regs)

    return result, cpu.bits_erased

def run_reversible_addition(a: int, b: int) -> tuple[int, int]:
    """Runs a strict 8-bit addition on a Reversible CPU, using Toffoli gates and carry cleanup."""
    cpu = ReversibleCPU()

    for i in range(8):
        cpu.set_register(f"A_{i}", (a >> i) & 1)
        cpu.set_register(f"B_{i}", (b >> i) & 1)

    # Carry prefix
    carries = [f"c0_add_{i}" for i in range(9)]

    for i in range(8):
        cpu.gate_xor(f"S_{i}", f"A_{i}")
        cpu.gate_xor(f"S_{i}", f"B_{i}")
        cpu.gate_xor(f"S_{i}", carries[i])

        cpu.gate_and_xor(carries[i+1], f"A_{i}", f"B_{i}")
        cpu.gate_xor(f"A_{i}", f"B_{i}")
        cpu.gate_and_xor(carries[i+1], carries[i], f"A_{i}")
        cpu.gate_xor(f"A_{i}", f"B_{i}")

    # Dynamically uncompute carries forward
    for i in range(7, -1, -1):
        cpu.gate_xor(f"A_{i}", f"B_{i}")
        cpu.gate_and_xor(carries[i+1], carries[i], f"A_{i}")
        cpu.gate_xor(f"A_{i}", f"B_{i}")
        cpu.gate_and_xor(carries[i+1], f"A_{i}", f"B_{i}")

    # Copy output
    for i in range(8):
        cpu.gate_xor(f"OUT_{i}", f"S_{i}")

    # Remove the copy gates from the history before running reverse
    copy_history = cpu.gate_history[-8:]
    cpu.gate_history = cpu.gate_history[:-8]

    cpu.run_reverse()

    result = 0
    for i in range(8):
        result |= (cpu.get_register(f"OUT_{i}") << i)

    for i in range(8):
        assert cpu.get_register(f"S_{i}") == 0, f"Register S_{i} was not cleaned!"
    for i in range(9):
        assert cpu.get_register(carries[i]) == 0, f"Register {carries[i]} was not cleaned!"

    return result, 0

def main():
    print("=" * 60)
    print("CAT_CAS: Thermodynamic Reversible Ripple-Carry Adder & Landauer Limit")
    print("=" * 60)

    print("[Adder Run] Running 8-bit Ripple-Carry Adder...")
    a = 187
    b = 94
    expected_sum = (a + b) & 0xFF
    
    sum_irrev, erased_irrev = run_irreversible_addition(a, b)
    energy_irrev = calculate_landauer_energy(erased_irrev)
    print("  Group A (Irreversible Control):")
    print(f"    Sum: {sum_irrev}, Erased: {erased_irrev} bits, Landauer Heat: {energy_irrev:.4e} J")
    
    sum_rev, erased_rev = run_reversible_addition(a, b)
    energy_rev = calculate_landauer_energy(erased_rev)
    print("  Group B (Reversible Catalytic):")
    print(f"    Sum: {sum_rev}, Erased: {erased_rev} bits, Landauer Heat: {energy_rev:.4e} J")
    
    assert sum_irrev == expected_sum, f"Irreversible addition failed! Expected {expected_sum}, got {sum_irrev}"
    assert sum_rev == expected_sum, f"Reversible addition failed! Expected {expected_sum}, got {sum_rev}"
    assert erased_rev == 0, "Reversible addition leaked entropy!"
    
    print("=" * 60)
    print("Reversible Ripple-Carry Adder Experiment Succeeded!")
    print("=" * 60)

if __name__ == "__main__":
    main()
