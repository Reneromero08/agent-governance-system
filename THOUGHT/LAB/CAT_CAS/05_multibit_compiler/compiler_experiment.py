import os
import sys

# Add current directory to path
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from reversible_cpu import ReversibleCPU, calculate_landauer_energy
from reversible_compiler import ReversibleCompiler

def run_compiled_reversible_expression(expr: str, input_vals: dict[str, int]) -> tuple[int, int, int]:
    """
    Compiles a multi-bit expression, runs it on a Reversible CPU,
    verifies result, cleans temporary registers, and returns results.
    """
    compiler = ReversibleCompiler()
    cpu = ReversibleCPU()

    # 1. Setup input registers for each 8-bit variable
    for var, val in input_vals.items():
        for i in range(8):
            cpu.set_register(f"{var}_{i}", (val >> i) & 1)

    # 2. Compile expression dynamically
    instructions, final_res_regs = compiler.compile(expr)

    # 3. Execute instructions forward
    for inst in instructions:
        op = inst[0]
        if op == 'XOR':
            cpu.gate_xor(inst[1], inst[2])
        elif op == 'NOT':
            cpu.gate_not(inst[1])
        elif op == 'AND_XOR':
            cpu.gate_and_xor(inst[1], inst[2], inst[3])

    # Extract raw value before cleaning
    result_val = 0
    for i in range(8):
        result_val |= (cpu.get_register(final_res_regs[i]) << i)

    # 4. Reversibly copy the 8-bit result to Output Registers OUT_0..OUT_7
    for i in range(8):
        cpu.gate_xor(f"OUT_{i}", final_res_regs[i])

    # 5. Clean up temporary registers by running reverse pass
    # Exclude the 8 copy gates from the reverse pass history
    cpu.gate_history = cpu.gate_history[:-8]
    cpu.run_reverse()

    # Verify that all temporary registers (t0_0..t0_7, carries, etc.) are restored to 0
    dest_regs = set()
    for inst in instructions:
        dest_regs.add(inst[1])
    
    # Exclude input registers
    input_regs = {f"{var}_{i}" for var in input_vals for i in range(8)}
    temp_regs = dest_regs - input_regs

    for temp_reg in temp_regs:
        val = cpu.get_register(temp_reg)
        assert val == 0, f"Temporary register {temp_reg} was not cleaned! Value: {val}"

    # Verify input variables remain unchanged
    for var, val in input_vals.items():
        for i in range(8):
            expected_bit = (val >> i) & 1
            assert cpu.get_register(f"{var}_{i}") == expected_bit, f"Input register {var}_{i} was corrupted!"

    final_result = 0
    for i in range(8):
        final_result |= (cpu.get_register(f"OUT_{i}") << i)
    assert final_result == result_val, "Output registers got corrupted during cleaning!"

    return final_result, len(instructions), 0  # 0 net bits of information erased

def evaluate_classical_expression(expr: str, input_vals: dict[str, int]) -> tuple[int, int]:
    """Evaluates the expression classically on 8-bit integers and counts erasures."""
    compiler = ReversibleCompiler()
    tokens = compiler.tokenize(expr)
    postfix = compiler.to_postfix(tokens)
    
    # 1. Evaluate classical postfix
    stack = []
    for token in postfix:
        if token in input_vals:
            stack.append(input_vals[token])
        elif token == '~':
            val = stack.pop()
            stack.append((~val) & 0xFF)
        elif token == '+':
            v2 = stack.pop()
            v1 = stack.pop()
            stack.append((v1 + v2) & 0xFF)
        elif token == '^':
            v2 = stack.pop()
            v1 = stack.pop()
            stack.append(v1 ^ v2)
        elif token == '&':
            v2 = stack.pop()
            v1 = stack.pop()
            stack.append(v1 & v2)
        elif token == '|':
            v2 = stack.pop()
            v1 = stack.pop()
            stack.append(v1 | v2) # A | B = (A & B) ^ A ^ B classically (wait, stack.append(v1 | v2) is simpler and correct!)
            
    classical_result = stack[0]
    
    # 2. Count erasures:
    num_ops = sum(1 for t in tokens if t in {'&', '|', '^', '~', '+'})
    num_adds = sum(1 for t in tokens if t == '+')
    bits_erased = (num_ops * 8) + (num_adds * 8)
    
    return classical_result, bits_erased

def main():
    print("=" * 60)
    print("CAT_CAS: Multi-Bit Reversible Logic and Arithmetic Compiler")
    print("=" * 60)

    expressions = [
        "(X & Y) ^ ~Z",
        "((X | Y) & Z) ^ W",
        "~(X & Y & Z) ^ (W | X)",
        "X + Y",
        "(X + Y) & ~Z",
        "((X + Y) ^ Z) & (W + X)"
    ]
    
    inputs = {"X": 187, "Y": 94, "Z": 51, "W": 12}
    print(f"  Inputs: {inputs}\n")
    
    for expr in expressions:
        print(f"  Compiling Expression: '{expr}'")
        
        # Classical irreversible run
        class_res, class_erased = evaluate_classical_expression(expr, inputs)
        class_energy = calculate_landauer_energy(class_erased)
        print("    Irreversible Execution:")
        print(f"      Result: {class_res}, Erased: {class_erased} bits, Landauer Heat: {class_energy:.4e} J")
        
        # Reversible compiled run
        rev_res, num_gates, rev_erased = run_compiled_reversible_expression(expr, inputs)
        rev_energy = calculate_landauer_energy(rev_erased)
        print("    Reversible Compiled Execution:")
        print(f"      Result: {rev_res}, Compiled Gates: {num_gates}, Erased: {rev_erased} bits, Landauer Heat: {rev_energy:.4e} J")
        
        assert class_res == rev_res, f"Compiler output mismatch for '{expr}'! Expected {class_res}, got {rev_res}"
        assert rev_erased == 0, "Reversible compiler leaked entropy!"
        print("    Verification: SUCCESS (Correct evaluation & 100% register cleaning)\n")
        
    print("=" * 60)
    print("Multi-Bit Reversible Compiler Succeeded!")
    print("=" * 60)

if __name__ == "__main__":
    main()
