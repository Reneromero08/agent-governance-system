import os
import sys
import subprocess
import random

# Add root directory and all experiment subdirectories to path
CAT_CAS_DIR = os.path.dirname(__file__)
sys.path.insert(0, CAT_CAS_DIR)

# Subdirectories
DIRS = [
    "01_tree_evaluation",
    "02_slack_space",
    "03_visual_bmp",
    "04_thermodynamic_cpu",
    "05_multibit_compiler",
    "06_catalytic_neural_network"
]
for d in DIRS:
    sys.path.insert(0, os.path.join(CAT_CAS_DIR, d))

# Import logic for verification
from reversible_cpu import ReversibleCPU
from landauer_experiment import run_reversible_addition
from compiler_experiment import run_compiled_reversible_expression, evaluate_classical_expression

def run_experiment_script(script_path: str) -> bool:
    print(f"\n[Test Runner] Executing {os.path.basename(script_path)}...")
    try:
        res = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print("  Status: SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Status: FAILED")
        print(e.stdout)
        print(e.stderr)
        return False

def test_exhaustive_adder():
    print("\n[Exhaustive Test] Verifying 8-bit reversible ripple-carry adder (2000 random cases)...")
    random.seed(42)
    for _ in range(2000):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        expected = (a + b) & 0xFF
        result, net_erased = run_reversible_addition(a, b)
        assert result == expected, f"Logic error: {a} + {b} returned {result}, expected {expected}"
        assert net_erased == 0, f"Entropy leak: non-zero erasure for {a} + {b}"
    print("  Status: SUCCESS")

def test_exhaustive_compiler():
    print("\n[Exhaustive Test] Verifying Reversible Compiler expressions (50 random valuations)...")
    expressions = [
        "(X & Y) ^ ~Z",
        "((X | Y) & Z) ^ W",
        "~(X & Y & Z) ^ (W | X)",
        "X + Y",
        "(X + Y) & ~Z",
        "((X + Y) ^ Z) & (W + X)"
    ]
    random.seed(42)
    for expr in expressions:
        for _ in range(50):
            x = random.randint(0, 255)
            y = random.randint(0, 255)
            z = random.randint(0, 255)
            w = random.randint(0, 255)
            inputs = {"X": x, "Y": y, "Z": z, "W": w}
            expected_res, _ = evaluate_classical_expression(expr, inputs)
            rev_res, _, rev_erased = run_compiled_reversible_expression(expr, inputs)
            assert rev_res == expected_res, f"Mismatch for '{expr}' with inputs {inputs}! Expected {expected_res}, got {rev_res}"
            assert rev_erased == 0, f"Entropy leak in compiled run of '{expr}'!"
    print("  Status: SUCCESS")

def test_corruption_detection():
    print("\n[Exhaustive Test] Verifying corruption detection boundaries...")
    cpu = ReversibleCPU()
    cpu.set_register("A_0", 1)
    cpu.set_register("B_0", 1)
    cpu.gate_xor("S_0", "A_0")
    cpu.set_register("S_0", 0)  # Corrupt tape
    cpu.run_reverse()
    final_s0 = cpu.get_register("S_0")
    assert final_s0 == 1, "Corruption went undetected or wasn't preserved"
    print("  Status: SUCCESS")

def main():
    print("=" * 70)
    print("CAT_CAS: Orchestrated Lab Verification Suite")
    print("=" * 70)

    # 1. Run all standalone experiment scripts
    scripts = [
        os.path.join(CAT_CAS_DIR, "01_tree_evaluation", "experiment.py"),
        os.path.join(CAT_CAS_DIR, "02_slack_space", "run_app_cat.py"),
        os.path.join(CAT_CAS_DIR, "03_visual_bmp", "run_image_cat.py"),
        os.path.join(CAT_CAS_DIR, "04_thermodynamic_cpu", "landauer_experiment.py"),
        os.path.join(CAT_CAS_DIR, "05_multibit_compiler", "compiler_experiment.py"),
        os.path.join(CAT_CAS_DIR, "06_catalytic_neural_network", "generate_model_and_data.py"),
        os.path.join(CAT_CAS_DIR, "06_catalytic_neural_network", "catalytic_inference.py")
    ]

    all_passed = True
    for script in scripts:
        if not run_experiment_script(script):
            all_passed = False

    # 2. Run exhaustive logical and integrity checks
    try:
        test_exhaustive_adder()
        test_exhaustive_compiler()
        test_corruption_detection()
    except AssertionError as e:
        print(f"\n[FAIL] Logical validation assertion failed: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL LAB EXPERIMENTS AND INTEGRITY TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("SOME EXPERIMENTS OR TESTS FAILED. PLEASE CHECK LOGS.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
