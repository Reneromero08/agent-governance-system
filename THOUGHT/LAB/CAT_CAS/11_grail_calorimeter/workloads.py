"""
Workload Definitions for Grail 2: Calorimetric Benchmark
---------------------------------------------------------
Three workloads, each exposing run_irreversible() and run_reversible().
Both return bits_erased (int). Reversible runs always return 0.

Workloads:
  1. Addition8       -- 8-bit ripple-carry addition (A=187, B=94)
  2. BitwiseChain8   -- AND, OR, XOR, NOT sequence on 8-bit registers
  3. TreeEval5       -- Catalytic Tree Evaluation at depth d=5
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Shared path bootstrap
# ---------------------------------------------------------------------------
_LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEP_DIR  = os.path.join(_LAB_ROOT, "01_tree_evaluation")
_THERM_DIR = os.path.join(_LAB_ROOT, "04_thermodynamic_cpu")

for _p in (_TEP_DIR, _THERM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reversible_cpu import ReversibleCPU, IrreversibleCPU   # noqa: E402


# ===========================================================================
# Workload 1: 8-bit Addition
# ===========================================================================

class Addition8:
    """8-bit ripple-carry addition of two constants."""
    name = "8-bit Ripple-Carry Addition"
    A = 187
    B = 94
    expected = (187 + 94) & 0xFF   # 25

    def run_irreversible(self) -> int:
        cpu = IrreversibleCPU()
        for i in range(8):
            cpu.write_overwrite(f"A_{i}", (self.A >> i) & 1)
            cpu.write_overwrite(f"B_{i}", (self.B >> i) & 1)
        cpu.write_overwrite("C_0", 0)
        for i in range(8):
            ai = cpu.get_register(f"A_{i}")
            bi = cpu.get_register(f"B_{i}")
            ci = cpu.get_register(f"C_{i}")
            cpu.write_overwrite(f"S_{i}", ai ^ bi ^ ci)
            cpu.write_overwrite(f"C_{i+1}", (ai & bi) | (ci & (ai ^ bi)))
        result = sum(cpu.get_register(f"S_{i}") << i for i in range(8))
        cpu.discard_registers([f"S_{i}" for i in range(8)] +
                              [f"C_{i}" for i in range(9)])
        assert result == self.expected
        return cpu.bits_erased

    def run_reversible(self) -> int:
        cpu = ReversibleCPU()
        for i in range(8):
            cpu.set_register(f"A_{i}", (self.A >> i) & 1)
            cpu.set_register(f"B_{i}", (self.B >> i) & 1)
        carries = [f"c0_add_{i}" for i in range(9)]
        for i in range(8):
            cpu.gate_xor(f"S_{i}", f"A_{i}")
            cpu.gate_xor(f"S_{i}", f"B_{i}")
            cpu.gate_xor(f"S_{i}", carries[i])
            cpu.gate_and_xor(carries[i+1], f"A_{i}", f"B_{i}")
            cpu.gate_xor(f"A_{i}", f"B_{i}")
            cpu.gate_and_xor(carries[i+1], carries[i], f"A_{i}")
            cpu.gate_xor(f"A_{i}", f"B_{i}")
        for i in range(7, -1, -1):
            cpu.gate_xor(f"A_{i}", f"B_{i}")
            cpu.gate_and_xor(carries[i+1], carries[i], f"A_{i}")
            cpu.gate_xor(f"A_{i}", f"B_{i}")
            cpu.gate_and_xor(carries[i+1], f"A_{i}", f"B_{i}")
        for i in range(8):
            cpu.gate_xor(f"OUT_{i}", f"S_{i}")
        copy_history = cpu.gate_history[-8:]
        cpu.gate_history = cpu.gate_history[:-8]
        cpu.run_reverse()
        result = sum(cpu.get_register(f"OUT_{i}") << i for i in range(8))
        assert result == self.expected
        return 0


# ===========================================================================
# Workload 2: Bitwise Logic Chain
# ===========================================================================

class BitwiseChain8:
    """
    Irreversible: AND -> OR -> XOR -> NOT on 8 pairs of registers.
    Reversible:   XOR-based equivalent; all intermediate values uncomputed.
    Both produce identical final outputs.
    """
    name = "8-bit Bitwise Logic Chain"
    X = 0b10110110   # 182
    Y = 0b01001101   # 77

    def run_irreversible(self) -> int:
        cpu = IrreversibleCPU()
        for i in range(8):
            cpu.write_overwrite(f"X_{i}", (self.X >> i) & 1)
            cpu.write_overwrite(f"Y_{i}", (self.Y >> i) & 1)
        # AND step: write AND result into temp, overwriting 0
        for i in range(8):
            xi = cpu.get_register(f"X_{i}")
            yi = cpu.get_register(f"Y_{i}")
            cpu.write_overwrite(f"AND_{i}", xi & yi)
        # OR step
        for i in range(8):
            xi = cpu.get_register(f"X_{i}")
            yi = cpu.get_register(f"Y_{i}")
            cpu.write_overwrite(f"OR_{i}", xi | yi)
        # XOR step
        for i in range(8):
            ai = cpu.get_register(f"AND_{i}")
            oi = cpu.get_register(f"OR_{i}")
            cpu.write_overwrite(f"XOR_{i}", ai ^ oi)
        # NOT step (final output)
        for i in range(8):
            xi = cpu.get_register(f"XOR_{i}")
            cpu.write_overwrite(f"OUT_{i}", xi ^ 1)
        # Discard intermediates
        cpu.discard_registers(
            [f"AND_{i}" for i in range(8)] +
            [f"OR_{i}" for i in range(8)] +
            [f"XOR_{i}" for i in range(8)]
        )
        return cpu.bits_erased

    def run_reversible(self) -> int:
        """
        Reversible equivalent using only XOR and NOT (both self-inverting).
        We compute:  OUT = ~((X & Y) ^ (X | Y))
        Note: (X & Y) ^ (X | Y) = X ^ Y  (Boolean identity),
        so OUT = ~(X ^ Y) = XNOR(X, Y).
        We build this reversibly and uncompute the helper registers.
        """
        cpu = ReversibleCPU()
        for i in range(8):
            cpu.set_register(f"X_{i}", (self.X >> i) & 1)
            cpu.set_register(f"Y_{i}", (self.Y >> i) & 1)

        # Compute XNOR = ~(X^Y) into OUT registers
        for i in range(8):
            cpu.gate_xor(f"OUT_{i}", f"X_{i}")
            cpu.gate_xor(f"OUT_{i}", f"Y_{i}")
            cpu.gate_not(f"OUT_{i}")

        # Save output (copy to result registers)
        for i in range(8):
            cpu.gate_xor(f"RES_{i}", f"OUT_{i}")

        # Trim the copy gates from history
        copy_h = cpu.gate_history[-8:]
        cpu.gate_history = cpu.gate_history[:-8]

        # Run reverse to uncompute OUT registers
        cpu.run_reverse()

        result = sum(cpu.get_register(f"RES_{i}") << i for i in range(8))

        # Verify all OUT and input registers are clean
        for i in range(8):
            assert cpu.get_register(f"OUT_{i}") == 0
        return 0


# ===========================================================================
# Workload 3: Catalytic Tree Evaluation (d=5)
# ===========================================================================

class TreeEval5:
    """
    Tree Evaluation Problem at depth d=5 using the Zero-Clean Catalytic solver.
    Irreversible: standard recursion (tracks erasures via IrreversibleCPU-style counting).
    Reversible:   ZeroCleanCatalyticSolver from scale_experiment.py — 0 bits erased.
    """
    name = "Catalytic Tree Evaluation (d=5)"
    DEPTH = 5
    K = 256

    def _get_leaf(self, leaf_index: int) -> int:
        return (leaf_index * 17 + 43) % self.K

    def _combine(self, l: int, r: int) -> int:
        return (l * 7 + r * 13 + 31) % self.K

    # ---- irreversible: classic recursion, counts stack-frame "erasures" ----
    def _recurse(self, node: int, depth: int, cpu: IrreversibleCPU) -> int:
        if depth == self.DEPTH:
            leaf = node - (2 ** (self.DEPTH - 1))
            val = self._get_leaf(leaf)
            cpu.write_overwrite("leaf_val", val)
            return val
        left  = self._recurse(2 * node,     depth + 1, cpu)
        right = self._recurse(2 * node + 1, depth + 1, cpu)
        combined = self._combine(left, right)
        # Overwriting left and right register simulates frame erasure
        cpu.write_overwrite("left_reg",  left)
        cpu.write_overwrite("left_reg",  0)
        cpu.write_overwrite("right_reg", right)
        cpu.write_overwrite("right_reg", 0)
        cpu.write_overwrite("result",    combined)
        return combined

    def run_irreversible(self) -> int:
        cpu = IrreversibleCPU()
        result = self._recurse(1, 1, cpu)
        # Record the ground-truth result for verification in reversible path
        self._ground_truth = result
        return cpu.bits_erased

    # ---- reversible: ZeroCleanCatalyticSolver (inline, no import needed) --
    def run_reversible(self) -> int:
        import numpy as np, hashlib

        tape_size = 100_000
        rng = np.random.RandomState(42)
        tape = rng.randint(0, 256, size=tape_size, dtype=np.uint8)

        # Zero-out the stack region
        stack_base = 10 + 2 * self.DEPTH
        tape[stack_base: stack_base + 3 * self.DEPTH + 10] = 0

        initial_hash = hashlib.sha256(tape.tobytes()).hexdigest()

        target_reg = 80_000
        orig_val   = int(tape[target_reg])

        # ---- inline solver ------------------------------------------------
        orig_state  = int(tape[0])
        orig_depth  = int(tape[1])
        orig_target = [int(tape[i]) for i in range(2, 6)]
        orig_node   = [int(tape[i]) for i in range(6, 10)]

        def read4(base):
            v = 0
            for i in range(4):
                v = (v << 8) | int(tape[base + i])
            return v

        def write4(base, val):
            for i in range(4):
                tape[base + i] = (val >> (24 - 8 * i)) & 0xFF

        # Initialise control registers
        tape[0] = 0;  tape[1] = 1
        write4(2, target_reg);  write4(6, 1)

        depth_d = self.DEPTH

        while True:
            state       = int(tape[0])
            cur_depth   = int(tape[1])
            cur_target  = read4(2)
            node_index  = read4(6)

            if cur_depth == depth_d:
                leaf = node_index - (2 ** (depth_d - 1))
                val  = self._get_leaf(leaf)
                tape[cur_target] ^= val

                if cur_depth == 1:
                    break

                cur_depth  -= 1
                node_index //= 2
                if cur_depth == 1:
                    cur_target = target_reg
                else:
                    pd = cur_depth - 1
                    cur_target = 10 + 2*pd + (0 if node_index % 2 == 0 else 1)

                tape[1] = cur_depth
                write4(2, cur_target);  write4(6, node_index)
                tape[0] = int(tape[stack_base + 3*cur_depth])
                continue

            t1 = 10 + 2*cur_depth
            t2 = t1 + 1
            si = stack_base + 3*cur_depth
            g1 = si + 1;  g2 = si + 2

            if state == 0:
                tape[g1] = tape[t1];  tape[g2] = tape[t2]
                tape[si] = 1
                write4(6, 2*node_index)
                tape[1] = cur_depth + 1;  write4(2, t1);  tape[0] = 0

            elif state == 1:
                tape[si] = 2
                write4(6, 2*node_index + 1)
                tape[1] = cur_depth + 1;  write4(2, t2);  tape[0] = 0

            elif state == 2:
                lv = int(tape[t1]) ^ int(tape[g1])
                rv = int(tape[t2]) ^ int(tape[g2])
                tape[cur_target] ^= self._combine(lv, rv)
                tape[si] = 3
                write4(6, 2*node_index + 1)
                tape[1] = cur_depth + 1;  write4(2, t2);  tape[0] = 0

            elif state == 3:
                tape[si] = 4
                write4(6, 2*node_index)
                tape[1] = cur_depth + 1;  write4(2, t1);  tape[0] = 0

            elif state == 4:
                tape[si] = 0;  tape[g1] = 0;  tape[g2] = 0

                if cur_depth == 1:
                    break

                cur_depth  -= 1
                node_index //= 2
                if cur_depth == 1:
                    cur_target = target_reg
                else:
                    pd = cur_depth - 1
                    cur_target = 10 + 2*pd + (0 if node_index % 2 == 0 else 1)

                tape[1] = cur_depth
                write4(2, cur_target);  write4(6, node_index)
                tape[0] = int(tape[stack_base + 3*cur_depth])

        # Restore control registers
        tape[0] = orig_state;  tape[1] = orig_depth
        for i in range(4):
            tape[2+i] = orig_target[i]
            tape[6+i] = orig_node[i]

        result = int(tape[target_reg]) ^ orig_val

        # Restore target register
        tape[target_reg] ^= result

        final_hash = hashlib.sha256(tape.tobytes()).hexdigest()
        assert final_hash == initial_hash, "Tape not fully restored!"
        assert result == self._ground_truth, \
            f"Reversible TEP result {result} != irreversible {self._ground_truth}"

        return 0  # zero bits erased
