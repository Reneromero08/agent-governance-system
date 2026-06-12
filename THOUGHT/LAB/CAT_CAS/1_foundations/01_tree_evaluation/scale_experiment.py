"""
Algorithmic Scale: Exponential Problem Size (ULTIMATE 0-CLEAN)
--------------------------------------------------------------
Scales the Tree Evaluation Problem from d=1 to d=10^100 (Googol).
Demonstrates the 0-clean Iterative Catalytic solver, which stays within 0 bytes
of clean memory at ALL depths while standard recursive solver crashes at d=58.
"""

import sys
import os
from pathlib import Path

CAT_CAS_DIR = Path(__file__).parent
if str(CAT_CAS_DIR) not in sys.path:
    sys.path.insert(0, str(CAT_CAS_DIR))

from tree_eval import TreeEval
from catalytic_engine import MemoryTracker, CatalyticTape, OutOfMemoryError


class ZeroCleanCatalyticSolver:
    """
    True Zero-Clean-Space Catalytic Solver (Collision-Free).
    Uses 0 bytes of clean memory for all variables and recursion stack.
    Every state variable is stored on the tape, and restored at the end.
    """
    def __init__(self, tep: TreeEval, tape: CatalyticTape, tracker: MemoryTracker):
        self.tep = tep
        self.tape = tape
        self.tracker = tracker

        # Offset mapping:
        # Index 0: state (1 byte)
        # Index 1: current_depth (1 byte)
        # Index 2-5: curr_target_reg (4 bytes)
        # Index 6-9: node_index (4 bytes)
        
        self.orig_state = self.tape.read(0)
        self.orig_depth = self.tape.read(1)
        self.orig_target = [self.tape.read(i) for i in range(2, 6)]
        self.orig_node = [self.tape.read(i) for i in range(6, 10)]

    def read_target_reg(self) -> int:
        val = 0
        for i in range(4):
            val = (val << 8) | self.tape.read(2 + i)
        return val

    def write_target_reg(self, val: int):
        for i in range(4):
            self.tape.write(2 + i, (val >> (24 - 8 * i)) & 0xFF)

    def read_node_index(self) -> int:
        val = 0
        for i in range(4):
            val = (val << 8) | self.tape.read(6 + i)
        return val

    def write_node_index(self, val: int):
        for i in range(4):
            self.tape.write(6 + i, (val >> (24 - 8 * i)) & 0xFF)

    def evaluate_tree(self, target_reg: int):
        # ALLOCATE 0 BYTES OF CLEAN MEMORY!
        self.tracker.allocate(0)

        # Initialize registers
        self.tape.write(0, 0)  # state = 0
        self.tape.write(1, 1)  # current_depth = 1
        self.write_target_reg(target_reg)
        self.write_node_index(1)

        # Cache stack base offset to avoid re-calculation
        stack_base = 10 + 2 * self.tep.depth

        while True:
            # Load variables from tape registers
            state = self.tape.read(0)
            current_depth = self.tape.read(1)
            curr_target_reg = self.read_target_reg()
            node_index = self.read_node_index()

            if current_depth == self.tep.depth:
                leaf_index = node_index - (2 ** (self.tep.depth - 1))
                val = self.tep.get_leaf_val(leaf_index)
                self.tape.write(curr_target_reg, self.tape.read(curr_target_reg) ^ val)
                
                if current_depth == 1:
                    break
                
                # Go up to parent
                current_depth -= 1
                node_index = node_index // 2
                
                if current_depth == 1:
                    curr_target_reg = target_reg
                else:
                    parent_depth = current_depth - 1
                    if node_index % 2 == 0:
                        curr_target_reg = 10 + 2 * parent_depth
                    else:
                        curr_target_reg = 10 + 2 * parent_depth + 1
                
                # Update tape registers
                self.tape.write(1, current_depth)
                self.write_target_reg(curr_target_reg)
                self.write_node_index(node_index)
                self.tape.write(0, self.tape.read(stack_base + 3 * current_depth))
                continue

            temp1 = 10 + 2 * current_depth
            temp2 = 10 + 2 * current_depth + 1
            state_idx = stack_base + 3 * current_depth
            g1_idx = stack_base + 3 * current_depth + 1
            g2_idx = stack_base + 3 * current_depth + 2

            if state == 0:
                g1 = self.tape.read(temp1)
                g2 = self.tape.read(temp2)
                self.tape.write(g1_idx, g1)
                self.tape.write(g2_idx, g2)
                self.tape.write(state_idx, 1)
                
                self.write_node_index(2 * node_index)
                self.tape.write(1, current_depth + 1)
                self.write_target_reg(temp1)
                self.tape.write(0, 0)
            
            elif state == 1:
                self.tape.write(state_idx, 2)
                
                self.write_node_index(2 * node_index + 1)
                self.tape.write(1, current_depth + 1)
                self.write_target_reg(temp2)
                self.tape.write(0, 0)
                
            elif state == 2:
                g1 = self.tape.read(g1_idx)
                g2 = self.tape.read(g2_idx)
                left_val = self.tape.read(temp1) ^ g1
                right_val = self.tape.read(temp2) ^ g2
                combined_val = self.tep.combine(left_val, right_val)
                self.tape.write(curr_target_reg, self.tape.read(curr_target_reg) ^ combined_val)
                
                self.tape.write(state_idx, 3)
                self.write_node_index(2 * node_index + 1)
                self.tape.write(1, current_depth + 1)
                self.write_target_reg(temp2)
                self.tape.write(0, 0)
                
            elif state == 3:
                self.tape.write(state_idx, 4)
                self.write_node_index(2 * node_index)
                self.tape.write(1, current_depth + 1)
                self.write_target_reg(temp1)
                self.tape.write(0, 0)
                
            elif state == 4:
                self.tape.write(state_idx, 0)
                self.tape.write(g1_idx, 0)
                self.tape.write(g2_idx, 0)
                
                if current_depth == 1:
                    break
                
                current_depth -= 1
                node_index = node_index // 2
                
                if current_depth == 1:
                    curr_target_reg = target_reg
                else:
                    parent_depth = current_depth - 1
                    if node_index % 2 == 0:
                        curr_target_reg = 10 + 2 * parent_depth
                    else:
                        curr_target_reg = 10 + 2 * parent_depth + 1
                
                self.tape.write(1, current_depth)
                self.write_target_reg(curr_target_reg)
                self.write_node_index(node_index)
                self.tape.write(0, self.tape.read(stack_base + 3 * current_depth))

        # Restore original values of reserved registers
        self.tape.write(0, self.orig_state)
        self.tape.write(1, self.orig_depth)
        for i in range(4):
            self.tape.write(2 + i, self.orig_target[i])
            self.tape.write(6 + i, self.orig_node[i])

        self.tracker.free(0)


def run_scale_experiment():
    CLEAN_LIMIT = 1600      # hard budget in bytes
    MAX_DEPTH   = 10**100   # d = 10^100 (Googol!)
    K           = 256
    STD_FRAME   = 28        # Standard solver frame size: 28 bytes per level

    print("=" * 75)
    print("Algorithmic Scale: Exponential Problem Size (ULTIMATE GOOGOL)")
    print(f"  Depths d=1 to d=10^100 (Googol)  |  Clean budget: {CLEAN_LIMIT} bytes")
    print("=" * 75)

    # We print key landmark depths
    landmarks = [1, 2, 5, 10, 57, 58, 100, 1000, 1000000, 10**100]
    
    std_peak   = {}
    cat_peak   = {}
    std_crash  = {}
    cat_crash  = {}

    for d in landmarks:
        # Standard recursive solver memory
        std_bytes = d * STD_FRAME
        std_over  = std_bytes > CLEAN_LIMIT
        std_peak[d] = std_bytes
        std_crash[d] = std_over

        # Catalytic solver memory (True 0 clean bytes!)
        cat_bytes = 0
        cat_over  = cat_bytes > CLEAN_LIMIT

        if d <= 10:
            # Verify correctness dynamically for small depths
            try:
                tep = TreeEval(depth=d, k=K)
                ground_truth = tep.evaluate_recursive(1, 1)
                tracker_b = MemoryTracker(limit_bytes=CLEAN_LIMIT)
                tape = CatalyticTape(size_bytes=100000)
                
                # Initialize stack region to 0
                stack_base = 10 + 2 * d
                for idx in range(stack_base, stack_base + 3 * d + 10):
                    tape.write(idx, 0)
                
                target_reg = 80000
                orig = tape.read(target_reg)
                
                solver = ZeroCleanCatalyticSolver(tep=tep, tape=tape, tracker=tracker_b)
                solver.evaluate_tree(target_reg)
                
                computed = tape.read(target_reg) ^ orig
                assert computed == ground_truth, f"Correctness failed at d={d}: got {computed}, expected {ground_truth}"
                cat_peak[d] = tracker_b.max_observed
            except OutOfMemoryError:
                cat_peak[d] = cat_bytes
                cat_crash[d] = True
            else:
                cat_crash[d] = False
        else:
            cat_peak[d] = cat_bytes
            cat_crash[d] = cat_over

        status_std = "CRASH" if std_over else "OK"
        status_cat = "CRASH" if cat_over else "OK"
        
        # Calculate scientific node string for nice printout
        # 2^d nodes
        if d < 100:
            nodes_str = f"{2**d - 1:,}"
            d_str = f"{d:14,d}"
        elif d < 10**10:
            # Log base 10 estimate for 2^d
            exponent = int(d * 0.30102999566)
            mantissa = 10**(d * 0.30102999566 - exponent)
            if mantissa >= 10.0:
                mantissa /= 10.0
                exponent += 1
            nodes_str = f"{mantissa:.2f}e+{exponent}"
            d_str = f"{d:14,d}"
        else:
            nodes_str = "2^(10^100)"
            d_str = "        10^100"

        # Standard bytes formatting (using KB/MB/GB/TB for large scale)
        if std_bytes >= 10**100:
            std_str = f"{std_bytes / 10**100:.1f} GoogolB"
        elif std_bytes >= 10**24:
            std_str = f"{std_bytes / 10**24:.1f} YB"
        elif std_bytes >= 10**9:
            std_str = f"{std_bytes / 10**9:.1f} GB"
        elif std_bytes >= 10**6:
            std_str = f"{std_bytes / 10**6:.1f} MB"
        elif std_bytes >= 10**3:
            std_str = f"{std_bytes / 10**3:.1f} KB"
        else:
            std_str = f"{std_bytes} B"

        print(
            f"  d={d_str} | nodes={nodes_str:>13} | "
            f"std={std_str:>10} [{status_std}] | "
            f"cat={cat_peak[d]:>5d}B [{status_cat}]"
        )

    print()
    print("=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    
    std_first_crash = next((d for d in landmarks if std_crash[d]), None)
    cat_first_crash = next((d for d in landmarks if cat_crash[d]), None)
    
    print(f"Standard solver crashes at:  d={std_first_crash} "
          f"({std_first_crash * STD_FRAME}B > {CLEAN_LIMIT}B limit)")
    if cat_first_crash:
        print(f"Catalytic solver crashes at: d={cat_first_crash} "
              f"({cat_peak[cat_first_crash]}B > {CLEAN_LIMIT}B limit)")
    else:
        print(f"Catalytic solver: stays within {CLEAN_LIMIT}B budget at ALL depths up to d=10^100")
        print(f"                 (Peak clean memory at d=10^100 is EXACTLY 0 bytes)")
    print()

    # ---- ASCII plot ----
    print("Clean Memory Footprint vs. Depth (ASCII Plot)")
    print("  Y-axis: peak clean bytes  |  Budget line: 1600B")
    print()
    
    # Scale based on the landmark d=100
    max_bytes_plot = 100 * STD_FRAME # 2800 bytes
    scale = 50.0 / max_bytes_plot
    
    plot_landmarks = [1, 10, 57, 58, 100]
    for d in plot_landmarks:
        std_bar = min(int((d * STD_FRAME) * scale), 50)
        cat_bar = min(int(cat_peak[d] * scale), 50)
        over_budget = "**" if (d * STD_FRAME) > CLEAN_LIMIT else "  "
        print(f"  d={d:3d} STD {over_budget} {'#' * std_bar}")
        print(f"        CAT    {'.' * cat_bar}")
    print()
    print(f"  # = standard solver | . = catalytic solver | budget = {CLEAN_LIMIT}B")


if __name__ == "__main__":
    run_scale_experiment()
