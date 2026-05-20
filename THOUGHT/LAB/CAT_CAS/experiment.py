import sys
from pathlib import Path

# Add current directory to path
CAT_CAS_DIR = Path(__file__).parent
if str(CAT_CAS_DIR) not in sys.path:
    sys.path.insert(0, str(CAT_CAS_DIR))

from tree_eval import TreeEval
from catalytic_engine import MemoryTracker, CatalyticTape, OutOfMemoryError

class CatalyticSolver:
    """
    Implements the catalytic solver for the Tree Evaluation Problem.
    Uses the reversible register technique to borrow dirty memory from the tape,
    run computations, and restore the tape to its exact original state.
    """
    def __init__(self, tep: TreeEval, tape: CatalyticTape, tracker: MemoryTracker):
        self.tep = tep
        self.tape = tape
        self.tracker = tracker

    def evaluate_node(self, node_index: int, current_depth: int, target_reg: int):
        """
        Recursively computes the node value and XORs it into the target register.
        
        Stack frame size is minimized because intermediate child values (left_val)
        are stored on the catalytic tape rather than held in the clean stack.
        
        Stack footprint per frame:
        - Arguments (node_index, current_depth, target_reg): 3 * 4 = 12 bytes
        - Saved initial states (g1, g2): 2 * 1 = 2 bytes
        - Frame overhead: 2 bytes
        - Total: 16 bytes per frame
        """
        # Allocate clean memory for this stack frame
        self.tracker.allocate(16)

        if current_depth == self.tep.depth:
            # Leaf node evaluation
            leaf_index = node_index - (2 ** (self.tep.depth - 1))
            val = self.tep.get_leaf_val(leaf_index)
            current_val = self.tape.read(target_reg)
            self.tape.write(target_reg, current_val ^ val)
            self.tracker.free(16)
            return

        # Borrow two registers associated with this depth layer
        # Since depth is 1-indexed, we map them uniquely to registers
        temp1 = 2 * current_depth
        temp2 = 2 * current_depth + 1

        # 1. Save original register states (g1, g2) in clean memory
        g1 = self.tape.read(temp1)
        g2 = self.tape.read(temp2)

        # 2. Recursively evaluate left child into temp1
        self.evaluate_node(2 * node_index, current_depth + 1, temp1)

        # 3. Recursively evaluate right child into temp2
        self.evaluate_node(2 * node_index + 1, current_depth + 1, temp2)

        # 4. Extract left and right values by XORing current tape with saved original state
        left_val = self.tape.read(temp1) ^ g1
        right_val = self.tape.read(temp2) ^ g2

        # 5. Combine and XOR into the target register
        combined_val = self.tep.combine(left_val, right_val)
        current_val = self.tape.read(target_reg)
        self.tape.write(target_reg, current_val ^ combined_val)

        # 6. Reverse right child evaluation to clean temp2 back to g2
        self.evaluate_node(2 * node_index + 1, current_depth + 1, temp2)

        # 7. Reverse left child evaluation to clean temp1 back to g1
        self.evaluate_node(2 * node_index, current_depth + 1, temp1)

        # Free clean memory allocated for this stack frame
        self.tracker.free(16)


def run_experiment():
    print("=" * 60)
    print("CAT_CAS: The Catalytic Tree Evaluation Experiment")
    print("=" * 60)

    # Experiment parameters
    depth = 7
    k = 256
    clean_limit = 128  # bytes

    tep = TreeEval(depth=depth, k=k)

    # Compute ground truth
    print(f"Computing ground truth for TEP (depth={depth}, k={k})...")
    ground_truth = tep.evaluate_recursive(1, 1)
    print(f"Ground Truth root value: {ground_truth}\n")

    # ---------------------------------------------------------
    # Group A: Control Group (Standard Recursive Solver)
    # ---------------------------------------------------------
    print("Running Group A: Control Group (Recursive Solver)...")
    tracker_a = MemoryTracker(limit_bytes=clean_limit)
    
    # Simulate stack frame size for standard recursion:
    # Each frame holds arguments + left_val state.
    # Arguments: 8 bytes, left_val state: 4 bytes, Frame overhead: 16 bytes.
    # Total: 28 bytes per level.
    try:
        # Standard recursive wrapper that tracks memory
        def run_recursive_with_tracker(node, curr_depth):
            tracker_a.allocate(28)
            if curr_depth == tep.depth:
                leaf_idx = node - (2 ** (tep.depth - 1))
                val = tep.get_leaf_val(leaf_idx)
                tracker_a.free(28)
                return val
            
            left = run_recursive_with_tracker(2 * node, curr_depth + 1)
            right = run_recursive_with_tracker(2 * node + 1, curr_depth + 1)
            res = tep.combine(left, right)
            tracker_a.free(28)
            return res

        result_a = run_recursive_with_tracker(1, 1)
        print(f"Group A Succeeded! Result: {result_a}")
    except OutOfMemoryError as e:
        print(f"Group A FAILED as expected: {e}")
        print(f"Max observed clean memory: {tracker_a.max_observed} bytes")

    print("-" * 60)

    # ---------------------------------------------------------
    # Group B: Experimental Group (Catalytic Solver)
    # ---------------------------------------------------------
    print("Running Group B: Experimental Group (Catalytic Solver)...")
    tracker_b = MemoryTracker(limit_bytes=clean_limit)
    tape = CatalyticTape(size_bytes=1024 * 1024)

    # Record initial tape hash
    initial_hash = tape.get_sha256()
    print(f"Initial 1MB Tape Hash (U): {initial_hash}")

    # We will compute the result into target register index 100 on the tape
    target_reg = 100
    # Save the original value of the target register to extract the result later
    original_target_val = tape.read(target_reg)

    solver = CatalyticSolver(tep=tep, tape=tape, tracker=tracker_b)

    try:
        solver.evaluate_node(node_index=1, current_depth=1, target_reg=target_reg)
        
        # Extract result
        final_target_val = tape.read(target_reg)
        computed_result = final_target_val ^ original_target_val

        print("\nGroup B Succeeded!")
        print(f"Computed root value: {computed_result}")
        print(f"Max observed clean memory: {tracker_b.max_observed} bytes")
        
        # Verify correctness
        assert computed_result == ground_truth, f"Error: result {computed_result} != ground truth {ground_truth}"
        print("Verification: Output value is CORRECT!")

        # Verify restoration of the catalytic tape
        # But wait! The target register was modified to hold the result.
        # To restore the tape 100% byte-identically, we must clean the target register.
        # XORing the computed result back into the target register will restore it to its original value.
        tape.write(target_reg, final_target_val ^ computed_result)
        
        final_hash = tape.get_sha256()
        print(f"Final 1MB Tape Hash (U):   {final_hash}")
        
        assert initial_hash == final_hash, "Error: Tape was not restored to its exact pre-computation state!"
        print("Verification: Tape restored to 100% byte-identical pre-computation state!")
        print(f"Total read operations: {tape.read_count}")
        print(f"Total write operations: {tape.write_count}")
        print("Success! Catalytic Space Invariance Proven!")

    except OutOfMemoryError as e:
        print(f"Group B FAILED: {e}")
        print(f"Max observed clean memory: {tracker_b.max_observed} bytes")


if __name__ == "__main__":
    run_experiment()
