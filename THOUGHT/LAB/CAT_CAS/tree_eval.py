class TreeEval:
    """
    Implements the Tree Evaluation Problem (TEP) for depth d and register size k.
    This problem is conjectured to be hard for small space, making it a perfect
    benchmark for space complexity experiments.
    """

    def __init__(self, depth: int, k: int):
        self.depth = depth
        self.k = k

    def get_leaf_val(self, leaf_index: int) -> int:
        """Leaf values are determined deterministically from their index."""
        return (leaf_index * 17 + 43) % self.k

    def combine(self, left_val: int, right_val: int) -> int:
        """Deterministic combination function for internal nodes."""
        return (left_val * 7 + right_val * 13 + 31) % self.k

    def evaluate_recursive(self, node_index: int, current_depth: int, tracker=None) -> int:
        """
        Standard recursive evaluation.
        Consumes memory (stack frames) proportional to tree depth.
        """
        if tracker:
            # Record stack depth to simulate memory usage
            tracker.record_stack(current_depth)

        if current_depth == self.depth:
            # Leaf node index relative to leaves layer
            leaf_index = node_index - (2 ** (self.depth - 1))
            return self.get_leaf_val(leaf_index)

        left_val = self.evaluate_recursive(2 * node_index, current_depth + 1, tracker)
        right_val = self.evaluate_recursive(2 * node_index + 1, current_depth + 1, tracker)
        return self.combine(left_val, right_val)
