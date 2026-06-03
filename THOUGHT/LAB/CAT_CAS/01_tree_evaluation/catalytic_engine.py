import hashlib
import numpy as np

class OutOfMemoryError(Exception):
    """Raised when clean memory usage exceeds the allowed limit."""
    pass

class MemoryTracker:
    """
    Enforces memory limits for clean space W.
    We track clean memory in bytes.
    Each recursive call stack frame is simulated as taking 16 bytes.
    Each local variable allocation is tracked explicitly.
    """
    def __init__(self, limit_bytes: int = 128):
        self.limit_bytes = limit_bytes
        self.current_bytes = 0
        self.max_observed = 0

    def allocate(self, num_bytes: int):
        self.current_bytes += num_bytes
        if self.current_bytes > self.max_observed:
            self.max_observed = self.current_bytes
        if self.current_bytes > self.limit_bytes:
            raise OutOfMemoryError(
                f"Clean memory limit exceeded! Allocated {self.current_bytes} bytes, "
                f"limit is {self.limit_bytes} bytes."
            )

    def free(self, num_bytes: int):
        self.current_bytes = max(0, self.current_bytes - num_bytes)

    def record_stack(self, depth: int):
        # Each stack frame consumes 16 bytes (simulating return address + frame pointer)
        stack_bytes = depth * 16
        if stack_bytes > self.limit_bytes:
            raise OutOfMemoryError(
                f"Stack overflow! Stack depth {depth} requires {stack_bytes} bytes, "
                f"limit is {self.limit_bytes} bytes."
            )
        if stack_bytes > self.max_observed:
            self.max_observed = stack_bytes


class CatalyticTape:
    """
    A 1 MB catalytic memory tape initialized with random data.
    Simulates the large 'dirty' workspace U.
    We track read and write operations.
    """
    def __init__(self, size_bytes: int = 1024 * 1024):
        self.size_bytes = size_bytes
        # Seed for reproducibility
        rng = np.random.default_rng(42)
        # Random integers in [0, 255]
        self.tape = rng.integers(0, 256, size=size_bytes, dtype=np.uint8)
        self.write_count = 0
        self.read_count = 0

    def read(self, index: int) -> int:
        self.read_count += 1
        return int(self.tape[index])

    def write(self, index: int, val: int):
        self.write_count += 1
        self.tape[index] = val & 0xFF

    def get_sha256(self) -> str:
        """Compute the SHA-256 hash of the entire tape."""
        return hashlib.sha256(self.tape.tobytes()).hexdigest()
