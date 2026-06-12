import os
import sys

DIR = os.path.dirname(__file__)
DATA_FILE = os.path.abspath(os.path.join(DIR, "data", "user_video.mp4"))

# Simulated Custom Allocator
class StrictMemoryAllocator:
    def __init__(self, limit_bytes: int):
        self.limit = limit_bytes
        self.allocated = 0

    def allocate(self, size: int):
        if self.allocated + size > self.limit:
            raise MemoryError(f"OOM: Attempted to allocate {size} bytes. Limit: {self.limit} bytes.")
        self.allocated += size
        return bytearray(size)

def run_classical_inference():
    print("=" * 60)
    print("CLASSICAL NEURAL NETWORK INFERENCE (Out-Of-Core)")
    print("=" * 60)
    
    allocator = StrictMemoryAllocator(100 * 1024) # 100 KB strict limit!
    print(f"[System] Clean RAM Limit strictly enforced at: 100 KB")
    
    try:
        print("[Network] Loading 2MB state vector into memory...")
        # A classical network requires allocating memory to store the layer input
        # and another block for the output
        layer_input = allocator.allocate(2 * 1024 * 1024)
        print("[Network] State loaded successfully.")
    except MemoryError as e:
        print(f"\n[CRASH] {e}")
        print("[CRASH] The classical computer cannot even store the layer activations!")
        print("[CRASH] Inference failed due to lack of RAM.")
        sys.exit(1)

if __name__ == "__main__":
    run_classical_inference()
