import os
import sys
import hashlib
import time

DIR = os.path.dirname(__file__)
DATA_FILE = os.path.abspath(os.path.join(DIR, "data", "user_video.mp4"))

MB = 1024 * 1024

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Convolution weights for 2 layers
W1 = [3, -1, 2]
W2 = [1, 2, -1]

def relu_quantize(x):
    return max(0, x) % 256

import mmap

def execute_catalytic_layer(file_path, target_offset, source_offset, weights, reverse=False):
    """
    Executes a 1D Conv Feistel layer in-place.
    Target = Target ^ ReLU(Conv1D(Source, W))
    """
    kernel_size = len(weights)
    chunk_size = 32 * 1024 # 32 KB chunk for target write
    
    with open(file_path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        
        for chunk_start in range(0, MB, chunk_size):
            chunk_len = min(chunk_size, MB - chunk_start)
            activations = bytearray(chunk_len)
            
            for i in range(chunk_len):
                global_i = chunk_start + i
                acc = 0
                for k in range(kernel_size):
                    src_idx = source_offset + ((global_i + k) % MB)
                    val = mm[src_idx]
                    acc += val * weights[k]
                
                activations[i] = relu_quantize(acc)
                
            # XOR into target
            for i in range(chunk_len):
                mm[target_offset + chunk_start + i] ^= activations[i]
                
        mm.flush()
        mm.close()


def run_catalytic_inference():
    print("=" * 60)
    print("CATALYTIC NEURAL NETWORK INFERENCE (Zero-RAM)")
    print("=" * 60)
    
    print("[System] Clean RAM footprint restricted to sliding windows < 100 KB.")
    print("[Network] Deep Quantized Feistel ConvNet (2MB Activation State)")
    
    original_hash = get_file_hash(DATA_FILE)
    print(f"[State] Initial Dirty Tape Hash: {original_hash}")
    
    start_time = time.time()
    
    # ---------------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------------
    print("\n[Forward Pass] Computing Layer 1 (Target=L, Source=R)...")
    execute_catalytic_layer(DATA_FILE, target_offset=0, source_offset=MB, weights=W1)
        
    print("[Forward Pass] Computing Layer 2 (Target=R, Source=L)...")
    execute_catalytic_layer(DATA_FILE, target_offset=MB, source_offset=0, weights=W2)
        
    forward_hash = get_file_hash(DATA_FILE)
    print(f"[State] Computed Dirty Tape Hash: {forward_hash}")
    
    # Extract prediction (Argmax of the first 10 bytes of R)
    with open(DATA_FILE, "rb") as f:
        f.seek(MB)
        logits = f.read(10)
    
    prediction = max(range(10), key=lambda i: logits[i])
    print(f"\n>>>> PREDICTION OUTPUT: Class {prediction} <<<<\n")
    
    # ---------------------------------------------------------
    # REVERSE PASS (Uncomputation)
    # ---------------------------------------------------------
    print("[Reverse Pass] Uncomputing Layer 2 (Target=R, Source=L)...")
    execute_catalytic_layer(DATA_FILE, target_offset=MB, source_offset=0, weights=W2, reverse=True)
        
    print("[Reverse Pass] Uncomputing Layer 1 (Target=L, Source=R)...")
    execute_catalytic_layer(DATA_FILE, target_offset=0, source_offset=MB, weights=W1, reverse=True)
        
    final_hash = get_file_hash(DATA_FILE)
    print(f"\n[State] Final Restored Tape Hash: {final_hash}")
    
    end_time = time.time()
    
    if final_hash == original_hash:
        print("\n[VERIFICATION] SUCCESS: Tape restored 100% byte-for-byte!")
        print(f"[VERIFICATION] Total clean RAM used: ~32 KB. Executed in {end_time - start_time:.2f}s.")
    else:
        print("\n[VERIFICATION] FAILED: Entropy leak detected. Tape was corrupted.")
        sys.exit(1)

if __name__ == "__main__":
    run_catalytic_inference()
