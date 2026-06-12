import os
import sys
import shutil
import hashlib

# Add current directory to path
CAT_CAS_DIR = os.path.dirname(__file__)
if CAT_CAS_DIR not in sys.path:
    sys.path.insert(0, CAT_CAS_DIR)

from generate_bmp import generate_gradient_bmp
from real_sorter import CatalyticBmpTape, CleanMemoryTracker, solve_maze_catalytic

def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 60)
    print("CAT_CAS: Visual BMP Catalytic Memory Experiment")
    print("=" * 60)

    workspace_dir = os.path.join(CAT_CAS_DIR, "workspace")
    bmp_path = os.path.join(workspace_dir, "fractal.bmp")
    dirty_bmp_path = os.path.join(workspace_dir, "fractal_dirty.bmp")

    # 1. Generate clean BMP gradient image (512x512 = 786,432 pixel bytes)
    print("[Host] Generating initial clean BMP image...")
    generate_gradient_bmp(bmp_path)
    
    initial_hash = compute_sha256(bmp_path)
    print(f"[Host] Initial BMP Image Hash: {initial_hash}")
    print(f"  File size on disk: {os.path.getsize(bmp_path)} bytes\n")

    # 2. Setup the catalytic tape and tracker
    tracker = CleanMemoryTracker(limit_bytes=64)
    tape = CatalyticBmpTape(bmp_path)

    dirty_saved = False
    max_sp_observed = 0

    # 3. Define callback to capture the image *during* calculation
    def step_callback(sp):
        nonlocal dirty_saved, max_sp_observed
        if sp > max_sp_observed:
            max_sp_observed = sp
            
        # When stack is deep, save a snapshot of the modified image
        if sp > 80 and not dirty_saved:
            print(f"[Host] DFS Stack pointer reached {sp}. Saving dirty snapshot to fractal_dirty.bmp...")
            # Flush file to disk before copying
            tape.f.flush()
            os.fsync(tape.f.fileno())
            shutil.copy2(bmp_path, dirty_bmp_path)
            dirty_saved = True

    # 4. Run the maze solver
    try:
        print("[Host] Running DFS Maze Solver (40x40 grid, 1600 nodes)...")
        path = solve_maze_catalytic(tape, tracker, step_callback=step_callback)
        print(f"[Host] Solver finished. Path found of length: {len(path)} steps.")
        print(f"[Host] Max stack depth reached: {max_sp_observed}")
        print(f"[Host] Clean memory tracker max: {tracker.max_observed} / 64 bytes.")
    finally:
        tape.close()

    # 5. Verify image restoration
    final_hash = compute_sha256(bmp_path)
    print(f"\n[Host] Final BMP Image Hash:   {final_hash}")
    
    assert initial_hash == final_hash, "Error: BMP Image was not restored to its exact pre-computation state!"
    print("Verification: BMP image is 100% byte-identical to initial state!")
    
    # Check dirty snapshot
    if os.path.exists(dirty_bmp_path):
        dirty_hash = compute_sha256(dirty_bmp_path)
        print(f"Verification: Dirty BMP Image Hash: {dirty_hash} (Modified)")
        assert initial_hash != dirty_hash, "Error: Dirty image hash is identical to initial!"
        print("Verification: Computational noise successfully recorded during run!")
    else:
        print("Warning: Did not save a dirty snapshot (stack pointer did not reach target depth).")

    print("\nSuccess! Visual BMP Catalytic Space Invariance Proven!")

if __name__ == "__main__":
    main()
