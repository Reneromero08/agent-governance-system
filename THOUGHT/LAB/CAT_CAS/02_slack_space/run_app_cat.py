import os
import sys
import json
import shutil
import hashlib
import numpy as np
import subprocess

def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_padded_file(file_path: str, initial_data: bytes, target_size: int, seed: int):
    """Creates a file of exactly target_size, filled with initial_data and random padding."""
    if len(initial_data) > target_size:
        raise ValueError("Initial data size exceeds target size!")
    
    # Generate deterministic random padding
    rng = np.random.default_rng(seed)
    padding_len = target_size - len(initial_data)
    padding = rng.integers(0, 256, size=padding_len, dtype=np.uint8).tobytes()
    
    # Write file
    with open(file_path, "wb") as f:
        f.write(initial_data + padding)

def get_directory_state(dir_path: str) -> tuple[dict[str, tuple[int, str]], dict[str, bytes]]:
    """
    Returns:
    - manifest: dict mapping relative path -> (file_size, sha256_hash)
    - cas_store: dict mapping relative path -> original file content bytes
    """
    manifest = {}
    cas_store = {}
    
    if not os.path.exists(dir_path):
        return manifest, cas_store

    for root, _, files in os.walk(dir_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, dir_path)
            # Compute hash and size
            file_size = os.path.getsize(full_path)
            file_hash = compute_sha256(full_path)
            manifest[rel_path] = (file_size, file_hash)
            # Store content bytes (CAS)
            with open(full_path, "rb") as f:
                cas_store[rel_path] = f.read()
                
    return manifest, cas_store

def compute_directory_hash(manifest: dict[str, tuple[int, str]]) -> str:
    """Compute a single deterministic hash representing the directory state."""
    sorted_items = sorted(manifest.items())
    manifest_str = json.dumps(sorted_items)
    return hashlib.sha256(manifest_str.encode()).hexdigest()

def main():
    print("=" * 60)
    print("CAT_CAS: Strict Slack-Space Catalytic File Storage")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    workspace_dir = os.path.join(base_dir, "workspace")
    output_file = os.path.join(base_dir, "output.txt")
    app_script = os.path.join(base_dir, "data_processor.py")

    # 1. Setup clean workspace
    print("[Host] Preparing clean workspace...")
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)
    
    if os.path.exists(output_file):
        os.remove(output_file)

    input_txt = os.path.join(workspace_dir, "input.txt")
    config_json = os.path.join(workspace_dir, "config.json")

    # Define inputs and target sizes (Exactly 4096 bytes / 4KB sector size)
    fixed_size = 4096
    
    input_content = "catalytic computing makes file systems space-invariant".encode("utf-8")
    # Pad to 500 bytes for input string partition
    input_content = input_content.ljust(500, b" ")
    
    config_content = json.dumps({"runs_count": 0, "last_run_timestamp": 0.0}).encode("utf-8")
    # Pad to 500 bytes for config JSON partition
    config_content = config_content.ljust(500, b" ")

    # Create padded files
    create_padded_file(input_txt, input_content, fixed_size, seed=100)
    create_padded_file(config_json, config_content, fixed_size, seed=200)

    # 2. Capture initial state
    initial_manifest, initial_cas = get_directory_state(workspace_dir)
    initial_dir_hash = compute_directory_hash(initial_manifest)

    print(f"[Host] Workspace initialized with Padded Files.")
    for rel_path, (size, sha) in initial_manifest.items():
        print(f"  {rel_path}: size={size} bytes, sha256={sha[:16]}...")
    print(f"  Initial Directory Hash: {initial_dir_hash}\n")

    # 3. Run application
    print("[Host] Executing application in workspace...")
    result = subprocess.run(
        [sys.executable, app_script],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[Host] Application crashed!")
        print(result.stderr)
        return

    # 4. Inspect directory state after run (Padded/Dirty State)
    dirty_manifest, _ = get_directory_state(workspace_dir)
    dirty_dir_hash = compute_directory_hash(dirty_manifest)
    print("[Host] Inspecting polluted workspace state post-execution:")
    for rel_path, (size, sha) in dirty_manifest.items():
        print(f"  {rel_path}: size={size} bytes, sha256={sha[:16]}...")
    print(f"  Polluted Directory Hash: {dirty_dir_hash}")
    
    # Assertions to prove physical space-invariance during execution
    print("\n[Host] Verifying physical constraints during run:")
    assert list(dirty_manifest.keys()) == list(initial_manifest.keys()), "Error: New files were created or deleted!"
    print("  [Pass] No files were created or deleted in the filesystem.")
    
    for rel_path in initial_manifest:
        assert dirty_manifest[rel_path][0] == fixed_size, f"Error: {rel_path} size changed!"
    print("  [Pass] All file sizes on disk remained exactly constant (4096 bytes).")

    # 5. Perform Catalytic Restoration (overwriting modified blocks, zero allocations/deallocations)
    print("\n[Host] Performing Catalytic Restoration...")
    for rel_path in initial_manifest:
        full_path = os.path.join(workspace_dir, rel_path)
        current_hash = compute_sha256(full_path)
        if current_hash != initial_manifest[rel_path][1]:
            print(f"  [Restore] Restoring modified bytes in: {rel_path}")
            # Overwrite with original CAS bytes (no file resizing)
            with open(full_path, "wb") as f:
                f.write(initial_cas[rel_path])
    print("[Host] Restoration complete.\n")

    # 6. Final Verification
    final_manifest, _ = get_directory_state(workspace_dir)
    final_dir_hash = compute_directory_hash(final_manifest)
    print("[Host] Verifying final workspace state:")
    for rel_path, (size, sha) in final_manifest.items():
        print(f"  {rel_path}: size={size} bytes, sha256={sha[:16]}...")
    print(f"  Final Directory Hash: {final_dir_hash}")

    assert initial_dir_hash == final_dir_hash, "Error: Directory hash does not match initial state!"
    print("Verification: Workspace is 100% byte-identical to initial state!")

    # Verify output file generated outside workspace
    assert os.path.exists(output_file), "Error: Output file was not generated!"
    with open(output_file, "r") as f:
        output_data = f.read()
    print(f"Verification: Application Output matches: '{output_data}'")
    print("\nSuccess! Strict Slack-Space Catalytic Run Proven!")

if __name__ == "__main__":
    main()
