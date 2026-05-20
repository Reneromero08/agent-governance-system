import os
import json
import time

def read_padded_config(config_file: str) -> dict:
    """Reads config from the first 500 bytes of the file."""
    with open(config_file, "rb") as f:
        config_bytes = f.read(500)
    config_str = config_bytes.decode("utf-8").strip()
    return json.loads(config_str)

def write_padded_config(config_file: str, config: dict):
    """Writes config back to the first 500 bytes of the file, preserving the rest."""
    config_str = json.dumps(config)
    config_bytes = config_str.encode("utf-8")
    if len(config_bytes) > 500:
        raise ValueError("Config size exceeds 500 bytes limit!")
    
    # Pad with spaces
    config_bytes = config_bytes.ljust(500, b" ")
    
    # Write in-place without changing file size
    with open(config_file, "r+b") as f:
        f.seek(0)
        f.write(config_bytes)

def main():
    workspace_dir = os.path.join(os.path.dirname(__file__), "workspace")
    input_file = os.path.join(workspace_dir, "input.txt")
    config_file = os.path.join(workspace_dir, "config.json")
    output_file = os.path.join(os.path.dirname(__file__), "output.txt")

    print("[App] Starting application in strict slack-space mode.")

    # 1. Read input text from the first 500 bytes of input.txt
    print(f"[App] Reading input from {input_file} (first 500 bytes)...")
    with open(input_file, "rb") as f:
        input_bytes = f.read(500)
    input_text = input_bytes.decode("utf-8").strip()

    # 2. Read and modify config.json within its first 500 bytes
    print(f"[App] Modifying config data in {config_file}...")
    config = read_padded_config(config_file)
    config["runs_count"] += 1
    config["last_run_timestamp"] = time.time()
    write_padded_config(config_file, config)

    # 3. Simulate lockfile creation: Write lock status into config.json padding (bytes 500-510)
    print(f"[App] Writing temporary lock status to {config_file} offset 500...")
    with open(config_file, "r+b") as f:
        f.seek(500)
        f.write(b"LOCKED_FLAG")

    # 4. Simulate chunk files: Write intermediate chunks into input.txt padding (bytes 1000-2000 and 2000-3000)
    chunks = [input_text[:len(input_text)//2], input_text[len(input_text)//2:]]
    for i, chunk in enumerate(chunks):
        offset = 1000 + i * 1000
        print(f"[App] Writing intermediate chunk {i} to {input_file} offset {offset}...")
        chunk_bytes = chunk.encode("utf-8").ljust(1000, b" ")
        with open(input_file, "r+b") as f:
            f.seek(offset)
            f.write(chunk_bytes[:1000])
        time.sleep(0.1)

    # 5. Write final output to output.txt (outside the workspace)
    print(f"[App] Writing final output to {output_file}...")
    processed_data = input_text.upper()
    with open(output_file, "w") as f:
        f.write(processed_data)

    print("[App] Application execution finished successfully.")

if __name__ == "__main__":
    main()
