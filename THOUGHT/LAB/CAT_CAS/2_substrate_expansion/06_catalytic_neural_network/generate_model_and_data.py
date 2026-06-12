import os
import random

DIR = os.path.dirname(__file__)
DATA_FILE = os.path.abspath(os.path.join(DIR, "data", "user_video.mp4"))

def generate_video_file():
    print(f"Generating 2MB dirty video file at {DATA_FILE}...")
    random.seed(42) # Deterministic for verification
    size = 2 * 1024 * 1024 # 2 Megabytes
    
    # We write 2MB of random bytes
    chunk_size = 1024 * 1024
    with open(DATA_FILE, "wb") as f:
        for _ in range(2):
            chunk = bytearray(random.getrandbits(8) for _ in range(chunk_size))
            f.write(chunk)
            
    print("Generation complete.")

if __name__ == "__main__":
    generate_video_file()
