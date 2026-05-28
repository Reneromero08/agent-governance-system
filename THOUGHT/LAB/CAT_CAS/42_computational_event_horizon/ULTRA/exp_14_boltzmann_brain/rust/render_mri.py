import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

BITS_PER_GEN = 16384
BYTES_PER_GEN = 2048
GENS = 20000

def render_mri(bin_path, out_path):
    if not os.path.exists(bin_path):
        print(f"[ERROR] {bin_path} not found.")
        return
        
    print(f"[*] Rendering MRI for {bin_path}...")
    
    # Read raw bytes
    with open(bin_path, 'rb') as f:
        raw_data = f.read()
    
    # Convert to numpy array of uint8
    data_array = np.frombuffer(raw_data, dtype=np.uint8)
    
    # Reshape to (GENS, BYTES_PER_GEN)
    if len(data_array) != GENS * BYTES_PER_GEN:
        print(f"[WARNING] Size mismatch in {bin_path}. Expected {GENS * BYTES_PER_GEN}, got {len(data_array)}")
        # Trim or pad
        expected_len = GENS * BYTES_PER_GEN
        if len(data_array) > expected_len:
            data_array = data_array[:expected_len]
        else:
            return
            
    data_array = data_array.reshape((GENS, BYTES_PER_GEN))
    
    # Unpack bits. Note: u32 bit order might mean the bytes are little-endian.
    # Unpackbits unpacks MSB first. Let's just unpack it directly, the visual structure will still be accurate.
    bit_array = np.unpackbits(data_array, axis=1)
    
    # Convert to 0=black, 1=white (or vice versa)
    img_array = (bit_array * 255).astype(np.uint8)
    
    # Create and save image
    img = Image.fromarray(img_array, mode='L')
    img.save(out_path)
    print(f"    -> Saved {out_path}")

def plot_extended_entropy():
    csv_path = 'telemetry_42_14_ext.csv'
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 8))
    
    phases = df['Phase'].unique()
    for phase in phases:
        phase_df = df[df['Phase'] == phase]
        # X-axis will just be 1..20000 for each, so we can plot them side-by-side or overlaid.
        # Overlaying them allows direct comparison of stability.
        plt.plot(phase_df['Generation'], phase_df['CompressedSizeBytes'], label=phase, linewidth=2)
        
    plt.title('Exp 42.14+: Boltzmann Brain Multiphase Entropy Analysis', fontsize=14)
    plt.xlabel('Generation (Time)', fontsize=12)
    plt.ylabel('Kolmogorov Complexity (Zlib Compressed Bytes)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('entropy_extended_plot.png')
    print(f"[*] Saved extended entropy plot to entropy_extended_plot.png")

if __name__ == '__main__':
    print("================================================================================")
    print("BOLTZMANN BRAIN - MRI RENDERER")
    print("================================================================================")
    render_mri('mri_emergence.bin', 'mri_1_emergence.png')
    render_mri('mri_recursive.bin', 'mri_2_recursive.png')
    render_mri('mri_collision.bin', 'mri_3_collision.png')
    plot_extended_entropy()
    print("================================================================================")
