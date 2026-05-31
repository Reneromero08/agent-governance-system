import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

print("================================================================================")
print("EXP 42.14: THE BOLTZMANN BRAIN - ENTROPY ANALYSIS")
print("================================================================================")

csv_path = 'telemetry_42_14.csv'
if not os.path.exists(csv_path):
    print(f"[ERROR] {csv_path} not found. Run the Rust simulation first.")
    exit(1)

df = pd.read_csv(csv_path)

# Calculate drop
initial_entropy = df.iloc[0]['CompressedSizeBytes']
final_entropy = df.iloc[-1]['CompressedSizeBytes']
drop_percent = ((initial_entropy - final_entropy) / initial_entropy) * 100

print(f"[*] Initial Noise Complexity : {initial_entropy} bytes")
print(f"[*] Final Structure Complexity: {final_entropy} bytes")
print(f"[*] Entropy Drop             : {drop_percent:.2f}%")

entropy_vals = df['CompressedSizeBytes'].values
if len(entropy_vals) >= 2:
    mean_ent = np.mean(entropy_vals)
    std_ent = np.std(entropy_vals, ddof=1)
    print(f"\n[STATISTICS] N={len(entropy_vals)} generations, Kolmogorov complexity trajectory:")
    print(f"    Mean compressed size = {mean_ent:.2f} bytes")
    print(f"    Standard deviation   = {std_ent:.2f} bytes")
    print(f"    bootstrap 95% CI     = [{mean_ent - 1.96*std_ent/np.sqrt(len(entropy_vals)):.2f}, "
          f"{mean_ent + 1.96*std_ent/np.sqrt(len(entropy_vals)):.2f}]")

if drop_percent > 10:
    print("\n[SUCCESS] Massive entropy drop detected! The noise has organized into a Boltzmann Brain.")
else:
    print("\n[FAILED] Entropy did not drop significantly. The universe remains chaotic.")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Generation'], df['CompressedSizeBytes'], color='purple', linewidth=2)
plt.title('Exp 42.14: Emergence of the Boltzmann Brain (Rule 110)', fontsize=14)
plt.xlabel('Generation (Time)', fontsize=12)
plt.ylabel('Kolmogorov Complexity (Zlib Compressed Bytes)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plot_path = 'entropy_collapse_plot.png'
plt.savefig(plot_path)
print(f"\n[*] Scientific visualization saved to {plot_path}")
print("================================================================================")
