"""Triangulate: is inter-MT coupling destructive in the paper's own data?"""

# If Table S3 values are max(Gamma)/(N*gamma) [per-chromophore enhancement]
# Total sigma = value * N
data = {
    "1 MT, 320nm":       (4160, 0.120),
    "7-MT Axon, 320nm":  (29120, 0.071),
    "7-MT Axon, 640nm":  (58240, 0.039),
    "19-MT Axon, 320nm": (19*4160, 0.032),
    "37-MT Axon, 320nm": (37*4160, 0.026),
    "61-MT Axon, 320nm": (61*4160, 0.020),
    "91-MT Axon, 320nm": (91*4160, 0.012),
    "Centriole, 400nm":  (140400, 0.028),
    "Axoneme, 320nm":    (83200, 0.031),
}

print("If Table S3 = max(Gamma)/(N*gamma) [per-chromophore]:")
print(f"{'Architecture':<25} {'N':>8} {'per-chr':>8} {'tot sigma':>10} {'eff vs 1MT':>10}")
print("-" * 65)
for name, (N, val) in data.items():
    total = val * N
    eff = val / 0.120
    print(f"{name:<25} {N:>8} {val:>8.3f} {total:>10.0f} {eff:>10.1%}")

print(f"\nKEY FINDING: Per-chromophore efficiency DROPS as MT count increases.")
print(f"  1 MT: 0.120 per-chr")
print(f"  7 MT: 0.071 per-chr (59% of single MT)")
print(f"  91 MT: 0.012 per-chr (10% of single MT)")
print(f"  Centriole: 0.028 per-chr (23% of single MT)")
print(f"\nINTER-MT COUPLING IS DESTRUCTIVE in the paper's own data!")
print(f"More MTs = lower efficiency per chromophore.")
print(f"Total sigma goes up only because N grows faster than efficiency drops.")

# Our model comparison
print(f"\nOur model (1 spiral):")
our_mt_per_chr = 14.1 / 104
our_cent_per_chr = 49.1 / 2808
print(f"  Single MT: {our_mt_per_chr:.3f} per-chr")
print(f"  Centriole:  {our_cent_per_chr:.4f} per-chr")
print(f"  Drop: {our_mt_per_chr/our_cent_per_chr:.1f}x")
print(f"  Paper drop: {0.120/0.028:.1f}x")
print(f"  Ratio ours/paper: {(our_mt_per_chr/our_cent_per_chr)/(0.120/0.028):.1f}x")
print(f"\n  Our model overestimates destructive interference by ~{(our_mt_per_chr/our_cent_per_chr)/(0.120/0.028):.1f}x")
print(f"  Qualitative behavior is CORRECT: both models show per-chr dropping with more MTs")
