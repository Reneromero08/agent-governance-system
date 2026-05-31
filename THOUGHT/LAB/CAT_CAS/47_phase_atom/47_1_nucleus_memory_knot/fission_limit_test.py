import gc
import time
import sys

def measure_knot_friction(N):
    # Disable GC so we can isolate the collection time
    gc.disable()
    gc.collect()
    
    # Build a deep recursive knot (The Strong Force)
    nucleons = [[] for _ in range(N)]
    for i in range(N):
        nucleons[i].append(bytearray(10**3))
        nucleons[i].append(nucleons[(i+1)%N])
        
    del nucleons
    
    start = time.perf_counter_ns()
    collected = gc.collect()
    end = time.perf_counter_ns()
    
    return end - start, collected

print(f"System Recursion Limit: {sys.getrecursionlimit()}")
print("Running Fission Limit Test (Massive Topological Density)...")

for N in [1_000, 10_000, 50_000, 100_000, 500_000]:
    try:
        latency, collected = measure_knot_friction(N)
        print(f"N = {N:,} -> GC Latency: {latency / 1_000_000:,.2f} ms | Objects Collected: {collected:,}")
    except Exception as e:
        print(f"N = {N:,} -> KERNEL FRACTURE (FISSION): {e}")
        break
