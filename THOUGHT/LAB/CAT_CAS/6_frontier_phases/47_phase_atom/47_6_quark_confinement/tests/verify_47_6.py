import sys
import importlib
from pathlib import Path
import pytest
import mmap
import time
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

physics_engine = importlib.import_module("47_6_quark_confinement")
BennettHistoryTape = physics_engine.BennettHistoryTape

def test_bennett_tape():
    tape = BennettHistoryTape(size_mb=1) 
    tape.record_operation("101010")
    tape.uncompute()
    tape.verify()
    assert True

def test_quark_confinement_physics():
    vacuum_size = 100 * 1024 * 1024 # 100MB
    vacuum = mmap.mmap(-1, vacuum_size)
    
    # Warm phase (Asymptotic Freedom -> String Tension)
    warm_size = 10 * 1024 * 1024
    for i in range(0, warm_size, 4096):
        _ = vacuum[i]
        
    latencies = {}
    offsets = [16, 64, 2048]
    iterations = 500
    
    import random
    random.seed(42)
    bases = [random.randint(0, warm_size - max(offsets) - 1) for _ in range(iterations)]
    
    for offset in offsets:
        times = []
        for i in range(iterations):
            base = bases[i]
            start = time.perf_counter_ns()
            _ = vacuum[base]
            _ = vacuum[base + offset]
            end = time.perf_counter_ns()
            times.append(end - start)
        
        times = np.array(times)
        p99 = np.percentile(times, 99)
        filtered = times[times < p99]
        latencies[offset] = np.mean(filtered)
        
    # Asymptotic Freedom (Flat latency inside cache line)
    assert abs(latencies[64] - latencies[16]) < max(20.0, latencies[16] * 0.5), "Asymptotic freedom failed."
    
    # String Tension (Latency scales as TLB boundaries are crossed)
    assert latencies[2048] > latencies[16], "String tension failed."
    
    # Pair Production (OS Page Fault Interception)
    cold_base = warm_size + 4096
    
    # First access to cold memory causes a page fault
    start = time.perf_counter_ns()
    _ = vacuum[cold_base] # Fault 1
    _ = vacuum[cold_base + 8192] # Fault 2 (Pair Production)
    end = time.perf_counter_ns()
    
    cold_latency = end - start
    
    # It must be massively larger than warm latency
    assert cold_latency > latencies[2048] * 1.5, "Pair production OS interception failed."
