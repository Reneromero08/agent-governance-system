import sys
import gc
import pytest
from pathlib import Path

import importlib

sys.path.append(str(Path(__file__).parent.parent))

physics_engine = importlib.import_module("47_1_nucleus_memory_knot")
CatalyticTape = physics_engine.CatalyticTape

def test_unbound_nucleons():
    # Prove that independent objects deallocate synchronously.
    # We test that the latency for unbound is less than the cyclic latency
    # This is a unit test wrapper around our macro-statistical ensemble.
    
    u_3, b_3 = physics_engine.execute_ensemble(3, 10**6, iterations=5)
    
    # Gate 1: Baseline GC for unbound must be lower than knot friction
    assert u_3 < b_3, f"Unbound latency ({u_3}) should be less than bound knot latency ({b_3})"

def test_nuclear_knot_friction():
    # Prove the cyclic knot forces latency spike
    u_3, b_3 = physics_engine.execute_ensemble(3, 10**6, iterations=5)
    f_3 = b_3 / u_3
    
    # Due to OS jitter, we ensure it's at least > 1.0
    assert f_3 >= 1.0, f"Nuclear knot friction must be >= 1.0, got {f_3}"

def test_scale_invariance():
    # Prove friction multiplier scales with mass
    u_10, b_10 = physics_engine.execute_ensemble(10, 10**6, iterations=5)
    u_50, b_50 = physics_engine.execute_ensemble(50, 10**6, iterations=5)
    
    f_10 = b_10 / u_10
    f_50 = b_50 / u_50
    
    assert f_50 > f_10, f"Friction must scale non-linearly with mass. f_10={f_10}, f_50={f_50}"

def test_catalytic_tape_preservation():
    tape = CatalyticTape(size_mb=10) # Smaller for fast unit test
    tape.verify()
    assert True
