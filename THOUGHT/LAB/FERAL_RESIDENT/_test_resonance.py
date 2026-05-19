"""Test daemon Core resonance vs cosine similarity."""
import os, sys, numpy as np, time
os.environ['FERAL_EIGEN'] = '1'
sys.path.insert(0, r'THOUGHT/LAB/FERAL_RESIDENT')
from cognition.native_eigen_reasoner import NativeEigenReasoner

eigen = NativeEigenReasoner(d=64, heads=4, layers=4, cycles=4)
eigen._init_core()

from geometric_reasoner import GeometricState

# Compare cosine vs Core resonance
pairs = [
    ("Deep compression reduces neural network size", "Model pruning and quantization"),
    ("Deep compression reduces neural network size", "The weather today is sunny"),
    ("Transformer attention mechanism", "Multi-head self-attention for sequences"),
    ("Transformer attention mechanism", "Baking a chocolate cake recipe"),
]

print("Resonance comparison: Cosine vs Core (phase coherence)")
print(f"Critical threshold: 1/(2pi) = {1/(2*np.pi):.4f}")
print("-" * 55)
for a, b in pairs:
    s1 = eigen.initialize(a)
    s2 = eigen.initialize(b)
    cosine = s1.vector.dot(s2.vector) / (np.linalg.norm(s1.vector) * np.linalg.norm(s2.vector) + 1e-8)
    core_E = eigen.E_with(s1, s2)
    print(f"cos={cosine:.3f}  core={core_E:.3f}  |  {a[:40]} <-> {b[:40]}")
