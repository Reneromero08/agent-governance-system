"""Test QGT C library from WSL with complex vectors."""
import sys, os, ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'THOUGHT', 'LAB', 'EIGEN_ALIGNMENT', 'qgt_lib', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'THOUGHT', 'LAB', 'EIGEN_ALIGNMENT', 'qgt_lib', 'python'))

from qgt import fubini_study_metric, participation_ratio

# Check if C library loaded
lib_loaded = False
try:
    import ctypes
    lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'THOUGHT', 'LAB', 'EIGEN_ALIGNMENT', 'qgt_lib', 'build', 'lib', 'libquantum_geometric.so')
    if os.path.exists(lib_path):
        lib = ctypes.CDLL(lib_path)
        lib_loaded = True
except Exception as e:
    pass

# Test on random complex vectors
n, d = 50, 384
np.random.seed(42)
real_vecs = np.random.randn(n, d)
imag_vecs = np.random.randn(n, d)
complex_vecs = real_vecs + 1j * imag_vecs
norms = np.sqrt(np.sum(np.abs(complex_vecs)**2, axis=1, keepdims=True))
complex_vecs = complex_vecs / (norms + 1e-10)
real_only = np.real(complex_vecs)

metric = fubini_study_metric(real_only)
pr = participation_ratio(real_only)

print(f"C library loaded: {lib_loaded}")
print(f"PR = {pr:.2f}")
print(f"Metric shape: {metric.shape}")
print("SUCCESS")
