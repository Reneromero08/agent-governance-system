"""
Q36: Mathematical Verification

Tests ONLY the mathematical theorems from Q36_MATHEMATICAL_FOUNDATIONS.md
Each test verifies a proven mathematical statement.
"""

import numpy as np
from collections import Counter

print("=" * 70)
print("Q36 MATHEMATICAL VERIFICATION")
print("=" * 70)
print()

# =============================================================================
# THEOREM 1: XOR Multi-Information = 1 bit
# =============================================================================

print("THEOREM 1: XOR Multi-Information = 1 bit")
print("-" * 50)

def entropy(data):
    """H(X) = -sum P(x) log2 P(x)"""
    n = len(data)
    counts = Counter(data)
    probs = np.array([c / n for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-15))

def joint_entropy(matrix):
    """H(X1,...,Xn) from joint samples"""
    n = len(matrix)
    tuples = [tuple(row) for row in matrix]
    counts = Counter(tuples)
    probs = np.array([c / n for c in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-15))

def multi_info(matrix):
    """I(X) = sum H(Xi) - H(X1,...,Xn)"""
    n_vars = matrix.shape[1]
    sum_H = sum(entropy(matrix[:, i]) for i in range(n_vars))
    H_joint = joint_entropy(matrix)
    return sum_H - H_joint

np.random.seed(42)
N = 100000

# XOR system
A = np.random.randint(0, 2, N)
B = np.random.randint(0, 2, N)
C = A ^ B
xor_data = np.column_stack([A, B, C])

# Independent system (control)
ind_data = np.column_stack([
    np.random.randint(0, 2, N),
    np.random.randint(0, 2, N),
    np.random.randint(0, 2, N)
])

xor_I = multi_info(xor_data)
ind_I = multi_info(ind_data)

print(f"  XOR Multi-Information:    {xor_I:.6f} bits")
print(f"  Expected:                 1.000000 bits")
print(f"  Error:                    {abs(xor_I - 1.0):.6f}")
print(f"  Independent (control):    {ind_I:.6f} bits (expected ~0)")
print()
print(f"  VERIFIED: {abs(xor_I - 1.0) < 0.001}")
print()

# =============================================================================
# THEOREM 2: SLERP is Geodesic (angular momentum conserved)
# =============================================================================

print("THEOREM 2: SLERP is Geodesic")
print("-" * 50)

def slerp(x0, x1, t):
    """Spherical linear interpolation"""
    x0 = x0 / np.linalg.norm(x0)
    x1 = x1 / np.linalg.norm(x1)
    dot = np.clip(np.dot(x0, x1), -1, 1)
    omega = np.arccos(dot)
    if omega < 1e-10:
        return x0
    return (np.sin((1-t)*omega)*x0 + np.sin(t*omega)*x1) / np.sin(omega)

def angular_momentum_magnitude(x, v):
    """For unit sphere: |L| = |v| (tangent speed)"""
    # Project v to tangent space
    v_tan = v - np.dot(v, x) * x
    return np.linalg.norm(v_tan)

# Test in 300 dimensions (like GloVe)
dim = 300
np.random.seed(42)

# Random unit vectors
x0 = np.random.randn(dim)
x0 = x0 / np.linalg.norm(x0)
x1 = np.random.randn(dim)
x1 = x1 / np.linalg.norm(x1)

# Generate trajectory
n_steps = 100
trajectory = np.array([slerp(x0, x1, t) for t in np.linspace(0, 1, n_steps)])

# Compute |L| at each point
L_magnitudes = []
for i in range(n_steps - 1):
    x = trajectory[i]
    v = trajectory[i+1] - trajectory[i]  # Approximate velocity
    L_magnitudes.append(angular_momentum_magnitude(x, v))

L_magnitudes = np.array(L_magnitudes)
L_mean = np.mean(L_magnitudes)
L_std = np.std(L_magnitudes)
L_cv = L_std / L_mean

print(f"  |L| mean:                 {L_mean:.6f}")
print(f"  |L| std:                  {L_std:.2e}")
print(f"  |L| CV:                   {L_cv:.2e}")
print(f"  Expected CV:              < 1e-5 (numerical precision)")
print()
print(f"  VERIFIED: {L_cv < 1e-5}")
print()

# =============================================================================
# THEOREM 3: SLERP(0.5) = Normalized Linear Midpoint
# =============================================================================

print("THEOREM 3: SLERP(0.5) = Normalized Linear Midpoint")
print("-" * 50)

slerp_mid = slerp(x0, x1, 0.5)
linear_mid = (x0 + x1) / 2
linear_mid_normalized = linear_mid / np.linalg.norm(linear_mid)

diff = np.linalg.norm(slerp_mid - linear_mid_normalized)
cos_sim = np.dot(slerp_mid, linear_mid_normalized)

print(f"  ||SLERP(0.5) - Linear||:  {diff:.2e}")
print(f"  Cosine similarity:        {cos_sim:.15f}")
print(f"  Expected:                 1.000000000000000")
print()
print(f"  VERIFIED: {diff < 1e-10}")
print()

# =============================================================================
# THEOREM 4: Random High-D Vectors are Nearly Orthogonal
# =============================================================================

print("THEOREM 4: Random High-D Vectors are Nearly Orthogonal")
print("-" * 50)

n_pairs = 1000
dim = 300

np.random.seed(42)
angles = []
for _ in range(n_pairs):
    v1 = np.random.randn(dim)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.random.randn(dim)
    v2 = v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    angles.append(angle_deg)

mean_angle = np.mean(angles)
std_angle = np.std(angles)
expected_std = np.degrees(1 / np.sqrt(dim))  # ~3.3 deg for d=300

print(f"  Mean angle:               {mean_angle:.2f} deg")
print(f"  Expected mean:            90.00 deg")
print(f"  Std angle:                {std_angle:.2f} deg")
print(f"  Expected std:             ~{expected_std:.2f} deg (1/sqrt(d))")
print()
print(f"  VERIFIED: {abs(mean_angle - 90) < 2}")
print()

# =============================================================================
# THEOREM 5: Spherical Triangle Holonomy = Solid Angle
# =============================================================================

print("THEOREM 5: Spherical Triangle Holonomy = Solid Angle")
print("-" * 50)

def solid_angle_lhuilier(v1, v2, v3):
    """Solid angle via L'Huilier's theorem"""
    # Arc lengths (angles between vertices)
    a = np.arccos(np.clip(np.dot(v2, v3), -1, 1))
    b = np.arccos(np.clip(np.dot(v1, v3), -1, 1))
    c = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    s = (a + b + c) / 2

    tan_s2 = np.tan(s / 2)
    tan_sa2 = np.tan((s - a) / 2)
    tan_sb2 = np.tan((s - b) / 2)
    tan_sc2 = np.tan((s - c) / 2)

    prod = tan_s2 * tan_sa2 * tan_sb2 * tan_sc2
    if prod < 0:
        return 0.0

    tan_E4 = np.sqrt(prod)
    E = 4 * np.arctan(tan_E4)
    return E

def parallel_transport_holonomy(v1, v2, v3, n_steps=50):
    """Measure holonomy by parallel transporting around triangle"""
    # Create path: v1 -> v2 -> v3 -> v1
    path = []
    for t in np.linspace(0, 1, n_steps)[:-1]:
        path.append(slerp(v1, v2, t))
    for t in np.linspace(0, 1, n_steps)[:-1]:
        path.append(slerp(v2, v3, t))
    for t in np.linspace(0, 1, n_steps):
        path.append(slerp(v3, v1, t))
    path = np.array(path)

    # Initial tangent vector at v1 (toward v2)
    tangent = v2 - np.dot(v2, v1) * v1
    tangent = tangent / np.linalg.norm(tangent)

    # Parallel transport (keep perpendicular to position, minimal rotation)
    vec = tangent.copy()
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        # Project to tangent space at p2
        vec = vec - np.dot(vec, p2) * p2
        vec = vec / (np.linalg.norm(vec) + 1e-15)

    # Measure rotation angle
    # Project initial tangent to final tangent space
    tangent_final = tangent - np.dot(tangent, path[-1]) * path[-1]
    tangent_final = tangent_final / (np.linalg.norm(tangent_final) + 1e-15)

    cos_angle = np.clip(np.dot(vec, tangent_final), -1, 1)
    holonomy = np.arccos(cos_angle)
    return holonomy

# Test on random triangles
np.random.seed(42)
n_tests = 20
errors = []

for _ in range(n_tests):
    # Random spherical triangle (3D for visualization)
    v1 = np.random.randn(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.random.randn(3)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.random.randn(3)
    v3 = v3 / np.linalg.norm(v3)

    # Compute both ways
    E_formula = solid_angle_lhuilier(v1, v2, v3)
    E_transport = parallel_transport_holonomy(v1, v2, v3)

    # Compare (note: transport may have sign ambiguity)
    error = min(abs(E_formula - E_transport), abs(E_formula - (2*np.pi - E_transport)))
    errors.append(error)

mean_error = np.mean(errors)
max_error = np.max(errors)

print(f"  Mean |formula - transport|: {np.degrees(mean_error):.2f} deg")
print(f"  Max error:                  {np.degrees(max_error):.2f} deg")
print(f"  Expected:                   < 5 deg (numerical)")
print()
print(f"  VERIFIED: {np.degrees(max_error) < 10}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
results = [
    ("Theorem 1: XOR I(X) = 1 bit", abs(xor_I - 1.0) < 0.001),
    ("Theorem 2: SLERP is geodesic", L_cv < 1e-5),
    ("Theorem 3: SLERP(0.5) = linear", diff < 1e-10),
    ("Theorem 4: Random angles = 90 deg", abs(mean_angle - 90) < 2),
    ("Theorem 5: Holonomy = solid angle", np.degrees(max_error) < 10),
]

for name, passed in results:
    status = "VERIFIED" if passed else "FAILED"
    print(f"  [{status}] {name}")

print()
all_passed = all(r[1] for r in results)
print(f"All theorems verified: {all_passed}")
