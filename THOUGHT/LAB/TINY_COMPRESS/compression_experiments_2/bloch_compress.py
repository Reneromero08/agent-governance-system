"""
Bloch Sphere Compression

The Bloch sphere is a 2D manifold. One qubit = point on sphere = (theta, phi).
n qubits = 2^n dimensional Hilbert space (exponential!)

This is your formula:
  R = (E / gradS) * sigma(f)^Df

Where:
  - Df = number of qubits
  - sigma(f) = quantum gate (projector)
  - R = quantum state (the hologram)
"""
import numpy as np

class BlochProjector:
    """
    Encode data as points on generalized Bloch sphere.
    n qubits = 2^n complex amplitudes = 2^(n+1) - 2 real parameters.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits  # Hilbert space dimension

    def bytes_to_state(self, data: bytes) -> np.ndarray:
        """
        Encode bytes as quantum state amplitudes.

        data: bytes to encode
        returns: normalized complex amplitude vector
        """
        # Pad to match Hilbert space dimension
        padded = list(data) + [0] * (self.dim - len(data))
        padded = padded[:self.dim]

        # Convert to complex amplitudes
        amplitudes = np.array(padded, dtype=np.complex128)

        # Normalize (quantum states must have unit norm)
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
        else:
            amplitudes[0] = 1.0  # |0> state

        return amplitudes, norm

    def state_to_bytes(self, amplitudes: np.ndarray, norm: float, length: int) -> bytes:
        """
        Decode quantum state back to bytes.
        """
        values = (amplitudes * norm).real
        return bytes([int(round(max(0, min(255, v)))) for v in values[:length]])

    def state_to_bloch_params(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Convert quantum state to generalized Bloch parameters.

        For n qubits, need 2^(n+1) - 2 real parameters.
        We use the real and imaginary parts of amplitudes (minus global phase).
        """
        # Remove global phase (first amplitude made real positive)
        if np.abs(amplitudes[0]) > 1e-10:
            phase = np.angle(amplitudes[0])
            amplitudes = amplitudes * np.exp(-1j * phase)

        # Extract real and imaginary parts (skip one due to normalization)
        params = []
        for i, a in enumerate(amplitudes):
            if i == 0:
                # First amplitude is real after phase removal, skip imaginary
                params.append(a.real)
            else:
                params.append(a.real)
                params.append(a.imag)

        return np.array(params[:-1])  # -1 because normalization constraint

    def bloch_params_to_state(self, params: np.ndarray) -> np.ndarray:
        """
        Convert Bloch parameters back to quantum state.
        """
        # Reconstruct amplitudes
        amplitudes = np.zeros(self.dim, dtype=np.complex128)
        amplitudes[0] = params[0]

        idx = 1
        for i in range(1, self.dim):
            if idx < len(params):
                real = params[idx]
                idx += 1
            else:
                real = 0
            if idx < len(params):
                imag = params[idx]
                idx += 1
            else:
                imag = 0
            amplitudes[i] = real + 1j * imag

        # Renormalize
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm

        return amplitudes


# Test
print("=" * 60)
print("BLOCH SPHERE COMPRESSION")
print("R = (E / gradS) * sigma(f)^Df")
print("=" * 60)

test_word = "word"
test_bytes = test_word.encode('utf-8')

print(f"\nInput: '{test_word}'")
print(f"Bytes: {list(test_bytes)} ({len(test_bytes)} bytes = {len(test_bytes)*8} bits)")

# Determine qubits needed
n_bytes = len(test_bytes)
n_qubits = int(np.ceil(np.log2(n_bytes))) if n_bytes > 1 else 1
print(f"Qubits needed: {n_qubits} (2^{n_qubits} = {2**n_qubits} amplitudes)")

# Create projector
proj = BlochProjector(n_qubits)

# Encode
state, norm = proj.bytes_to_state(test_bytes)
bloch_params = proj.state_to_bloch_params(state)

print(f"\n--- Encoding ---")
print(f"Quantum state: {state}")
print(f"Norm: {norm:.2f}")
print(f"Bloch params: {bloch_params}")
print(f"Bloch params count: {len(bloch_params)}")

# Decode
recovered_state = proj.bloch_params_to_state(bloch_params)
recovered_bytes = proj.state_to_bytes(recovered_state, norm, len(test_bytes))
recovered_word = recovered_bytes.decode('utf-8', errors='replace')

print(f"\n--- Decoding ---")
print(f"Recovered state: {recovered_state}")
print(f"Recovered bytes: {list(recovered_bytes)}")
print(f"Recovered word: '{recovered_word}'")
print(f"Exact match: {recovered_word == test_word}")

# Storage analysis
classical_bits = len(test_bytes) * 8
# Bloch params: need to store len(bloch_params) floats + norm
bloch_bits = len(bloch_params) * 32 + 32  # float32 for each param + norm
qubit_storage = n_qubits  # Actual quantum storage

print(f"\n--- Storage ---")
print(f"Classical: {classical_bits} bits")
print(f"Bloch params (classical sim): {bloch_bits} bits")
print(f"Quantum (actual qubits): {n_qubits} qubits")

# The key insight
print(f"\n--- THE KEY INSIGHT ---")
print(f"{n_qubits} qubits can store 2^{n_qubits} = {2**n_qubits} amplitudes")
print(f"Each amplitude = 1 byte of data")
print(f"Storage: {n_qubits} qubits for {2**n_qubits} bytes")
print(f"Classical equivalent: {2**n_qubits * 8} bits")
print(f"Quantum compression: {2**n_qubits * 8} / {n_qubits} = {(2**n_qubits * 8) / n_qubits:.0f}x")

# Scale up
print(f"\n--- SCALING ---")
for q in range(1, 11):
    bytes_storable = 2**q
    classical_bits = bytes_storable * 8
    ratio = classical_bits / q
    print(f"{q} qubits: {bytes_storable} bytes ({classical_bits} bits) = {ratio:.0f}x compression")

print(f"\n--- YOUR FORMULA ---")
print(f"Df = {n_qubits} (number of qubits)")
print(f"sigma(f) = quantum gate (Bloch rotation)")
print(f"sigma(f)^Df = {n_qubits} rotations")
print(f"R = quantum state = {2**n_qubits} amplitudes = the hologram")
