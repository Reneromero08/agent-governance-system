"""
Quantum compression experiment: encode classical data in qubit amplitudes

The idea: n qubits have 2^n amplitudes. Each amplitude can store information.
This is "amplitude encoding" - a potential beyond-Shannon trick.

Classical: 32 bits = 32 bits storage
Quantum: 32 values encoded in 5 qubits (2^5 = 32 amplitudes)

BUT: reading requires measurement, which is probabilistic.
For deterministic readout, need quantum tomography = many measurements.
"""
import numpy as np

def classical_to_amplitudes(data: bytes) -> np.ndarray:
    """
    Encode bytes into quantum amplitudes.

    Each byte (0-255) becomes one amplitude.
    Normalize to unit vector (quantum state requirement).
    """
    values = np.array([b for b in data], dtype=np.float64)
    # Normalize to unit vector
    norm = np.linalg.norm(values)
    if norm > 0:
        amplitudes = values / norm
    else:
        amplitudes = values
    return amplitudes, norm

def amplitudes_to_classical(amplitudes: np.ndarray, norm: float) -> bytes:
    """
    Decode amplitudes back to bytes.

    Denormalize and round to nearest integer.
    """
    values = amplitudes * norm
    return bytes([int(round(max(0, min(255, v)))) for v in values])

def num_qubits_needed(n_values: int) -> int:
    """Number of qubits to store n values in amplitudes."""
    return int(np.ceil(np.log2(n_values)))

# Test with "word"
test_word = "word"
test_bytes = test_word.encode('utf-8')

print("=" * 60)
print("QUANTUM AMPLITUDE ENCODING")
print("=" * 60)

print(f"\nInput: '{test_word}'")
print(f"Bytes: {list(test_bytes)}")
print(f"Classical size: {len(test_bytes)} bytes = {len(test_bytes) * 8} bits")

# Encode
amplitudes, norm = classical_to_amplitudes(test_bytes)
n_qubits = num_qubits_needed(len(test_bytes))

print(f"\n--- Quantum Encoding ---")
print(f"Amplitudes: {amplitudes}")
print(f"Norm (stored separately): {norm:.2f}")
print(f"Qubits needed: {n_qubits} (2^{n_qubits} = {2**n_qubits} amplitudes)")

# Decode
recovered_bytes = amplitudes_to_classical(amplitudes, norm)
recovered_word = recovered_bytes.decode('utf-8', errors='replace')

print(f"\n--- Decode ---")
print(f"Recovered bytes: {list(recovered_bytes)}")
print(f"Recovered word: '{recovered_word}'")
print(f"Exact match: {recovered_word == test_word}")

# Compression analysis
classical_bits = len(test_bytes) * 8
quantum_bits = n_qubits  # Number of qubits
# But we also need the norm (1 float = 32 bits)
total_quantum_bits = quantum_bits + 32

print(f"\n--- Compression Analysis ---")
print(f"Classical: {classical_bits} bits")
print(f"Quantum (qubits only): {n_qubits} qubits")
print(f"Quantum (qubits + norm): {n_qubits} + 32 = {total_quantum_bits} bits")

# The catch
print(f"\n--- THE CATCH ---")
print(f"To READ amplitudes deterministically, need quantum tomography:")
print(f"  - Many repeated measurements")
print(f"  - O(2^n) measurements for n qubits")
print(f"  - This negates the compression!")
print(f"\nQuantum amplitude encoding is NOT beyond Shannon for")
print(f"classical-in → classical-out compression.")

# What DOES work
print(f"\n--- WHAT QUANTUM CAN DO ---")
print(f"1. Superdense coding: 2 classical bits per qubit (needs entanglement)")
print(f"2. Quantum fingerprinting: compare data with log(n) qubits")
print(f"3. Quantum RAM: query in superposition (O(1) vs O(n))")
print(f"4. Keep data quantum: avoid measurement until needed")

# The Df connection
print(f"\n--- THE Df CONNECTION ---")
print(f"Your formula R = (E / gradS) * sigma(f)^Df")
print(f"If Df = 1 (The One), the manifold is 1D.")
print(f"A 1D manifold has 1 continuous parameter.")
print(f"Quantum: encode that parameter in a qubit phase?")
print(f"Phase: e^(i * theta) has infinite precision in theta")
print(f"But measurement collapses to discrete outcomes.")
print(f"\nThe quantum advantage is in PROCESSING, not storage.")
