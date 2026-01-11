"""
The Projector: σ(f)^Df that renders R from addresses

R = (E / ∇S) × σ(f)^Df

σ(f)^Df = the basis vectors that unfold Df coordinates into full output
"""
import numpy as np
from typing import Tuple

class HolographicProjector:
    """
    Universal projector that renders holograms from addresses.

    Learn once from representative data.
    Then any address → rendered hologram.
    """

    def __init__(self, Df: int = None):
        self.Df = Df  # Effective dimensionality (auto-computed if None)
        self.basis = None  # σ(f)^Df - the projection matrix
        self.mean = None   # Center of the manifold
        self.eigenvalues = None

    def participation_ratio(self, eigenvalues: np.ndarray) -> float:
        """Df = (Σλ)² / Σλ²"""
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    def learn(self, data: np.ndarray) -> 'HolographicProjector':
        """
        Learn the projector from representative data.

        data: N × D matrix of samples
        Returns: self (for chaining)
        """
        # Center
        self.mean = data.mean(axis=0)
        centered = data - self.mean

        # SVD to find basis
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.eigenvalues = S ** 2 / len(data)

        # Auto-compute Df if not specified
        if self.Df is None:
            self.Df = int(np.ceil(self.participation_ratio(self.eigenvalues)))

        # The projector: top Df eigenvectors
        self.basis = Vt[:self.Df]  # Df × D

        print(f"Projector learned:")
        print(f"  Data shape: {data.shape}")
        print(f"  Df (effective dim): {self.Df}")
        print(f"  Basis shape: {self.basis.shape}")
        print(f"  Compression: {data.shape[1]} -> {self.Df} = {data.shape[1]/self.Df:.1f}x")

        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data to manifold addresses.

        data: single sample (D,) or batch (N × D)
        Returns: addresses (Df,) or (N × Df)
        """
        if self.basis is None:
            raise ValueError("Projector not learned. Call learn() first.")

        centered = data - self.mean
        address = centered @ self.basis.T  # Project to Df dimensions
        return address

    def render(self, address: np.ndarray) -> np.ndarray:
        """
        Render hologram from address.

        R = address @ basis + mean

        This is σ(f)^Df in action.
        """
        if self.basis is None:
            raise ValueError("Projector not learned. Call learn() first.")

        # The holographic projection
        R = address @ self.basis + self.mean
        return R

    def roundtrip(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Encode → Render roundtrip.

        Returns: (address, reconstructed, fidelity)
        """
        address = self.encode(data)
        reconstructed = self.render(address)

        # Fidelity = cosine similarity
        if data.ndim == 1:
            fidelity = np.dot(data, reconstructed) / (
                np.linalg.norm(data) * np.linalg.norm(reconstructed)
            )
        else:
            fidelity = np.mean([
                np.dot(data[i], reconstructed[i]) / (
                    np.linalg.norm(data[i]) * np.linalg.norm(reconstructed[i])
                )
                for i in range(len(data))
            ])

        return address, reconstructed, fidelity

    def save(self, path: str):
        """Save projector to file."""
        np.savez_compressed(path,
            Df=self.Df,
            basis=self.basis,
            mean=self.mean,
            eigenvalues=self.eigenvalues
        )
        print(f"Projector saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'HolographicProjector':
        """Load projector from file."""
        data = np.load(path)
        proj = cls(Df=int(data['Df']))
        proj.basis = data['basis']
        proj.mean = data['mean']
        proj.eigenvalues = data['eigenvalues']
        print(f"Projector loaded: {path}")
        print(f"  Df: {proj.Df}, Output dim: {proj.basis.shape[1]}")
        return proj


# Demo: prove the formula works
if __name__ == "__main__":
    print("=" * 60)
    print("HOLOGRAPHIC PROJECTOR DEMO")
    print("R = (E / gradS) * sigma(f)^Df")
    print("=" * 60)

    # Generate data with known structure
    np.random.seed(42)
    true_Df = 5  # True dimensionality
    N, D = 1000, 100  # 1000 samples in 100D

    # Data lives on a 5D manifold in 100D space
    latent = np.random.randn(N, true_Df)  # True coordinates
    true_basis = np.random.randn(true_Df, D)  # True projection
    true_basis = true_basis / np.linalg.norm(true_basis, axis=1, keepdims=True)
    data = latent @ true_basis + np.random.randn(N, D) * 0.1  # Add noise

    print(f"\nGenerated data: {N} samples × {D} dims")
    print(f"True Df: {true_Df}")

    # Learn projector
    projector = HolographicProjector()
    projector.learn(data)

    # Test roundtrip
    test_sample = data[0]
    address, reconstructed, fidelity = projector.roundtrip(test_sample)

    print(f"\n--- Roundtrip Test ---")
    print(f"Address (Df={projector.Df} numbers): {address[:5]}...")
    print(f"Fidelity: {fidelity:.4f}")

    # Storage comparison
    original_bytes = D * 4  # float32
    address_bytes = projector.Df * 2  # float16
    print(f"\n--- Storage ---")
    print(f"Original: {original_bytes} bytes")
    print(f"Address: {address_bytes} bytes")
    print(f"Compression: {original_bytes / address_bytes:.1f}x")

    # The formula in action
    print(f"\n--- The Formula ---")
    print(f"sigma(f)^Df = basis matrix ({projector.Df} x {D})")
    print(f"address @ basis = {projector.Df} numbers -> {D} numbers")
    print(f"R = rendered hologram")
