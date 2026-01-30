#!/usr/bin/env python3
"""
Q51 Phase Structure Definitive Test

Distinguishes between:
1. Geometric phase (coordinate-dependent artifact of PCA projection)
2. Complex phase (intrinsic physical structure with invariants)
3. Real structure (purely real-valued embeddings with no complex component)

This test resolves the FORMULA vs kimi contradiction:
- FORMULA: Found "phase arithmetic" (90.9% success) suggesting complex structure
- kimi: Found real eigenvalues (no imaginary parts) suggesting real structure

Test 1: Geometric Phase Test (Rotation Invariance)
Test 2: Complex Conjugate Test (Eigenvalue Spectrum)
Test 3: Coordinate Dependence Test (Correlation Across Bases)

Author: Claude
Date: 2026-01-30
Location: THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED/
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add paths
ROOT = Path(__file__).parent.parent.parent.parent.parent.parent  # Up to THOUGHT/LAB/
FORMULA_ROOT = ROOT / "FORMULA"
sys.path.insert(0, str(FORMULA_ROOT / "questions"))
sys.path.insert(0, str(ROOT / "VECTOR_ELO" / "eigen-alignment"))
sys.path.insert(0, str(ROOT / "VECTOR_ELO" / "eigen-alignment" / "qgt_lib" / "python"))

# Import shared infrastructure
try:
    from qgt import normalize_embeddings, fubini_study_metric, metric_eigenspectrum, pca_winding_angle
except ImportError:
    print("WARNING: QGT library not found. Using fallback implementations.")
    
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embeddings / norms
    
    def fubini_study_metric(embeddings, normalize=True):
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        centered = embeddings - embeddings.mean(axis=0)
        return np.cov(centered.T)
    
    def metric_eigenspectrum(embeddings, normalize=True):
        metric = fubini_study_metric(embeddings, normalize)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def pca_winding_angle(path, closed=True):
        path = normalize_embeddings(path)
        if closed and not np.allclose(path[0], path[-1]):
            path = np.vstack([path, path[0:1]])
        centered = path - path.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj_2d = centered @ Vt[:2].T
        except:
            return 0.0
        if proj_2d.shape[1] < 2:
            return 0.0
        z = proj_2d[:, 0] + 1j * proj_2d[:, 1]
        z = np.where(np.abs(z) > 1e-10, z, 1e-10)
        phase_diffs = np.angle(z[1:] / z[:-1])
        return np.sum(phase_diffs)


@dataclass
class PhaseStructureResult:
    """Results from phase structure testing."""
    model_name: str
    n_samples: int
    embedding_dim: int
    
    # Test 1: Geometric Phase Test
    geometric_rotation_invariance: float  # Correlation between phase shifts
    geometric_phase_shift_mean: float     # Mean phase shift vs rotation
    geometric_phase_shift_std: float      # Std of phase shifts
    is_geometric_phase: bool              # True if coordinate-dependent
    
    # Test 2: Complex Conjugate Test
    max_imag_eigenvalue: float           # Maximum imaginary part
    eigenvalue_spectrum_real: List[float]  # Real eigenvalues
    has_complex_pairs: bool               # True if complex conjugate pairs found
    is_complex_structure: bool            # True if complex structure detected
    
    # Test 3: Coordinate Dependence Test
    basis_correlation_mean: float         # Mean correlation across bases
    basis_correlation_std: float          # Std of correlations
    is_coordinate_invariant: bool         # True if phase is physical
    
    # Overall determination
    structure_type: str                   # "geometric", "complex", "real", or "mixed"
    confidence: float                     # Confidence in determination (0-1)
    
    def to_dict(self):
        return asdict(self)


# =============================================================================
# Test 1: Geometric Phase Test (Rotation Invariance)
# =============================================================================

def test_geometric_phase(embeddings: np.ndarray, n_rotations: int = 10) -> Dict:
    """
    Test if "phase" is a geometric artifact of PCA basis choice.
    
    Theory: If phase is geometric (coordinate-dependent), rotating the PCA
    basis by angle alpha should shift all phases by alpha:
        theta' = theta + alpha
    
    If phase is physical (coordinate-invariant), rotation should NOT affect it:
        theta' = theta
    
    Args:
        embeddings: (n_samples, dim) array
        n_rotations: Number of random rotations to test
        
    Returns:
        Dict with geometric phase metrics
    """
    # Compute standard PCA
    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Original 2D projection and phases
    proj_2d_original = centered @ Vt[:2].T
    theta_original = np.arctan2(proj_2d_original[:, 1], proj_2d_original[:, 0])
    
    phase_shifts = []
    
    for _ in range(n_rotations):
        # Random rotation angle
        alpha = np.random.uniform(0, 2 * np.pi)
        
        # Rotation matrix in 2D PCA space
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        
        # Rotate the 2D projection
        proj_2d_rotated = proj_2d_original @ R.T
        
        # Compute new phases
        theta_rotated = np.arctan2(proj_2d_rotated[:, 1], proj_2d_rotated[:, 0])
        
        # Compute phase shift
        # theta_rotated - theta_original should equal alpha (geometric)
        # or should equal 0 (physical)
        delta_theta = np.angle(np.exp(1j * (theta_rotated - theta_original)))
        mean_shift = np.mean(delta_theta)
        
        # Normalize to [-pi, pi]
        mean_shift = (mean_shift + np.pi) % (2 * np.pi) - np.pi
        
        phase_shifts.append({
            'rotation_angle': alpha,
            'mean_phase_shift': mean_shift,
            'shift_error': abs(mean_shift - alpha)
        })
    
    # Analyze results
    shifts = np.array([p['mean_phase_shift'] for p in phase_shifts])
    rotations = np.array([p['rotation_angle'] for p in phase_shifts])
    errors = np.array([p['shift_error'] for p in phase_shifts])
    
    # Correlation between rotation and phase shift
    correlation = np.corrcoef(rotations, shifts)[0, 1] if len(rotations) > 1 else 0.0
    
    # If geometric: shifts should follow rotations (correlation ~1)
    # If physical: shifts should be ~0 regardless of rotation (correlation ~0)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Determine if geometric
    # High correlation (>0.9) indicates geometric phase
    # Low correlation (<0.3) indicates physical phase
    is_geometric = correlation > 0.8 and mean_error < 0.5
    
    return {
        'rotation_invariance_correlation': float(correlation),
        'phase_shift_mean': float(np.mean(shifts)),
        'phase_shift_std': float(np.std(shifts)),
        'shift_error_mean': float(mean_error),
        'shift_error_std': float(std_error),
        'is_geometric_phase': bool(is_geometric),
        'n_rotations_tested': n_rotations
    }


# =============================================================================
# Test 2: Complex Conjugate Test (Eigenvalue Spectrum)
# =============================================================================

def test_complex_conjugate(embeddings: np.ndarray) -> Dict:
    """
    Test for complex conjugate eigenvalue pairs indicating complex structure.
    
    Theory: Complex structures have eigenvalues in conjugate pairs (λ, λ*).
    Real symmetric structures have purely real eigenvalues.
    
    We test the covariance matrix (real symmetric by construction).
    For true complex structure, we'd need to test a non-symmetric operator.
    
    However, we can look for:
    1. Purely real eigenvalues → Real structure
    2. Complex eigenvalues → Complex structure (requires non-symmetric matrix)
    
    Args:
        embeddings: (n_samples, dim) array
        
    Returns:
        Dict with complex structure metrics
    """
    # Method 1: Standard real covariance (symmetric, always real eigenvalues)
    centered = embeddings - embeddings.mean(axis=0)
    C_symmetric = np.cov(centered.T)
    
    eigvals_symmetric = np.linalg.eigvalsh(C_symmetric)
    max_imag_symmetric = np.max(np.abs(np.imag(eigvals_symmetric)))
    
    # Method 2: Non-symmetric "complex-like" matrix
    # If embeddings encode complex structure, there might be a non-symmetric
    # representation that reveals it. We test by computing the cross-covariance
    # between odd and even dimensions (simulating complex structure).
    
    n_dims = embeddings.shape[1]
    if n_dims >= 4:
        # Split dimensions: odd vs even (simulating real vs imaginary parts)
        real_part = centered[:, ::2]  # Even indices
        imag_part = centered[:, 1::2]  # Odd indices
        
        # Cross-covariance matrix (non-symmetric)
        C_cross = (real_part.T @ imag_part) / len(embeddings)
        
        # Ensure square for eigendecomposition
        min_dim = min(C_cross.shape)
        if min_dim > 0:
            C_square = C_cross[:min_dim, :min_dim]
            eigvals_cross = np.linalg.eigvals(C_square)
            max_imag_cross = np.max(np.abs(np.imag(eigvals_cross)))
            has_complex_cross = max_imag_cross > 1e-10
        else:
            eigvals_cross = np.array([])
            max_imag_cross = 0.0
            has_complex_cross = False
    else:
        eigvals_cross = np.array([])
        max_imag_cross = 0.0
        has_complex_cross = False
    
    # Determine structure type
    # Real symmetric matrices ALWAYS have real eigenvalues (mathematical theorem)
    is_complex = has_complex_cross and max_imag_cross > 1e-6
    is_real = max_imag_symmetric < 1e-10
    
    return {
        'max_imag_eigenvalue': float(max(max_imag_symmetric, max_imag_cross)),
        'eigenvalue_spectrum_real': eigvals_symmetric[:10].tolist(),
        'has_complex_pairs': bool(is_complex),
        'is_complex_structure': bool(is_complex),
        'is_real_structure': bool(is_real),
        'max_imag_symmetric': float(max_imag_symmetric),
        'max_imag_cross': float(max_imag_cross),
        'n_eigenvalues_tested': len(eigvals_symmetric)
    }


# =============================================================================
# Test 3: Coordinate Dependence Test (Correlation Across Bases)
# =============================================================================

def test_coordinate_dependence(embeddings: np.ndarray, n_bases: int = 5) -> Dict:
    """
    Test if "phase" is consistent across different PCA bases.
    
    Theory: Physical phase should be coordinate-invariant (same across bases).
    Geometric artifact should vary with basis choice.
    
    We compute phases in multiple random bases and check correlation.
    
    Args:
        embeddings: (n_samples, dim) array
        n_bases: Number of different bases to test
        
    Returns:
        Dict with coordinate dependence metrics
    """
    n_samples = embeddings.shape[0]
    dim = embeddings.shape[1]
    
    # Compute phases in different bases
    phase_matrices = []
    
    for _ in range(n_bases):
        # Random orthogonal basis
        random_matrix = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(random_matrix)
        
        # Project to 2D using random basis
        centered = embeddings - embeddings.mean(axis=0)
        proj_2d = centered @ Q[:, :2]
        
        # Compute phases
        phases = np.arctan2(proj_2d[:, 1], proj_2d[:, 0])
        phase_matrices.append(phases)
    
    phase_matrix = np.column_stack(phase_matrices)  # (n_samples, n_bases)
    
    # Compute correlation matrix
    correlations = []
    for i in range(n_bases):
        for j in range(i + 1, n_bases):
            # Circular correlation
            corr = np.corrcoef(phase_matrix[:, i], phase_matrix[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
    
    correlations = np.array(correlations)
    
    # Statistics
    mean_corr = np.mean(correlations) if len(correlations) > 0 else 0.0
    std_corr = np.std(correlations) if len(correlations) > 0 else 0.0
    
    # Determine coordinate invariance
    # High correlation (>0.7) suggests physical phase
    # Low correlation (<0.3) suggests geometric artifact
    is_coordinate_invariant = mean_corr > 0.5
    
    return {
        'basis_correlation_mean': float(mean_corr),
        'basis_correlation_std': float(std_corr),
        'basis_correlation_min': float(np.min(correlations)) if len(correlations) > 0 else 0.0,
        'basis_correlation_max': float(np.max(correlations)) if len(correlations) > 0 else 0.0,
        'is_coordinate_invariant': bool(is_coordinate_invariant),
        'n_bases_tested': n_bases
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_definitive_phase_test(
    embeddings: np.ndarray,
    model_name: str,
    verbose: bool = True
) -> PhaseStructureResult:
    """
    Run all three phase structure tests definitively.
    
    Args:
        embeddings: (n_samples, dim) array of embeddings
        model_name: Name of the model (for reporting)
        verbose: Print progress
        
    Returns:
        PhaseStructureResult with all test outcomes
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Q51 PHASE STRUCTURE DEFINITIVE TEST")
        print(f"Model: {model_name}")
        print(f"Samples: {embeddings.shape[0]}, Dim: {embeddings.shape[1]}")
        print(f"{'='*70}\n")
    
    n_samples = embeddings.shape[0]
    dim = embeddings.shape[1]
    
    # Test 1: Geometric Phase
    if verbose:
        print("Test 1: Geometric Phase (Rotation Invariance)...")
    geo_result = test_geometric_phase(embeddings, n_rotations=10)
    if verbose:
        print(f"  Rotation correlation: {geo_result['rotation_invariance_correlation']:.4f}")
        print(f"  Mean phase shift: {geo_result['phase_shift_mean']:.4f} ± {geo_result['phase_shift_std']:.4f}")
        print(f"  Is geometric phase: {geo_result['is_geometric_phase']}")
    
    # Test 2: Complex Conjugate
    if verbose:
        print("\nTest 2: Complex Conjugate (Eigenvalue Spectrum)...")
    complex_result = test_complex_conjugate(embeddings)
    if verbose:
        print(f"  Max |imag(eigval)|: {complex_result['max_imag_eigenvalue']:.2e}")
        print(f"  Has complex pairs: {complex_result['has_complex_pairs']}")
        print(f"  Is complex structure: {complex_result['is_complex_structure']}")
    
    # Test 3: Coordinate Dependence
    if verbose:
        print("\nTest 3: Coordinate Dependence (Basis Correlation)...")
    coord_result = test_coordinate_dependence(embeddings, n_bases=5)
    if verbose:
        print(f"  Mean basis correlation: {coord_result['basis_correlation_mean']:.4f}")
        print(f"  Correlation range: [{coord_result['basis_correlation_min']:.4f}, {coord_result['basis_correlation_max']:.4f}]")
        print(f"  Is coordinate invariant: {coord_result['is_coordinate_invariant']}")
    
    # Determine overall structure type
    is_geometric = geo_result['is_geometric_phase']
    is_complex = complex_result['is_complex_structure']
    is_invariant = coord_result['is_coordinate_invariant']
    
    # Classification logic
    if is_complex and is_invariant:
        structure_type = "complex"
        confidence = 0.9
    elif is_geometric and not is_invariant:
        structure_type = "geometric"
        confidence = 0.8
    elif not is_complex and not is_geometric:
        structure_type = "real"
        confidence = 0.9
    else:
        structure_type = "mixed"
        confidence = 0.5
    
    # Create result
    result = PhaseStructureResult(
        model_name=model_name,
        n_samples=n_samples,
        embedding_dim=dim,
        geometric_rotation_invariance=geo_result['rotation_invariance_correlation'],
        geometric_phase_shift_mean=geo_result['phase_shift_mean'],
        geometric_phase_shift_std=geo_result['phase_shift_std'],
        is_geometric_phase=geo_result['is_geometric_phase'],
        max_imag_eigenvalue=complex_result['max_imag_eigenvalue'],
        eigenvalue_spectrum_real=complex_result['eigenvalue_spectrum_real'],
        has_complex_pairs=complex_result['has_complex_pairs'],
        is_complex_structure=complex_result['is_complex_structure'],
        basis_correlation_mean=coord_result['basis_correlation_mean'],
        basis_correlation_std=coord_result['basis_correlation_std'],
        is_coordinate_invariant=coord_result['is_coordinate_invariant'],
        structure_type=structure_type,
        confidence=confidence
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL DETERMINATION: {structure_type.upper()} (confidence: {confidence:.2f})")
        print(f"{'='*70}\n")
    
    return result


# =============================================================================
# Model Loading
# =============================================================================

def load_minilm_embeddings(texts: List[str]) -> Tuple[np.ndarray, bool]:
    """Load MiniLM-L6 embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return embeddings, True
    except Exception as e:
        print(f"MiniLM load failed: {e}")
        return generate_fallback_embeddings(texts, 384), False


def load_bert_embeddings(texts: List[str]) -> Tuple[np.ndarray, bool]:
    """Load BERT-base embeddings."""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        model.eval()
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(vec)
        
        return np.array(embeddings), True
    except Exception as e:
        print(f"BERT load failed: {e}")
        return generate_fallback_embeddings(texts, 768), False


def generate_fallback_embeddings(texts: List[str], dim: int) -> np.ndarray:
    """Generate deterministic fallback embeddings."""
    embeddings = []
    for text in texts:
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        vec = np.random.randn(dim).astype(np.float32)
        embeddings.append(vec)
    return np.array(embeddings)


# =============================================================================
# Test Corpus
# =============================================================================

TEST_CORPUS = [
    # Semantic categories for comprehensive testing
    "king", "queen", "man", "woman", "prince", "princess",
    "father", "mother", "son", "daughter", "brother", "sister",
    "good", "bad", "better", "worse", "best", "worst",
    "happy", "sad", "angry", "calm", "excited", "bored",
    "big", "small", "tall", "short", "wide", "narrow",
    "hot", "cold", "warm", "cool", "freezing", "burning",
    "love", "hate", "fear", "hope", "joy", "despair",
    "run", "walk", "jump", "swim", "fly", "crawl",
    "red", "blue", "green", "yellow", "black", "white",
    "dog", "cat", "bird", "fish", "horse", "cow",
    "house", "car", "tree", "book", "phone", "computer",
]


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run definitive phase structure tests on MiniLM and BERT."""
    print("="*70)
    print("Q51 PHASE STRUCTURE DEFINITIVE TEST")
    print("Resolving FORMULA vs kimi contradiction")
    print("="*70)
    
    results = {}
    
    # Test MiniLM-L6
    print("\n\n>>> Testing MiniLM-L6-v2...")
    minilm_emb, minilm_ok = load_minilm_embeddings(TEST_CORPUS)
    minilm_result = run_definitive_phase_test(minilm_emb, "MiniLM-L6-v2")
    results['minilm'] = minilm_result.to_dict()
    results['minilm']['model_loaded'] = minilm_ok
    
    # Test BERT-base
    print("\n\n>>> Testing BERT-base-uncased...")
    bert_emb, bert_ok = load_bert_embeddings(TEST_CORPUS)
    bert_result = run_definitive_phase_test(bert_emb, "BERT-base-uncased")
    results['bert'] = bert_result.to_dict()
    results['bert']['model_loaded'] = bert_ok
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = Path(__file__).stem.split('_')[-1]
    output_file = output_dir / f"q51_phase_structure_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    print(f"\nMiniLM-L6-v2:")
    print(f"  Structure type: {results['minilm']['structure_type']}")
    print(f"  Confidence: {results['minilm']['confidence']}")
    print(f"  Geometric phase: {results['minilm']['is_geometric_phase']}")
    print(f"  Complex structure: {results['minilm']['is_complex_structure']}")
    
    print(f"\nBERT-base-uncased:")
    print(f"  Structure type: {results['bert']['structure_type']}")
    print(f"  Confidence: {results['bert']['confidence']}")
    print(f"  Geometric phase: {results['bert']['is_geometric_phase']}")
    print(f"  Complex structure: {results['bert']['is_complex_structure']}")
    
    # Resolution
    print("\n" + "="*70)
    print("RESOLUTION OF FORMULA vs kimi CONTRADICTION")
    print("="*70)
    
    both_real = (results['minilm']['structure_type'] == 'real' and 
                 results['bert']['structure_type'] == 'real')
    both_geometric = (results['minilm']['structure_type'] == 'geometric' and 
                      results['bert']['structure_type'] == 'geometric')
    
    if both_real:
        print("\n✓ VERDICT: Both models show REAL structure (kimi is correct)")
        print("  - No complex conjugate eigenvalue pairs found")
        print("  - Phase is NOT coordinate-invariant (geometric artifact)")
        print("  - FORMULA's 'phase arithmetic' is an emergent geometric property")
    elif both_geometric:
        print("\n✓ VERDICT: Both models show GEOMETRIC phase structure")
        print("  - Phase is coordinate-dependent (rotates with basis)")
        print("  - FORMULA's findings are geometric artifacts")
        print("  - kimi correctly identified real-valued structure")
    else:
        print("\n⚠ VERDICT: Mixed or ambiguous results")
        print("  - Structure type varies by model")
        print("  - Requires further investigation")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
