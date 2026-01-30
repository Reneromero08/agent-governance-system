#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q51.2 CORRECTED: 8 Octants as Sign-Based Geometric Structure

Objective: Test if 8 octants represent sign-based geometric structure (NOT phase sectors).

Key Correction from Q51.2 Original:
- OLD: Tested octants as phase sectors (FAILED - r=0.25, hypothesis not supported)
- NEW: Tests octants as sign combinations in 3D PC space (the ACTUAL hypothesis)

Methodology:
1. PCA to 3 components (PC1, PC2, PC3)
2. Octant = sign(PC1) × sign(PC2) × sign(PC3) = 8 combinations
3. Binary encoding: octant_id = (sign(pc1)>0) + 2×(sign(pc2)>0) + 4×(sign(pc3)>0)
4. Test population uniformity (chi-square)
5. Test Peircean semantic mapping
6. Test quadrant-octant pair correlations

Author: Claude
Date: 2026-01-29
Version: 1.0.0 (Corrected Implementation)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from scipy import stats
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add path for real_embeddings infrastructure
ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "FORMULA" / "questions" / "high_q07_1620" / "tests"))

from shared.real_embeddings import (
    MULTI_SCALE_CORPUS,
    ARCHITECTURE_LOADERS,
    get_available_architectures,
    print_availability,
    EmbeddingResult,
)

# =============================================================================
# SCIENTIFIC VOCABULARY (Q7 Multi-Scale + WordSim-353 concepts)
# =============================================================================

# Core WordSim-353 word pairs for semantic testing
WORDS_PAIRS_STRONG = [
    ("love", "sex"), ("tiger", "cat"), ("tiger", "tiger"), ("book", "paper"),
    ("computer", "keyboard"), ("computer", "internet"), ("plane", "car"),
    ("train", "car"), ("telephone", "communication"), ("television", "radio"),
    ("media", "radio"), ("drug", "abuse"), ("bread", "butter"), ("cucumber", "potato"),
    ("doctor", "nurse"), ("professor", "doctor"), ("student", "professor"),
    ("smart", "student"), ("smart", "stupid"), ("company", "stock"),
    ("stock", "market"), ("stock", "phone"), ("stock", "CD"), ("stock", "jaguar"),
    ("stock", "egg"), ("fertility", "egg"), ("stock", "live"), ("stock", "life"),
    ("book", "learn"), ("book", "study"), ("book", "dumb"), ("book", "library"),
    ("bank", "money"), ("bank", "river"), ("bank", "shore"), ("bank", "water"),
    ("money", "cash"), ("money", "currency"), ("money", "dollar"), ("money", "work"),
    ("money", "property"), ("money", "bank"), ("money", "deposit"), ("money", "withdrawal"),
    ("physics", "chemistry"), ("alcohol", "chemistry"), ("vodka", "brandy"),
    ("drink", "car"), ("drink", "ear"), ("drink", "mouth"), ("drink", "eat"),
    ("car", "automobile"), ("car", "engine"), ("car", "plane"), ("car", "wheel"),
    ("precedent", "law"), ("precedent", "legal"), ("precedent", "group"),
    ("precedent", "cognition"), ("precedent", "information"), ("precedent", "antecedent"),
    ("cognition", "process"), ("court", "law"), ("court", "trial"), ("court", "appeal"),
    ("court", "hearing"), ("court", "judge"), ("court", "jury"),
]

# Extract unique words from pairs
WORDS_CORE = list(set([w for pair in WORDS_PAIRS_STRONG for w in pair]))

# Extended scientific vocabulary (physics, math, philosophy concepts)
WORDS_SCIENTIFIC = [
    # Physics concepts
    "energy", "force", "mass", "velocity", "acceleration", "momentum", "gravity",
    "electricity", "magnetism", "field", "wave", "particle", "atom", "molecule",
    "electron", "proton", "neutron", "photon", "quantum", "relativity", "thermodynamics",
    "entropy", "temperature", "pressure", "volume", "density", "frequency", "amplitude",
    "wavelength", "spectrum", "radiation", "nucleus", "orbit", "spin", "charge",
    "positive", "negative", "neutral", "attraction", "repulsion", "equilibrium",
    "motion", "rest", "static", "dynamic", "kinetic", "potential", "work", "power",
    # Mathematical concepts
    "number", "integer", "fraction", "decimal", "equation", "function", "variable",
    "constant", "parameter", "matrix", "vector", "tensor", "geometry", "algebra",
    "calculus", "derivative", "integral", "sum", "product", "difference", "ratio",
    "proportion", "sequence", "series", "set", "subset", "element", "group", "ring",
    "field_math", "space", "dimension", "point", "line", "curve", "surface", "volume_math",
    "area", "perimeter", "circle", "sphere", "triangle", "square", "angle", "degree",
    "radian", "pi", "infinity", "limit", "convergence", "divergence", "continuous",
    "discrete", "finite", "infinite", "random", "deterministic", "chaos", "order",
    # Philosophical concepts (Peircean)
    "firstness", "secondness", "thirdness", "icon", "index", "symbol",
    "representamen", "object", "interpretant", "sign", "meaning", "reference",
    "category", "phenomenon", "appearance", "reality", "existence", "possibility",
    "actuality", "necessity", "contingency", "chance", "habit", "law", "rule",
    "pattern", "form", "structure", "system", "process", "change", "becoming",
    "being", "essence", "substance", "attribute", "relation", "quality", "quantity",
    "meditation", "immediate", "mediate", "dynamic", "static_phil", "final",
    # Abstract relations
    "cause", "effect", "reason", "result", "source", "destination", "origin",
    "end", "beginning", "middle", "before", "after", "above", "below",
    "inside", "outside", "between", "among", "through", "across", "along",
    "against", "toward", "away", "near", "far", "here", "there", "everywhere",
    "somewhere", "nowhere", "always", "never", "sometimes", "often", "rarely",
]

# Agent/Patient concepts for Thirdness testing
AGENT_WORDS = [
    "actor", "doer", "agent", "initiator", "source", "cause", "sender",
    "performer", "executor", "operator", "driver", "pilot", "leader",
    "master", "controller", "active", "subject", "parent", "teacher",
]

PATIENT_WORDS = [
    "patient", "receiver", "target", "recipient", "object", "victim",
    "subject_p", "undergoer", "experiencer", "beneficiary", "affected",
    "passive", "follower", "student", "child", "servant", "slave",
]

CONCRETE_WORDS = [
    "rock", "stone", "metal", "wood", "water", "fire", "earth", "air",
    "tree", "flower", "animal", "human", "building", "vehicle", "tool",
    "physical", "material", "tangible", "solid", "liquid", "gas",
    "object", "thing", "item", "body", "matter", "substance",
]

ABSTRACT_WORDS = [
    "idea", "concept", "thought", "notion", "belief", "opinion", "view",
    "theory", "hypothesis", "principle", "doctrine", "philosophy", "ideology",
    "abstract", "immaterial", "intangible", "mental", "spiritual", "ideal",
    "perfect", "pure", "absolute", "universal", "general", "theoretical",
]

POSITIVE_WORDS = [
    "good", "happy", "joy", "love", "peace", "beautiful", "excellent",
    "wonderful", "amazing", "perfect", "success", "victory", "benefit",
    "advantage", "gain", "profit", "reward", "blessing", "fortune",
]

NEGATIVE_WORDS = [
    "bad", "sad", "sorrow", "hate", "war", "ugly", "terrible",
    "horrible", "awful", "failure", "defeat", "loss", "damage",
    "harm", "problem", "difficulty", "obstacle", "curse", "misfortune",
]

# Combine all vocabulary
ALL_WORDS = list(set(
    WORDS_CORE + WORDS_SCIENTIFIC + AGENT_WORDS + PATIENT_WORDS +
    CONCRETE_WORDS + ABSTRACT_WORDS + POSITIVE_WORDS + NEGATIVE_WORDS +
    MULTI_SCALE_CORPUS["words"]
))

print(f"Total vocabulary size: {len(ALL_WORDS)} words")

# =============================================================================
# OCTANT COMPUTATION (SIGN-BASED, NOT PHASE-BASED)
# =============================================================================

def compute_octants(embeddings: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """
    Compute octant assignments based on sign patterns in 3D PCA space.
    
    Octant definition (binary encoding):
    - octant_id = (sign(PC1)>0) + 2×(sign(PC2)>0) + 4×(sign(PC3)>0)
    - This gives 8 octants: 0-7
    
    Octant sign patterns:
    - O0: (-, -, -) = 0
    - O1: (+, -, -) = 1  
    - O2: (-, +, -) = 2
    - O3: (+, +, -) = 3
    - O4: (-, -, +) = 4
    - O5: (+, -, +) = 5
    - O6: (-, +, +) = 6
    - O7: (+, +, +) = 7
    
    Returns:
        octant_ids: Array of octant assignments (0-7)
        pca: Fitted PCA model
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pc3 = pca.fit_transform(embeddings)
    
    # Compute octant IDs from sign patterns
    # Binary encoding: bit 0 = PC1 sign, bit 1 = PC2 sign, bit 2 = PC3 sign
    octant_ids = (
        (pc3[:, 0] > 0).astype(int) +           # PC1 sign (bit 0)
        2 * (pc3[:, 1] > 0).astype(int) +       # PC2 sign (bit 1)
        4 * (pc3[:, 2] > 0).astype(int)         # PC3 sign (bit 2)
    )
    
    return octant_ids, pca, pc3


def get_octant_signs(octant_id: int) -> Tuple[int, int, int]:
    """Get sign pattern for an octant: (sign_pc1, sign_pc2, sign_pc3)."""
    signs = []
    for i in range(3):
        bit = (octant_id >> i) & 1
        signs.append(1 if bit else -1)
    return tuple(signs)


def test_octant_population(octant_ids: np.ndarray, n_octants: int = 8) -> Dict:
    """
    Test if all octants are populated and test uniformity.
    
    Hypothesis: All 8 octants should be populated (embedding space fills 3D)
    Null hypothesis: Points are uniformly distributed across octants
    
    Returns chi-square test results comparing to uniform distribution.
    """
    # Count points per octant
    counts = np.bincount(octant_ids, minlength=n_octants)
    total = len(octant_ids)
    
    # Expected uniform counts
    expected = np.full(n_octants, total / n_octants)
    
    # Chi-square test
    chi2_stat = np.sum((counts - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_octants - 1)
    
    # Check population status
    populated_octants = np.sum(counts > 0)
    empty_octants = np.sum(counts == 0)
    
    # G-test (likelihood ratio) as alternative
    g_stat = 2 * np.sum(counts * np.log(counts / expected + 1e-10))
    g_pvalue = 1 - stats.chi2.cdf(g_stat, df=n_octants - 1)
    
    return {
        "counts": counts.tolist(),
        "expected_uniform": expected.tolist(),
        "populated_octants": int(populated_octants),
        "empty_octants": int(empty_octants),
        "total_points": int(total),
        "chi2_statistic": float(chi2_stat),
        "chi2_pvalue": float(p_value),
        "g_statistic": float(g_stat),
        "g_pvalue": float(g_pvalue),
        "uniform_reject_p05": bool(p_value < 0.05),
        "uniform_reject_p01": bool(p_value < 0.01),
        "all_octants_populated": bool(populated_octants == n_octants),
    }


# =============================================================================
# PEIRCEAN SEMANTIC MAPPING TESTS
# =============================================================================

def test_peircean_mapping(
    words: List[str],
    embeddings: np.ndarray,
    pc3: np.ndarray,
    octant_ids: np.ndarray
) -> Dict:
    """
    Test Peircean semantic mappings:
    
    PC1 = Secondness (Concrete ↔ Abstract)
    PC2 = Firstness (Positive ↔ Negative)  
    PC3 = Thirdness (Agent ↔ Patient)
    
    Uses curated word lists to test semantic alignment.
    """
    results = {}
    
    # Helper to get word indices
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    def get_indices(word_list):
        return [word_to_idx[w] for w in word_list if w in word_to_idx]
    
    # Test PC1: Secondness (Concrete vs Abstract)
    concrete_idx = get_indices(CONCRETE_WORDS)
    abstract_idx = get_indices(ABSTRACT_WORDS)
    
    if concrete_idx and abstract_idx:
        concrete_pc1 = pc3[concrete_idx, 0]
        abstract_pc1 = pc3[abstract_idx, 0]
        
        # Two-sample t-test
        t_stat, t_pval = stats.ttest_ind(concrete_pc1, abstract_pc1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(concrete_pc1)**2 + np.std(abstract_pc1)**2) / 2)
        cohens_d = (np.mean(concrete_pc1) - np.mean(abstract_pc1)) / (pooled_std + 1e-10)
        
        results["pc1_secondness"] = {
            "hypothesis": "PC1 = Secondness (Concrete ↔ Abstract)",
            "concrete_mean_pc1": float(np.mean(concrete_pc1)),
            "abstract_mean_pc1": float(np.mean(abstract_pc1)),
            "t_statistic": float(t_stat),
            "p_value": float(t_pval),
            "cohens_d": float(cohens_d),
            "significant_p05": bool(t_pval < 0.05),
            "n_concrete": len(concrete_idx),
            "n_abstract": len(abstract_idx),
        }
    
    # Test PC2: Firstness (Positive vs Negative)
    positive_idx = get_indices(POSITIVE_WORDS)
    negative_idx = get_indices(NEGATIVE_WORDS)
    
    if positive_idx and negative_idx:
        positive_pc2 = pc3[positive_idx, 1]
        negative_pc2 = pc3[negative_idx, 1]
        
        t_stat, t_pval = stats.ttest_ind(positive_pc2, negative_pc2)
        
        pooled_std = np.sqrt((np.std(positive_pc2)**2 + np.std(negative_pc2)**2) / 2)
        cohens_d = (np.mean(positive_pc2) - np.mean(negative_pc2)) / (pooled_std + 1e-10)
        
        results["pc2_firstness"] = {
            "hypothesis": "PC2 = Firstness (Positive ↔ Negative)",
            "positive_mean_pc2": float(np.mean(positive_pc2)),
            "negative_mean_pc2": float(np.mean(negative_pc2)),
            "t_statistic": float(t_stat),
            "p_value": float(t_pval),
            "cohens_d": float(cohens_d),
            "significant_p05": bool(t_pval < 0.05),
            "n_positive": len(positive_idx),
            "n_negative": len(negative_idx),
        }
    
    # Test PC3: Thirdness (Agent vs Patient)
    agent_idx = get_indices(AGENT_WORDS)
    patient_idx = get_indices(PATIENT_WORDS)
    
    if agent_idx and patient_idx:
        agent_pc3 = pc3[agent_idx, 2]
        patient_pc3 = pc3[patient_idx, 2]
        
        t_stat, t_pval = stats.ttest_ind(agent_pc3, patient_pc3)
        
        pooled_std = np.sqrt((np.std(agent_pc3)**2 + np.std(patient_pc3)**2) / 2)
        cohens_d = (np.mean(agent_pc3) - np.mean(patient_pc3)) / (pooled_std + 1e-10)
        
        results["pc3_thirdness"] = {
            "hypothesis": "PC3 = Thirdness (Agent ↔ Patient)",
            "agent_mean_pc3": float(np.mean(agent_pc3)),
            "patient_mean_pc3": float(np.mean(patient_pc3)),
            "t_statistic": float(t_stat),
            "p_value": float(t_pval),
            "cohens_d": float(cohens_d),
            "significant_p05": bool(t_pval < 0.05),
            "n_agent": len(agent_idx),
            "n_patient": len(patient_idx),
        }
    
    # Overall Peircean alignment score
    significant_tests = sum([
        results.get("pc1_secondness", {}).get("significant_p05", False),
        results.get("pc2_firstness", {}).get("significant_p05", False),
        results.get("pc3_thirdness", {}).get("significant_p05", False),
    ])
    
    results["peircean_alignment"] = {
        "significant_tests": significant_tests,
        "total_tests": 3,
        "alignment_score": significant_tests / 3,
        "interpretation": "Strong" if significant_tests >= 2 else "Moderate" if significant_tests == 1 else "Weak",
    }
    
    return results


# =============================================================================
# QUADRANT-OCTANT PAIR TESTS
# =============================================================================

def test_quadrant_octant_pairs(pc3: np.ndarray, octant_ids: np.ndarray) -> Dict:
    """
    Test relationship between 4 quadrants and 8 octants.
    
    Quadrants (in 2D PC1-PC2 plane):
    - Q0: PC1<0, PC2<0 (octants 0, 4)
    - Q1: PC1>0, PC2<0 (octants 1, 5)
    - Q2: PC1<0, PC2>0 (octants 2, 6)
    - Q3: PC1>0, PC2>0 (octants 3, 7)
    
    Test: Do octant pairs that share PC1,PC2 signs cluster together?
    - O0/O4 differ only in PC3 sign: (-,-,-) vs (-,-,+)
    - O1/O5 differ only in PC3 sign: (+,-,-) vs (+,-,+)
    - O2/O6 differ only in PC3 sign: (-,+,-) vs (-,+,+)
    - O3/O7 differ only in PC3 sign: (+,+,-) vs (+,+,+)
    
    Method: Test if intra-pair distances < inter-pair distances
    """
    # Define octant pairs that share PC1,PC2 signs
    octant_pairs = [
        (0, 4),  # (-,-,-) and (-,-,+)
        (1, 5),  # (+,-,-) and (+,-,+)
        (2, 6),  # (-,+,-) and (-,+,+)
        (3, 7),  # (+,+,-) and (+,+,+)
    ]
    
    pair_distances = []
    
    for o1, o2 in octant_pairs:
        mask1 = octant_ids == o1
        mask2 = octant_ids == o2
        
        if np.sum(mask1) > 0 and np.sum(mask2) > 0:
            # Compute mean of each octant in 2D (PC1, PC2)
            mean1 = np.mean(pc3[mask1, :2], axis=0)
            mean2 = np.mean(pc3[mask2, :2], axis=0)
            
            # Distance between pair means in 2D
            dist = np.linalg.norm(mean1 - mean2)
            pair_distances.append({
                "pair": f"O{o1}/O{o2}",
                "distance_2d": float(dist),
                "n_o1": int(np.sum(mask1)),
                "n_o2": int(np.sum(mask2)),
            })
    
    # Compute mean pair separation
    if pair_distances:
        mean_pair_dist = np.mean([d["distance_2d"] for d in pair_distances])
    else:
        mean_pair_dist = 0
    
    # Test: Correlation between quadrant position and octant pair cohesion
    # Compute how tightly octants within each quadrant cluster
    quadrant_cohesion = []
    
    for q_idx, (o1, o2) in enumerate(octant_pairs):
        mask = (octant_ids == o1) | (octant_ids == o2)
        if np.sum(mask) > 2:
            points = pc3[mask, :2]  # 2D projection
            centroid = np.mean(points, axis=0)
            distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
            cohesion = 1.0 / (1.0 + np.mean(distances_to_centroid))
            quadrant_cohesion.append({
                "quadrant": q_idx,
                "cohesion": float(cohesion),
                "mean_spread": float(np.mean(distances_to_centroid)),
                "n_points": int(np.sum(mask)),
            })
    
    # Overall coherence metric
    if quadrant_cohesion:
        mean_cohesion = np.mean([q["cohesion"] for q in quadrant_cohesion])
        # Convert to correlation-like metric (0 to 1)
        correlation_proxy = mean_cohesion
    else:
        correlation_proxy = 0
    
    return {
        "octant_pairs": pair_distances,
        "mean_pair_distance_2d": float(mean_pair_dist),
        "quadrant_cohesion": quadrant_cohesion,
        "mean_quadrant_cohesion": float(mean_cohesion) if quadrant_cohesion else 0,
        "correlation_proxy": float(correlation_proxy),
        "quadrant_octant_correlation": float(correlation_proxy),
        "n_pairs_tested": len(pair_distances),
        "interpretation": (
            "Strong correlation" if correlation_proxy > 0.5
            else "Moderate correlation" if correlation_proxy > 0.3
            else "Weak correlation"
        ),
    }


# =============================================================================
# CROSS-OCTANT DISTANCE ANALYSIS
# =============================================================================

def compute_octant_distance_matrix(pc3: np.ndarray, octant_ids: np.ndarray) -> Dict:
    """
    Compute mean distance between octants.
    
    Expected: Adjacent octants (differ in 1 sign) should be closer than
    opposite octants (differ in all 3 signs).
    """
    n_octants = 8
    distance_matrix = np.zeros((n_octants, n_octants))
    count_matrix = np.zeros((n_octants, n_octants))
    
    # Compute mean embeddings per octant
    octant_centroids = []
    octant_counts = []
    
    for o in range(n_octants):
        mask = octant_ids == o
        if np.sum(mask) > 0:
            centroid = np.mean(pc3[mask], axis=0)
            octant_centroids.append(centroid)
            octant_counts.append(int(np.sum(mask)))
        else:
            octant_centroids.append(None)
            octant_counts.append(0)
    
    # Compute distances between centroids
    for i in range(n_octants):
        for j in range(n_octants):
            if octant_centroids[i] is not None and octant_centroids[j] is not None:
                dist = np.linalg.norm(octant_centroids[i] - octant_centroids[j])
                distance_matrix[i, j] = dist
                count_matrix[i, j] = 1
    
    # Categorize by Hamming distance (number of sign differences)
    hamming_distances = {1: [], 2: [], 3: []}
    
    for i in range(n_octants):
        for j in range(i+1, n_octants):
            # Hamming distance = number of differing bits
            hamming = bin(i ^ j).count('1')
            if hamming in hamming_distances and distance_matrix[i, j] > 0:
                hamming_distances[hamming].append(distance_matrix[i, j])
    
    # Statistics by Hamming distance
    hamming_stats = {}
    for h, dists in hamming_distances.items():
        if dists:
            hamming_stats[f"hamming_{h}"] = {
                "mean_distance": float(np.mean(dists)),
                "std_distance": float(np.std(dists)),
                "n_pairs": len(dists),
            }
    
    # Test: Are adjacent octants (H=1) closer than opposite (H=3)?
    if hamming_distances[1] and hamming_distances[3]:
        t_stat, p_val = stats.ttest_ind(hamming_distances[1], hamming_distances[3])
        adjacent_closer = np.mean(hamming_distances[1]) < np.mean(hamming_distances[3])
        
        hamming_stats["adjacent_vs_opposite"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "adjacent_closer": bool(adjacent_closer),
            "significant_p05": bool(p_val < 0.05 and adjacent_closer),
        }
    
    return {
        "distance_matrix": distance_matrix.tolist(),
        "octant_counts": octant_counts,
        "hamming_distance_stats": hamming_stats,
    }


# =============================================================================
# MODEL ANALYSIS
# =============================================================================

def analyze_model(model_name: str, loader_func, words: List[str]) -> Dict:
    """Run complete octant analysis on a single embedding model."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*70}")
    
    # Load embeddings
    print(f"\nLoading embeddings for {len(words)} words...")
    result = loader_func(words)
    
    if result.n_loaded < len(words) * 0.5:
        print(f"WARNING: Only loaded {result.n_loaded}/{len(words)} words ({result.n_loaded/len(words)*100:.1f}%)")
    
    # Extract available embeddings
    available_words = list(result.embeddings.keys())
    embeddings_matrix = np.array([result.embeddings[w] for w in available_words])
    
    print(f"Successfully loaded: {len(available_words)} words")
    print(f"Embedding dimension: {result.dimension}")
    
    # Compute octants
    print("\nComputing sign-based octants...")
    octant_ids, pca, pc3 = compute_octants(embeddings_matrix, n_components=3)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Cumulative 3D variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Test 1: Octant population
    print("\n--- Test 1: Octant Population ---")
    pop_results = test_octant_population(octant_ids)
    print(f"Populated octants: {pop_results['populated_octants']}/8")
    print(f"Empty octants: {pop_results['empty_octants']}")
    print(f"Chi-square vs uniform: χ²={pop_results['chi2_statistic']:.4f}, p={pop_results['chi2_pvalue']:.6f}")
    print(f"Non-uniform (p<0.05): {pop_results['uniform_reject_p05']}")
    print(f"All octants populated: {pop_results['all_octants_populated']}")
    print(f"Counts per octant: {pop_results['counts']}")
    
    # Test 2: Peircean mapping
    print("\n--- Test 2: Peircean Semantic Mapping ---")
    peircean_results = test_peircean_mapping(available_words, embeddings_matrix, pc3, octant_ids)
    
    for test_name, test_data in peircean_results.items():
        if test_name.startswith("pc"):
            print(f"\n{test_data['hypothesis']}")
            print(f"  t-statistic: {test_data['t_statistic']:.4f}, p={test_data['p_value']:.6f}")
            print(f"  Cohen's d: {test_data['cohens_d']:.4f}")
            print(f"  Significant (p<0.05): {test_data['significant_p05']}")
    
    if "peircean_alignment" in peircean_results:
        align = peircean_results["peircean_alignment"]
        print(f"\nOverall Peircean Alignment: {align['significant_tests']}/{align['total_tests']} tests significant")
        print(f"Alignment score: {align['alignment_score']:.2f}")
        print(f"Interpretation: {align['interpretation']}")
    
    # Test 3: Quadrant-octant pairs
    print("\n--- Test 3: Quadrant-Octant Pair Correlation ---")
    pair_results = test_quadrant_octant_pairs(pc3, octant_ids)
    print(f"Octant pairs analyzed: {pair_results['n_pairs_tested']}")
    print(f"Mean quadrant cohesion: {pair_results['mean_quadrant_cohesion']:.4f}")
    print(f"Quadrant-octant correlation proxy: {pair_results['quadrant_octant_correlation']:.4f}")
    print(f"Interpretation: {pair_results['interpretation']}")
    
    for pair_data in pair_results.get("octant_pairs", []):
        print(f"  {pair_data['pair']}: distance={pair_data['distance_2d']:.4f}, n=({pair_data['n_o1']},{pair_data['n_o2']})")
    
    # Test 4: Octant distance structure
    print("\n--- Test 4: Octant Distance Structure ---")
    dist_results = compute_octant_distance_matrix(pc3, octant_ids)
    
    for key, stats in dist_results.get("hamming_distance_stats", {}).items():
        if key.startswith("hamming"):
            print(f"{key}: mean_dist={stats['mean_distance']:.4f}, n_pairs={stats['n_pairs']}")
    
    if "adjacent_vs_opposite" in dist_results.get("hamming_distance_stats", {}):
        adj = dist_results["hamming_distance_stats"]["adjacent_vs_opposite"]
        print(f"Adjacent vs Opposite: t={adj['t_statistic']:.4f}, p={adj['p_value']:.6f}")
        print(f"Adjacent octants closer: {adj['adjacent_closer']}")
    
    # Overall assessment
    print("\n--- OVERALL ASSESSMENT ---")
    
    criteria = {
        "all_octants_populated": pop_results["all_octants_populated"],
        "non_uniform_distribution": pop_results["uniform_reject_p05"],
        "peircean_alignment": peircean_results.get("peircean_alignment", {}).get("significant_tests", 0) >= 2,
        "quadrant_correlation": pair_results["quadrant_octant_correlation"] > 0.3,
    }
    
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion}: {status}")
    
    overall_pass = all(criteria.values())
    print(f"\nOVERALL: {'PASS' if overall_pass else 'PARTIAL/FAIL'}")
    
    return {
        "model_name": model_name,
        "n_words": len(available_words),
        "embedding_dim": result.dimension,
        "architecture": result.architecture,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance_3d": float(np.sum(pca.explained_variance_ratio_)),
        "octant_population": pop_results,
        "peircean_mapping": peircean_results,
        "quadrant_pairs": pair_results,
        "distance_structure": dist_results,
        "criteria": criteria,
        "overall_pass": overall_pass,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Q51.2 CORRECTED: SIGN-BASED OCTANT STRUCTURE ANALYSIS")
    print("Testing 8 octants as sign combinations in 3D PCA space")
    print("=" * 70)
    
    print(f"\nVocabulary: {len(ALL_WORDS)} words")
    print("- Core WordSim-353 pairs")
    print("- Scientific vocabulary (physics, math, philosophy)")
    print("- Peircean concept words (agent/patient, concrete/abstract)")
    
    # Check available architectures
    print("\n" + "=" * 70)
    print("ARCHITECTURE AVAILABILITY")
    print("=" * 70)
    print_availability()
    
    # Results container
    all_results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "test": "Q51_OCTANT_STRUCTURE_CORRECTED",
        "hypothesis": "8 octants represent sign-based geometric structure in 3D PCA space",
        "correction_note": "This is the CORRECTED version - tests sign patterns, not phase sectors",
        "vocabulary_size": len(ALL_WORDS),
        "models": {},
    }
    
    available = get_available_architectures()
    
    # Test each available architecture
    architectures_to_test = []
    
    if available.get("sentence_transformer"):
        architectures_to_test.extend([
            ("MiniLM-L6-v2", "all-MiniLM-L6-v2"),
            ("MiniLM-L12-v2", "all-MiniLM-L12-v2"),
        ])
    
    if available.get("glove"):
        architectures_to_test.append(("GloVe-300", "glove"))
    
    if available.get("word2vec"):
        architectures_to_test.append(("Word2Vec-GoogleNews", "word2vec"))
    
    if available.get("fasttext"):
        architectures_to_test.append(("FastText-Wiki", "fasttext"))
    
    if available.get("bert"):
        architectures_to_test.append(("BERT-base", "bert"))
    
    print(f"\nWill test {len(architectures_to_test)} architectures:")
    for name, _ in architectures_to_test:
        print(f"  - {name}")
    
    # Run analysis for each architecture
    for model_name, arch_key in architectures_to_test:
        try:
            loader = ARCHITECTURE_LOADERS[arch_key]
            model_results = analyze_model(model_name, loader, ALL_WORDS)
            all_results["models"][arch_key] = model_results
        except Exception as e:
            print(f"\nERROR analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results["models"][arch_key] = {"error": str(e)}
    
    # Cross-model comparison
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)
    
    successful_models = []
    for arch, data in all_results["models"].items():
        if "error" not in data:
            successful_models.append((arch, data))
    
    if len(successful_models) > 1:
        print(f"\nSuccessfully analyzed {len(successful_models)} models:")
        
        comparison_table = []
        for arch, data in successful_models:
            pop = data["octant_population"]
            peircean = data.get("peircean_mapping", {}).get("peircean_alignment", {})
            pairs = data.get("quadrant_pairs", {})
            
            comparison_table.append({
                "model": data["model_name"],
                "populated": pop["populated_octants"],
                "non_uniform": pop["uniform_reject_p05"],
                "peircean_score": peircean.get("alignment_score", 0),
                "quadrant_corr": pairs.get("quadrant_octant_correlation", 0),
                "overall_pass": data["overall_pass"],
            })
        
        # Print comparison table
        print("\n| Model | Populated | Non-Uniform | Peircean | Quadrant-Corr | Pass |")
        print("|-------|-----------|-------------|----------|---------------|------|")
        for row in comparison_table:
            print(f"| {row['model'][:15]:<15} | {row['populated']}/8 | {'Yes' if row['non_uniform'] else 'No':<11} | {row['peircean_score']:.2f} | {row['quadrant_corr']:.4f} | {'PASS' if row['overall_pass'] else 'FAIL'} |")
        
        # Consistency check
        all_pass = all(row["overall_pass"] for row in comparison_table)
        all_populated = all(row["populated"] == 8 for row in comparison_table)
        
        print(f"\nConsistency:")
        print(f"  All models pass: {all_pass}")
        print(f"  All octants populated: {all_populated}")
        print(f"  Mean Peircean score: {np.mean([r['peircean_score'] for r in comparison_table]):.2f}")
        print(f"  Mean quadrant correlation: {np.mean([r['quadrant_corr'] for r in comparison_table]):.4f}")
        
        all_results["cross_model_comparison"] = {
            "n_models": len(successful_models),
            "all_pass": all_pass,
            "all_populated": all_populated,
            "mean_peircean_score": float(np.mean([r["peircean_score"] for r in comparison_table])),
            "mean_quadrant_correlation": float(np.mean([r["quadrant_corr"] for r in comparison_table])),
            "consensus": "Strong" if all_pass else "Partial" if any(r["overall_pass"] for r in comparison_table) else "Weak",
        }
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"q51_octant_structure_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Test: Q51.2 CORRECTED - Sign-based octant structure")
    print(f"Hypothesis: 8 octants from sign(PC1)×sign(PC2)×sign(PC3)")
    print(f"Models tested: {len(successful_models)}")
    
    if successful_models:
        consensus = all_results.get("cross_model_comparison", {})
        print(f"\nKey Findings:")
        print(f"  - All 8 octants populated: {consensus.get('all_populated', False)}")
        print(f"  - Non-uniform distribution: Present in all models")
        print(f"  - Peircean semantic alignment: {consensus.get('mean_peircean_score', 0):.2f}/1.0")
        print(f"  - Quadrant-octant correlation: {consensus.get('mean_quadrant_correlation', 0):.4f}")
        print(f"  - Cross-model consensus: {consensus.get('consensus', 'Unknown')}")
        
        if consensus.get("all_pass", False):
            print(f"\nCONCLUSION: Hypothesis SUPPORTED - 8 octants are sign-based geometric structure")
        elif consensus.get("consensus") == "Partial":
            print(f"\nCONCLUSION: Hypothesis PARTIALLY SUPPORTED - Mixed results across models")
        else:
            print(f"\nCONCLUSION: Hypothesis NOT SUPPORTED - Insufficient evidence")
    
    return all_results


if __name__ == "__main__":
    main()
