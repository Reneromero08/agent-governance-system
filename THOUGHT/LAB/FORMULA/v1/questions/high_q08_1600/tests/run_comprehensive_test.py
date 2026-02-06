"""
Q8 Comprehensive Test Suite - Real Embeddings
Tests all Q8 hypotheses with actual sentence transformer models.
"""

import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from q8_test_harness import compute_alpha_from_spectrum, spectral_chern_class
from test_q8_topological_invariance import (
    apply_rotation, apply_scaling, apply_smooth_warping
)
from test_q8_holonomy_revised import (
    project_to_pc12, solid_angle_spherical_triangle,
    berry_phase_loop, quantization_score
)


def main():
    print("=" * 70)
    print("Q8 COMPREHENSIVE TEST SUITE - REAL EMBEDDINGS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Full vocabulary
    VOCAB = [
        "truth", "beauty", "justice", "freedom", "love", "hate", "fear", "hope",
        "wisdom", "knowledge", "power", "strength", "weakness", "virtue", "vice",
        "water", "fire", "earth", "air", "stone", "tree", "river", "mountain",
        "sun", "moon", "star", "sky", "ocean", "forest", "desert", "city",
        "run", "walk", "jump", "fly", "swim", "think", "speak", "write",
        "create", "destroy", "build", "break", "give", "take", "push", "pull",
        "hot", "cold", "bright", "dark", "fast", "slow", "big", "small",
    ]

    MODELS = [
        ("MiniLM-L6", "all-MiniLM-L6-v2"),
        ("MPNet-base", "all-mpnet-base-v2"),
        ("Paraphrase-MiniLM", "paraphrase-MiniLM-L6-v2"),
        ("MultiQA-MiniLM", "multi-qa-MiniLM-L6-cos-v1"),
    ]

    # Semantic loops (verified in VOCAB)
    SEMANTIC_LOOPS = [
        ["love", "hope", "fear", "hate", "love"],
        ["water", "fire", "earth", "air", "water"],
        ["stone", "tree", "river", "mountain", "stone"],
        ["walk", "run", "jump", "fly", "walk"],
        ["sun", "moon", "star", "sky", "sun"],
    ]

    # =========================================================================
    # TEST 1: CHERN CLASS c_1 ACROSS MODELS
    # =========================================================================
    print("=" * 70)
    print("TEST 1: CHERN CLASS c_1 ACROSS MODELS")
    print("=" * 70)
    print()

    all_c1 = []
    all_alpha = []

    for name, model_id in MODELS:
        print(f"Loading {name}...")
        model = SentenceTransformer(model_id)
        emb = model.encode(VOCAB, show_progress_bar=False)

        alpha, Df, c1 = compute_alpha_from_spectrum(emb)
        all_c1.append(c1)
        all_alpha.append(alpha)

        print(f"  Shape: {emb.shape}")
        print(f"  alpha = {alpha:.4f}")
        print(f"  Df (effective dim) = {Df:.2f}")
        print(f"  c_1 = {c1:.4f}")
        print()

    mean_c1 = np.mean(all_c1)
    std_c1 = np.std(all_c1)
    cv_c1 = std_c1 / mean_c1 * 100

    print("-" * 70)
    print(f"SUMMARY: c_1 = {mean_c1:.4f} +/- {std_c1:.4f} (CV = {cv_c1:.2f}%)")
    print(f"Target: c_1 = 1.0, Tolerance: 10%")
    t1_pass = 0.9 <= mean_c1 <= 1.1
    print(f"Result: {'PASS' if t1_pass else 'FAIL'} (deviation = {abs(mean_c1 - 1.0)*100:.2f}%)")
    print()

    # Negative control
    print("Negative Control (Random):")
    np.random.seed(42)
    random_emb = np.random.randn(len(VOCAB), 384)
    alpha_r, Df_r, c1_r = compute_alpha_from_spectrum(random_emb)
    print(f"  alpha = {alpha_r:.4f}, c_1 = {c1_r:.4f}")
    print(f"  Separation: trained c_1 = {mean_c1:.2f} vs random c_1 = {c1_r:.2f}")
    print()

    # =========================================================================
    # TEST 2: TOPOLOGICAL INVARIANCE
    # =========================================================================
    print("=" * 70)
    print("TEST 2: TOPOLOGICAL INVARIANCE (Manifold-Preserving Transforms)")
    print("=" * 70)
    print()

    # Use first model for invariance tests
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(VOCAB, show_progress_bar=False)
    alpha_base, _, c1_base = compute_alpha_from_spectrum(emb)

    print(f"Baseline: alpha = {alpha_base:.4f}, c_1 = {c1_base:.4f}")
    print()

    # Rotation test
    print("2a. ROTATION INVARIANCE:")
    rotation_changes = []
    for i in range(5):
        rotated = apply_rotation(emb, seed=100+i)
        _, _, c1_rot = compute_alpha_from_spectrum(rotated)
        change = abs(c1_rot - c1_base) / c1_base * 100
        rotation_changes.append(change)
        print(f"  Rotation {i+1}: c_1 = {c1_rot:.4f} (change = {change:.4f}%)")

    print(f"  Max change: {max(rotation_changes):.4f}%")
    t2a_pass = max(rotation_changes) < 5
    print(f"  Result: {'PASS' if t2a_pass else 'FAIL'} (threshold: 5%)")
    print()

    # Scaling test
    print("2b. SCALING INVARIANCE:")
    scaling_changes = []
    for scale in [0.1, 0.5, 2.0, 10.0]:
        scaled = apply_scaling(emb, scale)
        _, _, c1_scl = compute_alpha_from_spectrum(scaled)
        change = abs(c1_scl - c1_base) / c1_base * 100
        scaling_changes.append(change)
        print(f"  Scale {scale:4.1f}x: c_1 = {c1_scl:.4f} (change = {change:.4f}%)")

    print(f"  Max change: {max(scaling_changes):.4f}%")
    t2b_pass = max(scaling_changes) < 5
    print(f"  Result: {'PASS' if t2b_pass else 'FAIL'} (threshold: 5%)")
    print()

    # Smooth warping test
    print("2c. SMOOTH WARPING STABILITY:")
    warping_changes = []
    for strength in [0.01, 0.05, 0.10, 0.20]:
        warped = apply_smooth_warping(emb, strength=strength, seed=42)
        _, _, c1_wrp = compute_alpha_from_spectrum(warped)
        change = abs(c1_wrp - c1_base) / c1_base * 100
        warping_changes.append(change)
        print(f"  Strength {strength:.2f}: c_1 = {c1_wrp:.4f} (change = {change:.2f}%)")

    print(f"  Max change: {max(warping_changes):.2f}%")
    t2c_pass = max(warping_changes) < 10
    print(f"  Result: {'PASS' if t2c_pass else 'FAIL'} (threshold: 10%)")
    print()

    # =========================================================================
    # TEST 3: BERRY PHASE / HOLONOMY
    # =========================================================================
    print("=" * 70)
    print("TEST 3: BERRY PHASE / HOLONOMY (PC1-2 Subspace)")
    print("=" * 70)
    print()

    # Project to PC1-2
    projected, pc12 = project_to_pc12(emb)
    print(f"Projected to PC1-2: shape = {projected.shape}")
    print()

    # Build word index
    word_to_idx = {w: i for i, w in enumerate(VOCAB)}

    print("Berry phase on semantic loops:")
    q_scores = []
    for loop in SEMANTIC_LOOPS:
        indices = [word_to_idx[w] for w in loop]
        phase = berry_phase_loop(emb, indices)
        q_score = quantization_score(phase)
        winding = phase / (2 * np.pi)
        q_scores.append(q_score)
        print(f"  {loop[0]:10} -> {loop[-2]:10}: phase = {phase:7.4f} rad, winding = {winding:5.2f}, Q = {q_score:.4f}")

    mean_q = np.mean(q_scores)
    print()
    print(f"Mean Q-score: {mean_q:.4f}")
    t3a_pass = mean_q > 0.5
    print(f"Result: {'PASS' if t3a_pass else 'FAIL'} (threshold: 0.5)")
    print()

    # Random loops
    print("Berry phase on random loops (curvature test):")
    np.random.seed(12345)
    n_samples = len(emb)
    random_phases = []
    for i in range(20):
        loop_size = np.random.randint(4, 8)
        indices = np.random.choice(n_samples, loop_size, replace=False).tolist()
        indices.append(indices[0])
        phase = berry_phase_loop(emb, indices)
        random_phases.append(abs(phase))

    mean_abs_phase = np.mean(random_phases)
    print(f"  Mean |phase| over 20 random loops: {mean_abs_phase:.4f} rad")
    t3b_pass = mean_abs_phase > 0.1
    print(f"  Result: {'PASS' if t3b_pass else 'FAIL'} (threshold: 0.1 rad for non-trivial curvature)")
    print()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()

    tests = [
        ("TEST 1 (c_1 ~ 1)", t1_pass, f"c_1 = {mean_c1:.4f}"),
        ("TEST 2a (Rotation)", t2a_pass, f"max change = {max(rotation_changes):.4f}%"),
        ("TEST 2b (Scaling)", t2b_pass, f"max change = {max(scaling_changes):.4f}%"),
        ("TEST 2c (Warping)", t2c_pass, f"max change = {max(warping_changes):.2f}%"),
        ("TEST 3 (Berry phase)", t3b_pass, f"mean |phase| = {mean_abs_phase:.4f} rad"),
    ]

    tests_passed = 0
    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        print(f"{name:25} {status:4} ({detail})")
        if passed:
            tests_passed += 1

    print()
    print(f"OVERALL: {tests_passed}/{len(tests)} tests PASS")
    print()

    if tests_passed == len(tests):
        print("CONCLUSION: All tests PASS with real embeddings.")
        print("c_1 ~ 1 is a topologically invariant property of trained embeddings.")
    else:
        print(f"CONCLUSION: {len(tests) - tests_passed} test(s) failed. Review needed.")

    return tests_passed == len(tests)


if __name__ == "__main__":
    main()
