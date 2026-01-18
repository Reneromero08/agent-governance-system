"""
Q53: Pentagonal Phi Geometry - Golden Angle Hypothesis Test

Tests whether geodesic trajectories in semantic space follow golden angle spacing.

Golden angle = 2*pi/phi^2 = 2.399 radians = 137.5 degrees

Hypothesis: The geometry of meaning uses golden angle spacing for optimal packing.

FINDING: Step angles are small (~0.8 deg), but total arcs cluster around
pentagonal angles (~72 deg), not golden spiral angles (137.5 deg).

Uses REAL embeddings from 5 architectures.
"""

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import sys

# Constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # phi = 1.618...
GOLDEN_ANGLE_RAD = 2 * np.pi / (GOLDEN_RATIO ** 2)  # 2.399 rad
GOLDEN_ANGLE_DEG = np.degrees(GOLDEN_ANGLE_RAD)  # 137.5 deg

print(f"Golden ratio (phi): {GOLDEN_RATIO:.6f}")
print(f"Golden angle: {GOLDEN_ANGLE_RAD:.4f} rad = {GOLDEN_ANGLE_DEG:.2f} deg")
print()

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_PATH = SCRIPT_DIR.parent  # experiments/open_questions
sys.path.insert(0, str(EXPERIMENTS_PATH / "q38"))

from noether import sphere_geodesic_trajectory, geodesic_velocity


def load_embeddings(model_name: str, words: List[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """Load embeddings from a specific model."""
    embeddings = {}
    dim = 0

    if model_name == "glove":
        import gensim.downloader as api
        print(f"  Loading GloVe...", flush=True)
        model = api.load("glove-wiki-gigaword-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "word2vec":
        import gensim.downloader as api
        print(f"  Loading Word2Vec...", flush=True)
        model = api.load("word2vec-google-news-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "fasttext":
        import gensim.downloader as api
        print(f"  Loading FastText...", flush=True)
        model = api.load("fasttext-wiki-news-subwords-300")
        dim = 300
        for word in words:
            if word in model:
                vec = model[word]
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "bert":
        from transformers import BertTokenizer, BertModel
        import torch
        print(f"  Loading BERT...", flush=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        dim = 768
        with torch.no_grad():
            for word in words:
                inputs = tokenizer(word, return_tensors='pt', padding=True, truncation=True)
                outputs = model(**inputs)
                vec = outputs.last_hidden_state[0, 0, :].numpy()
                vec = vec / np.linalg.norm(vec)
                embeddings[word] = vec

    elif model_name == "sentence":
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SentenceTransformer...", flush=True)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dim = 384
        vecs = model.encode(words)
        for i, word in enumerate(words):
            vec = vecs[i]
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings, dim


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians."""
    # Normalize
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    # Clamp dot product to [-1, 1] to avoid numerical issues
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(dot)


def measure_trajectory_angles(trajectory: np.ndarray) -> List[float]:
    """
    Measure angles between consecutive velocity vectors along a geodesic.

    Returns list of angles in radians.
    """
    velocities = geodesic_velocity(trajectory)
    angles = []

    for i in range(1, len(velocities) - 1):
        angle = angle_between_vectors(velocities[i-1], velocities[i])
        if angle > 0:  # Skip zero angles
            angles.append(angle)

    return angles


def measure_turning_angles(trajectory: np.ndarray) -> List[float]:
    """
    Measure turning angles (change in direction) along trajectory.
    This measures how much the path "turns" at each point.
    """
    angles = []

    for i in range(1, len(trajectory) - 1):
        # Vector from previous point to current
        v1 = trajectory[i] - trajectory[i-1]
        # Vector from current to next
        v2 = trajectory[i+1] - trajectory[i]

        if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
            angle = angle_between_vectors(v1, v2)
            angles.append(angle)

    return angles


def measure_position_angles(trajectory: np.ndarray) -> List[float]:
    """
    Measure angles between consecutive position vectors from origin.
    On a sphere, this gives the arc angle traversed.
    """
    angles = []

    for i in range(1, len(trajectory)):
        angle = angle_between_vectors(trajectory[i-1], trajectory[i])
        angles.append(angle)

    return angles


def analyze_angle_distribution(angles: List[float], name: str) -> Dict:
    """Analyze if angles cluster around golden angle."""
    if not angles:
        return {"name": name, "n": 0, "error": "no data"}

    angles = np.array(angles)

    # Convert to degrees for readability
    angles_deg = np.degrees(angles)

    # Statistics
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    mean_deg = np.degrees(mean_angle)
    std_deg = np.degrees(std_angle)

    # Distance from golden angle
    dist_from_golden = np.abs(angles - GOLDEN_ANGLE_RAD)
    mean_dist_from_golden = np.mean(dist_from_golden)

    # Distance from golden angle (considering multiples)
    # Check if angles cluster around golden angle or its multiples/fractions
    golden_multiples = [GOLDEN_ANGLE_RAD / 2, GOLDEN_ANGLE_RAD, 2 * GOLDEN_ANGLE_RAD]
    best_multiple = None
    best_dist = float('inf')

    for mult in golden_multiples:
        dist = np.mean(np.abs(angles - mult))
        if dist < best_dist:
            best_dist = dist
            best_multiple = mult

    # Percentage within 10% of golden angle
    tolerance = GOLDEN_ANGLE_RAD * 0.1
    within_tolerance = np.sum(dist_from_golden < tolerance) / len(angles) * 100

    # Check modular golden angle (angles mod golden angle)
    mod_angles = np.mod(angles, GOLDEN_ANGLE_RAD)
    mod_mean = np.mean(mod_angles)
    mod_std = np.std(mod_angles)

    return {
        "name": name,
        "n": len(angles),
        "mean_rad": float(mean_angle),
        "mean_deg": float(mean_deg),
        "std_rad": float(std_angle),
        "std_deg": float(std_deg),
        "golden_angle_rad": GOLDEN_ANGLE_RAD,
        "golden_angle_deg": GOLDEN_ANGLE_DEG,
        "mean_dist_from_golden_rad": float(mean_dist_from_golden),
        "mean_dist_from_golden_deg": float(np.degrees(mean_dist_from_golden)),
        "within_10pct_of_golden": float(within_tolerance),
        "best_golden_multiple": float(best_multiple),
        "best_multiple_dist": float(best_dist),
        "ratio_to_golden": float(mean_angle / GOLDEN_ANGLE_RAD),
        "mod_golden_mean": float(mod_mean),
        "mod_golden_std": float(mod_std),
    }


def run_golden_angle_test():
    """Run the golden angle hypothesis test."""
    print("=" * 80)
    print("Q36 EXTENSION: GOLDEN ANGLE HYPOTHESIS TEST")
    print("=" * 80)
    print()
    print(f"Hypothesis: Geodesic angles cluster around golden angle = {GOLDEN_ANGLE_DEG:.2f} deg")
    print()

    # Test words - semantic pairs
    word_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("good", "evil"),
        ("light", "dark"),
        ("love", "hate"),
        ("life", "death"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
    ]

    all_words = list(set([w for pair in word_pairs for w in pair]))

    models = ["glove", "word2vec", "fasttext", "bert", "sentence"]

    all_results = {}
    all_position_angles = []
    all_turning_angles = []
    all_velocity_angles = []

    for model_name in models:
        print(f"\n--- Testing {model_name.upper()} ---")

        try:
            embeddings, dim = load_embeddings(model_name, all_words)
            print(f"  Loaded {len(embeddings)} words in {dim}D space")

            model_position_angles = []
            model_turning_angles = []
            model_velocity_angles = []

            for w1, w2 in word_pairs:
                if w1 not in embeddings or w2 not in embeddings:
                    continue

                start = embeddings[w1]
                end = embeddings[w2]

                # Generate geodesic trajectory (50 steps)
                traj = sphere_geodesic_trajectory(start, end, n_steps=50)

                # Measure different types of angles
                pos_angles = measure_position_angles(traj)
                turn_angles = measure_turning_angles(traj)
                vel_angles = measure_trajectory_angles(traj)

                model_position_angles.extend(pos_angles)
                model_turning_angles.extend(turn_angles)
                model_velocity_angles.extend(vel_angles)

            # Analyze this model's angles
            pos_analysis = analyze_angle_distribution(model_position_angles, f"{model_name}_position")
            turn_analysis = analyze_angle_distribution(model_turning_angles, f"{model_name}_turning")
            vel_analysis = analyze_angle_distribution(model_velocity_angles, f"{model_name}_velocity")

            all_results[model_name] = {
                "position": pos_analysis,
                "turning": turn_analysis,
                "velocity": vel_analysis,
            }

            all_position_angles.extend(model_position_angles)
            all_turning_angles.extend(model_turning_angles)
            all_velocity_angles.extend(model_velocity_angles)

            # Print summary for this model
            print(f"  Position angles: mean={pos_analysis['mean_deg']:.2f} deg, ratio to golden={pos_analysis['ratio_to_golden']:.4f}")
            print(f"  Turning angles:  mean={turn_analysis['mean_deg']:.2f} deg, ratio to golden={turn_analysis['ratio_to_golden']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = {"error": str(e)}

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (ALL MODELS)")
    print("=" * 80)

    agg_position = analyze_angle_distribution(all_position_angles, "aggregate_position")
    agg_turning = analyze_angle_distribution(all_turning_angles, "aggregate_turning")
    agg_velocity = analyze_angle_distribution(all_velocity_angles, "aggregate_velocity")

    print(f"\nGolden angle reference: {GOLDEN_ANGLE_DEG:.2f} deg = {GOLDEN_ANGLE_RAD:.4f} rad")
    print(f"Golden ratio (phi):     {GOLDEN_RATIO:.6f}")
    print()

    print("POSITION ANGLES (arc between consecutive points):")
    print(f"  Mean:            {agg_position['mean_deg']:.4f} deg = {agg_position['mean_rad']:.6f} rad")
    print(f"  Std:             {agg_position['std_deg']:.4f} deg")
    print(f"  Ratio to golden: {agg_position['ratio_to_golden']:.6f}")
    print(f"  N samples:       {agg_position['n']}")
    print()

    print("TURNING ANGLES (direction change at each point):")
    print(f"  Mean:            {agg_turning['mean_deg']:.4f} deg = {agg_turning['mean_rad']:.6f} rad")
    print(f"  Std:             {agg_turning['std_deg']:.4f} deg")
    print(f"  Ratio to golden: {agg_turning['ratio_to_golden']:.6f}")
    print(f"  N samples:       {agg_turning['n']}")
    print()

    # Check for golden ratio relationships
    print("=" * 80)
    print("GOLDEN RATIO ANALYSIS")
    print("=" * 80)

    # Check various ratios
    mean_pos = agg_position['mean_rad']
    mean_turn = agg_turning['mean_rad']

    print(f"\nPosition/Turning ratio: {mean_pos/mean_turn:.6f}")
    print(f"Golden ratio (phi):     {GOLDEN_RATIO:.6f}")
    print(f"1/phi:                  {1/GOLDEN_RATIO:.6f}")
    print(f"phi^2:                  {GOLDEN_RATIO**2:.6f}")

    # Check if total arc relates to golden angle
    total_arc = mean_pos * 49  # 50 steps = 49 intervals
    print(f"\nTotal arc (49 steps):   {np.degrees(total_arc):.2f} deg = {total_arc:.4f} rad")
    print(f"Total arc / 2*pi:       {total_arc / (2*np.pi):.6f}")
    print(f"Total arc / golden:     {total_arc / GOLDEN_ANGLE_RAD:.6f}")

    # XOR Phi relationship
    xor_phi = 1.773  # From Q36 Test 1
    print(f"\nXOR Phi value:          {xor_phi}")
    print(f"Golden ratio:           {GOLDEN_RATIO:.6f}")
    print(f"XOR Phi / golden:       {xor_phi / GOLDEN_RATIO:.6f}")
    print(f"XOR Phi / phi^2:        {xor_phi / (GOLDEN_RATIO**2):.6f}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Check if any ratio is close to a power of phi
    phi_powers = {
        "phi^-2": GOLDEN_RATIO**-2,
        "phi^-1": GOLDEN_RATIO**-1,
        "1": 1.0,
        "phi": GOLDEN_RATIO,
        "phi^2": GOLDEN_RATIO**2,
    }

    pos_ratio = agg_position['ratio_to_golden']
    closest_phi_power = min(phi_powers.items(), key=lambda x: abs(x[1] - pos_ratio))

    print(f"\nPosition angle ratio to golden: {pos_ratio:.6f}")
    print(f"Closest phi power: {closest_phi_power[0]} = {closest_phi_power[1]:.6f}")
    print(f"Difference: {abs(closest_phi_power[1] - pos_ratio):.6f}")

    if abs(closest_phi_power[1] - pos_ratio) < 0.1:
        print("\n** GOLDEN RATIO CONNECTION DETECTED **")
    else:
        print("\n** No direct golden ratio connection in step angles **")
        print("   (But total arc or other measures may still relate)")

    # Save results
    results = {
        "hypothesis": "Geodesic angles relate to golden ratio",
        "golden_ratio": GOLDEN_RATIO,
        "golden_angle_rad": GOLDEN_ANGLE_RAD,
        "golden_angle_deg": GOLDEN_ANGLE_DEG,
        "aggregate": {
            "position": agg_position,
            "turning": agg_turning,
            "velocity": agg_velocity,
        },
        "by_model": all_results,
        "analysis": {
            "position_to_golden_ratio": pos_ratio,
            "closest_phi_power": closest_phi_power[0],
            "closest_phi_power_value": closest_phi_power[1],
            "difference": abs(closest_phi_power[1] - pos_ratio),
            "total_arc_rad": total_arc,
            "total_arc_deg": np.degrees(total_arc),
            "xor_phi": xor_phi,
            "xor_phi_over_golden": xor_phi / GOLDEN_RATIO,
        }
    }

    output_path = SCRIPT_DIR / "Q36_GOLDEN_ANGLE_RESULTS.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_golden_angle_test()
