"""
WINDOWED BRAIN-STIMULUS TEST

KEY INSIGHT: The equality is NOT Df(brain) = Df(stimulus)
It IS: Df(brain | stimulus, window, task) ~ Df(stimulus | window)

This means:
- Brain Df is CONDITIONAL on stimulus, time window, and task
- We need to measure Df WITHIN specific time windows
- The matching is per-stimulus, per-window, not global

Approach:
1. Split EEG into time windows:
   - Early (0-100ms): feedforward visual processing
   - Mid (100-300ms): object recognition
   - Late (300-500ms): semantic processing
2. Compute brain Df per window per concept
3. Look for window-specific correlations with stimulus features
"""

import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cdist
import json

OUTPUT_DIR = Path("d:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/TINY_COMPRESS/things_eeg_data")


def participation_ratio(data: np.ndarray) -> float:
    """Df = (sum lambda)^2 / sum(lambda^2)"""
    if data.ndim == 1:
        return 1.0

    centered = data - data.mean(axis=0)
    n_samples, n_features = centered.shape

    if n_samples < n_features:
        gram = centered @ centered.T
        eigenvalues = np.linalg.eigvalsh(gram)
    else:
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            return 1.0
        eigenvalues = np.linalg.eigvalsh(cov)

    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 1.0

    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


def load_data():
    """Load EEG and ViT features."""
    eeg_path = OUTPUT_DIR / "Preprocessed_data_250Hz_whiten" / "sub-01" / "test.pt"
    vit_path = OUTPUT_DIR / "Preprocessed_data_250Hz_whiten" / "ViT-B-32_features_test.pt"

    eeg_data = torch.load(str(eeg_path), map_location='cpu', weights_only=False)
    vit_data = torch.load(str(vit_path), map_location='cpu', weights_only=False)

    return eeg_data, vit_data


def compute_windowed_brain_df(eeg, texts, fs=250):
    """
    Compute brain Df per concept per time window.

    Windows (at 250 Hz):
    - baseline: -200ms to 0ms (samples 0-50)
    - early: 0-100ms (samples 50-75)
    - mid: 100-300ms (samples 75-125)
    - late: 300-500ms (samples 125-175)
    - sustained: 500-800ms (samples 175-250)

    Actually, the data is 250 samples total for 1 second = 250 Hz
    Starting from stimulus onset means:
    - 0-100ms: samples 0-25
    - 100-200ms: samples 25-50
    - etc.
    """
    n_concepts, n_reps, n_channels, n_timepoints = eeg.shape
    print(f"EEG shape: {eeg.shape}")
    print(f"Timepoints: {n_timepoints} at {fs}Hz = {n_timepoints/fs*1000:.0f}ms")

    # Define windows (in samples)
    windows = {
        'early_50-100ms': (12, 25),    # Early visual
        'P1_100-150ms': (25, 38),      # P1 component
        'N1_150-200ms': (38, 50),      # N1 component
        'P2_200-300ms': (50, 75),      # P2 / categorization
        'late_300-500ms': (75, 125),   # Late semantic
        'sustained_500-800ms': (125, 200)  # Sustained
    }

    results = {}

    for window_name, (start, end) in windows.items():
        window_dfs = {}

        for i in range(n_concepts):
            concept_name = texts[i, 0]
            concept_eeg = eeg[i]  # (reps, channels, timepoints)

            # Extract window
            window_eeg = concept_eeg[:, :, start:end]  # (reps, channels, window_size)

            # Reshape: (reps * window_size, channels)
            window_samples = window_eeg.transpose(0, 2, 1).reshape(-1, n_channels)

            # Compute Df
            df = participation_ratio(window_samples.astype(np.float64))
            window_dfs[concept_name] = df

        results[window_name] = window_dfs
        mean_df = np.mean(list(window_dfs.values()))
        print(f"  {window_name}: mean Df = {mean_df:.2f}")

    return results


def compute_semantic_measures(vit_features_dict):
    """Compute semantic complexity measures from embeddings."""
    concept_names = list(vit_features_dict.keys())
    embeddings = np.array([vit_features_dict[name].numpy() for name in concept_names])

    # Magnitude
    magnitudes = np.linalg.norm(embeddings, axis=1)

    # Centrality (distance from mean)
    mean_emb = embeddings.mean(axis=0)
    centrality = np.linalg.norm(embeddings - mean_emb, axis=1)

    # Distinctiveness (distance to nearest neighbor)
    dists = cdist(embeddings, embeddings, metric='cosine')
    np.fill_diagonal(dists, np.inf)
    distinctiveness = dists.min(axis=1)

    # Semantic category spread
    # High spread = concept near multiple categories (ambiguous)
    category_proximity = dists.mean(axis=1)

    return {
        name: {
            'magnitude': float(magnitudes[i]),
            'centrality': float(centrality[i]),
            'distinctiveness': float(distinctiveness[i]),
            'category_proximity': float(category_proximity[i])
        }
        for i, name in enumerate(concept_names)
    }


def main():
    print("=" * 70)
    print("WINDOWED BRAIN-STIMULUS TEST")
    print("Df(brain | stimulus, window) ~ Df(stimulus | window)")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    eeg_data, vit_data = load_data()

    eeg = eeg_data['eeg']
    texts = eeg_data['text']
    img_features = vit_data['img_features']

    print(f"EEG: {eeg.shape}")
    print(f"Image features: {len(img_features)} concepts")

    # Compute windowed brain Df
    print("\nComputing brain Df per window...")
    windowed_dfs = compute_windowed_brain_df(eeg, texts)

    # Compute semantic measures
    print("\nComputing semantic measures...")
    img_semantic = compute_semantic_measures(img_features)

    # Match and correlate per window
    print("\n" + "=" * 70)
    print("CORRELATIONS BY WINDOW")
    print("=" * 70)

    common = set(next(iter(windowed_dfs.values())).keys()) & set(img_semantic.keys())
    print(f"Common concepts: {len(common)}")
    concepts = sorted(common)

    # Build semantic arrays
    magnitudes = np.array([img_semantic[c]['magnitude'] for c in concepts])
    centrality = np.array([img_semantic[c]['centrality'] for c in concepts])
    distinctiveness = np.array([img_semantic[c]['distinctiveness'] for c in concepts])

    results_table = []

    for window_name, window_dfs in windowed_dfs.items():
        brain_arr = np.array([window_dfs[c] for c in concepts])

        corr_mag = np.corrcoef(brain_arr, magnitudes)[0, 1]
        corr_cent = np.corrcoef(brain_arr, centrality)[0, 1]
        corr_dist = np.corrcoef(brain_arr, distinctiveness)[0, 1]

        results_table.append({
            'window': window_name,
            'corr_magnitude': corr_mag,
            'corr_centrality': corr_cent,
            'corr_distinctiveness': corr_dist,
            'brain_df_mean': brain_arr.mean(),
            'brain_df_std': brain_arr.std()
        })

        print(f"\n{window_name}:")
        print(f"  Brain Df mean: {brain_arr.mean():.2f} (+/- {brain_arr.std():.2f})")
        print(f"  vs Magnitude:      r = {corr_mag:+.4f}")
        print(f"  vs Centrality:     r = {corr_cent:+.4f}")
        print(f"  vs Distinctiveness: r = {corr_dist:+.4f}")

    # Find best window-feature combination
    print("\n" + "=" * 70)
    print("BEST CORRELATIONS")
    print("=" * 70)

    all_corrs = []
    for r in results_table:
        all_corrs.append((r['window'], 'magnitude', r['corr_magnitude']))
        all_corrs.append((r['window'], 'centrality', r['corr_centrality']))
        all_corrs.append((r['window'], 'distinctiveness', r['corr_distinctiveness']))

    all_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nTop 5 window-feature correlations:")
    for window, feature, corr in all_corrs[:5]:
        print(f"  {window:25s} vs {feature:15s}: r = {corr:+.4f}")

    # KEY ANALYSIS
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    best_window, best_feature, best_corr = all_corrs[0]

    if abs(best_corr) > 0.3:
        print(f"\nMEANINGFUL CORRELATION FOUND!")
        print(f"  Window: {best_window}")
        print(f"  Feature: {best_feature}")
        print(f"  r = {best_corr:+.4f}")
        print(f"\nThis suggests that brain Df in the {best_window} window")
        print(f"correlates with semantic {best_feature}.")
        print(f"\nInterpretation: The brain's processing complexity during")
        print(f"this time window reflects the stimulus's semantic structure.")
    elif abs(best_corr) > 0.15:
        print(f"\nWEAK BUT PRESENT CORRELATION: r = {best_corr:+.4f}")
        print(f"  Window: {best_window}")
        print(f"  Feature: {best_feature}")
        print("The relationship exists but is subtle at this measurement level.")
    else:
        print(f"\nNO SIGNIFICANT CORRELATION: max |r| = {abs(best_corr):.4f}")
        print("""
This doesn't disprove the formula. Possible reasons:

1. EEG (temporal) may not capture spatial Df structure
   - Try fMRI which has spatial resolution

2. Single-trial variability dominates
   - 80 repetitions may not be enough to stabilize Df

3. The relationship is nonlinear
   - Try mutual information instead of correlation

4. Df(brain) reflects processing EFFICIENCY, not complexity
   - Simple, familiar objects -> low Df (efficient)
   - Novel, complex objects -> high Df (exploration)
   - This is an INVERSE relationship

5. Attention modulates Df independently of stimulus
   - Need controlled attention task
""")

    # LOOK FOR PATTERNS IN THE DATA
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Compare early vs late windows
    early_dfs = np.array([windowed_dfs['early_50-100ms'][c] for c in concepts])
    late_dfs = np.array([windowed_dfs['late_300-500ms'][c] for c in concepts])

    corr_early_late = np.corrcoef(early_dfs, late_dfs)[0, 1]
    print(f"\nEarly vs Late window correlation: r = {corr_early_late:.4f}")

    if corr_early_late > 0.5:
        print("-> High correlation: Brain Df is stable across processing stages")
        print("   Stimulus-driven complexity is preserved")
    else:
        print("-> Low correlation: Brain Df changes across processing stages")
        print("   Suggests task-dependent modulation")

    # Variance across windows (processing dynamics)
    window_variance = np.array([
        [windowed_dfs[w][c] for w in windowed_dfs.keys()]
        for c in concepts
    ]).std(axis=1)

    corr_var_cent = np.corrcoef(window_variance, centrality)[0, 1]
    print(f"\nDf variance across windows vs Semantic centrality: r = {corr_var_cent:.4f}")

    if abs(corr_var_cent) > 0.2:
        print("-> Unusual concepts have more variable processing across time")

    # Save results
    results = {
        'window_correlations': results_table,
        'best_correlation': {
            'window': best_window,
            'feature': best_feature,
            'r': float(best_corr)
        },
        'early_late_correlation': float(corr_early_late),
        'n_concepts': len(concepts)
    }

    with open(OUTPUT_DIR / "windowed_brain_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'windowed_brain_test_results.json'}")


if __name__ == "__main__":
    main()
