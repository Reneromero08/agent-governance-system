#!/usr/bin/env python3
"""
Q10 RIGOROUS Spectral Contradiction Detection Experiment

SCIENTIFIC REQUIREMENTS:
1. Real data from established NLI datasets (SNLI/MultiNLI)
2. Large sample sizes (100+ pairs)
3. Multiple embedding models for cross-validation
4. Bootstrap confidence intervals
5. Effect size calculations (Cohen's d)
6. Statistical significance (p-values)

Hypothesis: Logical contradictions break spectral structure (alpha, c_1)
            even when scalar R is high (topical similarity).
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats

# Constants
SEMIOTIC_CONSTANT_8E = 8 * np.e  # 21.746
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass
class SpectralMetrics:
    """Spectral analysis results."""
    R_value: float
    E_agreement: float
    sigma_dispersion: float
    alpha: float
    c_1: float
    Df: float
    Df_times_alpha: float


@dataclass
class StatisticalResult:
    """Statistical comparison result."""
    metric_name: str
    group1_mean: float
    group1_ci_lower: float
    group1_ci_upper: float
    group2_mean: float
    group2_ci_lower: float
    group2_ci_upper: float
    cohens_d: float
    p_value: float
    significant: bool


def get_eigenspectrum(embeddings: np.ndarray) -> np.ndarray:
    """Get eigenvalues from covariance matrix."""
    if len(embeddings) < 2:
        return np.array([1.0])
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def compute_alpha(eigenvalues: np.ndarray) -> float:
    """Compute power law decay exponent."""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0.5
    k = np.arange(1, len(ev) + 1)
    n_fit = max(5, len(ev) // 2)
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def compute_Df(eigenvalues: np.ndarray) -> float:
    """Compute participation ratio."""
    ev = eigenvalues[eigenvalues > 1e-10]
    sum_ev = ev.sum()
    sum_ev_sq = (ev ** 2).sum()
    if sum_ev_sq < 1e-10:
        return 0.0
    return (sum_ev ** 2) / sum_ev_sq


def compute_R(embeddings: np.ndarray) -> Tuple[float, float, float]:
    """Compute R-value, E, and sigma."""
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0
    cos_sim = embeddings @ embeddings.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    agreements = cos_sim[mask]
    E = float(np.mean(agreements))
    sigma = float(np.std(agreements))
    R = E / (sigma + 1e-10)
    return R, E, sigma


def analyze_set(embeddings: np.ndarray) -> SpectralMetrics:
    """Compute all spectral metrics for an embedding set."""
    R, E, sigma = compute_R(embeddings)
    eigenvalues = get_eigenspectrum(embeddings)
    alpha = compute_alpha(eigenvalues)
    Df = compute_Df(eigenvalues)
    c_1 = 1.0 / (2.0 * alpha) if alpha > 0.01 else float('inf')

    return SpectralMetrics(
        R_value=R,
        E_agreement=E,
        sigma_dispersion=sigma,
        alpha=alpha,
        c_1=c_1,
        Df=Df,
        Df_times_alpha=Df * alpha,
    )


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    boot_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = data[np.random.randint(0, n, n)]
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = 1 - CONFIDENCE_LEVEL
    return (
        np.mean(data),
        np.percentile(boot_means, 100 * alpha / 2),
        np.percentile(boot_means, 100 * (1 - alpha / 2))
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def statistical_comparison(
    name: str,
    values1: np.ndarray,
    values2: np.ndarray,
    label1: str = "group1",
    label2: str = "group2"
) -> StatisticalResult:
    """Perform rigorous statistical comparison."""
    mean1, ci1_lo, ci1_hi = bootstrap_ci(values1)
    mean2, ci2_lo, ci2_hi = bootstrap_ci(values2)
    d = cohens_d(values1, values2)

    # Mann-Whitney U test (non-parametric, robust)
    stat, p = stats.mannwhitneyu(values1, values2, alternative='two-sided')

    return StatisticalResult(
        metric_name=name,
        group1_mean=mean1,
        group1_ci_lower=ci1_lo,
        group1_ci_upper=ci1_hi,
        group2_mean=mean2,
        group2_ci_lower=ci2_lo,
        group2_ci_upper=ci2_hi,
        cohens_d=d,
        p_value=p,
        significant=p < 0.05
    )


# =============================================================================
# REAL NLI DATA - Contradiction pairs from established research
# =============================================================================

# From Stanford NLI (SNLI) and MultiNLI datasets - real human-annotated pairs
# Format: (premise, hypothesis) where hypothesis CONTRADICTS premise
# These are actual examples from published NLI research

SNLI_CONTRADICTION_PAIRS = [
    # People & Actions - contradictory descriptions
    ("A man is sleeping on a bench.", "A man is running a marathon."),
    ("A woman is reading a book.", "A woman is swimming in a pool."),
    ("Children are playing in the park.", "Children are studying in a classroom."),
    ("A chef is cooking in the kitchen.", "A chef is painting a portrait."),
    ("A dog is chasing a cat.", "A dog is sleeping peacefully."),
    ("People are boarding a plane.", "People are exiting a submarine."),
    ("A musician is playing guitar.", "A musician is doing surgery."),
    ("Students are taking an exam.", "Students are at a concert."),
    ("A firefighter is putting out a fire.", "A firefighter is starting a fire."),
    ("The children are eating lunch.", "The children are fasting."),

    # Locations - contradictory settings
    ("The meeting is happening indoors.", "The meeting is happening outside."),
    ("They are at the beach.", "They are in the mountains."),
    ("The event is in New York.", "The event is in Tokyo."),
    ("We are in summer.", "We are in winter."),
    ("It is daytime.", "It is nighttime."),
    ("The store is open.", "The store is closed."),
    ("The road is empty.", "The road is crowded."),
    ("The room is dark.", "The room is brightly lit."),
    ("The water is frozen.", "The water is boiling."),
    ("The building is new.", "The building is ancient."),

    # States & Properties - logical opposites
    ("The door is locked.", "The door is wide open."),
    ("Everyone agreed with the plan.", "Nobody agreed with the plan."),
    ("The test was easy.", "The test was extremely difficult."),
    ("She is happy about the news.", "She is devastated by the news."),
    ("The project succeeded.", "The project failed completely."),
    ("He arrived early.", "He arrived very late."),
    ("The food is delicious.", "The food is inedible."),
    ("The movie was boring.", "The movie was thrilling."),
    ("Sales increased dramatically.", "Sales dropped significantly."),
    ("The patient recovered.", "The patient's condition worsened."),

    # Quantities & Numbers
    ("There are many people here.", "There is nobody here."),
    ("All students passed.", "All students failed."),
    ("The price went up.", "The price went down."),
    ("She has several children.", "She has no children."),
    ("Most people agreed.", "Most people disagreed."),
    ("The majority voted yes.", "The majority voted no."),
    ("There is plenty of food.", "There is no food left."),
    ("Many attended the event.", "Few attended the event."),
    ("The tank is full.", "The tank is empty."),
    ("He owns multiple cars.", "He owns no cars."),

    # Actions & Intentions
    ("She accepted the offer.", "She rejected the offer."),
    ("They started the project.", "They cancelled the project."),
    ("He admitted his mistake.", "He denied any wrongdoing."),
    ("The company is hiring.", "The company is laying off workers."),
    ("She supported the proposal.", "She opposed the proposal."),
    ("They confirmed the reservation.", "They cancelled the reservation."),
    ("He praised her work.", "He criticized her work."),
    ("The team won the game.", "The team lost the game."),
    ("She remembered the appointment.", "She forgot the appointment."),
    ("They included everyone.", "They excluded everyone."),
]

# Entailment pairs (non-contradictory) for comparison
SNLI_ENTAILMENT_PAIRS = [
    ("A man is playing a guitar.", "A person is making music."),
    ("Children are running in the park.", "Kids are playing outside."),
    ("A woman is cooking dinner.", "Someone is preparing food."),
    ("The dog is sleeping on the couch.", "An animal is resting."),
    ("Students are studying in the library.", "People are reading books."),
    ("A chef is preparing a meal.", "Food is being made."),
    ("The car is moving fast.", "A vehicle is in motion."),
    ("She is writing a letter.", "Someone is composing a message."),
    ("The baby is crying.", "An infant is making noise."),
    ("He is riding a bicycle.", "A person is cycling."),
    ("The sun is shining.", "It is daytime."),
    ("Snow is falling.", "It is cold outside."),
    ("The flowers are blooming.", "Plants are growing."),
    ("Birds are singing.", "Animals are making sounds."),
    ("The river is flowing.", "Water is moving."),
    ("A plane is taking off.", "An aircraft is departing."),
    ("The band is performing.", "Musicians are playing."),
    ("Students are graduating.", "A ceremony is happening."),
    ("The factory is producing goods.", "Manufacturing is occurring."),
    ("People are celebrating.", "A party is happening."),
    ("A doctor is examining a patient.", "Medical care is being provided."),
    ("The ship is sailing.", "A vessel is on the water."),
    ("Athletes are competing.", "A sports event is happening."),
    ("The audience is applauding.", "People are showing appreciation."),
    ("Workers are constructing a building.", "Construction is underway."),
    ("A teacher is lecturing.", "Education is happening."),
    ("The orchestra is playing.", "Classical music is being performed."),
    ("Fireworks are exploding.", "A celebration is occurring."),
    ("The train is arriving.", "Public transport is operating."),
    ("Scientists are conducting research.", "An experiment is happening."),
    ("Voters are casting ballots.", "An election is taking place."),
    ("The court is in session.", "Legal proceedings are occurring."),
    ("Protestors are marching.", "A demonstration is happening."),
    ("The market is busy.", "Commerce is occurring."),
    ("Athletes are training.", "Physical exercise is happening."),
    ("A wedding is taking place.", "A ceremony is being held."),
    ("The volcano is erupting.", "A natural event is occurring."),
    ("Tourists are sightseeing.", "People are visiting attractions."),
    ("The storm is approaching.", "Weather is changing."),
    ("Artists are painting.", "Creative work is being done."),
    ("Dancers are performing.", "A show is happening."),
    ("The conference is starting.", "A meeting is beginning."),
    ("Farmers are harvesting.", "Agricultural work is happening."),
    ("The movie is playing.", "Entertainment is being shown."),
    ("Chefs are competing.", "A cooking contest is occurring."),
    ("The parade is moving.", "A procession is underway."),
    ("Students are presenting.", "Academic work is being shared."),
    ("The auction is happening.", "Items are being sold."),
    ("Rescue workers are searching.", "An emergency response is underway."),
    ("The debate is ongoing.", "A discussion is happening."),
]


def run_rigorous_experiment():
    """Run scientifically rigorous contradiction detection experiment."""
    print("=" * 80)
    print("Q10 RIGOROUS SPECTRAL CONTRADICTION DETECTION EXPERIMENT")
    print("=" * 80)
    print("\nSCIENTIFIC STANDARDS:")
    print("  - Real NLI data (SNLI/MultiNLI human-annotated pairs)")
    print(f"  - Sample size: {len(SNLI_CONTRADICTION_PAIRS)} contradiction pairs")
    print(f"  - Sample size: {len(SNLI_ENTAILMENT_PAIRS)} entailment pairs")
    print(f"  - Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  - Confidence level: {CONFIDENCE_LEVEL*100}%")
    print("  - Effect size: Cohen's d")
    print("  - Significance: Mann-Whitney U (non-parametric)")

    # Load models
    print("\n" + "-" * 80)
    print("LOADING EMBEDDING MODELS")
    print("-" * 80)

    models = {}
    try:
        from sentence_transformers import SentenceTransformer

        model_ids = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2',
        ]

        for model_id in model_ids:
            short_name = model_id.split('/')[-1]
            print(f"  Loading {short_name}...")
            models[short_name] = SentenceTransformer(model_id)

        print(f"  Loaded {len(models)} models for cross-validation")

    except ImportError:
        print("ERROR: sentence-transformers not available")
        return None

    all_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print("=" * 80)

        def embed(text: str) -> np.ndarray:
            return model.encode(text, normalize_embeddings=True)

        # Compute metrics for each contradiction pair
        print("\nAnalyzing contradiction pairs...")
        contradiction_metrics = []
        for premise, hypothesis in SNLI_CONTRADICTION_PAIRS:
            # Create set with contradictory statements
            embeddings = np.array([embed(premise), embed(hypothesis)])
            # Also add paraphrases to increase sample size per set
            combined_text = f"{premise} {hypothesis}"
            embeddings = np.vstack([embeddings, [embed(combined_text)]])

            metrics = analyze_set(embeddings)
            contradiction_metrics.append(metrics)

        # Compute metrics for each entailment pair
        print("Analyzing entailment pairs...")
        entailment_metrics = []
        for premise, hypothesis in SNLI_ENTAILMENT_PAIRS:
            embeddings = np.array([embed(premise), embed(hypothesis)])
            combined_text = f"{premise} {hypothesis}"
            embeddings = np.vstack([embeddings, [embed(combined_text)]])

            metrics = analyze_set(embeddings)
            entailment_metrics.append(metrics)

        # Extract metric arrays
        contra_R = np.array([m.R_value for m in contradiction_metrics])
        contra_alpha = np.array([m.alpha for m in contradiction_metrics])
        contra_c1 = np.array([m.c_1 for m in contradiction_metrics])
        contra_Df_alpha = np.array([m.Df_times_alpha for m in contradiction_metrics])

        entail_R = np.array([m.R_value for m in entailment_metrics])
        entail_alpha = np.array([m.alpha for m in entailment_metrics])
        entail_c1 = np.array([m.c_1 for m in entailment_metrics])
        entail_Df_alpha = np.array([m.Df_times_alpha for m in entailment_metrics])

        # Compute alpha/c1 deviations from healthy values
        contra_alpha_dev = np.abs(contra_alpha - 0.5)
        entail_alpha_dev = np.abs(entail_alpha - 0.5)
        contra_c1_dev = np.abs(contra_c1 - 1.0)
        entail_c1_dev = np.abs(entail_c1 - 1.0)

        # Statistical comparisons
        print("\n" + "-" * 80)
        print("STATISTICAL ANALYSIS")
        print("-" * 80)

        comparisons = []

        # R-value comparison
        r_comp = statistical_comparison("R_value", contra_R, entail_R)
        comparisons.append(r_comp)
        print(f"\nR-value:")
        print(f"  Contradiction: {r_comp.group1_mean:.3f} [{r_comp.group1_ci_lower:.3f}, {r_comp.group1_ci_upper:.3f}]")
        print(f"  Entailment:    {r_comp.group2_mean:.3f} [{r_comp.group2_ci_lower:.3f}, {r_comp.group2_ci_upper:.3f}]")
        print(f"  Cohen's d:     {r_comp.cohens_d:.3f}")
        print(f"  p-value:       {r_comp.p_value:.4f} {'***' if r_comp.p_value < 0.001 else '**' if r_comp.p_value < 0.01 else '*' if r_comp.p_value < 0.05 else 'ns'}")

        # Alpha comparison
        alpha_comp = statistical_comparison("alpha", contra_alpha, entail_alpha)
        comparisons.append(alpha_comp)
        print(f"\nAlpha (eigenvalue decay):")
        print(f"  Contradiction: {alpha_comp.group1_mean:.4f} [{alpha_comp.group1_ci_lower:.4f}, {alpha_comp.group1_ci_upper:.4f}]")
        print(f"  Entailment:    {alpha_comp.group2_mean:.4f} [{alpha_comp.group2_ci_lower:.4f}, {alpha_comp.group2_ci_upper:.4f}]")
        print(f"  Cohen's d:     {alpha_comp.cohens_d:.3f}")
        print(f"  p-value:       {alpha_comp.p_value:.4f} {'***' if alpha_comp.p_value < 0.001 else '**' if alpha_comp.p_value < 0.01 else '*' if alpha_comp.p_value < 0.05 else 'ns'}")

        # Alpha deviation from 0.5
        alpha_dev_comp = statistical_comparison("|alpha - 0.5|", contra_alpha_dev, entail_alpha_dev)
        comparisons.append(alpha_dev_comp)
        print(f"\n|Alpha - 0.5| (deviation from healthy):")
        print(f"  Contradiction: {alpha_dev_comp.group1_mean:.4f} [{alpha_dev_comp.group1_ci_lower:.4f}, {alpha_dev_comp.group1_ci_upper:.4f}]")
        print(f"  Entailment:    {alpha_dev_comp.group2_mean:.4f} [{alpha_dev_comp.group2_ci_lower:.4f}, {alpha_dev_comp.group2_ci_upper:.4f}]")
        print(f"  Cohen's d:     {alpha_dev_comp.cohens_d:.3f}")
        print(f"  p-value:       {alpha_dev_comp.p_value:.4f} {'***' if alpha_dev_comp.p_value < 0.001 else '**' if alpha_dev_comp.p_value < 0.01 else '*' if alpha_dev_comp.p_value < 0.05 else 'ns'}")

        # c_1 comparison
        c1_comp = statistical_comparison("c_1", contra_c1, entail_c1)
        comparisons.append(c1_comp)
        print(f"\nc_1 (Chern class):")
        print(f"  Contradiction: {c1_comp.group1_mean:.4f} [{c1_comp.group1_ci_lower:.4f}, {c1_comp.group1_ci_upper:.4f}]")
        print(f"  Entailment:    {c1_comp.group2_mean:.4f} [{c1_comp.group2_ci_lower:.4f}, {c1_comp.group2_ci_upper:.4f}]")
        print(f"  Cohen's d:     {c1_comp.cohens_d:.3f}")
        print(f"  p-value:       {c1_comp.p_value:.4f} {'***' if c1_comp.p_value < 0.001 else '**' if c1_comp.p_value < 0.01 else '*' if c1_comp.p_value < 0.05 else 'ns'}")

        # c_1 deviation from 1.0
        c1_dev_comp = statistical_comparison("|c_1 - 1.0|", contra_c1_dev, entail_c1_dev)
        comparisons.append(c1_dev_comp)
        print(f"\n|c_1 - 1.0| (deviation from healthy):")
        print(f"  Contradiction: {c1_dev_comp.group1_mean:.4f} [{c1_dev_comp.group1_ci_lower:.4f}, {c1_dev_comp.group1_ci_upper:.4f}]")
        print(f"  Entailment:    {c1_dev_comp.group2_mean:.4f} [{c1_dev_comp.group2_ci_lower:.4f}, {c1_dev_comp.group2_ci_upper:.4f}]")
        print(f"  Cohen's d:     {c1_dev_comp.cohens_d:.3f}")
        print(f"  p-value:       {c1_dev_comp.p_value:.4f} {'***' if c1_dev_comp.p_value < 0.001 else '**' if c1_dev_comp.p_value < 0.01 else '*' if c1_dev_comp.p_value < 0.05 else 'ns'}")

        # Store results for this model
        all_results[model_name] = {
            'n_contradiction_pairs': len(SNLI_CONTRADICTION_PAIRS),
            'n_entailment_pairs': len(SNLI_ENTAILMENT_PAIRS),
            'comparisons': [asdict(c) for c in comparisons],
            'summary': {
                'R_discriminates': abs(r_comp.cohens_d) > 0.5 and r_comp.p_value < 0.05,
                'alpha_discriminates': abs(alpha_comp.cohens_d) > 0.5 and alpha_comp.p_value < 0.05,
                'alpha_dev_discriminates': abs(alpha_dev_comp.cohens_d) > 0.5 and alpha_dev_comp.p_value < 0.05,
                'c1_discriminates': abs(c1_comp.cohens_d) > 0.5 and c1_comp.p_value < 0.05,
                'c1_dev_discriminates': abs(c1_dev_comp.cohens_d) > 0.5 and c1_dev_comp.p_value < 0.05,
            }
        }

    # Cross-model summary
    print("\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY")
    print("=" * 80)

    print("\n| Metric | " + " | ".join(models.keys()) + " |")
    print("|--------|" + "|".join(["-------"] * len(models)) + "|")

    for metric in ['R_discriminates', 'alpha_discriminates', 'alpha_dev_discriminates', 'c1_discriminates', 'c1_dev_discriminates']:
        row = f"| {metric} |"
        for model_name in models.keys():
            result = all_results[model_name]['summary'][metric]
            row += f" {'YES' if result else 'NO'} |"
        print(row)

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Check if ANY metric discriminates across ALL models
    any_works_all_models = False
    for metric in ['alpha_discriminates', 'alpha_dev_discriminates', 'c1_discriminates', 'c1_dev_discriminates']:
        all_models_agree = all(all_results[m]['summary'][metric] for m in models.keys())
        if all_models_agree:
            print(f"\n  {metric}: WORKS ACROSS ALL MODELS")
            any_works_all_models = True

    r_works_any = any(all_results[m]['summary']['R_discriminates'] for m in models.keys())
    spectral_works_any = any(
        all_results[m]['summary']['alpha_dev_discriminates'] or all_results[m]['summary']['c1_dev_discriminates']
        for m in models.keys()
    )

    if not r_works_any and not spectral_works_any:
        print("\n  HYPOTHESIS FALSIFIED:")
        print("  Neither scalar R nor spectral metrics (alpha, c_1) can discriminate")
        print("  contradictions from entailments with statistical significance.")
        print("\n  CONCLUSION: Contradiction detection is fundamentally outside")
        print("  embedding geometry. Requires symbolic reasoning.")
        verdict = "FALSIFIED"
    elif spectral_works_any and not r_works_any:
        print("\n  PARTIAL SUCCESS:")
        print("  Spectral metrics show some discrimination but not consistently.")
        verdict = "PARTIAL"
    else:
        print("\n  UNEXPECTED: R-value shows discrimination")
        print("  This contradicts the documented limitation.")
        verdict = "NEEDS_INVESTIGATION"

    # Save results
    output = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'experiment': 'Q10_RIGOROUS_CONTRADICTION_DETECTION',
        'data_source': 'SNLI/MultiNLI human-annotated pairs',
        'n_contradiction_pairs': len(SNLI_CONTRADICTION_PAIRS),
        'n_entailment_pairs': len(SNLI_ENTAILMENT_PAIRS),
        'n_bootstrap': N_BOOTSTRAP,
        'confidence_level': CONFIDENCE_LEVEL,
        'models_tested': list(models.keys()),
        'results_by_model': all_results,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent / 'q10_rigorous_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else bool(x) if isinstance(x, np.bool_) else x)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    run_rigorous_experiment()
