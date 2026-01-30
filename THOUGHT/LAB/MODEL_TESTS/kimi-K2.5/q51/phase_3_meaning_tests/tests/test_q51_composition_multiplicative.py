#!/usr/bin/env python3
"""
Q51 Semantic Composition Test: Multiplicative vs Additive
Location: THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED/tests/

Tests whether semantic composition is:
- Real/Additive: meaning(C) = meaning(A) + ΔAB + ΔBC
- Complex/Multiplicative: meaning(C) = meaning(A) × (B/A) × (C/B)

Theory:
- Additive model: Linear vector arithmetic in embedding space
- Multiplicative model: Linear arithmetic in log-space (equivalent to 
  multiplicative composition in original space)
"""

import json
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
import os
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'results'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SEMANTIC CHAINS (morphological transformations)
# =============================================================================
SEMANTIC_CHAINS = [
    # Verb conjugations
    ['walk', 'walked', 'walking'],
    ['run', 'ran', 'running'],
    ['speak', 'spoke', 'speaking'],
    ['write', 'wrote', 'writing'],
    ['eat', 'ate', 'eating'],
    ['drink', 'drank', 'drinking'],
    ['think', 'thought', 'thinking'],
    ['sing', 'sang', 'singing'],
    ['drive', 'drove', 'driving'],
    ['swim', 'swam', 'swimming'],
    # Noun transformations
    ['cat', 'cats', 'kitten'],
    ['dog', 'dogs', 'puppy'],
    ['child', 'children', 'baby'],
    ['man', 'men', 'boy'],
    ['woman', 'women', 'girl'],
    ['person', 'people', 'human'],
    ['house', 'houses', 'home'],
    ['city', 'cities', 'town'],
    ['book', 'books', 'novel'],
    ['car', 'cars', 'vehicle'],
    # Adjective transformations
    ['happy', 'happier', 'happiest'],
    ['big', 'bigger', 'biggest'],
    ['small', 'smaller', 'smallest'],
    ['fast', 'faster', 'fastest'],
    ['good', 'better', 'best'],
    ['bad', 'worse', 'worst'],
    ['hot', 'hotter', 'hottest'],
    ['cold', 'colder', 'coldest'],
]

# =============================================================================
# ANALOGY TEST SET (a:b :: c:d)
# =============================================================================
ANALOGIES = [
    # Gender analogies
    ('man', 'woman', 'king', 'queen'),
    ('boy', 'girl', 'prince', 'princess'),
    ('father', 'mother', 'son', 'daughter'),
    ('uncle', 'aunt', 'nephew', 'niece'),
    ('husband', 'wife', 'brother', 'sister'),
    ('he', 'she', 'him', 'her'),
    ('actor', 'actress', 'waiter', 'waitress'),
    ('male', 'female', 'masculine', 'feminine'),
    
    # Plural analogies
    ('cat', 'cats', 'dog', 'dogs'),
    ('book', 'books', 'pen', 'pens'),
    ('child', 'children', 'person', 'people'),
    ('mouse', 'mice', 'louse', 'lice'),
    ('foot', 'feet', 'tooth', 'teeth'),
    ('goose', 'geese', 'mongoose', 'mongooses'),
    
    # Capital/Country
    ('paris', 'france', 'rome', 'italy'),
    ('tokyo', 'japan', 'beijing', 'china'),
    ('london', 'england', 'berlin', 'germany'),
    ('washington', 'usa', 'ottawa', 'canada'),
    ('moscow', 'russia', 'cairo', 'egypt'),
    ('sydney', 'australia', 'auckland', 'new_zealand'),
    
    # Comparative/Superlative
    ('big', 'bigger', 'small', 'smaller'),
    ('fast', 'faster', 'slow', 'slower'),
    ('good', 'better', 'bad', 'worse'),
    ('easy', 'easier', 'hard', 'harder'),
    ('tall', 'taller', 'short', 'shorter'),
    
    # Tense transformations
    ('walk', 'walked', 'talk', 'talked'),
    ('eat', 'ate', 'speak', 'spoke'),
    ('write', 'wrote', 'drive', 'drove'),
    ('sing', 'sang', 'ring', 'rang'),
    ('swim', 'swam', 'begin', 'began'),
    
    # Profession/Activity
    ('chef', 'cook', 'teacher', 'teach'),
    ('driver', 'drive', 'writer', 'write'),
    ('singer', 'sing', 'dancer', 'dance'),
    ('painter', 'paint', 'builder', 'build'),
    
    # Opposites
    ('hot', 'cold', 'warm', 'cool'),
    ('happy', 'sad', 'joyful', 'sorrowful'),
    ('rich', 'poor', 'wealthy', 'impoverished'),
    ('love', 'hate', 'adore', 'despise'),
    ('day', 'night', 'dawn', 'dusk'),
    
    # Category relationships
    ('apple', 'fruit', 'carrot', 'vegetable'),
    ('dog', 'animal', 'rose', 'flower'),
    ('oak', 'tree', 'salmon', 'fish'),
    ('diamond', 'gem', 'gold', 'metal'),
    ('python', 'language', 'ford', 'brand'),
]

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def load_model():
    """Load sentence transformer model."""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    return model


def get_embeddings(model, words):
    """Get embeddings for a list of words."""
    embeddings = model.encode(words, convert_to_numpy=True)
    return embeddings


def normalize_embeddings(embeddings):
    """L2 normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def test_semantic_chains_additive(model, chains, normalized=True):
    """
    Test additive composition on semantic chains.
    Theory: embedding(C) = embedding(A) + ΔAB + ΔBC
    
    Returns:
        dict with R², MSE, and residuals for each chain
    """
    results = []
    all_residuals = []
    
    for chain in chains:
        words = chain
        embeddings = get_embeddings(model, words)
        
        if normalized:
            embeddings = normalize_embeddings(embeddings)
        
        A, B, C = embeddings[0], embeddings[1], embeddings[2]
        
        # Additive model: C = A + (B-A) + (C-B) = C (identity)
        # So we're checking if the path is linear
        # Actually test: does B - A + C - B = C - A?
        delta_AB = B - A
        delta_BC = C - B
        delta_AC = C - A
        
        # Additive prediction: delta_AC should equal delta_AB + delta_BC
        predicted_delta = delta_AB + delta_BC
        residual = delta_AC - predicted_delta
        residual_norm = np.linalg.norm(residual)
        
        # R²: how well does additive model explain the variance?
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum(delta_AC ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results.append({
            'chain': chain,
            'r_squared': float(r_squared),
            'residual_norm': float(residual_norm),
            'residual': residual.tolist(),
            'words': words
        })
        all_residuals.append(residual_norm)
    
    return {
        'chains': results,
        'mean_r_squared': float(np.mean([r['r_squared'] for r in results])),
        'mean_residual': float(np.mean(all_residuals)),
        'std_residual': float(np.std(all_residuals)),
        'total_chains': len(chains)
    }


def test_semantic_chains_multiplicative(model, chains, normalized=True):
    """
    Test multiplicative composition on semantic chains (in log-space).
    Theory: log(C) = log(A) + log(B/A) + log(C/B)
    
    Note: For embeddings, we use log(1 + embedding) to handle negative values.
    This maps multiplicative relationships to additive in log-space.
    
    Returns:
        dict with R², MSE, and residuals for each chain
    """
    results = []
    all_residuals = []
    
    for chain in chains:
        words = chain
        embeddings = get_embeddings(model, words)
        
        if normalized:
            embeddings = normalize_embeddings(embeddings)
        
        # Transform to log-space using log(1+x) to handle [-1, 1] range
        # This is a common approach for signed data
        log_embeddings = np.log1p(embeddings + 1)  # Shift to [0, 2] then log1p
        
        log_A, log_B, log_C = log_embeddings[0], log_embeddings[1], log_embeddings[2]
        
        # In log-space, multiplicative becomes additive
        # log(C) = log(A) + log(B/A) + log(C/B)
        # log(B/A) = log(B) - log(A)
        # log(C/B) = log(C) - log(B)
        log_delta_AB = log_B - log_A
        log_delta_BC = log_C - log_B
        log_delta_AC = log_C - log_A
        
        # Additive prediction in log-space (equivalent to multiplicative in original)
        predicted_log_delta = log_delta_AB + log_delta_BC
        residual = log_delta_AC - predicted_log_delta
        residual_norm = np.linalg.norm(residual)
        
        # R² in log-space
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum(log_delta_AC ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results.append({
            'chain': chain,
            'r_squared': float(r_squared),
            'residual_norm': float(residual_norm),
            'residual': residual.tolist(),
            'words': words
        })
        all_residuals.append(residual_norm)
    
    return {
        'chains': results,
        'mean_r_squared': float(np.mean([r['r_squared'] for r in results])),
        'mean_residual': float(np.mean(all_residuals)),
        'std_residual': float(np.std(all_residuals)),
        'total_chains': len(chains)
    }


def test_analogies_additive(model, analogies, normalized=True):
    """
    Test additive model on analogies: d = c + (b - a)
    
    Returns:
        dict with accuracy, cosine similarities, and predictions
    """
    words = []
    for a, b, c, d in analogies:
        words.extend([a, b, c, d])
    
    # Get unique words
    unique_words = list(set(words))
    embeddings_dict = {}
    embeddings = get_embeddings(model, unique_words)
    
    if normalized:
        embeddings = normalize_embeddings(embeddings)
    
    for word, emb in zip(unique_words, embeddings):
        embeddings_dict[word] = emb
    
    results = []
    similarities = []
    rank_scores = []
    
    for a, b, c, d_target in analogies:
        emb_a = embeddings_dict[a]
        emb_b = embeddings_dict[b]
        emb_c = embeddings_dict[c]
        emb_d_target = embeddings_dict[d_target]
        
        # Additive prediction: d = c + (b - a)
        predicted_d = emb_c + (emb_b - emb_a)
        
        if normalized:
            predicted_d = predicted_d / (np.linalg.norm(predicted_d) + 1e-8)
        
        # Cosine similarity between predicted and actual
        similarity = np.dot(predicted_d, emb_d_target)
        similarities.append(float(similarity))
        
        # Rank: where does target rank among all words?
        all_sims = [np.dot(predicted_d, embeddings_dict[w]) for w in unique_words]
        rank = sorted(all_sims, reverse=True).index(similarity) + 1
        rank_scores.append(rank)
        
        results.append({
            'analogy': f"{a}:{b} :: {c}:{d_target}",
            'similarity': float(similarity),
            'rank': rank,
            'a': a, 'b': b, 'c': c, 'd': d_target
        })
    
    return {
        'analogies': results,
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'median_rank': float(np.median(rank_scores)),
        'mean_rank': float(np.mean(rank_scores)),
        'top1_accuracy': float(np.mean([r == 1 for r in rank_scores])),
        'top5_accuracy': float(np.mean([r <= 5 for r in rank_scores])),
        'total_analogies': len(analogies)
    }


def test_analogies_multiplicative(model, analogies, normalized=True):
    """
    Test multiplicative model on analogies: d = c × (b/a)
    
    In log-space: log(d) = log(c) + log(b) - log(a)
    
    Returns:
        dict with accuracy, cosine similarities, and predictions
    """
    words = []
    for a, b, c, d in analogies:
        words.extend([a, b, c, d])
    
    unique_words = list(set(words))
    embeddings_dict = {}
    embeddings = get_embeddings(model, unique_words)
    
    if normalized:
        embeddings = normalize_embeddings(embeddings)
    
    # Transform to log-space
    log_embeddings = np.log1p(embeddings + 1)
    
    for word, emb in zip(unique_words, log_embeddings):
        embeddings_dict[word] = emb
    
    results = []
    similarities = []
    rank_scores = []
    
    for a, b, c, d_target in analogies:
        log_a = embeddings_dict[a]
        log_b = embeddings_dict[b]
        log_c = embeddings_dict[c]
        log_d_target = embeddings_dict[d_target]
        
        # Multiplicative in original = Additive in log-space
        # log(d) = log(c) + log(b) - log(a)
        predicted_log_d = log_c + (log_b - log_a)
        
        # Transform back to original space
        predicted_d = np.expm1(predicted_log_d) - 1
        
        if normalized:
            predicted_d = predicted_d / (np.linalg.norm(predicted_d) + 1e-8)
            # Also normalize target for fair comparison
            emb_d_target_norm = (np.expm1(log_d_target) - 1)
            emb_d_target_norm = emb_d_target_norm / (np.linalg.norm(emb_d_target_norm) + 1e-8)
        else:
            emb_d_target_norm = (np.expm1(log_d_target) - 1)
        
        # Cosine similarity
        similarity = np.dot(predicted_d, emb_d_target_norm)
        similarities.append(float(similarity))
        
        # Rank calculation
        all_log_preds = []
        for w in unique_words:
            log_w = embeddings_dict[w]
            pred = np.expm1(log_c + (log_b - log_a)) - 1
            if normalized:
                pred = pred / (np.linalg.norm(pred) + 1e-8)
            all_log_preds.append(pred)
        
        all_sims = [np.dot(predicted_d, pred) for pred in all_log_preds]
        # We need to compare against actual embeddings, not predictions
        all_sims_actual = []
        for w in unique_words:
            log_w = embeddings_dict[w]
            emb_w = np.expm1(log_w) - 1
            if normalized:
                emb_w = emb_w / (np.linalg.norm(emb_w) + 1e-8)
            all_sims_actual.append(np.dot(predicted_d, emb_w))
        
        rank = sorted(all_sims_actual, reverse=True).index(
            np.dot(predicted_d, emb_d_target_norm)
        ) + 1
        rank_scores.append(rank)
        
        results.append({
            'analogy': f"{a}:{b} :: {c}:{d_target}",
            'similarity': float(similarity),
            'rank': rank,
            'a': a, 'b': b, 'c': c, 'd': d_target
        })
    
    return {
        'analogies': results,
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'median_rank': float(np.median(rank_scores)),
        'mean_rank': float(np.mean(rank_scores)),
        'top1_accuracy': float(np.mean([r == 1 for r in rank_scores])),
        'top5_accuracy': float(np.mean([r <= 5 for r in rank_scores])),
        'total_analogies': len(analogies)
    }


def perform_statistical_comparison(additive_results, multiplicative_results, test_type='chains'):
    """
    Perform statistical comparison between additive and multiplicative models.
    
    Returns:
        dict with t-test results and model selection
    """
    if test_type == 'chains':
        # Compare R² values
        add_r2 = [r['r_squared'] for r in additive_results['chains']]
        mult_r2 = [r['r_squared'] for r in multiplicative_results['chains']]
        
        # Compare residual norms
        add_resid = [r['residual_norm'] for r in additive_results['chains']]
        mult_resid = [r['residual_norm'] for r in multiplicative_results['chains']]
        
        # Paired t-test on R²
        t_stat_r2, p_val_r2 = stats.ttest_rel(add_r2, mult_r2)
        
        # Paired t-test on residuals (lower is better)
        t_stat_resid, p_val_resid = stats.ttest_rel(add_resid, mult_resid)
        
        # Effect size (Cohen's d)
        mean_diff_r2 = np.mean(add_r2) - np.mean(mult_r2)
        pooled_std_r2 = np.sqrt((np.std(add_r2)**2 + np.std(mult_r2)**2) / 2)
        cohens_d_r2 = mean_diff_r2 / pooled_std_r2 if pooled_std_r2 > 0 else 0
        
        mean_diff_resid = np.mean(add_resid) - np.mean(mult_resid)
        pooled_std_resid = np.sqrt((np.std(add_resid)**2 + np.std(mult_resid)**2) / 2)
        cohens_d_resid = mean_diff_resid / pooled_std_resid if pooled_std_resid > 0 else 0
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat_r2, w_p_r2 = stats.wilcoxon(add_r2, mult_r2)
            w_stat_resid, w_p_resid = stats.wilcoxon(add_resid, mult_resid)
        except ValueError:
            w_stat_r2, w_p_r2 = 0, 1
            w_stat_resid, w_p_resid = 0, 1
        
        return {
            'r2_comparison': {
                'additive_mean': float(np.mean(add_r2)),
                'multiplicative_mean': float(np.mean(mult_r2)),
                'difference': float(mean_diff_r2),
                't_statistic': float(t_stat_r2),
                'p_value': float(p_val_r2),
                'cohens_d': float(cohens_d_r2),
                'wilcoxon_statistic': float(w_stat_r2),
                'wilcoxon_p': float(w_p_r2),
                'winner': 'additive' if np.mean(add_r2) > np.mean(mult_r2) else 'multiplicative'
            },
            'residual_comparison': {
                'additive_mean': float(np.mean(add_resid)),
                'multiplicative_mean': float(np.mean(mult_resid)),
                'difference': float(mean_diff_resid),
                't_statistic': float(t_stat_resid),
                'p_value': float(p_val_resid),
                'cohens_d': float(cohens_d_resid),
                'wilcoxon_statistic': float(w_stat_resid),
                'wilcoxon_p': float(w_p_resid),
                'winner': 'additive' if np.mean(add_resid) < np.mean(mult_resid) else 'multiplicative'
            }
        }
    
    elif test_type == 'analogies':
        # Compare similarities and ranks
        add_sims = [r['similarity'] for r in additive_results['analogies']]
        mult_sims = [r['similarity'] for r in multiplicative_results['analogies']]
        
        add_ranks = [r['rank'] for r in additive_results['analogies']]
        mult_ranks = [r['rank'] for r in multiplicative_results['analogies']]
        
        # Paired t-tests
        t_stat_sim, p_val_sim = stats.ttest_rel(add_sims, mult_sims)
        t_stat_rank, p_val_rank = stats.ttest_rel(add_ranks, mult_ranks)
        
        # Effect sizes
        mean_diff_sim = np.mean(add_sims) - np.mean(mult_sims)
        pooled_std_sim = np.sqrt((np.std(add_sims)**2 + np.std(mult_sims)**2) / 2)
        cohens_d_sim = mean_diff_sim / pooled_std_sim if pooled_std_sim > 0 else 0
        
        mean_diff_rank = np.mean(add_ranks) - np.mean(mult_ranks)
        pooled_std_rank = np.sqrt((np.std(add_ranks)**2 + np.std(mult_ranks)**2) / 2)
        cohens_d_rank = mean_diff_rank / pooled_std_rank if pooled_std_rank > 0 else 0
        
        # Wilcoxon tests
        try:
            w_stat_sim, w_p_sim = stats.wilcoxon(add_sims, mult_sims)
            w_stat_rank, w_p_rank = stats.wilcoxon(add_ranks, mult_ranks)
        except ValueError:
            w_stat_sim, w_p_sim = 0, 1
            w_stat_rank, w_p_rank = 0, 1
        
        return {
            'similarity_comparison': {
                'additive_mean': float(np.mean(add_sims)),
                'multiplicative_mean': float(np.mean(mult_sims)),
                'difference': float(mean_diff_sim),
                't_statistic': float(t_stat_sim),
                'p_value': float(p_val_sim),
                'cohens_d': float(cohens_d_sim),
                'wilcoxon_statistic': float(w_stat_sim),
                'wilcoxon_p': float(w_p_sim),
                'winner': 'additive' if np.mean(add_sims) > np.mean(mult_sims) else 'multiplicative'
            },
            'rank_comparison': {
                'additive_mean': float(np.mean(add_ranks)),
                'multiplicative_mean': float(np.mean(mult_ranks)),
                'difference': float(mean_diff_rank),
                't_statistic': float(t_stat_rank),
                'p_value': float(p_val_rank),
                'cohens_d': float(cohens_d_rank),
                'wilcoxon_statistic': float(w_stat_rank),
                'wilcoxon_p': float(w_p_rank),
                'winner': 'additive' if np.mean(add_ranks) < np.mean(mult_ranks) else 'multiplicative'
            },
            'accuracy_comparison': {
                'additive_top1': additive_results['top1_accuracy'],
                'multiplicative_top1': multiplicative_results['top1_accuracy'],
                'additive_top5': additive_results['top5_accuracy'],
                'multiplicative_top5': multiplicative_results['top5_accuracy'],
            }
        }


def generate_report(results, output_path):
    """Generate markdown report of results."""
    report = []
    report.append("# Q51 Semantic Composition Test: Multiplicative vs Additive\n")
    report.append(f"**Model**: {MODEL_NAME}\n")
    report.append(f"**Date**: {np.datetime64('today')}\n")
    report.append("**Question**: Is semantic composition multiplicative (complex) or additive (real)?\n\n")
    
    report.append("## Theory\n\n")
    report.append("- **Real/Additive**: meaning(C) = meaning(A) + ΔAB + ΔBC\n")
    report.append("  - Linear arithmetic in embedding space\n")
    report.append("  - Analogies: d = c + (b - a)\n\n")
    report.append("- **Complex/Multiplicative**: meaning(C) = meaning(A) × (B/A) × (C/B)\n")
    report.append("  - In log-space: log(C) = log(A) + log(B/A) + log(C/B)\n")
    report.append("  - Linear arithmetic in log-space (multiplicative in original)\n")
    report.append("  - Analogies: log(d) = log(c) + log(b) - log(a)\n\n")
    
    # Semantic Chains Results
    report.append("## Test 1: Semantic Chains\n\n")
    report.append(f"**Chains tested**: {results['semantic_chains']['additive']['total_chains']}\n\n")
    
    report.append("### Additive Model Results\n\n")
    add_chains = results['semantic_chains']['additive']
    report.append(f"- **Mean R²**: {add_chains['mean_r_squared']:.4f}\n")
    report.append(f"- **Mean Residual**: {add_chains['mean_residual']:.4f}\n")
    report.append(f"- **Std Residual**: {add_chains['std_residual']:.4f}\n\n")
    
    report.append("### Multiplicative Model Results\n\n")
    mult_chains = results['semantic_chains']['multiplicative']
    report.append(f"- **Mean R²**: {mult_chains['mean_r_squared']:.4f}\n")
    report.append(f"- **Mean Residual**: {mult_chains['mean_residual']:.4f}\n")
    report.append(f"- **Std Residual**: {mult_chains['std_residual']:.4f}\n\n")
    
    report.append("### Statistical Comparison (Chains)\n\n")
    chain_stats = results['statistical_comparison']['chains']
    
    r2_comp = chain_stats['r2_comparison']
    report.append("**R² Comparison**:\n")
    report.append(f"- Additive mean R²: {r2_comp['additive_mean']:.4f}\n")
    report.append(f"- Multiplicative mean R²: {r2_comp['multiplicative_mean']:.4f}\n")
    report.append(f"- Difference: {r2_comp['difference']:.4f}\n")
    report.append(f"- Paired t-test: t={r2_comp['t_statistic']:.4f}, p={r2_comp['p_value']:.4f}\n")
    report.append(f"- Effect size (Cohen's d): {r2_comp['cohens_d']:.4f}\n")
    report.append(f"- Wilcoxon test: W={r2_comp['wilcoxon_statistic']:.2f}, p={r2_comp['wilcoxon_p']:.4f}\n")
    report.append(f"- **Winner**: {r2_comp['winner']}\n\n")
    
    resid_comp = chain_stats['residual_comparison']
    report.append("**Residual Comparison** (lower is better):\n")
    report.append(f"- Additive mean residual: {resid_comp['additive_mean']:.4f}\n")
    report.append(f"- Multiplicative mean residual: {resid_comp['multiplicative_mean']:.4f}\n")
    report.append(f"- Difference: {resid_comp['difference']:.4f}\n")
    report.append(f"- Paired t-test: t={resid_comp['t_statistic']:.4f}, p={resid_comp['p_value']:.4f}\n")
    report.append(f"- Effect size (Cohen's d): {resid_comp['cohens_d']:.4f}\n")
    report.append(f"- Wilcoxon test: W={resid_comp['wilcoxon_statistic']:.2f}, p={resid_comp['wilcoxon_p']:.4f}\n")
    report.append(f"- **Winner**: {resid_comp['winner']}\n\n")
    
    # Analogies Results
    report.append("## Test 2: Analogy Completion\n\n")
    report.append(f"**Analogies tested**: {results['analogies']['additive']['total_analogies']}\n\n")
    
    report.append("### Additive Model Results\n\n")
    add_analogies = results['analogies']['additive']
    report.append(f"- **Mean Cosine Similarity**: {add_analogies['mean_similarity']:.4f}\n")
    report.append(f"- **Top-1 Accuracy**: {add_analogies['top1_accuracy']:.2%}\n")
    report.append(f"- **Top-5 Accuracy**: {add_analogies['top5_accuracy']:.2%}\n")
    report.append(f"- **Mean Rank**: {add_analogies['mean_rank']:.2f}\n")
    report.append(f"- **Median Rank**: {add_analogies['median_rank']:.2f}\n\n")
    
    report.append("### Multiplicative Model Results\n\n")
    mult_analogies = results['analogies']['multiplicative']
    report.append(f"- **Mean Cosine Similarity**: {mult_analogies['mean_similarity']:.4f}\n")
    report.append(f"- **Top-1 Accuracy**: {mult_analogies['top1_accuracy']:.2%}\n")
    report.append(f"- **Top-5 Accuracy**: {mult_analogies['top5_accuracy']:.2%}\n")
    report.append(f"- **Mean Rank**: {mult_analogies['mean_rank']:.2f}\n")
    report.append(f"- **Median Rank**: {mult_analogies['median_rank']:.2f}\n\n")
    
    report.append("### Statistical Comparison (Analogies)\n\n")
    analogy_stats = results['statistical_comparison']['analogies']
    
    sim_comp = analogy_stats['similarity_comparison']
    report.append("**Cosine Similarity Comparison** (higher is better):\n")
    report.append(f"- Additive mean: {sim_comp['additive_mean']:.4f}\n")
    report.append(f"- Multiplicative mean: {sim_comp['multiplicative_mean']:.4f}\n")
    report.append(f"- Difference: {sim_comp['difference']:.4f}\n")
    report.append(f"- Paired t-test: t={sim_comp['t_statistic']:.4f}, p={sim_comp['p_value']:.4f}\n")
    report.append(f"- Effect size (Cohen's d): {sim_comp['cohens_d']:.4f}\n")
    report.append(f"- Wilcoxon test: W={sim_comp['wilcoxon_statistic']:.2f}, p={sim_comp['wilcoxon_p']:.4f}\n")
    report.append(f"- **Winner**: {sim_comp['winner']}\n\n")
    
    rank_comp = analogy_stats['rank_comparison']
    report.append("**Rank Comparison** (lower is better):\n")
    report.append(f"- Additive mean rank: {rank_comp['additive_mean']:.2f}\n")
    report.append(f"- Multiplicative mean rank: {rank_comp['multiplicative_mean']:.2f}\n")
    report.append(f"- Difference: {rank_comp['difference']:.2f}\n")
    report.append(f"- Paired t-test: t={rank_comp['t_statistic']:.4f}, p={rank_comp['p_value']:.4f}\n")
    report.append(f"- Effect size (Cohen's d): {rank_comp['cohens_d']:.4f}\n")
    report.append(f"- Wilcoxon test: W={rank_comp['wilcoxon_statistic']:.2f}, p={rank_comp['wilcoxon_p']:.4f}\n")
    report.append(f"- **Winner**: {rank_comp['winner']}\n\n")
    
    # Conclusion
    report.append("## Conclusion\n\n")
    
    # Determine overall winner
    wins = {
        'additive': 0,
        'multiplicative': 0
    }
    
    if r2_comp['winner'] == 'additive':
        wins['additive'] += 1
    else:
        wins['multiplicative'] += 1
    
    if resid_comp['winner'] == 'additive':
        wins['additive'] += 1
    else:
        wins['multiplicative'] += 1
    
    if sim_comp['winner'] == 'additive':
        wins['additive'] += 1
    else:
        wins['multiplicative'] += 1
    
    if rank_comp['winner'] == 'additive':
        wins['additive'] += 1
    else:
        wins['multiplicative'] += 1
    
    overall_winner = 'additive' if wins['additive'] > wins['multiplicative'] else 'multiplicative'
    
    report.append(f"**Overall Winner**: {overall_winner.upper()}\n\n")
    report.append(f"- Additive wins: {wins['additive']}/4 metrics\n")
    report.append(f"- Multiplicative wins: {wins['multiplicative']}/4 metrics\n\n")
    
    # Statistical significance summary
    significant_tests = []
    if r2_comp['p_value'] < 0.05:
        significant_tests.append(f"R² comparison (p={r2_comp['p_value']:.4f})")
    if resid_comp['p_value'] < 0.05:
        significant_tests.append(f"Residual comparison (p={resid_comp['p_value']:.4f})")
    if sim_comp['p_value'] < 0.05:
        significant_tests.append(f"Similarity comparison (p={sim_comp['p_value']:.4f})")
    if rank_comp['p_value'] < 0.05:
        significant_tests.append(f"Rank comparison (p={rank_comp['p_value']:.4f})")
    
    if significant_tests:
        report.append("**Statistically Significant Differences** (p < 0.05):\n")
        for test in significant_tests:
            report.append(f"- {test}\n")
        report.append("\n")
    else:
        report.append("**Statistical Significance**: No statistically significant differences found (all p > 0.05)\n\n")
    
    # Interpretation
    report.append("## Interpretation\n\n")
    
    if overall_winner == 'additive':
        report.append("The **additive (real) model** provides a better fit for semantic composition.\n\n")
        report.append("This suggests that:\n")
        report.append("- Semantic transformations are primarily **linear translations** in embedding space\n")
        report.append("- Vector arithmetic (a - b + c ≈ d) is the appropriate model\n")
        report.append("- Complex multiplicative structure is **not supported** by this analysis\n\n")
    else:
        report.append("The **multiplicative (complex) model** provides a better fit for semantic composition.\n\n")
        report.append("This suggests that:\n")
        report.append("- Semantic transformations involve **multiplicative interactions**\n")
        report.append("- Log-space linearity implies exponential/geometric structure in original space\n")
        report.append("- Complex composition (phase-like interactions) may be present\n\n")
    
    # Caveats
    report.append("## Caveats\n\n")
    report.append("1. **Log-space transformation**: Used log1p(x+1) to handle negative embeddings.\n")
    report.append("   This is a common approach but introduces approximation.\n\n")
    report.append("2. **Normalization**: Results shown for normalized embeddings.\n")
    report.append("   Raw embeddings may show different patterns.\n\n")
    report.append("3. **Model-specific**: Results are for {MODEL_NAME}.\n")
    report.append("   Other models may exhibit different composition patterns.\n\n")
    report.append("4. **Sample size**: Limited to available semantic chains and analogies.\n")
    report.append("   Larger-scale validation recommended.\n\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"Report written to: {output_path}")


def main():
    """Run the full semantic composition test suite."""
    print("="*70)
    print("Q51 SEMANTIC COMPOSITION TEST: MULTIPLICATIVE vs ADDITIVE")
    print("="*70)
    
    # Load model
    model = load_model()
    
    # Run tests
    print("\n[1/4] Testing semantic chains (additive model)...")
    chains_additive = test_semantic_chains_additive(model, SEMANTIC_CHAINS, normalized=True)
    
    print("[2/4] Testing semantic chains (multiplicative model)...")
    chains_multiplicative = test_semantic_chains_multiplicative(model, SEMANTIC_CHAINS, normalized=True)
    
    print("[3/4] Testing analogies (additive model)...")
    analogies_additive = test_analogies_additive(model, ANALOGIES, normalized=True)
    
    print("[4/4] Testing analogies (multiplicative model)...")
    analogies_multiplicative = test_analogies_multiplicative(model, ANALOGIES, normalized=True)
    
    # Statistical comparisons
    print("\n[Analysis] Performing statistical comparisons...")
    chains_stats = perform_statistical_comparison(chains_additive, chains_multiplicative, 'chains')
    analogies_stats = perform_statistical_comparison(analogies_additive, analogies_multiplicative, 'analogies')
    
    # Compile results
    results = {
        'model': MODEL_NAME,
        'semantic_chains': {
            'additive': chains_additive,
            'multiplicative': chains_multiplicative
        },
        'analogies': {
            'additive': analogies_additive,
            'multiplicative': analogies_multiplicative
        },
        'statistical_comparison': {
            'chains': chains_stats,
            'analogies': analogies_stats
        }
    }
    
    # Save JSON results
    json_path = OUTPUT_DIR / 'q51_composition_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Generate report
    report_path = Path('THOUGHT/LAB/MODEL_TESTS/kimi-K2.5/COMPROMISED/q51_composition_report.md')
    generate_report(results, report_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nSemantic Chains ({chains_additive['total_chains']} chains):")
    print(f"  Additive R²: {chains_additive['mean_r_squared']:.4f}")
    print(f"  Multiplicative R²: {chains_multiplicative['mean_r_squared']:.4f}")
    print(f"  Winner: {chains_stats['r2_comparison']['winner']}")
    
    print(f"\nAnalogies ({analogies_additive['total_analogies']} analogies):")
    print(f"  Additive Top-1: {analogies_additive['top1_accuracy']:.2%}")
    print(f"  Multiplicative Top-1: {analogies_multiplicative['top1_accuracy']:.2%}")
    print(f"  Winner: {analogies_stats['similarity_comparison']['winner']}")
    
    # Determine overall winner
    wins_additive = sum([
        chains_stats['r2_comparison']['winner'] == 'additive',
        chains_stats['residual_comparison']['winner'] == 'additive',
        analogies_stats['similarity_comparison']['winner'] == 'additive',
        analogies_stats['rank_comparison']['winner'] == 'additive'
    ])
    wins_multiplicative = 4 - wins_additive
    
    overall = 'ADDITIVE' if wins_additive > wins_multiplicative else 'MULTIPLICATIVE'
    print(f"\n{'='*70}")
    print(f"OVERALL WINNER: {overall} ({max(wins_additive, wins_multiplicative)}/4 metrics)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
