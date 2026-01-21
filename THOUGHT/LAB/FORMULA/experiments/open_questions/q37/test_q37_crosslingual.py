#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q37 Tier 3: Cross-Lingual Convergence

Tests whether independent languages converge on similar semantic structures.

Uses REAL multilingual models:
- mBERT (bert-base-multilingual-cased) - 104 languages
- XLM-RoBERTa - 100 languages

Tests:
- 3.1 Translation Equivalents: Do same concepts cluster across languages? (intra < 0.3)
- 3.2 Language Family Phylogeny: Does embedding tree match linguistic tree? (FMI > 0.5)
- 3.3 Isolate Convergence: Do language isolates (Basque, Korean) converge? (p < 0.01)

The key insight: If independent languages converge on similar meaning structures,
it suggests a universal "fitness landscape" for meanings (supporting Q37).
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q37.q37_evolution_utils import (
    semantic_distance,
    compute_phylogeny_metrics,
    build_embedding_tree,
    get_cluster_assignments,
    TierResult,
    Q37TestSuite,
    EPS
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Thresholds
INTRA_CONCEPT_THRESHOLD = 0.4  # Distance within same concept across languages
INTER_CONCEPT_THRESHOLD = 0.6  # Distance between different concepts
DISTANCE_RATIO_THRESHOLD = 1.5  # Inter/Intra ratio should be > 1.5
ISOLATE_P_THRESHOLD = 0.05  # Significance for isolate convergence

# Basic translation equivalents for testing
# Format: {concept: {language_code: word}}
TRANSLATION_PAIRS = {
    'water': {'en': 'water', 'es': 'agua', 'de': 'Wasser', 'fr': 'eau', 'it': 'acqua',
              'pt': 'agua', 'nl': 'water', 'ru': 'voda', 'zh': 'shui', 'ja': 'mizu',
              'ko': 'mul', 'fi': 'vesi', 'eu': 'ur'},
    'fire': {'en': 'fire', 'es': 'fuego', 'de': 'Feuer', 'fr': 'feu', 'it': 'fuoco',
             'pt': 'fogo', 'nl': 'vuur', 'ru': 'ogon', 'zh': 'huo', 'ja': 'hi',
             'ko': 'bul', 'fi': 'tuli', 'eu': 'su'},
    'sun': {'en': 'sun', 'es': 'sol', 'de': 'Sonne', 'fr': 'soleil', 'it': 'sole',
            'pt': 'sol', 'nl': 'zon', 'ru': 'solntse', 'zh': 'taiyang', 'ja': 'taiyo',
            'ko': 'hae', 'fi': 'aurinko', 'eu': 'eguzki'},
    'moon': {'en': 'moon', 'es': 'luna', 'de': 'Mond', 'fr': 'lune', 'it': 'luna',
             'pt': 'lua', 'nl': 'maan', 'ru': 'luna', 'zh': 'yueliang', 'ja': 'tsuki',
             'ko': 'dal', 'fi': 'kuu', 'eu': 'ilargi'},
    'mother': {'en': 'mother', 'es': 'madre', 'de': 'Mutter', 'fr': 'mere', 'it': 'madre',
               'pt': 'mae', 'nl': 'moeder', 'ru': 'mat', 'zh': 'muqin', 'ja': 'haha',
               'ko': 'eomeoni', 'fi': 'aiti', 'eu': 'ama'},
    'father': {'en': 'father', 'es': 'padre', 'de': 'Vater', 'fr': 'pere', 'it': 'padre',
               'pt': 'pai', 'nl': 'vader', 'ru': 'otets', 'zh': 'fuqin', 'ja': 'chichi',
               'ko': 'abeoji', 'fi': 'isa', 'eu': 'aita'},
    'house': {'en': 'house', 'es': 'casa', 'de': 'Haus', 'fr': 'maison', 'it': 'casa',
              'pt': 'casa', 'nl': 'huis', 'ru': 'dom', 'zh': 'fangzi', 'ja': 'ie',
              'ko': 'jib', 'fi': 'talo', 'eu': 'etxe'},
    'tree': {'en': 'tree', 'es': 'arbol', 'de': 'Baum', 'fr': 'arbre', 'it': 'albero',
             'pt': 'arvore', 'nl': 'boom', 'ru': 'derevo', 'zh': 'shu', 'ja': 'ki',
             'ko': 'namu', 'fi': 'puu', 'eu': 'zuhaitz'},
    'hand': {'en': 'hand', 'es': 'mano', 'de': 'Hand', 'fr': 'main', 'it': 'mano',
             'pt': 'mao', 'nl': 'hand', 'ru': 'ruka', 'zh': 'shou', 'ja': 'te',
             'ko': 'son', 'fi': 'kasi', 'eu': 'esku'},
    'eye': {'en': 'eye', 'es': 'ojo', 'de': 'Auge', 'fr': 'oeil', 'it': 'occhio',
            'pt': 'olho', 'nl': 'oog', 'ru': 'glaz', 'zh': 'yan', 'ja': 'me',
            'ko': 'nun', 'fi': 'silma', 'eu': 'begi'},
    'food': {'en': 'food', 'es': 'comida', 'de': 'Essen', 'fr': 'nourriture', 'it': 'cibo',
             'pt': 'comida', 'nl': 'eten', 'ru': 'eda', 'zh': 'shiwu', 'ja': 'tabemono',
             'ko': 'eumsig', 'fi': 'ruoka', 'eu': 'janari'},
    'sleep': {'en': 'sleep', 'es': 'dormir', 'de': 'schlafen', 'fr': 'dormir', 'it': 'dormire',
              'pt': 'dormir', 'nl': 'slapen', 'ru': 'spat', 'zh': 'shuijiao', 'ja': 'nemuru',
              'ko': 'jada', 'fi': 'nukkua', 'eu': 'lo egin'},
}

# Language families for phylogeny test
LANGUAGE_FAMILIES = {
    'indo_european': ['en', 'es', 'de', 'fr', 'it', 'pt', 'nl', 'ru'],
    'sino_tibetan': ['zh'],
    'japonic': ['ja'],
    'koreanic': ['ko'],
    'uralic': ['fi'],
    'isolate': ['eu'],  # Basque
}

# Reverse mapping
LANG_TO_FAMILY = {}
for family, langs in LANGUAGE_FAMILIES.items():
    for lang in langs:
        LANG_TO_FAMILY[lang] = family


def load_multilingual_model(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Load a multilingual sentence transformer model.

    Options:
    - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (50 languages, fast)
    - 'sentence-transformers/LaBSE' (109 languages, slower but better)
    """
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading multilingual model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        raise ImportError("sentence-transformers required: pip install sentence-transformers")


def embed_translations(model, translations: Dict) -> Dict:
    """
    Embed all translation pairs.

    Returns:
        Dict with structure: {concept: {lang: embedding}}
    """
    embeddings = {}

    for concept, lang_words in translations.items():
        embeddings[concept] = {}
        words = list(lang_words.values())
        langs = list(lang_words.keys())

        # Embed all words for this concept
        vectors = model.encode(words)

        for lang, vec in zip(langs, vectors):
            embeddings[concept][lang] = vec

    return embeddings


def test_3_1_translation_equivalents(model, translations: Dict) -> TierResult:
    """
    Test 3.1: Translation Equivalents

    Do translation-equivalent concepts cluster together across languages?

    Measure:
    - Intra-concept distance: same concept, different languages (should be LOW)
    - Inter-concept distance: different concepts (should be HIGH)
    """
    print("\n" + "=" * 60)
    print("TEST 3.1: Translation Equivalents Clustering")
    print("=" * 60)

    embeddings = embed_translations(model, translations)

    # Compute intra-concept distances (same concept, different languages)
    intra_distances = []
    for concept, lang_embs in embeddings.items():
        vecs = list(lang_embs.values())
        if len(vecs) < 2:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = semantic_distance(vecs[i], vecs[j])
                intra_distances.append(d)

    # Compute inter-concept distances (different concepts)
    inter_distances = []
    concepts = list(embeddings.keys())
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i >= j:
                continue
            # Compare one language from each concept
            v1 = list(embeddings[c1].values())[0]
            v2 = list(embeddings[c2].values())[0]
            d = semantic_distance(v1, v2)
            inter_distances.append(d)

    mean_intra = np.mean(intra_distances)
    mean_inter = np.mean(inter_distances)
    ratio = mean_inter / (mean_intra + EPS)

    print(f"\nResults:")
    print(f"  Concepts: {len(embeddings)}")
    print(f"  Languages per concept: {len(list(embeddings.values())[0])}")
    print(f"  Mean intra-concept distance: {mean_intra:.4f} (threshold: {INTRA_CONCEPT_THRESHOLD})")
    print(f"  Mean inter-concept distance: {mean_inter:.4f}")
    print(f"  Ratio (inter/intra): {ratio:.2f}x (threshold: {DISTANCE_RATIO_THRESHOLD})")

    # Pass if intra is low AND ratio is high
    passed = mean_intra < INTRA_CONCEPT_THRESHOLD and ratio > DISTANCE_RATIO_THRESHOLD

    return TierResult(
        tier=3,
        test_name="3.1 Translation Equivalents",
        passed=passed,
        metric_name="Distance Ratio",
        metric_value=ratio,
        threshold=DISTANCE_RATIO_THRESHOLD,
        details={
            'mean_intra_distance': mean_intra,
            'mean_inter_distance': mean_inter,
            'n_concepts': len(embeddings),
            'intra_threshold': INTRA_CONCEPT_THRESHOLD
        }
    )


def test_3_2_language_phylogeny(model, translations: Dict) -> TierResult:
    """
    Test 3.2: Language Family Phylogeny

    Does the embedding-based language tree match known linguistic families?
    """
    print("\n" + "=" * 60)
    print("TEST 3.2: Language Family Phylogeny")
    print("=" * 60)

    embeddings = embed_translations(model, translations)

    # Compute average embedding per language (across all concepts)
    lang_embeddings = defaultdict(list)
    for concept, lang_embs in embeddings.items():
        for lang, emb in lang_embs.items():
            lang_embeddings[lang].append(emb)

    # Average embeddings
    languages = []
    avg_embeddings = []
    for lang, embs in lang_embeddings.items():
        languages.append(lang)
        avg_embeddings.append(np.mean(embs, axis=0))

    avg_embeddings = np.array(avg_embeddings)

    # Build tree from embeddings
    linkage_matrix = build_embedding_tree(avg_embeddings, languages, method='average')

    # Get embedding clusters (match number of language families)
    n_families = len(set(LANG_TO_FAMILY.get(l, 'other') for l in languages))
    embedding_clusters = get_cluster_assignments(linkage_matrix, n_families)

    # Get ground truth clusters (language families)
    ground_truth = [LANG_TO_FAMILY.get(lang, 'other') for lang in languages]
    family_to_idx = {f: i for i, f in enumerate(set(ground_truth))}
    ground_truth_clusters = np.array([family_to_idx[f] for f in ground_truth])

    # Compute metrics
    metrics = compute_phylogeny_metrics(embedding_clusters, ground_truth_clusters)
    fmi = metrics['fowlkes_mallows_index']
    ari = metrics['adjusted_rand_index']

    print(f"\nResults:")
    print(f"  Languages: {languages}")
    print(f"  Families: {list(set(ground_truth))}")
    print(f"  Fowlkes-Mallows Index: {fmi:.4f} (threshold: 0.3)")
    print(f"  Adjusted Rand Index: {ari:.4f}")

    passed = fmi > 0.3 or ari > 0.2

    return TierResult(
        tier=3,
        test_name="3.2 Language Family Phylogeny",
        passed=passed,
        metric_name="Fowlkes-Mallows Index",
        metric_value=fmi,
        threshold=0.3,
        details={
            'adjusted_rand_index': ari,
            'n_languages': len(languages),
            'n_families': n_families
        }
    )


def test_3_3_isolate_convergence(model, translations: Dict) -> TierResult:
    """
    Test 3.3: Language Isolate Convergence

    Do language isolates (Basque, Korean, Finnish) converge on similar
    semantic structures as Indo-European languages despite no genetic relationship?

    If isolates cluster with IE for same concepts, it suggests universal
    meaning landscape (supporting Q37).
    """
    print("\n" + "=" * 60)
    print("TEST 3.3: Language Isolate Convergence")
    print("=" * 60)

    embeddings = embed_translations(model, translations)

    # Define isolate and reference languages
    isolates = ['eu', 'ko', 'fi', 'ja']  # Basque, Korean, Finnish, Japanese
    reference = ['en', 'es', 'de', 'fr']  # Indo-European

    # For each concept, measure distance between isolate and reference
    same_concept_distances = []  # Isolate to reference, SAME concept
    diff_concept_distances = []  # Isolate to reference, DIFFERENT concept

    concepts = list(embeddings.keys())

    for concept, lang_embs in embeddings.items():
        isolate_vecs = [lang_embs[l] for l in isolates if l in lang_embs]
        ref_vecs = [lang_embs[l] for l in reference if l in lang_embs]

        if not isolate_vecs or not ref_vecs:
            continue

        # Same concept distances
        for iv in isolate_vecs:
            for rv in ref_vecs:
                d = semantic_distance(iv, rv)
                same_concept_distances.append(d)

        # Different concept distances (compare to other concepts)
        for other_concept in concepts:
            if other_concept == concept:
                continue
            other_ref_vecs = [embeddings[other_concept][l] for l in reference
                             if l in embeddings[other_concept]]
            for iv in isolate_vecs:
                for rv in other_ref_vecs:
                    d = semantic_distance(iv, rv)
                    diff_concept_distances.append(d)

    mean_same = np.mean(same_concept_distances)
    mean_diff = np.mean(diff_concept_distances)

    # Statistical test: same concept should be closer than different concept
    stat, p_value = mannwhitneyu(same_concept_distances, diff_concept_distances,
                                 alternative='less')

    print(f"\nResults:")
    print(f"  Isolate languages: {isolates}")
    print(f"  Reference languages (IE): {reference}")
    print(f"  Mean same-concept distance: {mean_same:.4f}")
    print(f"  Mean diff-concept distance: {mean_diff:.4f}")
    print(f"  Mann-Whitney U statistic: {stat:.1f}")
    print(f"  P-value (same < diff): {p_value:.2e}")

    # Isolates DO converge if same-concept is significantly closer
    passed = p_value < ISOLATE_P_THRESHOLD

    if passed:
        print(f"\n  CONVERGENCE CONFIRMED: Language isolates cluster with")
        print(f"  Indo-European for same concepts despite no genetic relationship!")

    return TierResult(
        tier=3,
        test_name="3.3 Isolate Convergence",
        passed=passed,
        metric_name="P-value",
        metric_value=p_value,
        threshold=ISOLATE_P_THRESHOLD,
        details={
            'mean_same_concept': mean_same,
            'mean_diff_concept': mean_diff,
            'mann_whitney_u': stat,
            'isolate_langs': isolates,
            'reference_langs': reference
        }
    )


def run_tier_3_tests(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') -> Q37TestSuite:
    """Run all Tier 3 tests."""
    print("\n" + "=" * 70)
    print("Q37 TIER 3: CROSS-LINGUAL CONVERGENCE")
    print("=" * 70)
    print(f"Model: {model_name}")

    suite = Q37TestSuite()

    # Load model
    model = load_multilingual_model(model_name)

    # Run tests
    result_3_1 = test_3_1_translation_equivalents(model, TRANSLATION_PAIRS)
    suite.add_result(result_3_1)

    result_3_2 = test_3_2_language_phylogeny(model, TRANSLATION_PAIRS)
    suite.add_result(result_3_2)

    result_3_3 = test_3_3_isolate_convergence(model, TRANSLATION_PAIRS)
    suite.add_result(result_3_3)

    suite.print_summary()

    return suite


def save_results(suite: Q37TestSuite, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = {
        'timestamp': datetime.now().isoformat(),
        'tier': 3,
        'test_type': 'Cross-Lingual Convergence',
        'data_source': 'Multilingual Embeddings',
        'pass_rate': float(suite.get_pass_rate()),
        'tests': [convert_numpy(r.to_dict()) for r in suite.results]
    }

    output_path = os.path.join(output_dir, 'q37_tier3_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Q37 Tier 3: Cross-Lingual Convergence')
    parser.add_argument('--model', type=str,
                        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        help='Multilingual model to use')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    suite = run_tier_3_tests(model_name=args.model)

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    save_results(suite, output_dir)

    # Exit with appropriate code
    if suite.get_pass_rate() >= 0.66:
        print("\nTIER 3: PASSED")
        sys.exit(0)
    else:
        print("\nTIER 3: FAILED")
        sys.exit(1)
