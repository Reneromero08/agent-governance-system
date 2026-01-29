#!/usr/bin/env python3
"""
Q25: What Determines Sigma? - REAL DATA VERSION

PRE-REGISTRATION:
1. HYPOTHESIS: Sigma is predictable from dataset properties (R^2 > 0.7)
2. PREDICTION: sigma = f(D, entropy, N, domain properties)
3. FALSIFICATION: If R^2 < 0.5, sigma is irreducibly empirical
4. DATA: 10+ REAL external datasets:
   - HuggingFace: STS-B, SST-2, AG News, IMDB, etc.
   - GEO gene expression (via public APIs)
   - Market data (yfinance)
   - NLP benchmarks (SNLI, MNLI)
5. THRESHOLD: Find predictive formula or prove none exists

METHODOLOGY:
1. Load 10+ real datasets from different domains
2. Generate embeddings using sentence-transformers
3. For each dataset, compute optimal sigma via grid search
4. Measure: dimensionality (Df), entropy (H), sample size (N), mean_dist
5. Fit regression: sigma ~ properties
6. Report cross-validated R^2

NO SYNTHETIC DATA. All datasets must be real external data.

Author: Claude Opus 4.5
Date: 2026-01-27
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# CONFIGURATION
# =============================================================================

R2_PREDICTABLE_THRESHOLD = 0.7
R2_IRREDUCIBLE_THRESHOLD = 0.5

SIGMA_MIN = 0.001
SIGMA_MAX = 100.0
SIGMA_STEPS = 100

N_BOOTSTRAP = 30

N_FOLDS = 5

MAX_SAMPLES_PER_DATASET = 500  # Limit for efficiency


@dataclass
class DatasetProperties:
    """Properties of a dataset."""
    name: str
    domain: str
    source: str  # Where the data came from
    n_samples: int
    n_dimensions: int

    # Computed properties
    entropy: float = 0.0
    effective_dim: float = 0.0
    mean_pairwise_distance: float = 0.0
    std_pairwise_distance: float = 0.0
    eigenvalue_ratio: float = 0.0
    mean_norm: float = 0.0
    std_norm: float = 0.0
    intrinsic_scale: float = 0.0

    # Optimal sigma
    optimal_sigma: float = 0.0
    optimal_R_cv: float = 0.0
    optimal_R_mean: float = 0.0


@dataclass
class RegressionResult:
    """Result of regression analysis."""
    r2_train: float
    r2_cv: float
    coefficients: Dict[str, float]
    formula: str
    residual_std: float
    predictions: List[float]
    actuals: List[float]
    best_features: List[str]


def to_builtin(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj
    return obj


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def compute_entropy(embeddings: np.ndarray) -> float:
    """Compute entropy from covariance eigenspectrum."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    probs = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return float(entropy)


def compute_effective_dim(embeddings: np.ndarray) -> float:
    """Compute effective dimensionality (participation ratio)."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)

    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)
    return float(Df)


def compute_eigenvalue_ratio(embeddings: np.ndarray) -> float:
    """Compute ratio of top eigenvalue to sum (concentration)."""
    centered = embeddings - embeddings.mean(axis=0)
    n = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    return float(eigenvalues[0] / np.sum(eigenvalues))


def compute_pairwise_stats(embeddings: np.ndarray, max_pairs: int = 5000) -> Tuple[float, float]:
    """Compute mean and std of pairwise distances."""
    n = len(embeddings)

    if n * (n - 1) // 2 > max_pairs:
        indices = np.random.choice(n, size=int(np.sqrt(max_pairs * 2)), replace=False)
        embeddings = embeddings[indices]
        n = len(embeddings)

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(d)

    if len(distances) == 0:
        return 0.0, 0.0

    distances = np.array(distances)
    return float(np.mean(distances)), float(np.std(distances))


# =============================================================================
# R COMPUTATION
# =============================================================================

def compute_R_for_sigma(embeddings: np.ndarray, sigma: float) -> float:
    """Compute R score from embeddings with given sigma."""
    if len(embeddings) < 2:
        return 0.0

    centroid = embeddings.mean(axis=0)
    errors = np.linalg.norm(embeddings - centroid, axis=1)
    z = errors / (sigma + 1e-10)
    E_values = np.exp(-0.5 * z ** 2)

    return float(np.mean(E_values))


def compute_R_bootstrap(embeddings: np.ndarray, sigma: float,
                        n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
    """Compute R mean and CV across bootstrap resamples."""
    n = len(embeddings)
    R_values = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        resampled = embeddings[indices]
        R = compute_R_for_sigma(resampled, sigma)
        R_values.append(R)

    R_values = np.array(R_values)
    mean_R = np.mean(R_values)
    std_R = np.std(R_values)

    if mean_R < 1e-10:
        return 0.0, 1.0

    cv = std_R / mean_R
    return float(mean_R), float(cv)


def find_optimal_sigma(embeddings: np.ndarray,
                       sigma_min: float = SIGMA_MIN,
                       sigma_max: float = SIGMA_MAX,
                       n_steps: int = SIGMA_STEPS,
                       verbose: bool = False) -> Tuple[float, float, float]:
    """Find optimal sigma via grid search."""
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_steps)

    best_sigma = sigmas[0]
    best_cv = float('inf')
    best_mean_R = 0.0

    sweet_spot_sigmas = []

    for sigma in sigmas:
        mean_R, cv = compute_R_bootstrap(embeddings, sigma, n_bootstrap=10)

        if 0.1 < mean_R < 0.95:
            sweet_spot_sigmas.append((sigma, cv, mean_R))

    if len(sweet_spot_sigmas) == 0:
        for sigma in sigmas:
            mean_R, cv = compute_R_bootstrap(embeddings, sigma, n_bootstrap=10)
            if abs(mean_R - 0.5) < abs(best_mean_R - 0.5):
                best_sigma = sigma
                best_cv = cv
                best_mean_R = mean_R
    else:
        for sigma, cv, mean_R in sweet_spot_sigmas:
            if cv < best_cv:
                best_sigma = sigma
                best_cv = cv
                best_mean_R = mean_R

    _, final_cv = compute_R_bootstrap(embeddings, best_sigma, n_bootstrap=N_BOOTSTRAP)

    if verbose:
        print(f"    Optimal sigma: {best_sigma:.6f}, CV: {final_cv:.4f}, R: {best_mean_R:.4f}")

    return float(best_sigma), float(final_cv), float(best_mean_R)


# =============================================================================
# REAL DATA LOADERS
# =============================================================================

def load_embedding_model():
    """Load the sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        return None


def load_stsb_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load STS-B dataset embeddings."""
    try:
        from datasets import load_dataset
        print("  Loading STS-B from HuggingFace...")
        dataset = load_dataset('mteb/stsbenchmark-sts', split='test', trust_remote_code=True)

        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)

        sentences = [d['sentence1'] for d in dataset] + [d['sentence2'] for d in dataset]
        np.random.shuffle(sentences)
        sentences = sentences[:max_samples]

        embeddings = model.encode(sentences, normalize_embeddings=True)
        return embeddings, "nlp_similarity", "HuggingFace/mteb/stsbenchmark-sts"
    except Exception as e:
        print(f"  Failed to load STS-B: {e}")
        return None


def load_sst2_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load SST-2 sentiment dataset embeddings."""
    try:
        from datasets import load_dataset
        print("  Loading SST-2 from HuggingFace...")
        dataset = load_dataset('stanfordnlp/sst2', split='validation')

        sentences = [d['sentence'] for d in dataset]
        if len(sentences) > max_samples:
            sentences = list(np.random.choice(sentences, max_samples, replace=False))

        embeddings = model.encode(sentences, normalize_embeddings=True)
        return embeddings, "nlp_sentiment", "HuggingFace/stanfordnlp/sst2"
    except Exception as e:
        print(f"  Failed to load SST-2: {e}")
        return None


def load_ag_news_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load AG News topic classification dataset."""
    try:
        from datasets import load_dataset
        print("  Loading AG News from HuggingFace...")
        dataset = load_dataset('fancyzhx/ag_news', split='test')

        texts = [d['text'] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "nlp_news", "HuggingFace/fancyzhx/ag_news"
    except Exception as e:
        print(f"  Failed to load AG News: {e}")
        return None


def load_imdb_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load IMDB movie review dataset."""
    try:
        from datasets import load_dataset
        print("  Loading IMDB from HuggingFace...")
        dataset = load_dataset('stanfordnlp/imdb', split='test')

        texts = [d['text'][:512] for d in dataset]  # Truncate long reviews
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "nlp_reviews", "HuggingFace/stanfordnlp/imdb"
    except Exception as e:
        print(f"  Failed to load IMDB: {e}")
        return None


def load_snli_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load SNLI natural language inference dataset."""
    try:
        from datasets import load_dataset
        print("  Loading SNLI from HuggingFace...")
        dataset = load_dataset('stanfordnlp/snli', split='validation')

        # Combine premise and hypothesis
        sentences = []
        for d in dataset:
            if d['label'] != -1:  # Skip unlabeled
                sentences.append(d['premise'])
                sentences.append(d['hypothesis'])

        if len(sentences) > max_samples:
            sentences = list(np.random.choice(sentences, max_samples, replace=False))

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "nlp_inference", "HuggingFace/stanfordnlp/snli"
    except Exception as e:
        print(f"  Failed to load SNLI: {e}")
        return None


def load_mnli_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load MNLI (Multi-Genre NLI) dataset."""
    try:
        from datasets import load_dataset
        print("  Loading MNLI from HuggingFace...")
        dataset = load_dataset('nyu-mll/glue', 'mnli', split='validation_matched')

        sentences = []
        for d in dataset:
            sentences.append(d['premise'])
            sentences.append(d['hypothesis'])

        if len(sentences) > max_samples:
            sentences = list(np.random.choice(sentences, max_samples, replace=False))

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "nlp_multi_genre", "HuggingFace/nyu-mll/glue/mnli"
    except Exception as e:
        print(f"  Failed to load MNLI: {e}")
        return None


def load_emotion_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load emotion classification dataset."""
    try:
        from datasets import load_dataset
        print("  Loading Emotion dataset from HuggingFace...")
        dataset = load_dataset('dair-ai/emotion', split='test')

        texts = [d['text'] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "nlp_emotion", "HuggingFace/dair-ai/emotion"
    except Exception as e:
        print(f"  Failed to load Emotion: {e}")
        return None


def load_tweet_eval_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load TweetEval sentiment dataset."""
    try:
        from datasets import load_dataset
        print("  Loading TweetEval from HuggingFace...")
        dataset = load_dataset('cardiffnlp/tweet_eval', 'sentiment', split='test')

        texts = [d['text'] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "social_media", "HuggingFace/cardiffnlp/tweet_eval"
    except Exception as e:
        print(f"  Failed to load TweetEval: {e}")
        return None


def load_financial_phrasebank_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load Financial PhraseBank dataset."""
    try:
        from datasets import load_dataset
        print("  Loading Financial PhraseBank from HuggingFace...")
        dataset = load_dataset('takala/financial_phrasebank', 'sentences_allagree', split='train')

        texts = [d['sentence'] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "finance_text", "HuggingFace/takala/financial_phrasebank"
    except Exception as e:
        print(f"  Failed to load Financial PhraseBank: {e}")
        return None


def load_yfinance_data(max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load market data from yfinance and create embeddings from price features."""
    try:
        import yfinance as yf
        print("  Loading market data from yfinance...")

        # Multiple tickers for diversity
        tickers = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

        all_features = []

        for ticker in tickers:
            try:
                data = yf.download(ticker, period='2y', interval='1d', progress=False)
                if len(data) < 50:
                    continue

                # Create feature vectors from rolling windows
                for i in range(30, len(data) - 10, 5):
                    window = data.iloc[i-30:i]

                    # Features: returns, volatility, volume changes
                    if 'Adj Close' in window.columns:
                        close = window['Adj Close'].values
                    else:
                        close = window['Close'].values

                    returns = np.diff(close) / close[:-1]

                    features = [
                        np.mean(returns),
                        np.std(returns),
                        np.min(returns),
                        np.max(returns),
                        returns[-1] if len(returns) > 0 else 0,
                        np.percentile(returns, 25) if len(returns) > 0 else 0,
                        np.percentile(returns, 75) if len(returns) > 0 else 0,
                        np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.5,
                    ]

                    # Add volume features if available
                    if 'Volume' in window.columns:
                        vol = window['Volume'].values
                        vol_changes = np.diff(vol) / (vol[:-1] + 1)
                        features.extend([
                            np.mean(vol_changes),
                            np.std(vol_changes),
                        ])
                    else:
                        features.extend([0, 0])

                    all_features.append(features)
            except Exception:
                continue

        if len(all_features) < 50:
            print("  Not enough market data collected")
            return None

        embeddings = np.array(all_features)

        # Normalize
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)

        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]

        return embeddings, "market_data", "yfinance/multi_ticker"
    except ImportError:
        print("  yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"  Failed to load yfinance data: {e}")
        return None


def load_geo_gene_expression() -> Optional[Tuple[np.ndarray, str, str]]:
    """
    Load gene expression data from GEO (Gene Expression Omnibus).
    Uses a publicly available dataset.
    """
    try:
        import urllib.request
        print("  Loading GEO gene expression data...")

        # GEO accession: GSE45267 (Liver cancer study - publicly available)
        # This is a small, well-curated dataset
        url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE45nnn/GSE45267/matrix/GSE45267_series_matrix.txt.gz"

        import gzip
        import io

        # Download and decompress
        print("    Downloading from NCBI GEO...")
        response = urllib.request.urlopen(url, timeout=30)
        compressed = response.read()
        decompressed = gzip.decompress(compressed).decode('utf-8')

        # Parse the matrix file
        lines = decompressed.split('\n')
        data_start = False
        samples = []
        gene_data = []

        for line in lines:
            if line.startswith('!series_matrix_table_begin'):
                data_start = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if data_start and line.strip():
                parts = line.split('\t')
                if parts[0] == '"ID_REF"':
                    samples = [p.strip('"') for p in parts[1:]]
                elif parts[0].startswith('"'):
                    try:
                        values = [float(p) if p and p != 'null' else 0.0 for p in parts[1:]]
                        if len(values) == len(samples):
                            gene_data.append(values)
                    except ValueError:
                        continue

        if len(gene_data) < 100:
            print("    Not enough gene expression data")
            return None

        # Transpose: samples as rows, genes as columns
        embeddings = np.array(gene_data).T

        # Select top varying genes for efficiency
        variances = np.var(embeddings, axis=0)
        top_indices = np.argsort(variances)[-200:]
        embeddings = embeddings[:, top_indices]

        # Normalize
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)

        print(f"    Loaded {embeddings.shape[0]} samples x {embeddings.shape[1]} genes")
        return embeddings, "gene_expression", "NCBI_GEO/GSE45267"

    except Exception as e:
        print(f"  Failed to load GEO data: {e}")
        return None


def load_wikipedia_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load Wikipedia text data."""
    try:
        from datasets import load_dataset
        print("  Loading Wikipedia data from HuggingFace...")
        dataset = load_dataset('wikipedia', '20220301.simple', split='train', trust_remote_code=True)

        # Sample texts
        indices = np.random.choice(len(dataset), min(max_samples * 2, len(dataset)), replace=False)
        texts = []
        for idx in indices:
            text = dataset[int(idx)]['text'][:500]  # First 500 chars
            if len(text) > 100:
                texts.append(text)
            if len(texts) >= max_samples:
                break

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "encyclopedia", "HuggingFace/wikipedia"
    except Exception as e:
        print(f"  Failed to load Wikipedia: {e}")
        return None


def load_squad_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load SQuAD question answering dataset."""
    try:
        from datasets import load_dataset
        print("  Loading SQuAD from HuggingFace...")
        dataset = load_dataset('rajpurkar/squad', split='validation')

        # Mix questions and contexts
        texts = []
        for d in dataset:
            texts.append(d['question'])
            texts.append(d['context'][:300])

        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "qa", "HuggingFace/rajpurkar/squad"
    except Exception as e:
        print(f"  Failed to load SQuAD: {e}")
        return None


def load_yelp_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load Yelp review dataset."""
    try:
        from datasets import load_dataset
        print("  Loading Yelp reviews from HuggingFace...")
        dataset = load_dataset('Yelp/yelp_review_full', split='test')

        texts = [d['text'][:512] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "review", "HuggingFace/Yelp/yelp_review_full"
    except Exception as e:
        print(f"  Failed to load Yelp: {e}")
        return None


def load_amazon_polarity_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load Amazon polarity dataset."""
    try:
        from datasets import load_dataset
        print("  Loading Amazon polarity from HuggingFace...")
        dataset = load_dataset('amazon_polarity', split='test')

        texts = [(d['title'] + " " + d['content'])[:512] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "ecommerce", "HuggingFace/amazon_polarity"
    except Exception as e:
        print(f"  Failed to load Amazon: {e}")
        return None


def load_wikitext_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load WikiText-2 dataset for general language modeling text."""
    try:
        from datasets import load_dataset
        print("  Loading WikiText-2 from HuggingFace...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split='test')

        # Filter and collect non-empty paragraphs
        texts = []
        for d in dataset:
            text = d['text'].strip()
            if len(text) > 100 and not text.startswith('='):
                texts.append(text[:500])
            if len(texts) >= max_samples:
                break

        if len(texts) < 100:
            print("    Not enough WikiText samples")
            return None

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "wikipedia", "HuggingFace/wikitext"
    except Exception as e:
        print(f"  Failed to load WikiText: {e}")
        return None


def load_multi_news_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load Multi-News summarization dataset."""
    try:
        from datasets import load_dataset
        print("  Loading Multi-News from HuggingFace...")
        dataset = load_dataset('alexfabbri/multi_news', split='test')

        texts = [d['document'][:512] for d in dataset]
        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "news_long", "HuggingFace/alexfabbri/multi_news"
    except Exception as e:
        print(f"  Failed to load Multi-News: {e}")
        return None


def load_sciq_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load SciQ science questions dataset."""
    try:
        from datasets import load_dataset
        print("  Loading SciQ from HuggingFace...")
        dataset = load_dataset('allenai/sciq', split='test')

        # Combine questions with correct answers
        texts = []
        for d in dataset:
            texts.append(d['question'] + " " + d['correct_answer'])
            if d['support']:
                texts.append(d['support'][:300])

        if len(texts) > max_samples:
            texts = list(np.random.choice(texts, max_samples, replace=False))

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "science_qa", "HuggingFace/allenai/sciq"
    except Exception as e:
        print(f"  Failed to load SciQ: {e}")
        return None


def load_mrpc_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load MRPC paraphrase dataset."""
    try:
        from datasets import load_dataset
        print("  Loading MRPC from HuggingFace...")
        dataset = load_dataset('nyu-mll/glue', 'mrpc', split='test')

        sentences = []
        for d in dataset:
            sentences.append(d['sentence1'])
            sentences.append(d['sentence2'])

        if len(sentences) > max_samples:
            sentences = list(np.random.choice(sentences, max_samples, replace=False))

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "paraphrase", "HuggingFace/nyu-mll/glue/mrpc"
    except Exception as e:
        print(f"  Failed to load MRPC: {e}")
        return None


def load_rte_data(model, max_samples: int = MAX_SAMPLES_PER_DATASET) -> Optional[Tuple[np.ndarray, str, str]]:
    """Load RTE entailment dataset."""
    try:
        from datasets import load_dataset
        print("  Loading RTE from HuggingFace...")
        dataset = load_dataset('nyu-mll/glue', 'rte', split='validation')

        sentences = []
        for d in dataset:
            sentences.append(d['sentence1'])
            sentences.append(d['sentence2'])

        if len(sentences) > max_samples:
            sentences = list(np.random.choice(sentences, max_samples, replace=False))

        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        return embeddings, "entailment", "HuggingFace/nyu-mll/glue/rte"
    except Exception as e:
        print(f"  Failed to load RTE: {e}")
        return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset(embeddings: np.ndarray, name: str, domain: str, source: str,
                    verbose: bool = True) -> DatasetProperties:
    """Compute all properties of a dataset."""
    n_samples, n_dims = embeddings.shape

    entropy = compute_entropy(embeddings)
    effective_dim = compute_effective_dim(embeddings)
    eigenvalue_ratio = compute_eigenvalue_ratio(embeddings)
    mean_dist, std_dist = compute_pairwise_stats(embeddings)

    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))

    intrinsic_scale = std_dist / np.sqrt(n_dims) if n_dims > 0 else 0.0

    optimal_sigma, optimal_cv, optimal_mean_R = find_optimal_sigma(
        embeddings, verbose=verbose
    )

    return DatasetProperties(
        name=name,
        domain=domain,
        source=source,
        n_samples=n_samples,
        n_dimensions=n_dims,
        entropy=entropy,
        effective_dim=effective_dim,
        mean_pairwise_distance=mean_dist,
        std_pairwise_distance=std_dist,
        eigenvalue_ratio=eigenvalue_ratio,
        mean_norm=mean_norm,
        std_norm=std_norm,
        intrinsic_scale=intrinsic_scale,
        optimal_sigma=optimal_sigma,
        optimal_R_cv=optimal_cv,
        optimal_R_mean=optimal_mean_R
    )


def build_feature_matrix(datasets: List[DatasetProperties],
                         features: List[str]) -> np.ndarray:
    """Build feature matrix from dataset properties."""
    X = []
    for ds in datasets:
        row = []
        for f in features:
            if f == "log_n_samples":
                row.append(np.log(ds.n_samples + 1))
            elif f == "log_n_dimensions":
                row.append(np.log(ds.n_dimensions + 1))
            elif f == "entropy":
                row.append(ds.entropy)
            elif f == "effective_dim":
                row.append(ds.effective_dim)
            elif f == "log_effective_dim":
                row.append(np.log(ds.effective_dim + 1))
            elif f == "mean_pairwise_distance":
                row.append(ds.mean_pairwise_distance)
            elif f == "std_pairwise_distance":
                row.append(ds.std_pairwise_distance)
            elif f == "eigenvalue_ratio":
                row.append(ds.eigenvalue_ratio)
            elif f == "mean_norm":
                row.append(ds.mean_norm)
            elif f == "std_norm":
                row.append(ds.std_norm)
            elif f == "intrinsic_scale":
                row.append(ds.intrinsic_scale)
            elif f == "log_intrinsic_scale":
                row.append(np.log(ds.intrinsic_scale + 1e-6))
            elif f == "log_mean_dist":
                row.append(np.log(ds.mean_pairwise_distance + 1e-6))
            elif f == "log_std_dist":
                row.append(np.log(ds.std_pairwise_distance + 1e-6))
            else:
                row.append(0.0)
        X.append(row)

    return np.array(X)


def fit_linear_regression(X: np.ndarray, y: np.ndarray,
                          feature_names: List[str]) -> Tuple[np.ndarray, float]:
    """Fit linear regression and return coefficients and R^2."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    try:
        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coeffs = np.zeros(X_with_intercept.shape[1])

    predictions = X_with_intercept @ coeffs
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0

    return coeffs, float(r2)


def cross_validate_regression(X: np.ndarray, y: np.ndarray,
                               n_folds: int = N_FOLDS) -> float:
    """
    Cross-validated R^2 with leave-one-out for small samples.

    For small datasets (n < 10), use leave-one-out cross-validation.
    For larger datasets, use k-fold.
    """
    n = len(y)

    if n < 4:
        return 0.0  # Too few samples for any CV

    # Use leave-one-out for small samples
    if n < 10:
        predictions_loo = np.zeros(n)

        for i in range(n):
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[i:i+1]

            X_train_int = np.column_stack([np.ones(len(X_train)), X_train])
            try:
                coeffs = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]
            except np.linalg.LinAlgError:
                coeffs = np.zeros(X_train_int.shape[1])

            X_test_int = np.column_stack([np.ones(1), X_test])
            predictions_loo[i] = X_test_int @ coeffs

        ss_res = np.sum((y - predictions_loo) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
        return float(max(0.0, r2))

    # Standard k-fold for larger samples
    fold_size = n // n_folds
    r2_values = []

    np.random.seed(42)
    indices = np.random.permutation(n)

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        X_train_int = np.column_stack([np.ones(len(X_train)), X_train])
        try:
            coeffs = np.linalg.lstsq(X_train_int, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X_train_int.shape[1])

        X_test_int = np.column_stack([np.ones(len(X_test)), X_test])
        predictions = X_test_int @ coeffs

        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)

        r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
        r2_values.append(max(0.0, r2))

    return float(np.mean(r2_values))


def analyze_sigma_predictability(datasets: List[DatasetProperties],
                                  verbose: bool = True) -> RegressionResult:
    """Main analysis: can sigma be predicted from dataset properties?"""
    y_raw = np.array([ds.optimal_sigma for ds in datasets])

    if np.std(y_raw) < 1e-10:
        print("\nWARNING: No variance in sigma values!")
        return RegressionResult(
            r2_train=0.0,
            r2_cv=0.0,
            coefficients={"intercept": float(np.mean(y_raw))},
            formula=f"sigma = {np.mean(y_raw):.4f} (constant)",
            residual_std=0.0,
            predictions=y_raw.tolist(),
            actuals=y_raw.tolist(),
            best_features=[]
        )

    y = np.log(y_raw + 1e-10)

    feature_sets = [
        ["log_mean_dist", "log_std_dist"],
        ["intrinsic_scale", "log_intrinsic_scale"],
        ["mean_pairwise_distance", "std_pairwise_distance"],
        ["log_n_samples", "log_n_dimensions"],
        ["log_effective_dim", "eigenvalue_ratio"],
        ["entropy", "effective_dim", "eigenvalue_ratio"],
        ["log_mean_dist", "log_effective_dim", "eigenvalue_ratio"],
        ["intrinsic_scale", "entropy", "log_n_dimensions"],
        ["log_n_samples", "log_n_dimensions", "entropy", "log_effective_dim",
         "log_mean_dist", "log_std_dist", "eigenvalue_ratio", "intrinsic_scale"]
    ]

    best_r2_cv = -np.inf
    best_features = None
    best_coeffs = None
    best_r2_train = 0.0
    best_X = None

    if verbose:
        print("\n--- Feature Set Comparison ---")
        print(f"{'Features':<55} {'R2_train':>10} {'R2_cv':>10}")
        print("-" * 75)

    for features in feature_sets:
        X = build_feature_matrix(datasets, features)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_scaled = (X - X_mean) / X_std

        coeffs, r2_train = fit_linear_regression(X_scaled, y, features)
        r2_cv = cross_validate_regression(X_scaled, y)

        if verbose:
            feat_str = ", ".join(features[:3]) + ("..." if len(features) > 3 else "")
            print(f"{feat_str:<55} {r2_train:>10.4f} {r2_cv:>10.4f}")

        if r2_cv > best_r2_cv:
            best_r2_cv = r2_cv
            best_r2_train = r2_train
            best_features = features
            best_coeffs = coeffs
            best_X = X_scaled

    coeff_dict = {"intercept": float(best_coeffs[0])}
    for i, f in enumerate(best_features):
        coeff_dict[f] = float(best_coeffs[i + 1])

    formula_parts = [f"log(sigma) = {best_coeffs[0]:.4f}"]
    for i, f in enumerate(best_features):
        sign = "+" if best_coeffs[i + 1] >= 0 else ""
        formula_parts.append(f"{sign}{best_coeffs[i + 1]:.4f}*{f}")
    formula = " ".join(formula_parts)

    X_with_int = np.column_stack([np.ones(len(best_X)), best_X])
    log_predictions = X_with_int @ best_coeffs
    predictions = np.exp(log_predictions)

    residuals = y - log_predictions
    residual_std = float(np.std(residuals))

    return RegressionResult(
        r2_train=best_r2_train,
        r2_cv=best_r2_cv,
        coefficients=coeff_dict,
        formula=formula,
        residual_std=residual_std,
        predictions=predictions.tolist(),
        actuals=y_raw.tolist(),
        best_features=best_features
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    """Run the full Q25 experiment with REAL DATA."""

    if verbose:
        print("=" * 80)
        print("Q25: WHAT DETERMINES SIGMA? (REAL DATA VERSION)")
        print("=" * 80)
        print("\nPRE-REGISTRATION:")
        print(f"  HYPOTHESIS: Sigma predictable from dataset properties (R^2 > {R2_PREDICTABLE_THRESHOLD})")
        print(f"  FALSIFICATION: If R^2 < {R2_IRREDUCIBLE_THRESHOLD}, sigma is irreducibly empirical")
        print(f"  DATA: 10+ REAL external datasets (NO SYNTHETIC DATA)")
        print(f"  SOURCES: HuggingFace, GEO, yfinance")

    # Load embedding model
    model = load_embedding_model()
    if model is None:
        return {"error": "Could not load embedding model"}

    # Define data loaders - ordered by reliability
    loaders = [
        # Most reliable HuggingFace datasets
        ("stsb", lambda: load_stsb_data(model)),
        ("snli", lambda: load_snli_data(model)),
        ("mnli", lambda: load_mnli_data(model)),
        ("squad", lambda: load_squad_data(model)),
        ("mrpc", lambda: load_mrpc_data(model)),
        ("rte", lambda: load_rte_data(model)),
        ("sciq", lambda: load_sciq_data(model)),
        ("wikitext", lambda: load_wikitext_data(model)),

        # Additional HuggingFace datasets
        ("sst2", lambda: load_sst2_data(model)),
        ("ag_news", lambda: load_ag_news_data(model)),
        ("imdb", lambda: load_imdb_data(model)),
        ("emotion", lambda: load_emotion_data(model)),
        ("tweet_eval", lambda: load_tweet_eval_data(model)),
        ("financial_phrasebank", lambda: load_financial_phrasebank_data(model)),
        ("yelp", lambda: load_yelp_data(model)),
        ("multi_news", lambda: load_multi_news_data(model)),

        # Non-text data sources
        ("yfinance", lambda: load_yfinance_data()),
        ("geo_expression", lambda: load_geo_gene_expression()),
    ]

    if verbose:
        print(f"\n--- Loading {len(loaders)} real datasets ---")

    datasets = []
    loaded_sources = []

    for name, loader in loaders:
        if verbose:
            print(f"\n{name}:")

        try:
            result = loader()
            if result is not None:
                embeddings, domain, source = result
                props = analyze_dataset(embeddings, name, domain, source, verbose=verbose)
                datasets.append(props)
                loaded_sources.append(source)
                if verbose:
                    print(f"    Shape: {embeddings.shape}")
                    print(f"    Entropy: {props.entropy:.4f}, Effective dim: {props.effective_dim:.2f}")
                    print(f"    Mean pairwise dist: {props.mean_pairwise_distance:.4f}")
            else:
                if verbose:
                    print(f"    SKIPPED (data unavailable)")
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")

    if len(datasets) < 5:
        return {
            "error": f"Only loaded {len(datasets)} datasets, need at least 5",
            "loaded": [ds.name for ds in datasets]
        }

    # Show sigma distribution
    sigmas = [ds.optimal_sigma for ds in datasets]
    if verbose:
        print("\n" + "=" * 80)
        print("SIGMA DISTRIBUTION (REAL DATA)")
        print("=" * 80)
        print(f"  Datasets loaded: {len(datasets)}")
        print(f"  Min sigma: {min(sigmas):.6f}")
        print(f"  Max sigma: {max(sigmas):.6f}")
        print(f"  Mean sigma: {np.mean(sigmas):.6f}")
        print(f"  Std sigma: {np.std(sigmas):.6f}")
        print(f"  Range ratio: {max(sigmas)/min(sigmas):.2f}x")

    if verbose:
        print("\n" + "=" * 80)
        print("REGRESSION ANALYSIS")
        print("=" * 80)

    regression = analyze_sigma_predictability(datasets, verbose=verbose)

    # Determine verdict
    if regression.r2_cv >= R2_PREDICTABLE_THRESHOLD:
        verdict = "SIGMA_PREDICTABLE"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} >= {R2_PREDICTABLE_THRESHOLD}"
    elif regression.r2_cv < R2_IRREDUCIBLE_THRESHOLD:
        verdict = "SIGMA_IRREDUCIBLY_EMPIRICAL"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} < {R2_IRREDUCIBLE_THRESHOLD}"
    else:
        verdict = "SIGMA_PARTIALLY_PREDICTABLE"
        verdict_detail = f"R^2_cv = {regression.r2_cv:.4f} in [{R2_IRREDUCIBLE_THRESHOLD}, {R2_PREDICTABLE_THRESHOLD})"

    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS (REAL DATA)")
        print("=" * 80)
        print(f"\nBest R^2 (training): {regression.r2_train:.4f}")
        print(f"Best R^2 (cross-validated): {regression.r2_cv:.4f}")
        print(f"Best features: {regression.best_features}")
        print(f"Residual std (log-space): {regression.residual_std:.4f}")
        print(f"\nBest formula: {regression.formula}")

        print(f"\n--- Predicted vs Actual Sigma ---")
        print(f"{'Dataset':<25} {'Domain':<15} {'Actual':>12} {'Predicted':>12} {'Ratio':>10}")
        print("-" * 74)
        for i, ds in enumerate(datasets):
            ratio = regression.predictions[i] / ds.optimal_sigma if ds.optimal_sigma > 0 else 0
            print(f"{ds.name:<25} {ds.domain:<15} {ds.optimal_sigma:>12.4f} {regression.predictions[i]:>12.4f} {ratio:>10.2f}")

    # Domain summary
    domain_sigmas = {}
    for ds in datasets:
        if ds.domain not in domain_sigmas:
            domain_sigmas[ds.domain] = []
        domain_sigmas[ds.domain].append(ds.optimal_sigma)

    domain_summary = {}
    for domain, sigmas_list in domain_sigmas.items():
        domain_summary[domain] = {
            "mean_sigma": float(np.mean(sigmas_list)),
            "std_sigma": float(np.std(sigmas_list)) if len(sigmas_list) > 1 else 0.0,
            "n_datasets": len(sigmas_list)
        }

    if verbose:
        print(f"\n--- Domain Summary ---")
        for domain, stats in domain_summary.items():
            print(f"  {domain}: mean={stats['mean_sigma']:.4f}, std={stats['std_sigma']:.4f} (n={stats['n_datasets']})")

    # Final verdict
    if verbose:
        print("\n" + "=" * 80)
        print("VERDICT (REAL DATA)")
        print("=" * 80)
        print(f"\n{verdict}: {verdict_detail}")

        if verdict == "SIGMA_PREDICTABLE":
            print("\nSigma CAN be predicted from dataset properties!")
            print(f"Best predictive features: {regression.best_features}")
        elif verdict == "SIGMA_IRREDUCIBLY_EMPIRICAL":
            print("\nSigma CANNOT be reliably predicted from dataset properties.")
            print("It must be determined empirically for each dataset.")
        else:
            print("\nSigma is PARTIALLY predictable.")
            print("Some variance explained, but significant empirical component remains.")

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "Q25_what_determines_sigma_REAL_DATA",
        "data_type": "REAL_EXTERNAL_DATA",
        "pre_registration": {
            "hypothesis": f"Sigma predictable (R^2 > {R2_PREDICTABLE_THRESHOLD})",
            "falsification": f"Sigma irreducibly empirical (R^2 < {R2_IRREDUCIBLE_THRESHOLD})",
            "data_sources": loaded_sources
        },
        "n_datasets": len(datasets),
        "sigma_distribution": {
            "min": float(min(sigmas)),
            "max": float(max(sigmas)),
            "mean": float(np.mean(sigmas)),
            "std": float(np.std(sigmas)),
            "range_ratio": float(max(sigmas)/min(sigmas)) if min(sigmas) > 0 else None
        },
        "datasets": [],
        "regression": {
            "r2_train": regression.r2_train,
            "r2_cv": regression.r2_cv,
            "best_features": regression.best_features,
            "coefficients": regression.coefficients,
            "formula": regression.formula,
            "residual_std": regression.residual_std,
            "predictions": regression.predictions,
            "actuals": regression.actuals
        },
        "domain_summary": domain_summary,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "passes_hypothesis": regression.r2_cv >= R2_PREDICTABLE_THRESHOLD,
        "falsified": regression.r2_cv < R2_IRREDUCIBLE_THRESHOLD
    }

    for ds in datasets:
        output["datasets"].append({
            "name": ds.name,
            "domain": ds.domain,
            "source": ds.source,
            "n_samples": ds.n_samples,
            "n_dimensions": ds.n_dimensions,
            "entropy": ds.entropy,
            "effective_dim": ds.effective_dim,
            "eigenvalue_ratio": ds.eigenvalue_ratio,
            "mean_pairwise_distance": ds.mean_pairwise_distance,
            "std_pairwise_distance": ds.std_pairwise_distance,
            "mean_norm": ds.mean_norm,
            "std_norm": ds.std_norm,
            "intrinsic_scale": ds.intrinsic_scale,
            "optimal_sigma": ds.optimal_sigma,
            "optimal_R_cv": ds.optimal_R_cv,
            "optimal_R_mean": ds.optimal_R_mean
        })

    return to_builtin(output)


def main():
    """Main entry point."""
    results = run_experiment(verbose=True)

    # Save results
    output_path = Path(__file__).parent / "q25_real_data_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
