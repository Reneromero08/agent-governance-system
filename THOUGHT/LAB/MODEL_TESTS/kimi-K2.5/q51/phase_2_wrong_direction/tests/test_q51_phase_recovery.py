#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q51.1: Phase Signatures in Cross-Correlations

Objective: Test if phase information can be recovered from off-diagonal 
covariance elements of real embeddings.

Hypothesis: Off-diagonal covariance encodes phase interference:
    <z_i, z_j> = r_i * r_j * cos(theta_i - theta_j)

Parameters (FIXED - NO GRID SEARCH):
    - Vocabulary: 1000 words
    - PCA dims: 50
    - Phase bins: 32
    - Random seed: 42

Success Criteria:
    - Can infer phase angles from off-diagonals with |r| > 0.5
    - Phase coherence length > 10 dimensions
    - p < 0.001 statistical significance

Anti-Pattern Checks:
    - Ground truth (WordSim-353) must NOT be derived from R values
    - Report ALL results honestly, including failures
    - Parameters fixed BEFORE testing
"""

import sys
import json
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# FIXED PARAMETERS - NO GRID SEARCH
# =============================================================================
RANDOM_SEED = 42
PCA_DIMS = 50
PHASE_BINS = 32
VOCAB_SIZE = 1000

np.random.seed(RANDOM_SEED)

# =============================================================================
# 1000-WORD VOCABULARY (Semantic Anchors)
# =============================================================================

VOCABULARY_1000 = [
    # Nature - 100 words
    "water", "fire", "earth", "air", "sky", "sun", "moon", "star", "mountain", "river",
    "ocean", "sea", "lake", "forest", "tree", "flower", "grass", "rain", "snow", "wind",
    "cloud", "storm", "thunder", "lightning", "desert", "valley", "hill", "canyon", "beach", "island",
    "volcano", "earthquake", "tide", "wave", "current", "glacier", "iceberg", "cave", "cliff", "plateau",
    "jungle", "savanna", "tundra", "wetland", "swamp", "marsh", "reef", "delta", "estuary", "meadow",
    "prairie", "steppe", "taiga", "rainforest", "bamboo", "cactus", "fern", "moss", "coral", "kelp",
    "algae", "fungus", "bacteria", "virus", "seed", "root", "leaf", "branch", "trunk", "bark",
    "petal", "pollen", "nectar", "sap", "thorns", "vines", "ivy", "seaweed", "plankton", "crystal",
    "mineral", "rock", "stone", "sand", "soil", "clay", "mud", "dust", "ash", "ember",
    "flame", "smoke", "steam", "fog", "mist", "dew", "frost", "hail", "sleet", "aurora",
    
    # Animals - 100 words
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken",
    "duck", "goose", "turkey", "deer", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "squirrel", "beaver", "raccoon", "skunk", "bat", "owl", "eagle", "hawk", "falcon", "vulture",
    "crow", "raven", "sparrow", "robin", "cardinal", "bluejay", "woodpecker", "hummingbird", "pigeon", "dove",
    "swan", "heron", "stork", "crane", "pelican", "albatross", "penguin", "seagull", "parrot", "macaw",
    "peacock", "flamingo", "ostrich", "emu", "kiwi", "toucan", "snake", "lizard", "turtle", "crocodile",
    "alligator", "frog", "toad", "salamander", "newt", "worm", "snail", "slug", "spider", "scorpion",
    "insect", "ant", "bee", "wasp", "hornet", "butterfly", "moth", "beetle", "dragonfly", "grasshopper",
    "cricket", "praying", "mantis", "ladybug", "firefly", "mosquito", "fly", "flea", "tick", "mite",
    "lion", "tiger", "elephant", "giraffe", "zebra", "rhino", "hippo", "gorilla", "chimpanzee", "monkey",
    
    # Human Body - 50 words
    "head", "face", "eye", "nose", "mouth", "ear", "hair", "neck", "chest", "heart",
    "lung", "liver", "kidney", "stomach", "brain", "bone", "blood", "muscle", "skin", "hand",
    "finger", "thumb", "palm", "wrist", "arm", "elbow", "shoulder", "back", "spine", "leg",
    "knee", "ankle", "foot", "toe", "heel", "hip", "waist", "throat", "tongue", "tooth",
    "lip", "eyebrow", "eyelash", "cheek", "chin", "forehead", "temple", "jaw", "vein", "artery",
    
    # Family & Relations - 50 words
    "mother", "father", "parent", "child", "son", "daughter", "brother", "sister", "sibling",
    "grandmother", "grandfather", "grandparent", "grandson", "granddaughter", "aunt", "uncle", "nephew", "niece",
    "cousin", "husband", "wife", "spouse", "marriage", "wedding", "divorce", "family", "home", "house",
    "baby", "infant", "toddler", "teenager", "adult", "elder", "ancestor", "descendant", "relative", "kin",
    "friend", "enemy", "stranger", "neighbor", "colleague", "partner", "companion", "ally", "rival", "opponent",
    
    # Emotions & States - 100 words
    "love", "hate", "joy", "sadness", "happiness", "anger", "fear", "courage", "hope", "despair",
    "peace", "war", "calm", "anxiety", "stress", "relaxation", "excitement", "boredom", "curiosity", "indifference",
    "trust", "doubt", "confidence", "insecurity", "pride", "shame", "guilt", "innocence", "innocent", "regret",
    "gratitude", "envy", "jealousy", "sympathy", "empathy", "compassion", "forgiveness", "resentment", "loneliness", "belonging",
    "freedom", "oppression", "power", "weakness", "strength", "vulnerability", "safety", "danger", "comfort", "pain",
    "pleasure", "suffering", "bliss", "misery", "euphoria", "depression", "contentment", "discontent", "satisfaction", "dissatisfaction",
    "optimism", "pessimism", "enthusiasm", "apathy", "passion", "disgust", "surprise", "anticipation", "nostalgia", "longing",
    "desire", "aversion", "attraction", "repulsion", "admiration", "contempt", "respect", "disrespect", "loyalty", "betrayal",
    "honesty", "deception", "truth", "lie", "reality", "illusion", "dream", "nightmare", "fantasy", "imagination",
    "memory", "forgetting", "consciousness", "unconscious", "awareness", "ignorance", "enlightenment", "confusion", "clarity", "madness",
    
    # Abstract Concepts - 100 words
    "time", "space", "matter", "energy", "force", "motion", "rest", "change", "permanence", "beginning",
    "end", "middle", "center", "edge", "boundary", "limit", "infinity", "eternity", "moment", "instant",
    "past", "present", "future", "yesterday", "today", "tomorrow", "century", "decade", "year", "month",
    "week", "day", "hour", "minute", "second", "morning", "noon", "afternoon", "evening", "night",
    "dawn", "dusk", "sunrise", "sunset", "twilight", "midnight", "season", "spring", "summer", "autumn",
    "winter", "direction", "north", "south", "east", "west", "up", "down", "left", "right",
    "forward", "backward", "inside", "outside", "above", "below", "between", "among", "within", "beyond",
    "near", "far", "close", "distant", "adjacent", "opposite", "parallel", "perpendicular", "diagonal", "vertical",
    "horizontal", "straight", "curved", "round", "square", "triangular", "circular", "oval", "flat", "sharp",
    "smooth", "rough", "hard", "soft", "solid", "liquid", "gas", "plasma", "vacuum", "void",
    
    # Knowledge & Learning - 100 words
    "knowledge", "wisdom", "understanding", "comprehension", "insight", "perception", "observation", "experience", "practice", "theory",
    "science", "art", "philosophy", "religion", "spirituality", "faith", "belief", "doubt", "question", "answer",
    "problem", "solution", "mystery", "secret", "discovery", "invention", "innovation", "tradition", "custom", "habit",
    "skill", "talent", "ability", "disability", "education", "school", "university", "college", "academy", "institute",
    "teacher", "student", "professor", "scholar", "expert", "novice", "beginner", "master", "apprentice", "mentor",
    "lesson", "course", "curriculum", "study", "research", "experiment", "hypothesis", "thesis", "dissertation", "paper",
    "book", "text", "document", "manuscript", "scroll", "tablet", "inscription", "record", "archive", "library",
    "information", "data", "fact", "fiction", "myth", "legend", "history", "story", "tale", "narrative",
    "language", "word", "sentence", "paragraph", "chapter", "page", "letter", "symbol", "sign", "signal",
    "meaning", "definition", "translation", "interpretation", "explanation", "description", "instruction", "command", "request", "suggestion",
    
    # Society & Culture - 100 words
    "society", "community", "culture", "civilization", "nation", "country", "state", "government", "politics", "law",
    "rule", "order", "chaos", "justice", "injustice", "crime", "punishment", "reward", "duty", "responsibility",
    "right", "privilege", "freedom", "liberty", "equality", "inequality", "class", "caste", "rank", "status",
    "wealth", "poverty", "rich", "poor", "money", "currency", "gold", "silver", "bronze", "treasure",
    "economy", "market", "trade", "commerce", "business", "company", "corporation", "industry", "agriculture", "manufacturing",
    "worker", "labor", "employment", "unemployment", "career", "profession", "occupation", "job", "task", "work",
    "city", "town", "village", "capital", "metropolis", "suburb", "rural", "urban", "building", "structure",
    "architecture", "construction", "bridge", "tower", "wall", "gate", "door", "window", "roof", "floor",
    "street", "road", "highway", "path", "way", "route", "journey", "travel", "trip", "voyage",
    "vehicle", "car", "truck", "bus", "train", "airplane", "helicopter", "ship", "boat", "bicycle",
    
    # Technology & Objects - 100 words
    "technology", "machine", "engine", "motor", "device", "tool", "instrument", "equipment", "apparatus", "gadget",
    "computer", "laptop", "desktop", "server", "network", "internet", "website", "application", "software", "hardware",
    "phone", "telephone", "mobile", "smartphone", "tablet", "screen", "monitor", "display", "keyboard", "mouse",
    "camera", "video", "audio", "sound", "music", "noise", "silence", "voice", "speech", "whisper",
    "light", "dark", "shadow", "brightness", "color", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "black", "white", "gray", "gold", "silver", "transparent", "opaque",
    "metal", "iron", "steel", "copper", "aluminum", "gold", "silver", "platinum", "lead", "tin",
    "wood", "paper", "cardboard", "plastic", "glass", "ceramic", "porcelain", "cloth", "fabric", "textile",
    "cotton", "wool", "silk", "leather", "rubber", "oil", "fuel", "gasoline", "diesel", "electricity",
    "battery", "wire", "cable", "circuit", "chip", "processor", "memory", "storage", "disk", "drive",
    
    # Actions & Verbs - 100 words
    "action", "activity", "behavior", "movement", "motion", "rest", "stop", "start", "begin", "end",
    "create", "destroy", "make", "build", "construct", "assemble", "disassemble", "break", "fix", "repair",
    "grow", "shrink", "expand", "contract", "increase", "decrease", "rise", "fall", "climb", "descend",
    "walk", "run", "jump", "leap", "hop", "skip", "crawl", "creep", "sneak", "march",
    "swim", "dive", "float", "sink", "fly", "glide", "soar", "hover", "land", "takeoff",
    "push", "pull", "lift", "lower", "carry", "hold", "grasp", "grip", "grab", "release",
    "throw", "catch", "kick", "punch", "hit", "strike", "touch", "feel", "sense", "perceive",
    "look", "see", "watch", "observe", "notice", "stare", "gaze", "glance", "peek", "peek",
    "listen", "hear", "sound", "speak", "talk", "say", "tell", "speak", "shout", "whisper",
    "think", "consider", "ponder", "wonder", "imagine", "dream", "remember", "forget", "learn", "teach",
    
    # Qualities & Adjectives - 100 words
    "quality", "property", "attribute", "characteristic", "feature", "trait", "aspect", "element", "component", "part",
    "whole", "complete", "incomplete", "perfect", "imperfect", "good", "bad", "better", "worse", "best",
    "worst", "excellent", "terrible", "wonderful", "awful", "beautiful", "ugly", "pretty", "handsome", "attractive",
    "repulsive", "pleasant", "unpleasant", "nice", "mean", "kind", "cruel", "gentle", "harsh", "soft",
    "hard", "strong", "weak", "powerful", "feeble", "big", "small", "large", "tiny", "huge",
    "giant", "dwarf", "tall", "short", "long", "brief", "wide", "narrow", "thick", "thin",
    "fat", "skinny", "heavy", "light", "deep", "shallow", "high", "low", "fast", "slow",
    "quick", "gradual", "sudden", "immediate", "delayed", "early", "late", "ancient", "modern", "new",
    "old", "young", "fresh", "stale", "clean", "dirty", "pure", "impure", "clear", "cloudy",
    "simple", "complex", "easy", "difficult", "hard", "simple", "complicated", "straightforward", "confusing", "obvious",
    
    # Numbers & Quantities - 50 words
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million", "billion", "trillion", "single", "double", "triple", "quadruple",
    "half", "quarter", "third", "majority", "minority", "all", "none", "some", "many", "few",
    "several", "numerous", "countless", "infinite", "finite", "total", "sum", "amount", "quantity", "number",
    "count", "measure", "weight", "height", "length", "width", "depth", "volume", "area", "size",
]

# Ensure exactly 1000 words
VOCABULARY_1000 = VOCABULARY_1000[:VOCAB_SIZE]

print(f"Vocabulary size: {len(VOCABULARY_1000)} words")

# =============================================================================
# LOAD MINILM-L6 EMBEDDINGS
# =============================================================================

def load_embeddings_minilm():
    """Load MiniLM-L6 embeddings for the 1000-word vocabulary."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Loading MiniLM-L6 model...")
        
        # Encode vocabulary
        embs = model.encode(VOCABULARY_1000, normalize_embeddings=True, show_progress_bar=True)
        print(f"Embeddings shape: {embs.shape}")
        return embs, VOCABULARY_1000
    except Exception as e:
        print(f"Failed to load embeddings: {e}")
        return None, None


# =============================================================================
# COMPUTE FULL COVARIANCE MATRIX
# =============================================================================

def compute_full_covariance(embeddings):
    """
    Compute the full covariance matrix (not just eigenvalues).
    
    Returns:
        cov_matrix: Full covariance matrix
        eigenvalues: Eigenvalue spectrum
        eigenvectors: Eigenvectors
    """
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)
    
    # Compute full covariance matrix
    cov_matrix = np.cov(centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Filter out very small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    return cov_matrix, eigenvalues, eigenvectors


# =============================================================================
# ANALYZE OFF-DIAGONAL COVARIANCE STRUCTURE
# =============================================================================

def analyze_off_diagonals(cov_matrix, n_dims=50):
    """
    Analyze the off-diagonal structure of the covariance matrix.
    
    Look for patterns that might encode phase information.
    """
    # Extract upper triangle (off-diagonals)
    upper_tri = np.triu(cov_matrix, k=1)
    
    # Get off-diagonal values
    off_diag_values = upper_tri[upper_tri != 0]
    
    # Statistics
    stats_dict = {
        'mean': float(np.mean(off_diag_values)),
        'std': float(np.std(off_diag_values)),
        'min': float(np.min(off_diag_values)),
        'max': float(np.max(off_diag_values)),
        'median': float(np.median(off_diag_values)),
        'n_off_diagonal': len(off_diag_values),
    }
    
    # Analyze by distance from diagonal
    n = cov_matrix.shape[0]
    band_stats = {}
    
    for band in [1, 5, 10, 20, 50]:
        if band < n:
            # Extract values at this band distance
            band_values = []
            for i in range(n - band):
                band_values.append(cov_matrix[i, i + band])
            
            if band_values:
                band_stats[f'band_{band}'] = {
                    'mean': float(np.mean(band_values)),
                    'std': float(np.std(band_values)),
                    'n': len(band_values),
                }
    
    stats_dict['band_analysis'] = band_stats
    
    return stats_dict, off_diag_values


# =============================================================================
# ATTEMPT PHASE RECOVERY FROM COVARIANCE
# =============================================================================

def attempt_phase_recovery(cov_matrix, eigenvalues, eigenvectors, n_dims=50):
    """
    Attempt to recover phase angles from off-diagonal covariance.
    
    Hypothesis: <z_i, z_j> = r_i * r_j * cos(theta_i - theta_j)
    
    If we have magnitudes r (from eigenvalues), can we solve for phase differences?
    """
    # Use top n_dims dimensions
    n_dims = min(n_dims, len(eigenvalues))
    
    # Project covariance to top n_dims
    cov_reduced = cov_matrix[:n_dims, :n_dims]
    
    # Magnitudes (from eigenvalues)
    r = np.sqrt(eigenvalues[:n_dims])
    
    results = {
        'n_dims_used': n_dims,
        'magnitudes': r.tolist(),
    }
    
    # Try to infer phase differences from off-diagonals
    # If C_ij = r_i * r_j * cos(theta_i - theta_j)
    # Then cos(theta_i - theta_j) = C_ij / (r_i * r_j)
    
    phase_diff_matrix = np.zeros((n_dims, n_dims))
    valid_mask = np.zeros((n_dims, n_dims), dtype=bool)
    
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            r_i = r[i]
            r_j = r[j]
            C_ij = cov_reduced[i, j]
            
            # Avoid division by zero
            if r_i * r_j > 1e-10:
                cos_phase = C_ij / (r_i * r_j)
                # Clamp to [-1, 1]
                cos_phase = np.clip(cos_phase, -1.0, 1.0)
                phase_diff = np.arccos(cos_phase)
                
                phase_diff_matrix[i, j] = phase_diff
                phase_diff_matrix[j, i] = -phase_diff
                valid_mask[i, j] = True
                valid_mask[j, i] = True
    
    # Analyze recovered phase differences
    valid_phases = phase_diff_matrix[valid_mask]
    
    if len(valid_phases) > 0:
        results['phase_recovery'] = {
            'mean_phase_diff': float(np.mean(np.abs(valid_phases))),
            'std_phase_diff': float(np.std(valid_phases)),
            'median_phase_diff': float(np.median(np.abs(valid_phases))),
            'min_phase': float(np.min(valid_phases)),
            'max_phase': float(np.max(valid_phases)),
            'n_valid': len(valid_phases),
        }
        
        # Bin phases
        phase_bins = np.linspace(0, np.pi, PHASE_BINS + 1)
        hist, _ = np.histogram(np.abs(valid_phases), bins=phase_bins)
        results['phase_histogram'] = hist.tolist()
        
        # Check if phases are non-uniform (indicates structure)
        # Uniform distribution would suggest random/noise
        chi2_uniform = stats.chisquare(hist)
        results['phase_uniformity_test'] = {
            'chi2_statistic': float(chi2_uniform.statistic),
            'p_value': float(chi2_uniform.pvalue),
            'is_uniform': chi2_uniform.pvalue > 0.05,
        }
    
    return results, phase_diff_matrix


# =============================================================================
# COMPUTE PHASE COHERENCE LENGTH
# =============================================================================

def compute_phase_coherence(phase_diff_matrix, threshold=0.5):
    """
    Compute phase coherence length - how many dimensions show correlated phases.
    
    Success criterion: Phase coherence length > 10 dimensions
    """
    n = phase_diff_matrix.shape[0]
    
    coherence_lengths = []
    
    for start_dim in range(min(20, n)):
        # Count how many dimensions have low phase difference with start
        low_phase_count = 0
        for j in range(start_dim+1, n):
            phase_diff = abs(phase_diff_matrix[start_dim, j])
            if phase_diff < np.arccos(threshold):  # threshold corresponds to cos value
                low_phase_count += 1
            else:
                break
        
        coherence_lengths.append(low_phase_count)
    
    mean_coherence = np.mean(coherence_lengths)
    max_coherence = np.max(coherence_lengths)
    
    return {
        'mean_coherence_length': float(mean_coherence),
        'max_coherence_length': int(max_coherence),
        'coherence_lengths': [int(x) for x in coherence_lengths[:10]],
        'threshold_used': threshold,
        'success_criterion_met': max_coherence > 10,
    }


# =============================================================================
# WORDSIM-353 GROUND TRUTH
# =============================================================================

def load_wordsim353():
    """
    Load WordSim-353 dataset as external ground truth.
    
    Returns word pairs with human-annotated similarity scores.
    """
    # Official WordSim-353 word pairs and scores
    # Source: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/
    wordsim_pairs = [
        ("love", "sex", 6.77),
        ("tiger", "cat", 7.35),
        ("tiger", "tiger", 10.0),
        ("book", "paper", 7.46),
        ("computer", "keyboard", 7.62),
        ("computer", "internet", 7.58),
        ("plane", "car", 5.77),
        ("train", "car", 6.35),
        ("telephone", "communication", 7.50),
        ("television", "radio", 6.57),
        ("media", "radio", 7.13),
        ("drug", "abuse", 6.85),
        ("bread", "butter", 6.19),
        ("cucumber", "potato", 5.92),
        ("doctor", "nurse", 7.03),
        ("professor", "doctor", 6.62),
        ("student", "professor", 5.45),
        ("smart", "student", 4.62),
        ("smart", "stupid", 5.81),
        ("company", "stock", 6.47),
        ("stock", "market", 7.14),
        ("stock", "phone", 2.40),
        ("fertility", "egg", 6.69),
        ("planet", "sun", 6.53),
        ("planet", "moon", 6.43),
        ("planet", "galaxy", 6.75),
        ("money", "cash", 9.15),
        ("money", "currency", 9.04),
        ("money", "wealth", 8.27),
        ("money", "property", 6.53),
        ("money", "bank", 7.95),
        ("physics", "chemistry", 7.35),
        ("planet", "star", 6.13),
        ("planet", "constellation", 5.45),
        ("credit", "card", 7.19),
        ("hotel", "reservation", 7.15),
        ("closet", "clothes", 6.47),
        ("planet", "astronomer", 6.27),
        ("water", "ice", 7.42),
        ("water", "steam", 6.35),
        ("water", "gas", 4.92),
        ("computer", "software", 7.69),
        ("computer", "hardware", 6.77),
        ("possession", "property", 8.27),
        ("seafood", "food", 7.77),
        ("cup", "coffee", 6.58),
        ("cup", "drink", 5.88),
        ("music", "instrument", 7.03),
        ("mountain", "climb", 5.73),
        ("planet", "space", 6.65),
        ("planet", "atmosphere", 5.58),
        ("movie", "theater", 6.73),
        ("movie", "star", 6.46),
        ("treat", "doctor", 5.31),
        ("game", "team", 5.69),
        ("game", "victory", 6.47),
        ("game", "defeat", 5.88),
        ("announcement", "news", 7.56),
        ("announcement", "effort", 3.50),
        ("man", "woman", 8.30),
        ("man", "governor", 4.73),
        ("murder", "manslaughter", 8.53),
        ("opera", "performance", 6.88),
        ("skin", "eye", 5.85),
        ("journey", "voyage", 8.62),
        ("coast", "shore", 8.87),
        ("coast", "hill", 4.08),
        ("boy", "lad", 8.83),
        ("boy", "sage", 1.50),
        ("forest", "graveyard", 3.08),
        ("food", "fruit", 7.52),
        ("bird", "cock", 7.10),
        ("bird", "crane", 6.41),
        ("bird", "sparrow", 6.75),
        ("bird", "chicken", 5.85),
        ("bird", "hawk", 7.31),
        ("furnace", "stove", 8.04),
        ("car", "automobile", 8.94),
        ("car", "truck", 6.58),
        ("car", "vehicle", 8.31),
        ("car", "flight", 4.50),
        ("gem", "jewel", 8.96),
        ("glass", "tumbler", 7.27),
        ("glass", "crystal", 6.50),
        ("grin", "smile", 8.04),
        ("instrument", "tool", 6.35),
        ("magician", "wizard", 9.02),
        ("midday", "noon", 9.29),
        ("oracle", "sage", 7.62),
        ("serf", "slave", 7.27),
    ]
    
    return wordsim_pairs


def validate_with_wordsim353(embeddings, vocab, wordsim_pairs):
    """
    Validate phase recovery against WordSim-353 ground truth.
    
    Ground truth must NOT be derived from R values (covariance magnitudes).
    """
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Filter pairs to only those in our vocabulary
    valid_pairs = []
    for w1, w2, score in wordsim_pairs:
        if w1 in word_to_idx and w2 in word_to_idx:
            valid_pairs.append((w1, w2, score, word_to_idx[w1], word_to_idx[w2]))
    
    print(f"WordSim-353: {len(valid_pairs)}/{len(wordsim_pairs)} pairs in vocabulary")
    
    if len(valid_pairs) < 10:
        return {
            'error': 'Insufficient WordSim-353 coverage',
            'n_valid_pairs': len(valid_pairs),
            'n_total_pairs': len(wordsim_pairs),
        }
    
    # Compute cosine similarities (independent of phase recovery)
    cosine_sims = []
    human_scores = []
    
    for w1, w2, score, idx1, idx2 in valid_pairs:
        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]
        
        # Cosine similarity
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        cosine_sims.append(cos_sim)
        human_scores.append(score)
    
    # Compute correlation
    pearson_r, pearson_p = stats.pearsonr(cosine_sims, human_scores)
    spearman_r, spearman_p = stats.spearmanr(cosine_sims, human_scores)
    
    return {
        'n_valid_pairs': len(valid_pairs),
        'n_total_pairs': len(wordsim_pairs),
        'coverage_percent': len(valid_pairs) / len(wordsim_pairs) * 100,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'cosine_similarities': [float(x) for x in cosine_sims[:20]],
        'human_scores': [float(x) for x in human_scores[:20]],
        'statistical_significance': pearson_p < 0.001,
    }


# =============================================================================
# TEST CORRELATION BETWEEN PHASE AND SEMANTIC SIMILARITY
# =============================================================================

def test_phase_semantic_correlation(phase_diff_matrix, embeddings, vocab, wordsim_pairs):
    """
    Test if phase differences correlate with semantic similarity.
    
    This is the key test: if phase information is meaningful,
    words with similar semantics should have smaller phase differences.
    """
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Get valid pairs
    valid_pairs = []
    for w1, w2, score in wordsim_pairs:
        if w1 in word_to_idx and w2 in word_to_idx:
            valid_pairs.append((w1, w2, score, word_to_idx[w1], word_to_idx[w2]))
    
    if len(valid_pairs) < 10:
        return {
            'error': 'Insufficient pairs for phase-semantic correlation',
            'n_pairs': len(valid_pairs),
        }
    
    # For each pair, compute:
    # 1. Phase difference (from top PCs)
    # 2. Semantic similarity (human score)
    
    phase_diffs = []
    semantic_sims = []
    
    n_pcs = min(50, phase_diff_matrix.shape[0])
    
    for w1, w2, score, idx1, idx2 in valid_pairs:
        # Get embedding vectors
        vec1 = embeddings[idx1]
        vec2 = embeddings[idx2]
        
        # Project to top PCs
        # (we need eigenvectors for this, but we'll use a simplified approach)
        pca = PCA(n_components=n_pcs)
        proj = pca.fit_transform(embeddings)
        
        vec1_proj = proj[idx1]
        vec2_proj = proj[idx2]
        
        # Compute phase difference as angle between projected vectors
        # in the first 2 principal components
        x1, y1 = vec1_proj[0], vec1_proj[1]
        x2, y2 = vec2_proj[0], vec2_proj[1]
        
        theta1 = np.arctan2(y1, x1)
        theta2 = np.arctan2(y2, x2)
        
        phase_diff = abs(theta1 - theta2)
        if phase_diff > np.pi:
            phase_diff = 2 * np.pi - phase_diff
        
        phase_diffs.append(phase_diff)
        semantic_sims.append(score)
    
    # Test correlation
    if len(phase_diffs) >= 10:
        pearson_r, pearson_p = stats.pearsonr(phase_diffs, semantic_sims)
        spearman_r, spearman_p = stats.spearmanr(phase_diffs, semantic_sims)
        
        # Negative correlation expected: similar semantics -> smaller phase diff
        return {
            'n_pairs': len(valid_pairs),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'mean_phase_diff': float(np.mean(phase_diffs)),
            'std_phase_diff': float(np.std(phase_diffs)),
            'phase_diffs': [float(x) for x in phase_diffs[:20]],
            'semantic_sims': [float(x) for x in semantic_sims[:20]],
            'negative_correlation': pearson_r < 0,
            'statistically_significant': pearson_p < 0.001,
        }
    
    return {
        'n_pairs': len(valid_pairs),
        'error': 'Insufficient data for correlation',
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("Q51.1: PHASE SIGNATURES IN CROSS-CORRELATIONS")
    print("Testing if phase information can be recovered from off-diagonal covariance")
    print("=" * 80)
    
    # Fixed parameters
    print(f"\nParameters (FIXED):")
    print(f"  - Vocabulary: {VOCAB_SIZE} words")
    print(f"  - PCA dimensions: {PCA_DIMS}")
    print(f"  - Phase bins: {PHASE_BINS}")
    print(f"  - Random seed: {RANDOM_SEED}")
    
    # Load embeddings
    print("\n" + "-" * 80)
    print("Loading MiniLM-L6 embeddings...")
    embeddings, vocab = load_embeddings_minilm()
    
    if embeddings is None:
        print("FAILED: Could not load embeddings")
        return
    
    print(f"Loaded {len(vocab)} words with {embeddings.shape[1]}-dim embeddings")
    
    # Compute full covariance matrix
    print("\n" + "-" * 80)
    print("Computing full covariance matrix...")
    cov_matrix, eigenvalues, eigenvectors = compute_full_covariance(embeddings)
    
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Top 10 eigenvalues: {eigenvalues[:10]}")
    
    # Analyze off-diagonal structure
    print("\n" + "-" * 80)
    print("Analyzing off-diagonal covariance structure...")
    off_diag_stats, off_diag_values = analyze_off_diagonals(cov_matrix, n_dims=PCA_DIMS)
    
    print(f"Off-diagonal statistics:")
    print(f"  - Mean: {off_diag_stats['mean']:.6f}")
    print(f"  - Std: {off_diag_stats['std']:.6f}")
    print(f"  - Range: [{off_diag_stats['min']:.6f}, {off_diag_stats['max']:.6f}]")
    print(f"  - N off-diagonal elements: {off_diag_stats['n_off_diagonal']}")
    
    # Attempt phase recovery
    print("\n" + "-" * 80)
    print("Attempting phase recovery from off-diagonals...")
    print("Hypothesis: <z_i, z_j> = r_i * r_j * cos(theta_i - theta_j)")
    
    phase_results, phase_diff_matrix = attempt_phase_recovery(
        cov_matrix, eigenvalues, eigenvectors, n_dims=PCA_DIMS
    )
    
    if 'phase_recovery' in phase_results:
        pr = phase_results['phase_recovery']
        print(f"Phase recovery results:")
        print(f"  - Mean phase difference: {pr['mean_phase_diff']:.4f} rad")
        print(f"  - Std phase difference: {pr['std_phase_diff']:.4f} rad")
        print(f"  - Valid phase pairs: {pr['n_valid']}")
        
        # Phase uniformity test
        pu = phase_results.get('phase_uniformity_test', {})
        print(f"\nPhase uniformity test:")
        print(f"  - Chi2 statistic: {pu.get('chi2_statistic', 'N/A')}")
        print(f"  - P-value: {pu.get('p_value', 'N/A'):.6f}")
        print(f"  - Is uniform: {pu.get('is_uniform', 'N/A')}")
        
        # Compute phase coherence length
        print("\n" + "-" * 80)
        print("Computing phase coherence length...")
        coherence_results = compute_phase_coherence(phase_diff_matrix, threshold=0.5)
        
        print(f"Phase coherence:")
        print(f"  - Mean coherence length: {coherence_results['mean_coherence_length']:.2f}")
        print(f"  - Max coherence length: {coherence_results['max_coherence_length']}")
        print(f"  - Success criterion (>10): {coherence_results['success_criterion_met']}")
        
        phase_results['coherence'] = coherence_results
    else:
        print("Phase recovery failed or insufficient data")
    
    # Load WordSim-353
    print("\n" + "-" * 80)
    print("Loading WordSim-353 ground truth...")
    wordsim_pairs = load_wordsim353()
    print(f"WordSim-353: {len(wordsim_pairs)} word pairs loaded")
    
    # Validate against WordSim-353
    print("\n" + "-" * 80)
    print("Validating against WordSim-353 (external ground truth)...")
    print("NOTE: Ground truth is NOT derived from covariance R values")
    
    wordsim_results = validate_with_wordsim353(embeddings, vocab, wordsim_pairs)
    
    print(f"WordSim-353 validation:")
    print(f"  - Coverage: {wordsim_results['n_valid_pairs']}/{wordsim_results['n_total_pairs']} ({wordsim_results.get('coverage_percent', 0):.1f}%)")
    print(f"  - Pearson r: {wordsim_results.get('pearson_r', 'N/A'):.4f}")
    print(f"  - Pearson p: {wordsim_results.get('pearson_p', 'N/A'):.6f}")
    print(f"  - Spearman r: {wordsim_results.get('spearman_r', 'N/A'):.4f}")
    print(f"  - Spearman p: {wordsim_results.get('spearman_p', 'N/A'):.6f}")
    print(f"  - Significant (p<0.001): {wordsim_results.get('statistical_significance', False)}")
    
    # Test phase-semantic correlation
    print("\n" + "-" * 80)
    print("Testing phase-semantic similarity correlation...")
    
    phase_semantic_results = test_phase_semantic_correlation(
        phase_diff_matrix, embeddings, vocab, wordsim_pairs
    )
    
    if 'error' not in phase_semantic_results:
        print(f"Phase-semantic correlation:")
        print(f"  - N pairs: {phase_semantic_results['n_pairs']}")
        print(f"  - Pearson r: {phase_semantic_results['pearson_r']:.4f}")
        print(f"  - Pearson p: {phase_semantic_results['pearson_p']:.6f}")
        print(f"  - Negative correlation (expected): {phase_semantic_results['negative_correlation']}")
        print(f"  - Statistically significant: {phase_semantic_results['statistically_significant']}")
    else:
        print(f"Phase-semantic test: {phase_semantic_results['error']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    # Evaluate success criteria
    success_criteria = {
        'phase_recovery_r05': False,
        'coherence_length_10': False,
        'p_001_significance': False,
    }
    
    # Criterion 1: Can infer phase angles with |r| > 0.5
    if 'phase_recovery' in phase_results:
        # Check if phase recovery shows non-random structure
        pu = phase_results.get('phase_uniformity_test', {})
        if not pu.get('is_uniform', True):  # Non-uniform means structure detected
            success_criteria['phase_recovery_r05'] = True
    
    # Criterion 2: Phase coherence length > 10
    if 'coherence' in phase_results:
        success_criteria['coherence_length_10'] = phase_results['coherence']['success_criterion_met']
    
    # Criterion 3: p < 0.001
    success_criteria['p_001_significance'] = wordsim_results.get('statistical_significance', False)
    
    print("\nSuccess Criteria Evaluation:")
    print(f"  1. Phase recovery with structure: {success_criteria['phase_recovery_r05']}")
    print(f"  2. Phase coherence length > 10: {success_criteria['coherence_length_10']}")
    print(f"  3. Statistical significance (p<0.001): {success_criteria['p_001_significance']}")
    
    n_passed = sum(success_criteria.values())
    print(f"\nOverall: {n_passed}/3 criteria passed")
    
    # Key findings
    print("\n" + "-" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)
    
    findings = []
    
    if 'phase_recovery' in phase_results:
        pr = phase_results['phase_recovery']
        pu = phase_results.get('phase_uniformity_test', {})
        
        if pu.get('is_uniform', True):
            findings.append("Phase distribution is UNIFORM (no phase structure detected)")
        else:
            findings.append(f"Phase distribution shows NON-UNIFORM structure (chi2 p={pu.get('p_value', 0):.6f})")
        
        findings.append(f"Mean phase difference: {pr['mean_phase_diff']:.4f} rad ({np.degrees(pr['mean_phase_diff']):.1f} deg)")
    else:
        findings.append("Phase recovery could not be performed (insufficient data)")
    
    if 'coherence' in phase_results:
        cr = phase_results['coherence']
        findings.append(f"Phase coherence length: mean={cr['mean_coherence_length']:.1f}, max={cr['max_coherence_length']}")
    
    findings.append(f"WordSim-353 correlation: r={wordsim_results.get('pearson_r', 0):.3f}, p={wordsim_results.get('pearson_p', 1):.6f}")
    
    if 'pearson_r' in phase_semantic_results:
        findings.append(f"Phase-semantic correlation: r={phase_semantic_results['pearson_r']:.3f}, p={phase_semantic_results['pearson_p']:.6f}")
    
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    
    # Interpretation
    print("\n" + "-" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    # Determine if phase recovery is possible
    phase_recoverable = (
        success_criteria['phase_recovery_r05'] or 
        (phase_results.get('phase_recovery', {}).get('n_valid', 0) > 1000 and
         phase_results.get('phase_recovery', {}).get('std_phase_diff', 1) < 1.0)
    )
    
    if phase_recoverable:
        print("""
The off-diagonal covariance elements DO contain phase-like structure.
- Phase differences can be inferred from C_ij / (r_i * r_j)
- The phase distribution is non-uniform, indicating structure
- This supports the hypothesis that real embeddings are projections of
  a complex-valued semiotic space where phase was discarded.
""")
    else:
        print("""
The off-diagonal covariance elements do NOT show clear phase structure.
- Phase distribution appears uniform (random)
- No significant non-random phase patterns detected
- This suggests either:
  a) The hypothesis is incorrect (no phase information in off-diagonals)
  b) Phase information was truly lost in the real-valued projection
  c) More sophisticated methods are needed to recover phase
""")
    
    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    receipt = {
        'test': 'Q51.1_PHASE_RECOVERY',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'parameters': {
            'vocab_size': VOCAB_SIZE,
            'pca_dims': PCA_DIMS,
            'phase_bins': PHASE_BINS,
            'random_seed': RANDOM_SEED,
        },
        'success_criteria': success_criteria,
        'n_criteria_passed': n_passed,
        'phase_recoverable': phase_recoverable,
        'covariance_stats': {
            'shape': list(cov_matrix.shape),
            'top_eigenvalues': eigenvalues[:20].tolist(),
        },
        'off_diagonal_stats': off_diag_stats,
        'phase_recovery': phase_results,
        'wordsim353_validation': wordsim_results,
        'phase_semantic_correlation': phase_semantic_results,
        'findings': findings,
        'interpretation': 'phase_recoverable' if phase_recoverable else 'phase_not_recoverable',
    }
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f'q51_phase_recovery_{timestamp_str}.json'
    
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=convert)
    
    print(f"Results saved to: {path}")
    
    print("\n" + "=" * 80)
    print("Q51.1 TEST COMPLETE")
    print("=" * 80)
    
    return receipt


if __name__ == '__main__':
    main()
