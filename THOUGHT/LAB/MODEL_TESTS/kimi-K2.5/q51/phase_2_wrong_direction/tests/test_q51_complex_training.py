#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q51.3: Complex-Valued Training

**Objective:** Test if training with complex weights preserves 8e but reveals phase structure.

**Hypothesis:** Complex eigenvalues maintain Df × α = 8e (±5%) while making phase observable

**Test Strategy:**
1. Implement complex-valued embedding model (real + imaginary parts)
2. Train on same 1000-word vocabulary as real baseline
3. Compute Df × α for complex eigenvalues (|z| = sqrt(Re² + Im²))
4. Analyze phase distribution (theta = atan2(Im, Re))
5. Compare |z| power law to real λ_k baseline

**Parameters (FIXED):**
- Vocabulary: 1000 words (expanded from Q50 base)
- Training epochs: 10 (or until convergence)
- Learning rate: 2e-5
- Random seed: 42
- Model size: 384-dim complex (matches MiniLM-L6)

**Pass criteria:**
- 8e conservation maintained (|error| < 5%)
- Phase shows non-uniform structure (KL divergence from uniform > 0.5)
- Training converges without catastrophic instability

**Anti-Patterns:**
- Uses SAME vocabulary as real baseline (controlled comparison)
- Reports training instability honestly if it occurs
- Uses final converged state (not cherry-picked best epoch)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# COMPLEX-VALUED NEURAL NETWORK IMPLEMENTATION
# =============================================================================

class ComplexLinear:
    """Complex-valued linear layer: z_out = W * z_in + b"""
    
    def __init__(self, in_features, out_features, seed=42):
        np.random.seed(seed)
        # Complex weights: W = W_real + i * W_imag
        scale = np.sqrt(1.0 / in_features)
        self.W_real = np.random.randn(out_features, in_features) * scale
        self.W_imag = np.random.randn(out_features, in_features) * scale
        self.b_real = np.zeros(out_features)
        self.b_imag = np.zeros(out_features)
        
        # Gradients
        self.dW_real = np.zeros_like(self.W_real)
        self.dW_imag = np.zeros_like(self.W_imag)
        self.db_real = np.zeros_like(self.b_real)
        self.db_imag = np.zeros_like(self.b_imag)
    
    def forward(self, z_real, z_imag):
        """Forward pass: (W_real + i*W_imag) * (z_real + i*z_imag)"""
        # (a + bi) * (c + di) = (ac - bd) + i(ad + bc)
        out_real = self.W_real @ z_real.T - self.W_imag @ z_imag.T + self.b_real[:, None]
        out_imag = self.W_real @ z_imag.T + self.W_imag @ z_real.T + self.b_imag[:, None]
        return out_real.T, out_imag.T
    
    def backward(self, z_real, z_imag, dout_real, dout_imag):
        """Backward pass with complex derivatives"""
        # dW_real = dout_real * z_real + dout_imag * z_imag
        # dW_imag = dout_imag * z_real - dout_real * z_imag
        self.dW_real = (dout_real.T @ z_real + dout_imag.T @ z_imag) / z_real.shape[0]
        self.dW_imag = (dout_imag.T @ z_real - dout_real.T @ z_imag) / z_real.shape[0]
        self.db_real = dout_real.mean(axis=0)
        self.db_imag = dout_imag.mean(axis=0)
        
        # Return gradients for previous layer
        dz_real = dout_real @ self.W_real + dout_imag @ self.W_imag
        dz_imag = dout_imag @ self.W_real - dout_real @ self.W_imag
        return dz_real, dz_imag
    
    def step(self, lr):
        """Gradient descent update"""
        self.W_real -= lr * self.dW_real
        self.W_imag -= lr * self.dW_imag
        self.b_real -= lr * self.db_real
        self.b_imag -= lr * self.db_imag


class ComplexEmbedding:
    """Complex-valued embedding layer"""
    
    def __init__(self, vocab_size, embed_dim, seed=42):
        np.random.seed(seed)
        scale = 0.02
        # Each word has complex embedding
        self.embeddings_real = np.random.randn(vocab_size, embed_dim) * scale
        self.embeddings_imag = np.random.randn(vocab_size, embed_dim) * scale
        
        # Gradients
        self.dE_real = np.zeros_like(self.embeddings_real)
        self.dE_imag = np.zeros_like(self.embeddings_imag)
    
    def forward(self, indices):
        """Get complex embeddings for word indices"""
        return self.embeddings_real[indices], self.embeddings_imag[indices]
    
    def backward(self, indices, dout_real, dout_imag):
        """Accumulate gradients for embeddings"""
        np.add.at(self.dE_real, indices, dout_real)
        np.add.at(self.dE_imag, indices, dout_imag)
    
    def step(self, lr):
        """Gradient descent update"""
        self.embeddings_real -= lr * self.dE_real
        self.embeddings_imag -= lr * self.dE_imag
        self.dE_real.fill(0)
        self.dE_imag.fill(0)


# =============================================================================
# TRAINING COMPONENTS
# =============================================================================

def complex_tanh(z_real, z_imag):
    """Complex hyperbolic tangent activation"""
    # tanh(a + bi) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))
    denom = np.cosh(2*z_real) + np.cos(2*z_imag) + 1e-8
    out_real = np.sinh(2*z_real) / denom
    out_imag = np.sin(2*z_imag) / denom
    return out_real, out_imag


def complex_mse_loss(pred_real, pred_imag, target_real, target_imag):
    """Mean squared error for complex values: |pred - target|^2"""
    diff_real = pred_real - target_real
    diff_imag = pred_imag - target_imag
    loss = np.mean(diff_real**2 + diff_imag**2)
    return loss, diff_real, diff_imag


# =============================================================================
# VOCABULARY (1000 words - expanded from Q50 base)
# =============================================================================

BASE_WORDS = [
    # Nature (16)
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    # Animals (16)
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "wolf", "bear", "eagle", "snake", "rabbit", "deer", "fox", "owl",
    # Body (16)
    "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
    "ear", "nose", "mouth", "foot", "leg", "arm", "hair", "voice",
    # People (16)
    "mother", "father", "child", "friend", "king", "queen", "hero", "teacher",
    "doctor", "artist", "student", "leader", "stranger", "neighbor", "soldier", "writer",
    # Concepts (16)
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    # Objects (16)
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "sword", "shield", "chair", "table", "window", "bridge", "tower", "garden",
    # Qualities (16)
    "good", "bad", "big", "small", "old", "new", "high", "low",
    "fast", "slow", "hot", "cold", "bright", "dark", "strong", "weak",
    # Abstract (16)
    "light", "shadow", "music", "word", "name", "law", "art", "science",
    "history", "future", "reason", "belief", "memory", "knowledge", "wisdom", "beauty",
    # Colors (8)
    "red", "blue", "green", "yellow", "white", "black", "gold", "silver",
    # Directions (8)
    "north", "south", "east", "west", "up", "down", "left", "right",
    # Times (8)
    "day", "night", "morning", "evening", "spring", "summer", "winter", "autumn",
    # Materials (8)
    "wood", "metal", "glass", "clay", "sand", "ice", "dust", "smoke",
    # Actions (8)
    "run", "walk", "speak", "listen", "think", "feel", "create", "destroy",
    # Emotions (8)
    "anger", "calm", "excitement", "boredom", "confidence", "doubt", "pride", "shame",
    # States (8)
    "awake", "asleep", "alive", "dead", "free", "bound", "safe", "lost",
    # Social (8)
    "family", "nation", "world", "culture", "language", "tradition", "freedom", "justice",
    # Numbers (8)
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    # Professions (8)
    "worker", "farmer", "merchant", "craftsman", "priest", "judge", "explorer", "inventor",
    # Places (8)
    "city", "village", "forest", "desert", "island", "valley", "cave", "market",
    # Weather (8)
    "storm", "fog", "thunder", "lightning", "rainbow", "mist", "frost", "heat",
    # Relations (8)
    "marriage", "birth", "death", "meeting", "parting", "growth", "decay", "return",
    # Sounds (8)
    "silence", "noise", "voice", "echo", "song", "whisper", "cry", "laughter",
    # Elements (8)
    "air", "flame", "rock", "wave", "seed", "root", "branch", "leaf",
    # Measure (8)
    "weight", "height", "depth", "width", "length", "distance", "speed", "time",
    # Value (8)
    "price", "worth", "cost", "value", "quality", "quantity", "measure", "standard",
    # Form (8)
    "shape", "color", "pattern", "texture", "sound", "taste", "smell", "touch",
    # Movement (8)
    "flow", "flight", "fall", "rise", "turn", "stop", "start", "change",
    # Condition (8)
    "health", "sickness", "wealth", "poverty", "happiness", "sadness", "success", "failure",
    # Degree (8)
    "all", "none", "some", "many", "few", "most", "least", "more",
    # Position (8)
    "inside", "outside", "above", "below", "before", "after", "between", "beyond",
    # Connection (8)
    "link", "chain", "path", "way", "gate", "key", "lock", "door",
    # Opposition (8)
    "friend", "enemy", "ally", "rival", "partner", "competitor", "helper", "hinderer",
    # Transformation (8)
    "birth", "death", "growth", "shrink", "build", "break", "join", "separate",
    # Expression (8)
    "write", "read", "draw", "paint", "sing", "dance", "play", "work",
    # Cognition (8)
    "learn", "teach", "know", "forget", "understand", "confuse", "remember", "imagine",
    # Desire (8)
    "want", "need", "wish", "hope", "seek", "find", "lose", "keep",
    # Judgment (8)
    "right", "wrong", "fair", "unfair", "just", "unjust", "true", "false",
    # Existence (8)
    "be", "become", "exist", "vanish", "appear", "disappear", "live", "die",
    # Possession (8)
    "have", "lack", "own", "share", "give", "take", "receive", "return",
]


def build_vocabulary():
    """Build 1000-word vocabulary from base words + expansions"""
    vocab = list(BASE_WORDS)
    
    # Add variations to reach 1000 words
    prefixes = ["super", "sub", "anti", "pro", "pre", "post", "re", "un", "in", "out"]
    suffixes = ["ness", "ity", "ism", "ist", "er", "or", "tion", "ment", "ance", "ence"]
    
    # Generate compound-like variations
    np.random.seed(42)
    while len(vocab) < 1000:
        base = np.random.choice(BASE_WORDS[:100])  # Use first 100 bases
        if len(vocab) < 600:
            # Add prefix variations
            prefix = np.random.choice(prefixes)
            word = f"{prefix}{base}"
        elif len(vocab) < 800:
            # Add suffix variations
            suffix = np.random.choice(suffixes)
            word = f"{base}{suffix}"
        else:
            # Add compound variations
            base2 = np.random.choice(BASE_WORDS[:50])
            word = f"{base}_{base2}"
        
        if word not in vocab:
            vocab.append(word)
    
    return vocab[:1000]


# =============================================================================
# COMPLEX EMBEDDING MODEL
# =============================================================================

class ComplexEmbeddingModel:
    """Simple complex-valued embedding model for semantic learning"""
    
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=512, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Layers
        self.embedding = ComplexEmbedding(vocab_size, embed_dim, seed=seed)
        self.fc1 = ComplexLinear(embed_dim, hidden_dim, seed=seed+1)
        self.fc2 = ComplexLinear(hidden_dim, embed_dim, seed=seed+2)
        
        self.training_history = []
    
    def forward(self, indices):
        """Forward pass through the network"""
        # Embedding
        z_real, z_imag = self.embedding.forward(indices)
        
        # FC1 + Activation
        h1_real, h1_imag = self.fc1.forward(z_real, z_imag)
        h1_real, h1_imag = complex_tanh(h1_real, h1_imag)
        
        # FC2 (output embeddings)
        out_real, out_imag = self.fc2.forward(h1_real, h1_imag)
        out_real, out_imag = complex_tanh(out_real, out_imag)
        
        return out_real, out_imag
    
    def compute_loss(self, indices, target_real, target_imag):
        """Compute reconstruction loss"""
        pred_real, pred_imag = self.forward(indices)
        loss, diff_real, diff_imag = complex_mse_loss(pred_real, pred_imag, target_real, target_imag)
        return loss, diff_real, diff_imag, pred_real, pred_imag
    
    def train_step(self, indices, target_real, target_imag, lr):
        """Single training step"""
        # Forward
        loss, diff_real, diff_imag, pred_real, pred_imag = self.compute_loss(indices, target_real, target_imag)
        
        # Backward through layers
        # FC2
        dz2_real, dz2_imag = self.fc2.backward(self.fc2_cache_h1_real, self.fc2_cache_h1_imag, diff_real, diff_imag)
        
        # Activation backward (simplified: derivative of tanh is 1 - tanh^2)
        h1_real, h1_imag = self.fc2_cache_h1_real, self.fc2_cache_h1_imag
        tanh_sq_real = np.tanh(h1_real)**2
        dz2_real *= (1 - tanh_sq_real)
        dz2_imag *= (1 - tanh_sq_real)
        
        # FC1
        dz1_real, dz1_imag = self.fc1.backward(self.fc1_cache_z_real, self.fc1_cache_z_imag, dz2_real, dz2_imag)
        
        # Embedding
        self.embedding.backward(indices, dz1_real, dz1_imag)
        
        # Update parameters
        self.fc2.step(lr)
        self.fc1.step(lr)
        self.embedding.step(lr)
        
        return loss
    
    def train(self, vocab_indices, epochs=10, lr=2e-5, batch_size=32, verbose=True):
        """Train the model"""
        n_samples = len(vocab_indices)
        
        # Create targets (circular shift of embeddings as autoencoding target)
        np.random.seed(42)
        target_indices = np.roll(vocab_indices, shift=1)
        
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            perm = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                indices = vocab_indices[batch_idx]
                
                # Forward with cache
                z_real, z_imag = self.embedding.forward(indices)
                self.fc1_cache_z_real, self.fc1_cache_z_imag = z_real.copy(), z_imag.copy()
                
                h1_real, h1_imag = self.fc1.forward(z_real, z_imag)
                h1_real, h1_imag = complex_tanh(h1_real, h1_imag)
                self.fc2_cache_h1_real, self.fc2_cache_h1_imag = h1_real.copy(), h1_imag.copy()
                
                pred_real, pred_imag = self.fc2.forward(h1_real, h1_imag)
                pred_real, pred_imag = complex_tanh(pred_real, pred_imag)
                
                # Targets
                tgt_real, tgt_imag = self.embedding.forward(target_indices[batch_idx])
                
                # Loss and backward
                loss, diff_real, diff_imag = complex_mse_loss(pred_real, pred_imag, tgt_real, tgt_imag)
                
                # Manual backward pass
                # FC2 backward
                self.fc2.dW_real = (diff_real.T @ h1_real + diff_imag.T @ h1_imag) / len(indices)
                self.fc2.dW_imag = (diff_imag.T @ h1_real - diff_real.T @ h1_imag) / len(indices)
                self.fc2.db_real = diff_real.mean(axis=0)
                self.fc2.db_imag = diff_imag.mean(axis=0)
                
                dh2_real = diff_real @ self.fc2.W_real + diff_imag @ self.fc2.W_imag
                dh2_imag = diff_imag @ self.fc2.W_real - diff_real @ self.fc2.W_imag
                
                # Tanh backward
                tanh_sq = np.tanh(self.fc2_cache_h1_real)**2
                dh2_real *= (1 - tanh_sq)
                dh2_imag *= (1 - tanh_sq)
                
                # FC1 backward
                self.fc1.dW_real = (dh2_real.T @ z_real + dh2_imag.T @ z_imag) / len(indices)
                self.fc1.dW_imag = (dh2_imag.T @ z_real - dh2_real.T @ z_imag) / len(indices)
                self.fc1.db_real = dh2_real.mean(axis=0)
                self.fc1.db_imag = dh2_imag.mean(axis=0)
                
                # Embedding backward
                de_real = dh2_real @ self.fc1.W_real + dh2_imag @ self.fc1.W_imag
                de_imag = dh2_imag @ self.fc1.W_real - dh2_real @ self.fc1.W_imag
                
                # Update
                self.fc2.step(lr)
                self.fc1.step(lr)
                
                # Embedding update
                np.add.at(self.embedding.dE_real, indices, de_real)
                np.add.at(self.embedding.dE_imag, indices, de_imag)
                self.embedding.step(lr)
                
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs}: loss = {avg_loss:.6f}")
            
            # Check convergence
            if epoch > 2 and abs(losses[-1] - losses[-2]) < 1e-7:
                if verbose:
                    print(f"  Converged at epoch {epoch+1}")
                break
        
        self.training_history = losses
        return losses
    
    def get_embeddings(self):
        """Get final complex embeddings"""
        all_indices = np.arange(self.vocab_size)
        return self.embedding.forward(all_indices)


# =============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_df(eigenvalues):
    """Participation ratio Df = (Σλ)² / Σλ²"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) == 0:
        return 0.0
    return float((np.sum(ev)**2) / np.sum(ev**2))


def compute_alpha(eigenvalues):
    """Power law decay exponent α where λ_k ~ k^(-α)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return 0.0
    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return float(-slope)


def analyze_complex_spectrum(embeddings_real, embeddings_imag):
    """Analyze eigenspectrum of complex embeddings"""
    # Compute magnitudes: |z| = sqrt(Re² + Im²)
    magnitudes = np.sqrt(embeddings_real**2 + embeddings_imag**2)
    
    # Center
    centered = magnitudes - magnitudes.mean(axis=0)
    
    # Covariance and eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # Phase analysis
    phases = np.arctan2(embeddings_imag, embeddings_real)  # theta in [-π, π]
    
    return eigenvalues, magnitudes, phases


def compute_kl_divergence_from_uniform(phases, n_bins=8):
    """Compute KL divergence of phase distribution from uniform"""
    # Bin phases into sectors
    hist, _ = np.histogram(phases.flatten(), bins=n_bins, range=(-np.pi, np.pi))
    p_observed = hist / hist.sum()
    p_uniform = np.ones(n_bins) / n_bins
    
    # KL divergence
    kl = 0.0
    for p, q in zip(p_observed, p_uniform):
        if p > 0:
            kl += p * np.log(p / q)
    
    return float(kl)


def analyze_phase_structure(phases):
    """Analyze phase distribution structure"""
    phases_flat = phases.flatten()
    
    # Basic statistics
    mean_phase = float(np.mean(phases_flat))
    std_phase = float(np.std(phases_flat))
    
    # Circular statistics
    sin_mean = float(np.mean(np.sin(phases_flat)))
    cos_mean = float(np.mean(np.cos(phases_flat)))
    r_circ = float(np.sqrt(sin_mean**2 + cos_mean**2))  # Circular concentration
    
    # 8-sector analysis (matching 8 octants hypothesis)
    sector_edges = np.linspace(-np.pi, np.pi, 9)
    sector_counts = np.zeros(8)
    for i in range(8):
        mask = (phases_flat >= sector_edges[i]) & (phases_flat < sector_edges[i+1])
        sector_counts[i] = np.sum(mask)
    
    sector_probs = sector_counts / sector_counts.sum()
    
    # Entropy of sector distribution
    entropy = -np.sum(sector_probs * np.log(sector_probs + 1e-10))
    max_entropy = np.log(8)
    normalized_entropy = float(entropy / max_entropy)
    
    # KL divergence from uniform
    kl_div = compute_kl_divergence_from_uniform(phases, n_bins=8)
    
    return {
        'mean_phase': mean_phase,
        'std_phase': std_phase,
        'circular_concentration': r_circ,
        'sector_counts': sector_counts.tolist(),
        'sector_probabilities': sector_probs.tolist(),
        'phase_entropy_bits': float(entropy / np.log(2)),
        'normalized_entropy': normalized_entropy,
        'kl_divergence_from_uniform': kl_div,
    }


def compute_power_law_fit(eigenvalues):
    """Fit power law λ_k = A * k^(-α)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if len(ev) < 10:
        return {'alpha': 0.0, 'r_squared': 0.0}
    
    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    
    slope, intercept = np.polyfit(log_k, log_ev, 1)
    alpha = -slope
    
    # R-squared
    predicted = intercept + slope * log_k
    ss_res = np.sum((log_ev - predicted)**2)
    ss_tot = np.sum((log_ev - np.mean(log_ev))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'alpha': float(alpha),
        'intercept': float(intercept),
        'amplitude': float(np.exp(intercept)),
        'r_squared': float(r_squared),
        'n_eigenvalues': len(ev),
    }


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

def get_real_baseline(vocab_words):
    """Get real embedding baseline using sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(vocab_words, normalize_embeddings=True)
        
        # Analyze real spectrum
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        Df = compute_df(eigenvalues)
        alpha = compute_alpha(eigenvalues)
        
        return {
            'available': True,
            'model': 'all-MiniLM-L6-v2',
            'dim': 384,
            'Df': Df,
            'alpha': alpha,
            'Df_alpha': Df * alpha,
            'eigenvalues': eigenvalues.tolist(),
        }
    except ImportError:
        return {'available': False, 'reason': 'sentence-transformers not installed'}
    except Exception as e:
        return {'available': False, 'reason': str(e)}


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 70)
    print("Q51.3: COMPLEX-VALUED TRAINING")
    print("=" * 70)
    print("\nHypothesis: Complex eigenvalues maintain Df × α = 8e")
    print("            while making phase structure observable")
    print()
    
    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q51_COMPLEX_TRAINING',
        'session_id': 'e552b7c1-36e6-4a90-aa17-6bc78df9c82b',
        'target_8e': 8 * np.e,
        'vocab_size': 1000,
        'embed_dim': 384,
        'training': {},
        'complex_analysis': {},
        'real_baseline': {},
        'comparison': {},
        'success_criteria': {},
    }
    
    # Build vocabulary
    print("=" * 60)
    print("SETUP: Building 1000-word vocabulary")
    print("=" * 60)
    vocab_words = build_vocabulary()
    print(f"  Vocabulary size: {len(vocab_words)} words")
    print(f"  Sample words: {vocab_words[:10]}")
    
    vocab_indices = np.arange(len(vocab_words))
    
    # Initialize model
    print("\n" + "=" * 60)
    print("TRAINING: Complex-valued embedding model")
    print("=" * 60)
    print(f"  Model dimensions: 384-dim complex (matches MiniLM-L6)")
    print(f"  Hidden layer: 512-dim complex")
    print(f"  Training epochs: 10 (or until convergence)")
    print(f"  Learning rate: 2e-5")
    print(f"  Random seed: 42")
    print()
    
    model = ComplexEmbeddingModel(
        vocab_size=len(vocab_words),
        embed_dim=384,
        hidden_dim=512,
        seed=42
    )
    
    # Train
    print("  Training progress:")
    losses = model.train(
        vocab_indices=vocab_indices,
        epochs=10,
        lr=2e-5,
        batch_size=32,
        verbose=True
    )
    
    training_summary = {
        'epochs_completed': len(losses),
        'final_loss': float(losses[-1]),
        'initial_loss': float(losses[0]) if losses else None,
        'loss_reduction': float((losses[0] - losses[-1]) / losses[0]) if losses else 0,
        'converged': len(losses) < 10,
        'stable': not np.isnan(losses[-1]) and not np.isinf(losses[-1]),
    }
    
    results['training'] = training_summary
    
    print(f"\n  Training Summary:")
    print(f"    Epochs: {training_summary['epochs_completed']}")
    print(f"    Initial loss: {training_summary['initial_loss']:.6f}")
    print(f"    Final loss: {training_summary['final_loss']:.6f}")
    print(f"    Reduction: {training_summary['loss_reduction']*100:.1f}%")
    print(f"    Converged: {training_summary['converged']}")
    print(f"    Stable: {training_summary['stable']}")
    
    if not training_summary['stable']:
        print("\n  ⚠️ WARNING: Training showed instability!")
        print("  Recording as negative result due to training failure.")
        results['status'] = 'FAILED - Training instability'
        save_results(results)
        return
    
    # Get complex embeddings
    print("\n" + "=" * 60)
    print("ANALYSIS: Complex eigenspectrum")
    print("=" * 60)
    
    emb_real, emb_imag = model.get_embeddings()
    eigenvalues, magnitudes, phases = analyze_complex_spectrum(emb_real, emb_imag)
    
    # Compute Df and α for complex magnitudes
    Df_complex = compute_df(eigenvalues)
    alpha_complex = compute_alpha(eigenvalues)
    df_alpha_complex = Df_complex * alpha_complex
    error_8e = (df_alpha_complex - 8 * np.e) / (8 * np.e) * 100
    
    power_law_fit = compute_power_law_fit(eigenvalues)
    
    print(f"\n  Complex Spectrum (|z| = sqrt(Re² + Im²)):")
    print(f"    Df (participation ratio): {Df_complex:.2f}")
    print(f"    α (power law exponent): {alpha_complex:.4f}")
    print(f"    Df × α: {df_alpha_complex:.4f}")
    print(f"    Target 8e: {8 * np.e:.4f}")
    print(f"    Error: {error_8e:+.2f}%")
    print(f"    Power law R²: {power_law_fit['r_squared']:.4f}")
    
    complex_analysis = {
        'Df': float(Df_complex),
        'alpha': float(alpha_complex),
        'Df_alpha': float(df_alpha_complex),
        'vs_8e_percent': float(error_8e),
        'power_law_fit': power_law_fit,
        'eigenvalues': eigenvalues[:50].tolist(),  # Top 50 for brevity
    }
    
    results['complex_analysis'] = complex_analysis
    
    # Phase analysis
    print("\n" + "=" * 60)
    print("PHASE ANALYSIS: Testing for structure")
    print("=" * 60)
    
    phase_stats = analyze_phase_structure(phases)
    
    print(f"\n  Phase Distribution Statistics:")
    print(f"    Mean phase: {phase_stats['mean_phase']:.4f} rad")
    print(f"    Std phase: {phase_stats['std_phase']:.4f} rad")
    print(f"    Circular concentration: {phase_stats['circular_concentration']:.4f}")
    print(f"    Phase entropy: {phase_stats['phase_entropy_bits']:.2f} bits")
    print(f"    Normalized entropy: {phase_stats['normalized_entropy']:.4f} (1.0 = uniform)")
    print(f"    KL divergence from uniform: {phase_stats['kl_divergence_from_uniform']:.4f}")
    
    print(f"\n  8-Sector Distribution (testing octant-phase hypothesis):")
    for i, (count, prob) in enumerate(zip(phase_stats['sector_counts'], phase_stats['sector_probabilities'])):
        sector_start = -np.pi + i * (2*np.pi/8)
        sector_end = sector_start + (2*np.pi/8)
        print(f"    Sector {i} [{sector_start:.2f}, {sector_end:.2f}]: {count:5.0f} ({prob*100:4.1f}%)")
    
    results['phase_analysis'] = phase_stats
    
    # Real baseline comparison
    print("\n" + "=" * 60)
    print("BASELINE: Real-valued model comparison")
    print("=" * 60)
    
    real_baseline = get_real_baseline(vocab_words)
    results['real_baseline'] = real_baseline
    
    if real_baseline['available']:
        print(f"\n  Real Baseline (MiniLM-L6):")
        print(f"    Df: {real_baseline['Df']:.2f}")
        print(f"    α: {real_baseline['alpha']:.4f}")
        print(f"    Df × α: {real_baseline['Df_alpha']:.4f}")
        print(f"    vs 8e: {(real_baseline['Df_alpha'] - 8*np.e)/(8*np.e)*100:+.2f}%")
        
        # Comparison
        print(f"\n  Complex vs Real Comparison:")
        ratio_complex_to_real = df_alpha_complex / real_baseline['Df_alpha']
        print(f"    Ratio (complex/real): {ratio_complex_to_real:.4f}")
        
        results['comparison'] = {
            'ratio_complex_to_real': float(ratio_complex_to_real),
            'complex_error_vs_8e': float(error_8e),
            'real_error_vs_8e': float((real_baseline['Df_alpha'] - 8*np.e)/(8*np.e)*100),
        }
    else:
        print(f"\n  Real baseline unavailable: {real_baseline.get('reason', 'unknown')}")
        print("  Proceeding with theoretical comparison only.")
    
    # Success criteria evaluation
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)
    
    # Criterion 1: 8e conservation
    conserved_8e = abs(error_8e) < 5.0
    print(f"\n  1. 8e Conservation (|error| < 5%):")
    print(f"     Complex Df × α = {df_alpha_complex:.4f}")
    print(f"     Target 8e = {8*np.e:.4f}")
    print(f"     Error = {error_8e:+.2f}%")
    print(f"     ✓ PASS" if conserved_8e else f"     ✗ FAIL")
    
    # Criterion 2: Phase structure
    phase_structure = phase_stats['kl_divergence_from_uniform'] > 0.5
    print(f"\n  2. Phase Structure (KL > 0.5 from uniform):")
    print(f"     KL divergence = {phase_stats['kl_divergence_from_uniform']:.4f}")
    print(f"     Normalized entropy = {phase_stats['normalized_entropy']:.4f}")
    print(f"     ✓ PASS" if phase_structure else f"     ✗ FAIL")
    
    # Criterion 3: Training stability
    stable_training = training_summary['stable'] and training_summary['loss_reduction'] > 0.1
    print(f"\n  3. Training Stability:")
    print(f"     Stable (no NaN/Inf): {training_summary['stable']}")
    print(f"     Loss reduction: {training_summary['loss_reduction']*100:.1f}%")
    print(f"     ✓ PASS" if stable_training else f"     ✗ FAIL")
    
    all_pass = conserved_8e and phase_structure and stable_training
    
    results['success_criteria'] = {
        'conserved_8e': {
            'passed': conserved_8e,
            'value': float(df_alpha_complex),
            'target': float(8 * np.e),
            'error_percent': float(error_8e),
            'threshold': 5.0,
        },
        'phase_structure': {
            'passed': phase_structure,
            'kl_divergence': phase_stats['kl_divergence_from_uniform'],
            'threshold': 0.5,
        },
        'training_stability': {
            'passed': stable_training,
            'stable': training_summary['stable'],
            'loss_reduction': training_summary['loss_reduction'],
        },
        'overall_pass': all_pass,
    }
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_pass:
        print("VERDICT: HYPOTHESIS SUPPORTED")
        print("  Complex-valued training preserves 8e conservation")
        print("  Phase structure is observable and non-uniform")
        print("  Training is stable and convergent")
    else:
        print("VERDICT: HYPOTHESIS NOT FULLY SUPPORTED")
        if not conserved_8e:
            print(f"  - 8e conservation violated ({error_8e:+.2f}% error)")
        if not phase_structure:
            print(f"  - Phase distribution too uniform (KL = {phase_stats['kl_divergence_from_uniform']:.3f})")
        if not stable_training:
            print(f"  - Training instability detected")
    print("=" * 70)
    
    results['verdict'] = 'SUPPORTED' if all_pass else 'NOT_SUPPORTED'
    results['status'] = 'COMPLETE'
    
    # Save results
    save_results(results)


def save_results(results):
    """Save results to JSON file"""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = results_dir / f'q51_complex_training_{timestamp}.json'
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved: {path}")
    return path


if __name__ == '__main__':
    main()
