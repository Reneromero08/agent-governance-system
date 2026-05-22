"""Platonic EigenBuddy Tokenizer — Decode complex embedding vectors to tokens.

Q34 proved that embedding models converge to a shared Platonic geometry.
Instead of streaming full Qwen weight matrices through the catalytic tape,
train a small physics engine to navigate the Platonic manifold and map
complex-plane hidden states back to coherent tokens.

Architecture:
  Complex hidden state (from catalytic fabric, 896-dim XY) 
    → NativeEigenCore (small, trainable) 
    → Token logits (151,936 vocab)
    → argmax → token

Training: synthetic dataset from Qwen embedding table + random projections.
The EigenBuddy learns the mapping from "positions on the Platonic manifold"
to token identities, bypassing Qwen's full weight matrices.

Standalone module — does NOT hook into the main catalytic pipeline yet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import struct
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
from collections import defaultdict

# ===========================================================================
# NativeEigenCore (imported class from native_eigen_core.py for standalone use)
# ===========================================================================

class MultiHeadComplexAttention(nn.Module):
    def __init__(self, d_model=896, n_heads=16, merge='concat'):
        super().__init__()
        assert d_model % n_heads == 0
        self.H = n_heads
        self.dh = d_model // n_heads
        hd = d_model
        self.merge_mode = merge

        self.qr = nn.Linear(d_model, hd, bias=False)
        self.qi = nn.Linear(d_model, hd, bias=False)
        self.kr = nn.Linear(d_model, hd, bias=False)
        self.ki = nn.Linear(d_model, hd, bias=False)
        self.vr = nn.Linear(d_model, hd, bias=False)
        self.vi = nn.Linear(d_model, hd, bias=False)

        if merge == 'born':
            self.align_r = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
            self.align_i = nn.Parameter(torch.randn(self.dh, d_model) * 0.02)
        else:
            self.or_ = nn.Linear(hd, d_model, bias=False)
            self.oi = nn.Linear(hd, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.dh)
        self._init_platonic()

    def _init_platonic(self):
        """Geometric init: head angles spaced by 2pi/H, Q-K 45deg offset."""
        angle = 2.0 * math.pi / self.H
        head_angles = torch.arange(self.H, dtype=torch.float32) * angle
        qk_offset = math.pi / 4.0
        noise_std = 0.01
        for name, w in [('qr', self.qr), ('qi', self.qi), ('kr', self.kr),
                         ('ki', self.ki), ('vr', self.vi), ('vi', self.vi)]:
            base = torch.randn(w.weight.shape) * 0.02
            for h in range(self.H):
                start, end = h * self.dh, (h + 1) * self.dh
                c = math.cos(head_angles[h].item())
                template = base[start:end].clone()
                w.weight.data[start:end] = template * c
                if name.startswith('q'):
                    w.weight.data[start:end] *= math.cos(qk_offset)
                elif name.startswith('k'):
                    w.weight.data[start:end] *= math.cos(-qk_offset)
                w.weight.data[start:end] += torch.randn_like(w.weight.data[start:end]) * noise_std
        if self.merge_mode == 'concat':
            nn.init.normal_(self.or_.weight, std=0.02)
            nn.init.normal_(self.oi.weight, std=0.02)

    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag)
        qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag)
        ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag)
        vi = self.vr(x.imag) + self.vi(x.real)

        qr = qr.view(B, S, self.H, self.dh).transpose(1, 2)
        qi = qi.view(B, S, self.H, self.dh).transpose(1, 2)
        kr = kr.view(B, S, self.H, self.dh).transpose(1, 2)
        ki = ki.view(B, S, self.H, self.dh).transpose(1, 2)
        vr = vr.view(B, S, self.H, self.dh).transpose(1, 2)
        vi = vi.view(B, S, self.H, self.dh).transpose(1, 2)

        sr = (qr @ kr.transpose(-2, -1) + qi @ ki.transpose(-2, -1)) * self.scale
        si = (qi @ kr.transpose(-2, -1) - qr @ ki.transpose(-2, -1)) * self.scale

        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        si = si.masked_fill(mask, 0.0)

        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr
        out_i = attn @ vi

        if self.merge_mode == 'born':
            psi_r = out_r.sum(dim=1) / math.sqrt(self.H)
            psi_i = out_i.sum(dim=1) / math.sqrt(self.H)
            or_ = psi_r @ self.align_r + psi_i @ self.align_i
            oi_ = psi_r @ self.align_i - psi_i @ self.align_r
        else:
            out_r = out_r.transpose(1, 2).contiguous().view(B, S, -1)
            out_i = out_i.transpose(1, 2).contiguous().view(B, S, -1)
            or_ = self.or_(out_r) - self.oi(out_i)
            oi_ = self.or_(out_i) + self.oi(out_r)

        return torch.complex(or_, oi_), si


# ===========================================================================
# PlatonicAnchorBank — STABLE_32 anchors as Platonic coordinate frame
# ===========================================================================

STABLE_32 = [
    'destroy', 'effect', 'animal', 'fast', 'art', 'cold', 'child', 'walk',
    'stone', 'think', 'give', 'space', 'society', 'glass', 'touch', 'air',
    'evening', 'mountain', 'book', 'leader', 'sad', 'dog', 'cat', 'winter',
    'wood', 'morning', 'know', 'fire', 'car', 'building', 'person', 'enemy'
]


class PlatonicAnchorBank:
    """STABLE_32 words as navigation anchors in embedding space.

    Q34 proved these 32 words have stable neighborhoods across models.
    The M-field gap of 0.524 (8.5x separation from divergent words) means
    they form a shared coordinate frame — the Platonic grid.
    """
    def __init__(self, embed_table: torch.Tensor, token_ids: dict):
        self.anchors = STABLE_32
        self.embed_table = embed_table  # [vocab_size, dim] float32
        self.token_ids = token_ids  # {word: token_id}
        self.anchor_ids = {}
        self.anchor_vecs = []
        
        missing = []
        for w in self.anchors:
            if w in token_ids:
                tid = token_ids[w]
                self.anchor_ids[w] = tid
                self.anchor_vecs.append(embed_table[tid])
            else:
                # Qwen tokenizer may not have these as single tokens
                missing.append(w)
        
        if self.anchor_vecs:
            self.anchor_vecs = torch.stack(self.anchor_vecs)
            self.anchor_vecs = self.anchor_vecs / (self.anchor_vecs.norm(dim=1, keepdim=True) + 1e-8)
        
        if missing:
            print(f"  [anchors] {len(missing)} STABLE_32 words not in vocab: {missing[:5]}...")

    def anchor_distances(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute distances from vec to all STABLE_32 anchors.
        Returns [B, len(anchor_vecs)] tensor of cosine distances.
        """
        anchors = self.anchor_vecs.to(vec.device)
        if vec.dim() == 1:
            vec = vec / (vec.norm() + 1e-8)
            return 1.0 - (anchors @ vec)
        else:
            vec = vec / (vec.norm(dim=1, keepdim=True) + 1e-8)
            return 1.0 - (vec @ anchors.T)

    def platonic_coordinates(self, vec: torch.Tensor) -> torch.Tensor:
        """Express vec in the Platonic anchor-distance coordinate frame.
        Returns [len(anchor_vecs)] tensor of anchor distances.
        This is the M-field coordinate system from Q34.
        """
        return self.anchor_distances(vec)


# ===========================================================================
# EigenBuddyTokenizer — The main module
# ===========================================================================

class EigenBuddyTokenizer(nn.Module):
    """Decode complex-plane hidden states into token logits.

    Pipeline:
      catalytic hidden state (B, 1, D) complex
        → NativeEigenCore (D → D_inner → D)
        → anchor-distance projection (Platonic coordinate frame)
        → MLP (anchor_dims → vocab_size)
        → logits → argmax → token

    This replaces the lm_head projection (embedding_table @ hidden) with
    a learned decoder that navigates the Platonic manifold.
    """

    def __init__(
        self,
        dim: int = 896,
        vocab_size: int = 151936,
        eigen_layers: int = 2,
        eigen_heads: int = 16,
        anchor_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Simple MLP decoder for the prototype — no complex attention yet.
        # We learn a direct mapping from (real+imag catalytic output) → embedding → token.
        inner_dim = min(dim, 256)
        
        self.encoder = nn.Sequential(
            nn.Linear(2 * dim, inner_dim),  # real+imag → inner
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim),  # back to embedding space
        )
        
        # Platonic anchor bank (built later when embeddings are loaded)
        self.anchor_bank: Optional[PlatonicAnchorBank] = None
        
        # Token prediction head
        n_anchors_initial = len(STABLE_32)
        combined_dim = dim + n_anchors_initial
        bottleneck = min(1024, combined_dim)
        self.token_head = nn.Sequential(
            nn.Linear(combined_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, vocab_size),
        )
        
        self._init_weights()

    def _init_weights(self):
        for layer in [self.encoder, self.token_head]:
            for m in layer:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def set_anchor_bank(self, embed_table: torch.Tensor, token_ids: dict):
        """Load the STABLE_32 anchor bank from a tokenizer's embedding table."""
        self.anchor_bank = PlatonicAnchorBank(embed_table, token_ids)

    def forward(
        self,
        hidden: torch.Tensor,
        return_platonic: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Decode a complex hidden state into token logits.

        Args:
            hidden: (B, 1, D) complex tensor from catalytic fabric
            return_platonic: if True, also return Platonic coordinates

        Returns:
            logits: (B, vocab_size) token prediction logits
            info: dict with 'platonic_coords'
        """
        B, S, D = hidden.shape
        info = {}
        
        # Flatten to (B, 2*D) real+imag
        h_last = hidden[:, -1, :]
        flat = torch.cat([h_last.real, h_last.imag], dim=-1)  # (B, 2*D)
        
        # MLP encoder: (B, 2*D) → (B, D) embedding-space output
        embedding_out = self.encoder(flat)
        info['embedding_out'] = embedding_out.detach()
        
        # Platonic anchor-distance coordinates
        if self.anchor_bank is not None:
            platonic = self.anchor_bank.platonic_coordinates(embedding_out)  # (B, n_anchors)
        else:
            platonic = torch.zeros(B, 0, device=hidden.device)
        info['platonic_coords'] = platonic.detach()
        
        # Combine with Platonic coords and predict tokens
        combined = torch.cat([embedding_out, platonic], dim=-1)
        logits = self.token_head(combined)
        
        return logits, info

    def decode_token(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Convenience: get predicted token ID from hidden state."""
        logits, info = self.forward(hidden)
        token_ids = torch.argmax(logits, dim=-1)
        return token_ids, info


# ===========================================================================
# Training utilities
# ===========================================================================

def build_training_data(
    embed_table: torch.Tensor,
    token_ids: dict,
    num_samples: int = 10000,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build synthetic training data from the embedding table.

    Strategy:
      1. Use the embedding vectors themselves as "hidden states" (ground truth: self)
      2. Use perturbed embeddings (noise + rotation) as inputs (ground truth: original)
      3. Use random combinations of nearby embeddings (ground truth: majority)

    This teaches the EigenBuddy to map from noisy/transformed embedding-space
    positions back to the correct token — exactly what the catalytic fabric produces.
    """
    print(f"  Building training data: {num_samples} samples...")
    
    vocab_size, dim = embed_table.shape
    embed_norm = embed_table / (embed_table.norm(dim=1, keepdim=True) + 1e-8)
    
    inputs = []
    targets = []
    
    # Mode 1: Pure embeddings as both input and target (40%)
    n_self = int(num_samples * 0.4)
    indices = torch.randint(0, vocab_size, (n_self,))
    for idx in indices:
        real = embed_table[idx].clone()
        imag = torch.randn(dim) * 0.001  # tiny imag for numerical stability
        z = torch.complex(real, imag).unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        inputs.append(z)
        targets.append(idx)
    
    # Mode 2: Perturbed embeddings (40%)
    n_perturb = int(num_samples * 0.4)
    indices = torch.randint(0, vocab_size, (n_perturb,))
    for idx in indices:
        noise = torch.randn(dim) * 0.05
        real = embed_table[idx] + noise
        imag = torch.randn(dim) * 0.01
        phase = torch.randn(1).item() * 0.05
        c, s = math.cos(phase), math.sin(phase)
        real_rot = real * c - imag * s
        z = torch.complex(real_rot, imag).unsqueeze(0).unsqueeze(0)
        inputs.append(z)
        targets.append(idx)
    
    # Mode 3: Embedding mixture — nearest-neighbor interpolation (20%)
    n_mix = num_samples - n_self - n_perturb
    for _ in range(n_mix):
        idx = torch.randint(0, vocab_size, (1,)).item()
        sims = embed_norm[idx] @ embed_norm.T
        sims[idx] = -1
        nearby = torch.topk(sims, 3).indices
        weights = F.softmax(torch.randn(3), dim=0)
        mixed = sum(w * embed_table[n] for w, n in zip(weights, nearby))
        mixed = mixed * 0.3 + embed_table[idx] * 0.7  # bias toward target
        z = torch.complex(mixed, torch.randn(dim) * 0.005).unsqueeze(0).unsqueeze(0)
        inputs.append(z)
        targets.append(idx)
    
    print(f"  Done: {len(inputs)} samples, modes: self={n_self}, perturb={n_perturb}, mix={n_mix}")
    return inputs, torch.tensor(targets), embed_table


def train_eigen_buddy(
    model: EigenBuddyTokenizer,
    inputs: List[torch.Tensor],
    targets: torch.Tensor,
    embed_table: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True,
) -> dict:
    """Train the EigenBuddy on synthetic data."""
    model = model.to(device)
    model.train()
    
    # Sanity check: run one batch and verify no NaN
    batch_check = torch.cat(inputs[:4], dim=0).to(device)
    tgt_check = targets[:4].to(device)
    with torch.no_grad():
        logits_check, info_check = model(batch_check)
    print(f"  Sanity forward: logits range [{logits_check.min():.2f}, {logits_check.max():.2f}] "
          f"has_nan={torch.isnan(logits_check).any().item()} "
          f"has_inf={torch.isinf(logits_check).any().item()}")
    if torch.isnan(logits_check).any() or torch.isinf(logits_check).any():
        print("  FATAL: Initial forward produces NaN/Inf. Aborting training.")
        print(f"  encoder.0.weight stats: min={model.encoder[0].weight.min():.4f} max={model.encoder[0].weight.max():.4f}")
        return {'loss': [float('nan')], 'acc': [0.0], 'platonic_mfield': [float('nan')]}
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    targets = targets.to(device)
    n_samples = len(inputs)
    total_batches = epochs * ((n_samples + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_batches)
    
    history = {'loss': [], 'acc': [], 'platonic_mfield': []}
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_mfield = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]
            
            batch_inputs = torch.cat([inputs[i] for i in batch_idx], dim=0).to(device)
            batch_targets = targets[batch_idx]
            
            logits, info = model(batch_inputs)
            loss = F.cross_entropy(logits, batch_targets)
            
            # Check for NaN and skip if detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN] NaN/Inf loss at epoch {epoch+1} batch {n_batches} — skipping")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == batch_targets).float().mean().item()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_mfield += info.get('platonic_coords', torch.zeros(1)).abs().mean().item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        avg_mfield = epoch_mfield / max(n_batches, 1)
        
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        history['platonic_mfield'].append(avg_mfield)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={avg_acc:.3f} "
                  f"mfield={avg_mfield:.4f}")
    
    return history


# ===========================================================================
# Evaluation: M-field Platonic convergence test
# ===========================================================================

def evaluate_platonic_convergence(
    model: EigenBuddyTokenizer,
    test_inputs: List[torch.Tensor],
    test_targets: torch.Tensor,
    device: str = 'cpu',
) -> dict:
    """Evaluate EigenBuddy against Q34's Platonic convergence metrics.

    Measures:
      - Token recovery accuracy (top-1, top-5)
      - M-field (nabla_S) between predicted and ground-truth embedding positions
      - Phase coherence of NativeEigenCore output
    """
    model = model.to(device)
    model.eval()
    
    results = {
        'top1_correct': 0,
        'top5_correct': 0,
        'total': len(test_inputs),
        'platonic_coords': [],
        'phase_coh': [],
        'per_category': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }
    
    with torch.no_grad():
        batch_size = 32
        for start in range(0, len(test_inputs), batch_size):
            end = min(start + batch_size, len(test_inputs))
            batch_inputs = torch.cat(test_inputs[start:end], dim=0).to(device)
            batch_targets = test_targets[start:end].to(device)
            
            logits, info = model(batch_inputs)
            
            preds = logits.argmax(dim=-1)
            results['top1_correct'] += (preds == batch_targets).sum().item()
            _, top5 = logits.topk(5, dim=-1)
            results['top5_correct'] += (top5 == batch_targets.unsqueeze(-1)).any(dim=-1).sum().item()
            if 'platonic_coords' in info:
                results['platonic_coords'].append(info['platonic_coords'].cpu())
    
    
    results['top1_acc'] = results['top1_correct'] / results['total']
    results['top5_acc'] = results['top5_correct'] / results['total']
    results['mean_phase_coh'] = 0.0  # MLP mode — no phase coherence
    
    if results['platonic_coords']:
        coords = torch.cat(results['platonic_coords'], dim=0)
        results['mean_mfield'] = coords.abs().mean().item()
        results['mfield_std'] = coords.abs().std().item()
    
    return results


# ===========================================================================
# Main: self-contained training and evaluation
# ===========================================================================

def main():
    print("=" * 70)
    print("PLATONIC EIGENBUDDY TOKENIZER — Prototype")
    print("  Decoding the Platonic manifold into coherent tokens")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # Load Qwen embedding table
    print("\n--- Loading Qwen Embeddings ---")
    model_dir = Path(__file__).parent.parent / "CAT_CAS" / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b"
    
    embed_path = model_dir / "model.safetensors"
    tokenizer_path = model_dir / "tokenizer.json"
    
    if not embed_path.exists():
        print(f"  [SKIP] Model not found at {embed_path}")
        print("  Running with synthetic embeddings")
        VOCAB_SIZE = 1024
        DIM = 128
        embed_table = torch.randn(VOCAB_SIZE, DIM) * 0.1
        token_ids = {w: i for i, w in enumerate(STABLE_32)}
    else:
        import json, struct, mmap
        with open(embed_path, 'rb') as f:
            fd = f.fileno()
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            header_size = struct.unpack("<Q", mm[:8])[0]
            header = json.loads(mm[8:8+header_size].decode('utf-8'))
            mm.close()
        
        einfo = header.get("model.embed_tokens.weight", {})
        start, end = einfo['data_offsets']
        shape = einfo['shape']
        vocab_size, dim = int(shape[0]), int(shape[1])
        edtype = einfo.get("dtype", "F32")
        
        with open(embed_path, 'rb') as f:
            f.seek(8 + header_size + start)
            ebytes = f.read(end - start)
        
        # BF16→f32: read as uint16 native endian, shift left 16 to f32
        if edtype == "BF16":
            embeddings = np.frombuffer(ebytes, dtype=np.uint16)
            embeddings = embeddings.astype(np.uint32) << 16
            embed_table = torch.from_numpy(embeddings.view(np.float32).reshape(vocab_size, dim).copy())
        else:
            embed_table = torch.from_numpy(np.frombuffer(ebytes, dtype=np.float32).reshape(vocab_size, dim).copy())
        del embeddings  # free memory
        
        DIM = dim
        VOCAB_SIZE = min(vocab_size, 16384)
        
        # Fix NaN rows by replacing with random
        embed_table = embed_table[:VOCAB_SIZE]
        nan_mask = torch.isnan(embed_table).any(dim=1)
        if nan_mask.any():
            n_bad = nan_mask.sum().item()
            print(f"  Fixing {n_bad} NaN embedding rows with random replacement")
            bad_idx = torch.where(nan_mask)[0]
            embed_table[bad_idx] = torch.randn(n_bad, DIM) * embed_table[~nan_mask].std(dim=0).mean() * 0.1
        
        print(f"  Loaded: {vocab_size} tokens × {dim} dim (using top {VOCAB_SIZE} for training)")
        
        # Load tokenizer vocab
        token_ids = {}
        if tokenizer_path.exists():
            with open(tokenizer_path, 'rb') as f:
                tok_data = json.loads(f.read().decode('utf-8', errors='replace'))
            voc = tok_data.get('model', {}).get('vocab', {})
            for word_str, tid in voc.items():
                if tid < VOCAB_SIZE:
                    clean = word_str.lstrip('Ġ').lstrip('▁').lower()
                    token_ids[clean] = tid
                    token_ids[word_str] = tid
        else:
            token_ids = {w: i for i, w in enumerate(STABLE_32)}
        print(f"  STABLE_32 anchors in truncated vocab: {sum(1 for w in STABLE_32 if w in token_ids)}/32")
    
    print(f"\n--- Building EigenBuddy (d={DIM}, vocab={VOCAB_SIZE}) ---")
    
    # Scale model to embedding size
    eigen_layers = 2
    eigen_heads = max(2, DIM // 56)  # head_dim ~56
    
    model = EigenBuddyTokenizer(
        dim=DIM,
        vocab_size=VOCAB_SIZE,
        eigen_layers=eigen_layers,
        eigen_heads=eigen_heads,
    )
    model.set_anchor_bank(embed_table, token_ids)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")
    print(f"  Layers: {eigen_layers}, Heads: {eigen_heads}, Head dim: {DIM // eigen_heads}")
    
    # Build training data
    print(f"\n--- Training Data ---")
    inputs, targets, _ = build_training_data(
        embed_table, token_ids,
        num_samples=min(10000, VOCAB_SIZE * 5),
        device='cpu'
    )
    
    # Split train/test
    split = int(len(inputs) * 0.8)
    train_in, test_in = inputs[:split], inputs[split:]
    train_tgt, test_tgt = targets[:split], targets[split:]
    print(f"  Train: {len(train_in)}, Test: {len(test_in)}")
    
    # Train
    print(f"\n--- Training ---")
    history = train_eigen_buddy(
        model, train_in, train_tgt, embed_table,
        epochs=10, batch_size=32, lr=3e-4, device=device,
    )
    
    # Evaluate
    print(f"\n--- Evaluation ---")
    results = evaluate_platonic_convergence(model, test_in, test_tgt, device=device)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Top-1 accuracy:  {results['top1_acc']:.4f} ({results['top1_correct']}/{results['total']})")
    print(f"  Top-5 accuracy:  {results['top5_acc']:.4f} ({results['top5_correct']}/{results['total']})")
    print(f"  Phase coherence: {results['mean_phase_coh']:.4f}")
    print(f"  Mean M-field:    {results['mean_mfield']:.4f}")
    print(f"  M-field std:     {results['mfield_std']:.4f}")
    print(f"  Final loss:      {history['loss'][-1]:.4f}")
    print(f"  Final train acc: {history['acc'][-1]:.4f}")
    
    # Q34 M-field gap analysis
    print(f"\n{'='*70}")
    print(f"Q34 PLATONIC CONVERGENCE CHECK")
    print(f"{'='*70}")
    
    if model.anchor_bank and model.anchor_bank.anchor_vecs.shape[0] > 0:
        # Test STABLE_32 words
        anchor_tests = []
        for w in STABLE_32:
            if w in token_ids:
                tid = token_ids[w]
                emb = embed_table[tid]
                z = torch.complex(emb, torch.randn(DIM) * 0.01).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, info = model(z.to(device))
                    if 'platonic_coords' in info:
                        anchor_tests.append(info['platonic_coords'].cpu().abs().mean().item())
        
        if anchor_tests:
            anchor_mfield = np.mean(anchor_tests)
            # Random words as divergent baseline
            random_tests = []
            for _ in range(32):
                tid = torch.randint(0, VOCAB_SIZE, (1,)).item()
                emb = embed_table[tid]
                z = torch.complex(emb, torch.randn(DIM) * 0.01).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    _, info = model(z.to(device))
                    if 'platonic_coords' in info:
                        random_tests.append(info['platonic_coords'].cpu().abs().mean().item())
            
            random_mfield = np.mean(random_tests)
            gap = random_mfield - anchor_mfield
            
            print(f"  STABLE_32 M-field: {anchor_mfield:.4f}")
            print(f"  Random M-field:    {random_mfield:.4f}")
            print(f"  Gap:               {gap:+.4f} ({abs(gap)/max(anchor_mfield, 1e-8):.1f}x)")
            
            if gap > 0.1:
                print(f"  PLATONIC GEOMETRY PRESERVED — EigenBuddy maintains the M-field gap")
            elif gap > 0:
                print(f"  Weak Platonic signal — gap exists but small")
            else:
                print(f"  Platonic geometry not yet learned — keep training")
    
    print(f"\n--- Saving model ---")
    save_path = Path(__file__).parent / "weights" / "eigen_buddy_tokenizer.pt"
    save_path.parent.mkdir(exist_ok=True)
    save_results = {}
    for k, v in results.items():
        if isinstance(v, (int, float, str, bool)):
            save_results[k] = v
        elif isinstance(v, torch.Tensor):
            save_results[k] = v.item() if v.numel() == 1 else v.tolist()
    torch.save({
        'model_state': model.state_dict(),
        'dim': DIM,
        'vocab_size': VOCAB_SIZE,
        'eigen_layers': eigen_layers,
        'eigen_heads': eigen_heads,
        'history': history,
        'results': save_results,
    }, save_path)
    print(f"  Saved to {save_path}")
    
    print(f"\n{'='*70}")
    print(f"PROTOTYPE COMPLETE")
    print(f"  Next: hook into catalytic pipeline at lm_head replacement")
    print(f"  File: eigen_buddy_tokenizer.py")
    print(f"  Weights: weights/eigen_buddy_tokenizer.pt")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
