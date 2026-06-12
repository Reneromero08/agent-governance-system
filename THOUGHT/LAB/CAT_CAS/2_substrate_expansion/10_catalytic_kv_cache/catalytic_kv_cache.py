import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hashlib
from typing import Dict, Tuple, Optional, List

class EigenProjector(nn.Module):
    """
    Projects KV activations between high-dimensional model space and a low-dimensional manifold.
    Uses PCA-initialized, optionally learnable projections.
    """
    def __init__(self, full_dim: int, k: int):
        super().__init__()
        self.full_dim = full_dim
        self.k = k
        self.proj = nn.Parameter(torch.randn(k, full_dim) / math.sqrt(full_dim))
        self.mean = nn.Parameter(torch.zeros(full_dim), requires_grad=False)

    def init_from_pca(self, data: torch.Tensor):
        """Initialize projection matrix using SVD on calibration data."""
        with torch.no_grad():
            mean = data.mean(dim=0)
            centered = data - mean
            # SVD: centered = U * S * V^T
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            self.proj.copy_(Vt[:self.k])
            self.mean.copy_(mean)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Project high-dim states to low-dim manifold: (*, full_dim) -> (*, k)"""
        return F.linear(x - self.mean, self.proj)

    def decompress(self, x_compressed: torch.Tensor) -> torch.Tensor:
        """Lifts low-dim manifold coordinates back to full space: (*, k) -> (*, full_dim)"""
        return F.linear(x_compressed, self.proj.T) + self.mean


class HeavyHitterOracle:
    """
    Manages temporal pruning of the KV Cache.
    Keeps a sliding window of recent tokens and selects the top 'heavy hitters'
    from the older history based on accumulated attention scores.
    """
    def __init__(self, max_history: int, active_window: int):
        self.max_history = max_history       # Total historical tokens to keep (M)
        self.active_window = active_window   # Sliding local window tokens to keep (W)
        assert max_history >= active_window, "max_history must be larger than active_window"
        
        # Accumulator for attention scores per token index
        self.attn_scores: List[float] = []

    def update_scores(self, step_attn_weights: torch.Tensor):
        """
        Updates the running importance scores for each token index.
        step_attn_weights shape: (batch, num_heads, query_len, key_len)
        """
        # Average attention weights across batch, heads, and query tokens
        # shape: (key_len)
        avg_weights = step_attn_weights.detach().mean(dim=(0, 1, 2)).tolist()
        
        # Accumulate scores for existing tokens
        for i, w in enumerate(avg_weights):
            if i < len(self.attn_scores):
                self.attn_scores[i] += w
            else:
                self.attn_scores.append(w)

    def get_keep_indices(self, total_tokens: int) -> torch.Tensor:
        """
        Computes the indices of tokens to retain in the KV cache.
        Returns a sorted 1D tensor of indices.
        """
        if total_tokens <= self.max_history:
            return torch.arange(total_tokens, dtype=torch.long)

        # Split into active local window and historical window
        history_len = total_tokens - self.active_window
        
        # Candidates for pruning: historical tokens
        hist_scores = self.attn_scores[:history_len]
        
        # Find top heavy hitters in history
        keep_hist_count = self.max_history - self.active_window
        _, top_hist_indices = torch.topk(
            torch.tensor(hist_scores, dtype=torch.float32), 
            k=keep_hist_count, 
            sorted=False
        )
        
        # Local active window indices are always kept
        active_indices = torch.arange(history_len, total_tokens, dtype=torch.long)
        
        # Combine and sort indices to preserve causal order
        combined = torch.cat([top_hist_indices, active_indices])
        sorted_indices, _ = torch.sort(combined)
        return sorted_indices

    def prune_scores(self, keep_indices: torch.Tensor):
        """Synchronizes score accumulator with pruned indices."""
        idx_list = keep_indices.tolist()
        self.attn_scores = [self.attn_scores[i] for i in idx_list]


class CatalyticKVCache:
    """
    Manages spatial and temporal KV cache compression and stores the activations
    directly on a shared, pre-allocated VRAM tape.
    Uses bit-exact XOR operations to avoid floating point roundoff errors during tape restoration.
    """
    def __init__(
        self,
        tape: torch.Tensor,
        tape_background: torch.Tensor,
        k_projector: EigenProjector,
        v_projector: EigenProjector,
        num_heads: int,
        head_dim: int,
        max_history: int = 32,
        active_window: int = 16
    ):
        self.tape = tape                       # Pre-allocated VRAM tape
        self.tape_background = tape_background # Original dirty background values of the tape
        self.k_projector = k_projector
        self.v_projector = v_projector
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.oracle = HeavyHitterOracle(max_history, active_window)
        self.num_cached_tokens = 0
        
        self.compressed_dim = k_projector.k
        # We store 1 Key vector (k-dim) and 1 Value vector (k-dim) per token
        self.token_kv_elements = 2 * self.compressed_dim
        self.token_kv_bytes = self.token_kv_elements * tape.element_size()

    def add_step(self, k_step: torch.Tensor, v_step: torch.Tensor):
        """
        Compresses and adds the new key/value projection of the current step to the cache.
        k_step, v_step shape: (batch, num_heads, 1, head_dim)
        """
        batch, heads, _, h_dim = k_step.shape
        assert batch == 1, "Only batch size 1 supported in prototype"

        # 1. Spatial compression: project to k dimensions
        k_flat = k_step.squeeze(2).reshape(-1, heads * h_dim) # (1, heads * head_dim)
        v_flat = v_step.squeeze(2).reshape(-1, heads * h_dim)
        
        k_comp = self.k_projector.compress(k_flat) # (1, k)
        v_comp = self.v_projector.compress(v_flat) # (1, k)
        
        # Flat representation: concatenate compressed Key and Value
        kv_compressed = torch.cat([k_comp, v_comp], dim=1).flatten() # (2 * k)
        
        # 2. Write to the shared VRAM tape at the next slot by XORing with dirty background
        offset = self.num_cached_tokens * self.token_kv_elements
        tape_slice = self.tape[offset : offset + self.token_kv_elements].view(torch.int32)
        kv_view = kv_compressed.view(torch.int32)
        torch.bitwise_xor(tape_slice, kv_view, out=tape_slice)
        
        self.num_cached_tokens += 1

    def retrieve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves and decompresses the cached keys and values from the VRAM tape.
        Returns: k_full, v_full of shape (batch, num_heads, cached_len, head_dim)
        """
        # Read from the VRAM tape by XORing active with background
        total_elements = self.num_cached_tokens * self.token_kv_elements
        tape_active_view = self.tape[:total_elements].view(torch.int32)
        tape_bg_view = self.tape_background[:total_elements].view(torch.int32)
        tape_read = torch.bitwise_xor(tape_active_view, tape_bg_view).view(torch.float32)
        
        # Reshape to step segments
        kv_all = tape_read.reshape(self.num_cached_tokens, 2 * self.compressed_dim)
        
        # Split into key and value segments
        k_comp = kv_all[:, :self.compressed_dim] # (cached_len, k)
        v_comp = kv_all[:, self.compressed_dim:] # (cached_len, k)
        
        # 3. Spatial decompression
        k_full_flat = self.k_projector.decompress(k_comp) # (cached_len, heads * head_dim)
        v_full_flat = self.v_projector.decompress(v_comp)
        
        # Reshape to (batch, heads, cached_len, head_dim)
        k_full = k_full_flat.reshape(self.num_cached_tokens, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v_full = v_full_flat.reshape(self.num_cached_tokens, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        
        return k_full, v_full

    def prune(self):
        """
        Performs temporal pruning using HeavyHitterOracle.
        Rearranges the compressed states on the VRAM tape to compact them.
        """
        if self.num_cached_tokens <= self.oracle.max_history:
            return

        # 1. Determine which indices to keep
        keep_indices = self.oracle.get_keep_indices(self.num_cached_tokens)
        
        # 2. Extract, compact, and rewrite on the VRAM tape
        # Fetch current compressed states by XORing current tape with background
        total_elements = self.num_cached_tokens * self.token_kv_elements
        tape_active_view = self.tape[:total_elements].view(torch.int32)
        tape_bg_view = self.tape_background[:total_elements].view(torch.int32)
        current_comp_int = torch.bitwise_xor(tape_active_view, tape_bg_view).reshape(
            self.num_cached_tokens, self.token_kv_elements
        )
        
        # Select the retained steps
        retained_comp_int = current_comp_int[keep_indices]
        
        # Clear the old slots on the VRAM tape back to the background state
        self.tape[:total_elements] = self.tape_background[:total_elements]
        
        # Write the compacted steps back to the front of the tape
        new_cached_count = len(keep_indices)
        new_elements = new_cached_count * self.token_kv_elements
        
        tape_dest_view = self.tape[:new_elements].view(torch.int32)
        retained_flat_int = retained_comp_int.flatten()
        torch.bitwise_xor(tape_dest_view, retained_flat_int, out=tape_dest_view)
        
        # Update oracle state and token counter
        self.oracle.prune_scores(keep_indices)
        self.num_cached_tokens = new_cached_count

    def restore_tape(self):
        """
        Inverse operation: removes all perturbations from the VRAM tape,
        returning it to 100% of its original background state.
        """
        total_elements = self.num_cached_tokens * self.token_kv_elements
        if total_elements > 0:
            # XOR current tape view with current compressed values to restore background
            tape_active_view = self.tape[:total_elements].view(torch.int32)
            tape_bg_view = self.tape_background[:total_elements].view(torch.int32)
            current_comp_int = torch.bitwise_xor(tape_active_view, tape_bg_view)
            
            torch.bitwise_xor(tape_active_view, current_comp_int, out=tape_active_view)
            
        self.num_cached_tokens = 0
