"""Native text-only Qwen3.5/Qwen3.6 hybrid inference for flat v1 .holo.

The compressed path uses W ~= U @ SVh for the large projections and streams
the exact numerical shell from the original safetensors checkpoint.  The same
implementation can run entirely from exact tensors and acts as its own
reference engine.  Vision and MTP are intentionally outside this text engine.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


LAB = Path(__file__).resolve().parents[2]
DEFAULT_HOLO = LAB / "output" / "qwen_27b_k256.holo"
DEFAULT_MODEL = Path(
    "/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.6-27B"
)

HIDDEN = 5120
LAYERS = 64
VOCAB = 248320
EPS = 1e-6
FULL_Q_HEADS = 24
FULL_KV_HEADS = 4
FULL_HEAD_DIM = 256
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
GDN_QK_HEADS = 16
GDN_V_HEADS = 48
GDN_HEAD_DIM = 128
GDN_KEY_WIDTH = 2048
GDN_VALUE_WIDTH = 6144
GDN_CONV_WIDTH = 10240


def _apply_config(text_config: dict):
    """Set module-level architecture constants from a Qwen3.5 text_config."""
    global HIDDEN, LAYERS, VOCAB, FULL_Q_HEADS, FULL_KV_HEADS, FULL_HEAD_DIM
    global ROTARY_DIM, GDN_QK_HEADS, GDN_V_HEADS, GDN_HEAD_DIM
    global GDN_KEY_WIDTH, GDN_VALUE_WIDTH, GDN_CONV_WIDTH
    HIDDEN = int(text_config.get("hidden_size", HIDDEN))
    LAYERS = int(text_config.get("num_hidden_layers", LAYERS))
    VOCAB = int(text_config.get("vocab_size", VOCAB))
    FULL_Q_HEADS = int(text_config.get("num_attention_heads", FULL_Q_HEADS))
    FULL_KV_HEADS = int(text_config.get("num_key_value_heads", FULL_KV_HEADS))
    FULL_HEAD_DIM = int(text_config.get("head_dim", FULL_HEAD_DIM))
    prf = text_config.get("partial_rotary_factor", 0.25)
    ROTARY_DIM = int(FULL_HEAD_DIM * prf)
    GDN_QK_HEADS = int(text_config.get("linear_num_key_heads", GDN_QK_HEADS))
    GDN_V_HEADS = int(text_config.get("linear_num_value_heads", GDN_V_HEADS))
    GDN_HEAD_DIM = int(text_config.get("linear_key_head_dim", GDN_HEAD_DIM))
    GDN_KEY_WIDTH = GDN_QK_HEADS * GDN_HEAD_DIM
    GDN_VALUE_WIDTH = GDN_V_HEADS * GDN_HEAD_DIM
    GDN_CONV_WIDTH = 2 * GDN_KEY_WIDTH + GDN_VALUE_WIDTH


def _human_bytes(value: int) -> str:
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    raise AssertionError("unreachable")


class HoloWeights:
    """Memory-mapped flat-v1 .holo tensor dictionary."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = torch.load(
            self.path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(self.data, dict):
            raise TypeError("flat v1 .holo must contain a dictionary")
        self.config = self.data.get("_config", {})

    def has_factor(self, name: str) -> bool:
        return f"{name}.U" in self.data and f"{name}.SVh" in self.data

    def factors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[f"{name}.U"], self.data[f"{name}.SVh"]

    def get(self, name: str) -> torch.Tensor | None:
        value = self.data.get(name)
        return value if isinstance(value, torch.Tensor) else None


class OriginalWeights:
    """Lazy safetensors reader with persistent mmap handles and a small cache."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self._handles: dict[str, Any] = {}
        self._small: dict[str, torch.Tensor] = {}

    def _handle(self, name: str):
        shard = self.weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                self.model_dir / shard, framework="pt", device="cpu"
            )
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        if name in self._small:
            return self._small[name]
        if name not in self.weight_map:
            raise KeyError(f"original weight not found: {name}")
        value = self._handle(name).get_tensor(name)
        if value.numel() <= 300_000:
            self._small[name] = value
        return value

    def rows(self, name: str, start: int, end: int) -> torch.Tensor:
        if name in self._small:
            return self._small[name][start:end]
        return self._handle(name).get_slice(name)[start:end]

    def embedding(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Gather a small set of rows without materializing the 2.5 GB table."""
        flat = ids.detach().to("cpu", torch.long).reshape(-1)
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        rows = [self.rows(name, int(i), int(i) + 1) for i in unique.tolist()]
        table = torch.cat(rows, dim=0)
        return table[inverse].reshape(*ids.shape, table.shape[-1])


def load_holo(path: str | Path) -> HoloWeights:
    return HoloWeights(path)


def load_original(path: str | Path = DEFAULT_MODEL) -> OriginalWeights:
    _configure_from_dir(Path(path))
    return OriginalWeights(path)


def _configure_from_dir(model_dir: Path):
    """Apply the model's text_config to module constants (model-agnostic sizes)."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return
    import json

    cfg = json.loads(cfg_path.read_text())
    text = cfg.get("text_config", cfg)
    if "hidden_size" in text:
        _apply_config(text)


class ResonanceDecoder:
    """Cybernetic Truth control law: T = 1/(R + eps).

    R is the coherence of the generation trajectory, measured TorusOracle-style:
    the rolling hidden-state norms are mapped to phases on S^1 (norm/max * pi)
    and R = |mean(e^{i phi})|. High R -> low temperature -> deterministic,
    phase-locked output. Low R -> high temperature -> geodesic-seeking
    exploration. Data-free, no training - the lab's own decoder doctrine.
    """

    def __init__(self, buffer_size: int = 16, eps: float = 0.05, t_min: float = 0.01, t_max: float = 5.0):
        self.L = max(3, buffer_size)
        self.eps = eps
        self.t_min = t_min
        self.t_max = t_max
        self.buffer: list[float] = []
        self.trace: list[float] = []

    def reset(self) -> None:
        self.buffer.clear()
        self.trace.clear()

    def push(self, hidden: torch.Tensor) -> float:
        """Return coherence R in [0,1] after ingesting the current hidden state."""
        h = hidden.float()
        n = float(h.norm())
        self.buffer.append(n)
        if len(self.buffer) > self.L:
            self.buffer.pop(0)
        if len(self.buffer) < 3:
            return 0.5
        mx = max(self.buffer)
        if mx <= 1e-9:
            return 0.5
        phases = torch.tensor([v / mx * math.pi for v in self.buffer])
        z = torch.exp(1j * phases)
        r = float(z.mean().abs())
        self.trace.append(r)
        return r

    def temperature(self, r: float) -> float:
        t = 1.0 / (r + self.eps)
        return min(max(t, self.t_min), self.t_max)

    @property
    def mean_resonance(self) -> float:
        return sum(self.trace) / len(self.trace) if self.trace else 0.5


class Qwen35HoloEngine:
    def __init__(
        self,
        holo: HoloWeights | None,
        original: OriginalWeights,
        *,
        exact: bool = False,
        device: str | torch.device = "cpu",
        num_layers: int = LAYERS,
        lm_head_chunk: int = 32768,
        verbose: bool = True,
    ):
        if not exact and holo is None:
            raise ValueError("compressed mode requires a .holo source")
        self.holo = holo
        self.original = original
        self.exact = exact
        self.device = torch.device(device)
        self.num_layers = min(num_layers, LAYERS)  # respect configured model size
        self.lm_head_chunk = lm_head_chunk
        self.exact_tail = 0  # number of final layers run exactly (oracle tail)
        self._layer_exact = False
        self.verbose = verbose
        self.kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_hidden_rms: list[float] = []
        self.last_hidden_states: list[torch.Tensor] = []
        # Analytic low-rank calibration corrections (no training):
        #   corrections[layer] = {"mix": (A, B), "mlp": (A, B)}
        #   A: D x r, B: D x r; applied as  sub_out += A @ (B.T @ h_in)
        self.corrections: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        # Set this to {} before prefill to capture calibration activations.  Each
        # projection input is stored as [batch, sequence, input_features] on CPU.
        self.capture_io: dict[int, dict[str, Any]] | None = None

    @property
    def label(self) -> str:
        return "EXACT" if self.exact else "HOLO"

    def _exact_weight(self, name: str) -> torch.Tensor:
        # Plain tensors in a flat v1 cassette are exact and avoid extra I/O.
        if not self.exact and self.holo is not None:
            value = self.holo.get(name)
            if value is not None:
                return value
        return self.original.get(name)

    @staticmethod
    def _adapter(
        x: torch.Tensor, name: str, adapters: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if not adapters:
            return None
        entry = adapters.get(name)
        if isinstance(entry, dict):
            a, b = entry.get("A"), entry.get("B")
            alpha = float(entry.get("alpha", a.shape[0] if a is not None else 1))
        else:
            a, b = adapters.get(f"{name}.A"), adapters.get(f"{name}.B")
            alpha = float(adapters.get(f"{name}.alpha", a.shape[0] if a is not None else 1))
        if a is None or b is None:
            return None
        rank = a.shape[0]
        a = a.to(x.device)
        b = b.to(x.device)
        adapter_input = x.to(a.dtype)
        return ((adapter_input @ a.transpose(0, 1)) @ b.transpose(0, 1)) * (
            alpha / rank
        )

    def _linear(
        self,
        x: torch.Tensor,
        name: str,
        *,
        factorized: bool,
        adapters: dict[str, Any] | None,
    ) -> torch.Tensor:
        if factorized and not self.exact and not self._layer_exact and self.holo is not None and self.holo.has_factor(name):
            u, svh = self.holo.factors(name)
            u = u.to(self.device)
            svh = svh.to(self.device)
            base = (x.to(svh.dtype) @ svh.transpose(0, 1)) @ u.transpose(0, 1)
        else:
            weight = self._exact_weight(name).to(self.device)
            base = x.to(weight.dtype) @ weight.transpose(0, 1)
        correction = self._adapter(x, name, adapters)
        if correction is not None:
            base = base + correction.to(base.dtype)
        return base

    def _capture_projection(
        self, layer: int, name: str, projection_input: torch.Tensor
    ) -> None:
        if self.capture_io is None:
            return
        entry = self.capture_io.setdefault(layer, {})
        projections = entry.setdefault("proj", {})
        projections[name] = projection_input.detach().to("cpu").clone()

    @staticmethod
    def _rms_offset(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
        return (normed * (1.0 + weight.float())).to(x.dtype)

    @staticmethod
    def _rms_direct(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + EPS)
        return normed * weight.float()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _rope(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freq_index = torch.arange(0, ROTARY_DIM, 2, device=self.device).float()
        inv_freq = 1.0 / (ROPE_THETA ** (freq_index / ROTARY_DIM))
        angles = torch.outer(positions.float(), inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]
        q_rot, q_pass = q[..., :ROTARY_DIM], q[..., ROTARY_DIM:]
        k_rot, k_pass = k[..., :ROTARY_DIM], k[..., ROTARY_DIM:]
        q_rot = q_rot * cos + self._rotate_half(q_rot) * sin
        k_rot = k_rot * cos + self._rotate_half(k_rot) * sin
        return torch.cat((q_rot, q_pass), -1), torch.cat((k_rot, k_pass), -1)

    def _full_attention(
        self,
        x: torch.Tensor,
        layer: int,
        adapters: dict[str, Any] | None,
    ) -> torch.Tensor:
        prefix = f"model.language_model.layers.{layer}.self_attn"
        batch, length, _ = x.shape
        q_name = f"{prefix}.q_proj.weight"
        self._capture_projection(layer, q_name, x)
        q_gate = self._linear(
            x, q_name, factorized=True, adapters=adapters
        ).view(batch, length, FULL_Q_HEADS, 2 * FULL_HEAD_DIM)
        q, gate = q_gate.chunk(2, dim=-1)
        gate = gate.reshape(batch, length, -1)
        k = self._linear(
            x, f"{prefix}.k_proj.weight", factorized=False, adapters=None
        ).view(batch, length, FULL_KV_HEADS, FULL_HEAD_DIM)
        v = self._linear(
            x, f"{prefix}.v_proj.weight", factorized=False, adapters=None
        ).view(batch, length, FULL_KV_HEADS, FULL_HEAD_DIM)
        qn = self._exact_weight(f"{prefix}.q_norm.weight").to(self.device)
        kn = self._exact_weight(f"{prefix}.k_norm.weight").to(self.device)
        q = self._rms_offset(q, qn)
        k = self._rms_offset(k, kn)
        positions = torch.arange(length, device=self.device)
        q, k = self._rope(q.float(), k.float(), positions)

        # Keep the native four-head cache; generation currently rebuilds GDN
        # state from the complete context, so this cache is diagnostic/correct.
        self.kv_cache[layer] = (k.detach(), v.detach())
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(FULL_Q_HEADS // FULL_KV_HEADS, 1)
        v = v.float().transpose(1, 2).repeat_interleave(
            FULL_Q_HEADS // FULL_KV_HEADS, 1
        )
        scores = (q @ k.transpose(-2, -1)) * (FULL_HEAD_DIM**-0.5)
        causal = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)
        scores.masked_fill_(causal, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attended = (probabilities @ v).transpose(1, 2).contiguous()
        attended = attended.reshape(batch, length, -1) * torch.sigmoid(gate.float())
        attended = attended.to(x.dtype)
        o_name = f"{prefix}.o_proj.weight"
        self._capture_projection(layer, o_name, attended)
        return self._linear(
            attended,
            o_name,
            factorized=True,
            adapters=adapters,
        )

    def _gated_delta_net(
        self,
        x: torch.Tensor,
        layer: int,
        adapters: dict[str, Any] | None,
    ) -> torch.Tensor:
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        batch, length, _ = x.shape
        qkv_name = f"{prefix}.in_proj_qkv.weight"
        z_name = f"{prefix}.in_proj_z.weight"
        self._capture_projection(layer, qkv_name, x)
        self._capture_projection(layer, z_name, x)
        mixed = self._linear(
            x, qkv_name, factorized=True, adapters=adapters
        )
        z = self._linear(
            x, z_name, factorized=True, adapters=adapters
        ).view(batch, length, GDN_V_HEADS, GDN_HEAD_DIM)
        # Exact a/b are cheap and control the state dynamics.  beta is sigmoid(b);
        # dt_bias participates in the decay discretization, not the beta gate.
        b = self._linear(
            x, f"{prefix}.in_proj_b.weight", factorized=False, adapters=None
        ).float()
        a = self._linear(
            x, f"{prefix}.in_proj_a.weight", factorized=False, adapters=None
        ).float()
        conv = self._exact_weight(f"{prefix}.conv1d.weight").to(self.device).float()
        mixed = F.conv1d(
            mixed.float().transpose(1, 2),
            conv,
            bias=None,
            padding=conv.shape[-1] - 1,
            groups=GDN_CONV_WIDTH,
        )[..., :length]
        mixed = F.silu(mixed).transpose(1, 2)
        q, k, v = torch.split(
            mixed, (GDN_KEY_WIDTH, GDN_KEY_WIDTH, GDN_VALUE_WIDTH), dim=-1
        )
        q = q.view(batch, length, GDN_QK_HEADS, GDN_HEAD_DIM)
        k = k.view(batch, length, GDN_QK_HEADS, GDN_HEAD_DIM)
        v = v.view(batch, length, GDN_V_HEADS, GDN_HEAD_DIM)
        q = q.repeat_interleave(GDN_V_HEADS // GDN_QK_HEADS, dim=2)
        k = k.repeat_interleave(GDN_V_HEADS // GDN_QK_HEADS, dim=2)
        q = q.float()
        k = k.float()
        q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + EPS)
        k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + EPS)
        q = q * (GDN_HEAD_DIM**-0.5)
        v = v.float()
        beta = torch.sigmoid(b)
        a_log = self._exact_weight(f"{prefix}.A_log").to(self.device).float()
        dt_bias = self._exact_weight(f"{prefix}.dt_bias").to(self.device).float()
        g = -torch.exp(a_log)[None, None, :] * F.softplus(a + dt_bias)
        decay = torch.exp(g)

        state = torch.zeros(
            batch,
            GDN_V_HEADS,
            GDN_HEAD_DIM,
            GDN_HEAD_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        outputs = []
        for token in range(length):
            state = state * decay[:, token, :, None, None]
            kt, vt = k[:, token], v[:, token]
            prediction = (state * kt.unsqueeze(-1)).sum(dim=-2)
            delta = (vt - prediction) * beta[:, token, :, None]
            state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
            yt = (state * q[:, token].unsqueeze(-1)).sum(dim=-2)
            outputs.append(yt)
        y = torch.stack(outputs, dim=1)
        norm = self._exact_weight(f"{prefix}.norm.weight").to(self.device)
        y = self._rms_direct(y, norm) * F.silu(z.float())
        y = y.reshape(batch, length, GDN_VALUE_WIDTH).to(x.dtype)
        out_name = f"{prefix}.out_proj.weight"
        self._capture_projection(layer, out_name, y)
        return self._linear(
            y,
            out_name,
            factorized=True,
            adapters=adapters,
        )

    def _mlp(
        self,
        x: torch.Tensor,
        layer: int,
        adapters: dict[str, Any] | None,
    ) -> torch.Tensor:
        prefix = f"model.language_model.layers.{layer}.mlp"
        gate_name = f"{prefix}.gate_proj.weight"
        up_name = f"{prefix}.up_proj.weight"
        self._capture_projection(layer, gate_name, x)
        self._capture_projection(layer, up_name, x)
        gate = self._linear(
            x, gate_name, factorized=True, adapters=adapters
        )
        up = self._linear(
            x, up_name, factorized=True, adapters=adapters
        )
        activated = F.silu(gate.float()) * up.float()
        activated = activated.to(x.dtype)
        down_name = f"{prefix}.down_proj.weight"
        self._capture_projection(layer, down_name, activated)
        return self._linear(
            activated,
            down_name,
            factorized=True,
            adapters=adapters,
        )

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        name = "model.language_model.embed_tokens.weight"
        # The hybrid shell always uses original rows, even if embed factors exist.
        rows = self.original.embedding(name, input_ids)
        return rows.to(self.device)

    def _lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        chunks = []
        flat = hidden.reshape(-1, HIDDEN)
        # Tied-embedding models have no lm_head.weight; fall back to embed rows.
        head_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.original.weight_map
            else "model.language_model.embed_tokens.weight"
        )
        for start in range(0, VOCAB, self.lm_head_chunk):
            end = min(start + self.lm_head_chunk, VOCAB)
            weight = self.original.rows(head_name, start, end).to(self.device)
            chunks.append((flat.to(weight.dtype) @ weight.transpose(0, 1)).float())
        return torch.cat(chunks, dim=-1).reshape(*hidden.shape[:-1], VOCAB)

    @staticmethod
    def _apply_correction(
        sub_out: torch.Tensor, h_in: torch.Tensor, corr: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        if corr is None:
            return sub_out
        a, b = corr
        a = a.to(sub_out.device)
        b = b.to(sub_out.device)
        xf = h_in.to(a.dtype)
        correction = ((xf @ b) @ a.transpose(0, 1)).to(sub_out.dtype)
        return sub_out + correction

    def _prefill_layer_mixer(self, layer: int, normed: torch.Tensor, adapters) -> torch.Tensor:
        if layer % 4 == 3:
            return self._full_attention(normed, layer, adapters)
        return self._gated_delta_net(normed, layer, adapters)

    def prefill(
        self,
        input_ids: torch.Tensor,
        adapters: dict[str, Any] | None = None,
        *,
        capture_hidden: bool = False,
        compute_logits: bool = True,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(torch.long)
        hidden = self._embed(input_ids)
        self.kv_cache.clear()
        self.last_hidden_rms = []
        self.last_hidden_states = []
        if self.capture_io is not None:
            self.capture_io.clear()
        started = time.perf_counter()
        for layer in range(self.num_layers):
            prefix = f"model.language_model.layers.{layer}"
            self._layer_exact = self.exact or (layer >= self.num_layers - self.exact_tail)
            corr_entry = {} if self._layer_exact else self.corrections.get(layer, {})
            residual = hidden
            norm = self._exact_weight(f"{prefix}.input_layernorm.weight").to(self.device)
            mixed_input = self._rms_offset(hidden, norm)
            mixed_pre = self._prefill_layer_mixer(layer, mixed_input, adapters)
            mixed = self._apply_correction(
                mixed_pre, residual, corr_entry.get("mix")
            )
            block = "FULL" if layer % 4 == 3 else "GDN"
            hidden = residual + mixed.to(residual.dtype)
            residual = hidden
            post = self._exact_weight(f"{prefix}.post_attention_layernorm.weight").to(
                self.device
            )
            mlp_input = self._rms_offset(hidden, post)
            mlp_pre = self._mlp(mlp_input, layer, adapters)
            mlp = self._apply_correction(
                mlp_pre, residual, corr_entry.get("mlp")
            )
            hidden = residual + mlp.to(residual.dtype)
            if self.capture_io is not None:
                self.capture_io.setdefault(layer, {}).update({
                    "h_in": residual.detach().clone(),      # layer input (pre-norm)
                    "normed": mixed_input.detach().clone(), # mixer input (post-norm)
                    "mix_pre": mixed_pre.detach().clone(),  # mixer output before correction
                    "mlp_h": residual.detach().clone(),     # mlp stage input (post-mixer)
                    "mlp_in": mlp_input.detach().clone(),   # mlp input (post-norm)
                    "mlp_pre": mlp_pre.detach().clone(),    # mlp output before correction
                    "h_out": hidden.detach().clone(),
                })
            rms = float(hidden.float().square().mean().sqrt().item())
            self.last_hidden_rms.append(rms)
            if capture_hidden:
                self.last_hidden_states.append(hidden.detach().cpu().clone())
            if self.verbose:
                print(
                    f"  [{self.label}] L{layer:02d} {block:<4} rms={rms:.6f} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
            gc.collect()
        final_norm = self._exact_weight("model.language_model.norm.weight").to(self.device)
        hidden = self._rms_offset(hidden, final_norm)
        return self._lm_head(hidden) if compute_logits else hidden

    def hidden_rms_report(self) -> list[dict[str, float | int | str]]:
        return [
            {
                "layer": layer,
                "type": "full_attention" if layer % 4 == 3 else "linear_attention",
                "rms": rms,
            }
            for layer, rms in enumerate(self.last_hidden_rms)
        ]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        adapters: dict[str, Any] | None = None,
        resonance: bool = False,
        resonance_buffer: int = 16,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        tokens = input_ids.to(torch.long)
        dec = ResonanceDecoder(buffer_size=resonance_buffer) if resonance else None
        for _ in range(max_new_tokens):
            # GDN cache persistence is intentionally not implemented yet; rebuild
            # the complete context so generation remains mathematically correct.
            logits = self.prefill(tokens, adapters=adapters, capture_hidden=resonance)[:, -1]
            if resonance:
                h = self.last_hidden_states[-1]
                r = dec.push(h)
                t = dec.temperature(r)
                probabilities = torch.softmax(logits / t, dim=-1)
                if top_p < 1.0:
                    sorted_p, sorted_i = probabilities.sort(descending=True, dim=-1)
                    cumulative = sorted_p.cumsum(dim=-1)
                    remove = cumulative - sorted_p >= top_p
                    sorted_p = sorted_p.masked_fill(remove, 0.0)
                    sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_p, 1)
                    next_token = sorted_i.gather(-1, sampled).cpu()
                else:
                    next_token = torch.multinomial(probabilities, 1).cpu()
            elif temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True).cpu()
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_p, sorted_i = probabilities.sort(descending=True, dim=-1)
                    cumulative = sorted_p.cumsum(dim=-1)
                    remove = cumulative - sorted_p >= top_p
                    sorted_p = sorted_p.masked_fill(remove, 0.0)
                    sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                    sampled = torch.multinomial(sorted_p, 1)
                    next_token = sorted_i.gather(-1, sampled).cpu()
                else:
                    next_token = torch.multinomial(probabilities, 1).cpu()
            tokens = torch.cat((tokens.cpu(), next_token), dim=1)
        if resonance:
            self.resonance_trace = list(dec.trace)
        return tokens


def _self_test(args: argparse.Namespace) -> None:
    print("=" * 78)
    print("QWEN3.6-27B .HOLO NATIVE HYBRID SELF-TEST")
    print("=" * 78)
    print(f"Holo:  {args.holo} ({_human_bytes(Path(args.holo).stat().st_size)})")
    print(f"Model: {args.model_dir}")
    print(f"Device: {args.device}  layers={args.num_layers}")
    prompt = torch.tensor([[9707, 374, 264, 1296, 315, 16831, 323, 1146]])
    print(f"Prompt IDs ({prompt.numel()}): {prompt.tolist()[0]}")

    original = load_original(args.model_dir)
    holo = load_holo(args.holo)
    exact = Qwen35HoloEngine(
        None,
        original,
        exact=True,
        device=args.device,
        num_layers=args.num_layers,
        lm_head_chunk=args.lm_head_chunk,
        verbose=not args.quiet,
    )
    compressed = Qwen35HoloEngine(
        holo,
        original,
        exact=False,
        device=args.device,
        num_layers=args.num_layers,
        lm_head_chunk=args.lm_head_chunk,
        verbose=not args.quiet,
    )

    with torch.inference_mode():
        print("\nRunning exact safetensors reference ...", flush=True)
        t0 = time.perf_counter()
        exact_logits = exact.prefill(prompt, capture_hidden=True)
        exact_time = time.perf_counter() - t0
        print(f"Exact complete in {exact_time:.1f}s")

        print("\nRunning hybrid .holo engine ...", flush=True)
        t0 = time.perf_counter()
        holo_logits = compressed.prefill(prompt, capture_hidden=True)
        holo_time = time.perf_counter() - t0
        print(f"Holo complete in {holo_time:.1f}s")

        print("\nPer-layer hidden RMS comparison")
        print("  layer type      exact_rms     holo_rms      ratio   hidden_cos")
        hidden_cosines = []
        for layer, (reference_state, holo_state) in enumerate(
            zip(exact.last_hidden_states, compressed.last_hidden_states)
        ):
            reference_rms = exact.last_hidden_rms[layer]
            holo_rms = compressed.last_hidden_rms[layer]
            ratio = holo_rms / max(reference_rms, 1e-30)
            cosine = float(
                F.cosine_similarity(
                    reference_state.float().reshape(-1),
                    holo_state.float().reshape(-1),
                    dim=0,
                ).item()
            )
            hidden_cosines.append(cosine)
            block = "FULL" if layer % 4 == 3 else "GDN"
            print(
                f"  L{layer:02d}  {block:<4}  {reference_rms:12.6f} "
                f"{holo_rms:12.6f} {ratio:10.4f} {cosine:11.6f}"
            )

        ref_last = exact_logits[0, -1].float()
        holo_last = holo_logits[0, -1].float()
        logit_cosine = float(F.cosine_similarity(ref_last, holo_last, dim=0).item())
        exact_top = torch.topk(ref_last, 10).indices
        holo_top = torch.topk(holo_last, 10).indices
        overlap = len(set(exact_top.tolist()) & set(holo_top.tolist()))
        exact_argmax, holo_argmax = int(ref_last.argmax()), int(holo_last.argmax())
        finite = bool(torch.isfinite(holo_logits).all())
        print("\nFinal-logit agreement")
        print(f"  finite:         {finite}")
        print(f"  cosine:         {logit_cosine:.6f}")
        print(f"  top-10 overlap: {overlap}/10")
        print(f"  argmax exact:   {exact_argmax}")
        print(f"  argmax holo:    {holo_argmax}")
        print(f"  argmax match:   {exact_argmax == holo_argmax}")
        print(f"  hidden cosine:  first={hidden_cosines[0]:.6f} "
              f"last={hidden_cosines[-1]:.6f} mean={sum(hidden_cosines)/len(hidden_cosines):.6f}")
        print(f"  full-attn KV caches: {len(compressed.kv_cache)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holo", default=str(DEFAULT_HOLO))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=LAYERS)
    parser.add_argument("--lm-head-chunk", type=int, default=32768)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not Path(args.holo).is_file():
        parser.error(f"holo file not found: {args.holo}")
    if not (Path(args.model_dir) / "model.safetensors.index.json").is_file():
        parser.error(f"model index not found: {args.model_dir}")
    if not 1 <= args.num_layers <= LAYERS:
        parser.error("--num-layers must be in [1, 64]")
    return args


if __name__ == "__main__":
    _self_test(_parse_args())
