"""Build a wormhole_v1 cassette from a flat v1 Qwen .holo file.

For every repeated decoder weight type, the first U factor is retained as an
anchor.  Later U factors are represented by a k_eff x k_eff transport matrix
and a genuinely packed 2-bit residual.  The first layer's SVh is retained once
as the shared right factor for that type.

The fidelity report reconstructs weights against the original safetensors
checkpoint, not merely against the source .holo factors.  Full-row fidelity is
the default; --fidelity-rows provides a deterministic sampled-row mode for
quick smoke tests.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


LAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = LAB_ROOT / "output" / "qwen_27b_k256.holo"
DEFAULT_OUTPUT = LAB_ROOT / "output" / "qwen_27b_wormhole.holo"
DEFAULT_MODEL = Path(
    "/run/media/reneshizzle/Seagate/Reneshizzle SG/Models/Qwen/Qwen3.6-27B"
)

LAYER_FACTOR_RE = re.compile(
    r"^(?P<prefix>.+\.layers\.)(?P<layer>\d+)\."
    r"(?P<weight_type>.+)\.weight\.(?P<factor>U|SVh)$"
)
ANY_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)(?:\.|$)")
QUANT_LEVELS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


def tensor_nbytes(value: Any) -> int:
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0


def human_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024.0
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
    raise AssertionError("unreachable")


def parse_factor_key(key: str) -> tuple[int, str, str] | None:
    match = LAYER_FACTOR_RE.match(key)
    if match is None:
        return None
    return (
        int(match.group("layer")),
        match.group("weight_type"),
        match.group("factor"),
    )


def pack_2bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack four integer codes in [0, 3] into each uint8."""
    flat = codes.reshape(-1).to(device="cpu", dtype=torch.uint8).contiguous()
    pad = (-flat.numel()) % 4
    if pad:
        flat = F.pad(flat, (0, pad), value=0)
    grouped = flat.view(-1, 4)
    packed = (
        grouped[:, 0]
        | (grouped[:, 1] << 2)
        | (grouped[:, 2] << 4)
        | (grouped[:, 3] << 6)
    )
    return packed.contiguous()


def unpack_2bit(packed: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Unpack a packed residual index tensor for validation or decoding."""
    packed = packed.to(device=device, dtype=torch.uint8)
    codes = torch.stack(
        (
            packed & 0x03,
            (packed >> 2) & 0x03,
            (packed >> 4) & 0x03,
            (packed >> 6) & 0x03,
        ),
        dim=1,
    ).reshape(-1)
    return codes[: math.prod(shape)].reshape(shape)


def quantize_residual_2bit(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed codes, scalar scale, and the dequantized residual."""
    scale = residual.abs().amax().float()
    scale_value = float(scale.item())
    if not math.isfinite(scale_value):
        raise ValueError("non-finite residual scale")

    if scale_value == 0.0:
        codes = torch.zeros_like(residual, dtype=torch.uint8)
        dequantized = torch.zeros_like(residual)
    else:
        normalized = (residual / scale).clamp_(-1.0, 1.0).contiguous()
        boundaries = torch.tensor((-2.0 / 3.0, 0.0, 2.0 / 3.0), device=residual.device)
        codes = torch.bucketize(normalized, boundaries).to(torch.uint8)
        levels = torch.tensor(QUANT_LEVELS, dtype=residual.dtype, device=residual.device)
        dequantized = levels[codes.long()] * scale.to(residual.dtype)

    packed = pack_2bit(codes)
    return packed, scale.cpu(), dequantized


def select_fidelity_layers(layers: list[int], count: int) -> set[int]:
    candidates = layers[1:]
    if count <= 0 or not candidates:
        return set()
    if count >= len(candidates):
        return set(candidates)
    positions = torch.linspace(0, len(candidates) - 1, steps=count).round().to(torch.int64)
    return {candidates[int(position)] for position in positions}


class OriginalWeights:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as handle:
            self.weight_map = json.load(handle)["weight_map"]

    def load(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"original tensor is absent from index: {key}")
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


@torch.inference_mode()
def weight_cosine(
    original: torch.Tensor,
    u: torch.Tensor,
    svh: torch.Tensor,
    device: torch.device,
    row_chunk: int,
    fidelity_rows: int,
) -> float:
    """Cosine(original, U @ SVh) without materializing the full product."""
    if original.ndim != 2:
        raise ValueError(f"fidelity expects a matrix, got {tuple(original.shape)}")
    expected = (u.shape[0], svh.shape[1])
    if tuple(original.shape) != expected:
        raise ValueError(f"shape mismatch: original={tuple(original.shape)} reconstructed={expected}")

    if 0 < fidelity_rows < original.shape[0]:
        rows = torch.linspace(0, original.shape[0] - 1, fidelity_rows).round().long()
        rows = torch.unique(rows, sorted=True)
        original = original.index_select(0, rows)
        u = u.index_select(0, rows.to(u.device))

    svh_device = svh.to(device=device, dtype=torch.float32)
    dot = torch.zeros((), dtype=torch.float64, device=device)
    original_sq = torch.zeros((), dtype=torch.float64, device=device)
    reconstructed_sq = torch.zeros((), dtype=torch.float64, device=device)

    for start in range(0, original.shape[0], row_chunk):
        end = min(start + row_chunk, original.shape[0])
        target = original[start:end].to(device=device, dtype=torch.float32)
        reconstructed = u[start:end].to(device=device, dtype=torch.float32) @ svh_device
        dot += (target * reconstructed).sum(dtype=torch.float64)
        original_sq += target.square().sum(dtype=torch.float64)
        reconstructed_sq += reconstructed.square().sum(dtype=torch.float64)

    denominator = torch.sqrt(original_sq * reconstructed_sq).clamp_min_(1e-30)
    return float((dot / denominator).item())


def source_weight_key(input_u_key: str) -> str:
    if not input_u_key.endswith(".U"):
        raise ValueError(input_u_key)
    return input_u_key[:-2]


def copy_passthrough_group(
    output: dict[str, Any],
    entries: dict[int, dict[str, tuple[str, torch.Tensor]]],
) -> None:
    for layer_entries in entries.values():
        for key, tensor in layer_entries.values():
            output[key] = tensor


def validate_group(
    entries: dict[int, dict[str, tuple[str, torch.Tensor]]],
    minimum_k: int,
) -> tuple[bool, str]:
    if len(entries) < 2:
        return False, "fewer than two layers"
    if any("U" not in factors or "SVh" not in factors for factors in entries.values()):
        return False, "missing U/SVh pair"

    u_shapes = {tuple(factors["U"][1].shape) for factors in entries.values()}
    svh_shapes = {tuple(factors["SVh"][1].shape) for factors in entries.values()}
    if len(u_shapes) != 1 or len(svh_shapes) != 1:
        return False, f"non-uniform factor shapes U={sorted(u_shapes)} SVh={sorted(svh_shapes)}"

    u_shape = next(iter(u_shapes))
    svh_shape = next(iter(svh_shapes))
    if len(u_shape) != 2 or len(svh_shape) != 2 or u_shape[1] != svh_shape[0]:
        return False, f"invalid factor shapes U={u_shape} SVh={svh_shape}"
    if u_shape[1] < minimum_k:
        return False, f"k_eff={u_shape[1]} below minimum {minimum_k}"
    return True, ""


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def build_wormhole(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    device = choose_device(args.device)

    print(f"Input:       {input_path} ({human_bytes(input_path.stat().st_size)})")
    print(f"Model:       {model_dir}")
    print(f"Device:      {device}")
    print("Loading flat v1 .holo ...", flush=True)
    holo = torch.load(input_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(holo, dict):
        raise TypeError("input .holo must contain a dictionary")

    groups: dict[str, dict[int, dict[str, tuple[str, torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    discovered_layers: set[int] = set()
    for key, value in holo.items():
        parsed = parse_factor_key(key)
        if parsed is None or not isinstance(value, torch.Tensor):
            continue
        layer, weight_type, factor = parsed
        discovered_layers.add(layer)
        groups[weight_type][layer][factor] = (key, value)

    selected_layers = sorted(discovered_layers)
    if args.limit_layers > 0:
        selected_layers = selected_layers[: args.limit_layers]
    selected_layer_set = set(selected_layers)
    print(
        f"Selected layers: {len(selected_layers)}/{len(discovered_layers)}"
        + (f" {selected_layers}" if args.limit_layers else "")
    )

    output: dict[str, Any] = {}
    consumed_keys: set[str] = set()
    original_reader = OriginalWeights(model_dir) if args.fidelity_samples > 0 else None
    source_config = holo.get("_config", {})
    source_k = int(source_config.get("k", 0)) if isinstance(source_config, dict) else 0

    stats: dict[str, Any] = {
        "groups_compressed": 0,
        "groups_skipped": 0,
        "rotations": 0,
        "residual_values": 0,
        "residual_packed_bytes": 0,
        "fidelity": [],
        "group_meta": {},
        "device": str(device),
    }

    for weight_type in sorted(groups):
        entries = {
            layer: groups[weight_type][layer]
            for layer in sorted(groups[weight_type])
            if layer in selected_layer_set
        }
        if not entries:
            continue

        for factors in entries.values():
            consumed_keys.update(key for key, _ in factors.values())

        valid, reason = validate_group(entries, args.min_k_eff)
        if not valid:
            print(f"[{weight_type}] SKIP: {reason}", flush=True)
            copy_passthrough_group(output, entries)
            stats["groups_skipped"] += 1
            continue

        layers = sorted(entries)
        first_layer = layers[0]
        first_u = entries[first_layer]["U"][1]
        shared_svh = entries[first_layer]["SVh"][1]
        m, k_eff = first_u.shape
        sample_layers = select_fidelity_layers(layers, args.fidelity_samples)

        print(
            f"[{weight_type}] layers={len(layers)} U={tuple(first_u.shape)} "
            f"SVh={tuple(shared_svh.shape)} k_eff={k_eff}",
            flush=True,
        )

        output[f"{weight_type}.{first_layer}.U"] = first_u
        output[f"{weight_type}.SVh"] = shared_svh

        prev_actual = first_u.to(device=device, dtype=torch.float32)
        prev_decoded = prev_actual.clone()
        shared_svh_device = shared_svh.to(device=device, dtype=torch.float32)

        type_meta = {
            "layers": layers,
            "first_layer": first_layer,
            "u_shape": [m, k_eff],
            "svh_shape": list(shared_svh.shape),
            "k_eff": k_eff,
            "residual_shape": [m, k_eff],
            "residual_packing": "four_2bit_codes_per_uint8_lsb_first",
            "source_u_keys": {str(layer): entries[layer]["U"][0] for layer in layers},
        }

        for position, layer in enumerate(layers[1:], start=2):
            curr = entries[layer]["U"][1].to(device=device, dtype=torch.float32)

            # R follows the requested actual-to-actual layer transport.  The
            # residual is formed against the stored BF16 R so decoding matches.
            rotation = prev_actual.transpose(0, 1) @ curr
            rotation_stored = rotation.to(torch.bfloat16).cpu()
            rotation_device = rotation_stored.to(device=device, dtype=torch.float32)
            actual_base = prev_actual @ rotation_device
            residual = curr - actual_base
            packed, scale, residual_dequantized = quantize_residual_2bit(residual)

            key_prefix = f"{weight_type}.{layer}"
            output[f"{key_prefix}.R"] = rotation_stored
            output[f"{key_prefix}.res_q"] = packed
            output[f"{key_prefix}.res_scale"] = scale
            output[f"{key_prefix}.res_shape"] = torch.tensor(
                (m, k_eff), dtype=torch.int32
            )

            decoded_without = prev_decoded @ rotation_device
            decoded_with = decoded_without + residual_dequantized

            stats["rotations"] += 1
            stats["residual_values"] += residual.numel()
            stats["residual_packed_bytes"] += packed.numel()

            if layer in sample_layers and original_reader is not None:
                original_key = source_weight_key(entries[layer]["U"][0])
                original = original_reader.load(original_key)
                fid_without = weight_cosine(
                    original,
                    decoded_without,
                    shared_svh_device,
                    device,
                    args.fidelity_row_chunk,
                    args.fidelity_rows,
                )
                fid_with = weight_cosine(
                    original,
                    decoded_with,
                    shared_svh_device,
                    device,
                    args.fidelity_row_chunk,
                    args.fidelity_rows,
                )
                sampling = (
                    f"sampled_rows={min(args.fidelity_rows, original.shape[0])}"
                    if args.fidelity_rows > 0
                    else "all_rows"
                )
                print(
                    f"  L{layer}: cosine original vs shared-SVh reconstruction "
                    f"without_res={fid_without:.6f} with_res={fid_with:.6f} "
                    f"({sampling})",
                    flush=True,
                )
                stats["fidelity"].append(
                    {
                        "type": weight_type,
                        "layer": layer,
                        "without_residual": fid_without,
                        "with_residual": fid_with,
                        "rows": min(args.fidelity_rows, original.shape[0])
                        if args.fidelity_rows > 0
                        else original.shape[0],
                    }
                )
                del original

            prev_actual = curr
            prev_decoded = decoded_with
            del rotation, actual_base, residual, decoded_without

            if position == len(layers) or position % args.progress_every == 0:
                print(f"  progress {position}/{len(layers)} layers", flush=True)

        stats["group_meta"][weight_type] = type_meta
        stats["groups_compressed"] += 1
        del prev_actual, prev_decoded, shared_svh_device
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Preserve 1D tensors, global factors, and other metadata.  Under a limited
    # smoke test, omit unselected decoder-layer tensors rather than copying the
    # rest of the model into the test cassette.
    for key, value in holo.items():
        if key == "_config" or key in consumed_keys:
            continue
        layer_match = ANY_LAYER_RE.search(key)
        if args.limit_layers > 0 and layer_match is not None:
            if int(layer_match.group("layer")) not in selected_layer_set:
                continue
        output[key] = value

    input_config = dict(source_config) if isinstance(source_config, dict) else {}
    input_config.update(
        {
            "format": "wormhole_v1",
            "source_holo": str(input_path),
            "source_model": str(model_dir),
            "k": source_k,
            "limit_layers": args.limit_layers,
            "selected_layers": selected_layers,
            "quantization": {
                "bits": 2,
                "levels": list(QUANT_LEVELS),
                "scale": "per_tensor_absmax",
                "packed": True,
            },
            "wormhole_groups": stats["group_meta"],
        }
    )
    output["_format"] = "wormhole_v1"
    output["_k"] = source_k
    output["_config"] = input_config
    return output, stats


def validate_output(output: dict[str, Any], stats: dict[str, Any]) -> None:
    if output.get("_format") != "wormhole_v1":
        raise ValueError("missing wormhole_v1 format marker")

    residual_keys = [key for key in output if key.endswith(".res_q")]
    if len(residual_keys) != stats["rotations"]:
        raise ValueError("one packed residual is required per rotation")

    for residual_key in residual_keys:
        prefix = residual_key[: -len(".res_q")]
        packed = output[residual_key]
        shape_tensor = output[f"{prefix}.res_shape"]
        shape = tuple(int(value) for value in shape_tensor.tolist())
        expected_bytes = (math.prod(shape) + 3) // 4
        if packed.dtype != torch.uint8 or packed.numel() != expected_bytes:
            raise ValueError(
                f"invalid packed residual {residual_key}: "
                f"dtype={packed.dtype} bytes={packed.numel()} expected={expected_bytes}"
            )
        # Exercise the decoder path without retaining the unpacked tensor.
        decoded = unpack_2bit(packed, shape, torch.device("cpu"))
        if decoded.min().item() < 0 or decoded.max().item() > 3:
            raise ValueError(f"out-of-range residual code in {residual_key}")
        del decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--limit-layers",
        type=int,
        default=0,
        help="smoke test: process only the first N discovered decoder layer IDs",
    )
    parser.add_argument(
        "--min-k-eff",
        type=int,
        default=4,
        help="pass through groups whose effective SVD rank is smaller than this",
    )
    parser.add_argument(
        "--fidelity-samples",
        type=int,
        default=3,
        help="number of non-anchor layers sampled per compressed type (0 disables)",
    )
    parser.add_argument(
        "--fidelity-rows",
        type=int,
        default=0,
        help="deterministically sample this many matrix rows; 0 computes full fidelity",
    )
    parser.add_argument("--fidelity-row-chunk", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()
    for name in (
        "limit_layers",
        "fidelity_samples",
        "fidelity_rows",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.min_k_eff < 1 or args.fidelity_row_chunk < 1 or args.progress_every < 1:
        parser.error("minimum rank and chunk/progress sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"input .holo not found: {input_path}")
    if not (Path(args.model_dir) / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"original model index not found under: {args.model_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output, stats = build_wormhole(args)
    validate_output(output, stats)

    logical_bytes = sum(tensor_nbytes(value) for value in output.values())
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    print(f"Saving:      {output_path}", flush=True)
    torch.save(output, temporary)
    os.replace(temporary, output_path)

    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    elapsed = time.perf_counter() - started
    print("\nWORMHOLE COMPLETE")
    print(
        f"  groups: {stats['groups_compressed']} compressed, "
        f"{stats['groups_skipped']} passed through"
    )
    print(f"  rotations/residuals: {stats['rotations']}")
    print(
        f"  residual packing: {stats['residual_values']:,} 2-bit values -> "
        f"{human_bytes(stats['residual_packed_bytes'])}"
    )
    print(f"  input file:  {human_bytes(input_size)}")
    print(f"  output file: {human_bytes(output_size)}")
    print(f"  output tensor payload: {human_bytes(logical_bytes)}")
    print(f"  file ratio: {input_size / max(output_size, 1):.2f}x")
    print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
