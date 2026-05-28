#!/usr/bin/env python3
"""CAT_CAS topology-first image .holo.

This file format does not store image patches. It stores a phase topology:
dominant 2D spectral modes of a complex image field. The viewer illuminates that
topology at whatever resolution is requested by placing those modes onto a fresh
frequency lattice and sampling the inverse field.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = "CAT_CAS_TOPO_IMAGE_HOLO_V2"


def _rgb_to_phase(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    phase = 2.0 * np.pi * x
    return x.astype(np.complex64) * np.exp(1j * phase).astype(np.complex64)


def _phase_to_rgb(field: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.abs(field) * 255.0), 0, 255).astype(np.uint8)


def _signed_freq(index: int, size: int) -> int:
    half = size // 2
    return int(index if index <= half else index - size)


def _freq_index(freq: int, size: int) -> int:
    return int(freq % size)


def _spectral_metrics(energy: np.ndarray) -> dict[str, float | int]:
    total = float(np.sum(energy))
    if total <= 0:
        return {"d_pr": 0.0, "d_shannon": 0.0, "k95": 0}
    p = np.maximum(energy / total, 1e-30)
    cdf = np.cumsum(p)
    return {
        "d_pr": float(1.0 / np.sum(p * p)),
        "d_shannon": float(np.exp(-np.sum(p * np.log(p)))),
        "k95": int(np.searchsorted(cdf, 0.95) + 1),
    }


@dataclass(frozen=True)
class CatCasHoloImage:
    mode_y: np.ndarray
    mode_x: np.ndarray
    coeffs: np.ndarray
    energy: np.ndarray
    source_shape: tuple[int, int, int]

    @property
    def k(self) -> int:
        return int(self.coeffs.shape[0])

    def illuminate(
        self,
        render_k: int | None = None,
        output_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Sample the stored topology on a requested output lattice."""
        src_h, src_w, channels = self.source_shape
        if output_shape is None:
            out_h, out_w = src_h, src_w
        else:
            out_h, out_w = int(output_shape[0]), int(output_shape[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output resolution must be positive")

        if render_k is None:
            render_k = self.k
        render_k = max(1, min(int(render_k), self.k))

        spectrum = np.zeros((out_h, out_w, channels), dtype=np.complex64)
        scale = (out_h * out_w) / float(src_h * src_w)

        for i in range(render_k):
            y = _freq_index(int(self.mode_y[i]), out_h)
            x = _freq_index(int(self.mode_x[i]), out_w)
            spectrum[y, x, :] += self.coeffs[i, :] * scale

        field = np.empty((out_h, out_w, channels), dtype=np.complex64)
        for c in range(channels):
            field[:, :, c] = np.fft.ifft2(spectrum[:, :, c]).astype(np.complex64)
        return _phase_to_rgb(field)

    def save(self, path: Path) -> None:
        meta = {
            "magic": MAGIC,
            "source_shape": self.source_shape,
            "k": self.k,
            "codec": "complex_phase_field_sparse_2d_topology",
            "metrics": _spectral_metrics(self.energy),
        }
        path = Path(path)
        with path.open("wb") as f:
            np.savez_compressed(
                f,
                meta=np.asarray(json.dumps(meta), dtype=np.str_),
                mode_y=self.mode_y.astype(np.int32),
                mode_x=self.mode_x.astype(np.int32),
                coeffs=self.coeffs.astype(np.complex64),
                energy=self.energy.astype(np.float32),
            )

    @classmethod
    def load(cls, path: Path) -> "CatCasHoloImage":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "meta" not in data.files:
                raise ValueError(f"{path} is not a CAT_CAS topology .holo")
            meta = json.loads(str(data["meta"]))
            if meta.get("magic") != MAGIC:
                raise ValueError(f"{path} is not a CAT_CAS topology .holo")
            return cls(
                mode_y=data["mode_y"].astype(np.int32),
                mode_x=data["mode_x"].astype(np.int32),
                coeffs=data["coeffs"].astype(np.complex64),
                energy=data["energy"].astype(np.float32),
                source_shape=tuple(int(v) for v in meta["source_shape"]),
            )

    @classmethod
    def from_image(cls, path: Path, k: int) -> "CatCasHoloImage":
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        field = _rgb_to_phase(rgb)
        h, w, channels = field.shape

        fft = np.empty((h, w, channels), dtype=np.complex64)
        for c in range(channels):
            fft[:, :, c] = np.fft.fft2(field[:, :, c]).astype(np.complex64)

        energy_map = np.sum(np.abs(fft) ** 2, axis=2)
        flat_order = np.argsort(energy_map.reshape(-1))[::-1]
        k = max(1, min(int(k), flat_order.size))
        selected = flat_order[:k]
        ys, xs = np.unravel_index(selected, (h, w))

        mode_y = np.asarray([_signed_freq(int(y), h) for y in ys], dtype=np.int32)
        mode_x = np.asarray([_signed_freq(int(x), w) for x in xs], dtype=np.int32)
        coeffs = fft[ys, xs, :].astype(np.complex64)
        energy = energy_map[ys, xs].astype(np.float32)

        return cls(mode_y=mode_y, mode_x=mode_x, coeffs=coeffs, energy=energy, source_shape=rgb.shape)

    def stats(self, file_path: Path | None = None) -> dict[str, object]:
        raw_bytes = int(np.prod(self.source_shape))
        holo_bytes = int(Path(file_path).stat().st_size) if file_path else int(
            self.mode_y.nbytes + self.mode_x.nbytes + self.coeffs.nbytes + self.energy.nbytes
        )
        return {
            "codec": MAGIC,
            "source_shape": self.source_shape,
            "native_resolution": f"{self.source_shape[1]}x{self.source_shape[0]}",
            "k": self.k,
            "raw_rgb_bytes": raw_bytes,
            "holo_bytes": holo_bytes,
            "compression_ratio_vs_rgb": raw_bytes / max(1, holo_bytes),
            **_spectral_metrics(self.energy),
        }


def compress_cmd(args: argparse.Namespace) -> None:
    holo = CatCasHoloImage.from_image(Path(args.input), k=args.k)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".topo.holo")
    holo.save(out)
    stats = holo.stats(out)
    print(f"created={out}")
    print(f"codec={stats['codec']}")
    print(f"native_resolution={stats['native_resolution']} k={stats['k']}")
    print(f"D_pr={stats['d_pr']:.3f} D_shannon={stats['d_shannon']:.3f} k95={stats['k95']}")
    print(f"bytes={stats['holo_bytes']} ratio_vs_rgb={stats['compression_ratio_vs_rgb']:.3f}x")


def info_cmd(args: argparse.Namespace) -> None:
    holo = CatCasHoloImage.load(Path(args.input))
    for key, value in holo.stats(Path(args.input)).items():
        print(f"{key}={value}")


def view_cmd(args: argparse.Namespace) -> None:
    from holo_native_viewer import HoloNativeViewer

    HoloNativeViewer(Path(args.input), render_k=args.k, output_width=args.width, output_height=args.height).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT_CAS topology image .holo")
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("compress")
    comp.add_argument("input")
    comp.add_argument("-o", "--output")
    comp.add_argument("-k", type=int, default=4096)

    info = sub.add_parser("info")
    info.add_argument("input")

    view = sub.add_parser("view")
    view.add_argument("input")
    view.add_argument("-k", type=int)
    view.add_argument("--width", type=int)
    view.add_argument("--height", type=int)

    args = parser.parse_args()
    if args.command == "compress":
        compress_cmd(args)
    elif args.command == "info":
        info_cmd(args)
    elif args.command == "view":
        view_cmd(args)


if __name__ == "__main__":
    main()
