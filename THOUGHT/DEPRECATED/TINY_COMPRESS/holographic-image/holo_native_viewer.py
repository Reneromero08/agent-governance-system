#!/usr/bin/env python3
"""Native CAT_CAS topology .holo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from catcas_holo_image import CatCasHoloImage


class HoloNativeViewer:
    def __init__(
        self,
        holo_path: Path,
        render_k: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
    ):
        self.holo_path = Path(holo_path)
        self.holo = CatCasHoloImage.load(self.holo_path)
        src_h, src_w, _ = self.holo.source_shape

        self.root = tk.Tk()
        self.root.title(f"CAT_CAS topology .holo - {self.holo_path.name}")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.max_w = max(320, screen_w - 120)
        self.max_h = max(240, screen_h - 220)

        k0 = self.holo.k if render_k is None else max(1, min(int(render_k), self.holo.k))
        self.render_k = tk.IntVar(value=k0)
        self.width = tk.IntVar(value=int(output_width or src_w))
        self.height = tk.IntVar(value=int(output_height or src_h))
        self.photo = None

        self.canvas = tk.Canvas(self.root, width=min(src_w, self.max_w), height=min(src_h, self.max_h), bg="black")
        self.canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=8)

        ttk.Label(controls, text="k").pack(side="left")
        self.slider = ttk.Scale(controls, from_=1, to=self.holo.k, orient="horizontal", command=self.on_slider)
        self.slider.set(k0)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

        ttk.Label(controls, text="W").pack(side="left")
        self.width_entry = ttk.Entry(controls, textvariable=self.width, width=7)
        self.width_entry.pack(side="left", padx=(4, 8))
        ttk.Label(controls, text="H").pack(side="left")
        self.height_entry = ttk.Entry(controls, textvariable=self.height, width=7)
        self.height_entry.pack(side="left", padx=(4, 8))
        ttk.Button(controls, text="Illuminate", command=self.redraw).pack(side="left")

        presets = ttk.Frame(self.root)
        presets.pack(fill="x", padx=12, pady=(0, 8))
        ttk.Button(presets, text="Native", command=lambda: self.set_resolution(src_w, src_h)).pack(side="left")
        ttk.Button(presets, text="2x", command=lambda: self.set_resolution(src_w * 2, src_h * 2)).pack(side="left")
        ttk.Button(presets, text="4K", command=lambda: self.set_resolution(3840, 2160)).pack(side="left")
        ttk.Button(presets, text="8K", command=lambda: self.set_resolution(7680, 4320)).pack(side="left")
        self.status = ttk.Label(presets, text="")
        self.status.pack(side="left", padx=12)

        self.root.bind("<Left>", lambda _event: self.bump_k(-1))
        self.root.bind("<Right>", lambda _event: self.bump_k(1))
        self.root.bind("<Return>", lambda _event: self.redraw())
        self.root.after(0, self.redraw)
        self.root.after(150, self.bring_to_front)

    def bring_to_front(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.after(500, lambda: self.root.attributes("-topmost", False))

    def set_resolution(self, width: int, height: int) -> None:
        self.width.set(int(width))
        self.height.set(int(height))
        self.redraw()

    def bump_k(self, delta: int) -> None:
        k = max(1, min(self.holo.k, self.render_k.get() + delta))
        self.render_k.set(k)
        self.slider.set(k)
        self.redraw()

    def on_slider(self, value: str) -> None:
        self.render_k.set(max(1, min(self.holo.k, int(float(value)))))

    def redraw(self) -> None:
        k = self.render_k.get()
        width = max(1, int(self.width.get()))
        height = max(1, int(self.height.get()))
        rendered = self.holo.illuminate(render_k=k, output_shape=(height, width))
        pil = Image.fromarray(rendered)
        pil.thumbnail((self.max_w, self.max_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(pil.width // 2, pil.height // 2, image=self.photo)

        stats = self.holo.stats(self.holo_path)
        self.status.config(
            text=(
                f"sample={width}x{height} | k={k}/{self.holo.k} | "
                f"D_pr={stats['d_pr']:.2f} | {stats['holo_bytes'] / 1024:.1f} KB"
            )
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a CAT_CAS topology .holo.")
    parser.add_argument("holo", type=Path)
    parser.add_argument("-k", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    HoloNativeViewer(args.holo, render_k=args.k, output_width=args.width, output_height=args.height).run()


if __name__ == "__main__":
    main()
