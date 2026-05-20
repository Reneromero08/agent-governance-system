#!/usr/bin/env python3
"""Native .holo viewer.

This opens a .holo projection payload directly and renders it in memory. It does
not export PNGs or other bitmap files. The displayed pixels are computed from:

    coefficients[:, :k] @ basis[:k] + mean
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from holo import HolographicImage


class HoloNativeViewer:
    def __init__(self, holo_path: Path):
        self.holo_path = holo_path
        self.holo = HolographicImage.load(holo_path)
        self.root = tk.Tk()
        self.root.title(f".holo native viewer - {holo_path.name}")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        img_h, img_w = self.holo.image_shape[:2]

        self.max_w = min(screen_w - 120, img_w)
        self.max_h = min(screen_h - 180, img_h)
        self.render_k = tk.IntVar(value=self.holo.k)

        self.canvas = tk.Canvas(self.root, width=self.max_w, height=self.max_h, bg="black")
        self.canvas.pack(fill="both", expand=True)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=8)

        ttk.Label(controls, text="k").pack(side="left")
        self.slider = ttk.Scale(
            controls,
            from_=1,
            to=self.holo.k,
            orient="horizontal",
            command=self.on_slider,
        )
        self.slider.set(self.holo.k)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

        self.status = ttk.Label(controls, text="")
        self.status.pack(side="left")

        self.photo = None
        self.root.bind("<Left>", lambda _event: self.bump_k(-1))
        self.root.bind("<Right>", lambda _event: self.bump_k(1))
        self.root.after(0, self.redraw)

    def bump_k(self, delta: int) -> None:
        k = max(1, min(self.holo.k, self.render_k.get() + delta))
        self.render_k.set(k)
        self.slider.set(k)
        self.redraw()

    def on_slider(self, value: str) -> None:
        self.render_k.set(max(1, min(self.holo.k, int(float(value)))))
        self.redraw()

    def redraw(self) -> None:
        k = self.render_k.get()
        rendered = self.holo.render_full(render_k=k)
        pil = Image.fromarray(rendered)
        pil.thumbnail((self.max_w, self.max_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.config(width=pil.width, height=pil.height)
        self.canvas.create_image(pil.width // 2, pil.height // 2, image=self.photo)

        stats = self.holo.stats()
        self.status.config(
            text=(
                f"k={k}/{self.holo.k} | "
                f"{stats['image_shape'][1]}x{stats['image_shape'][0]} | "
                f"{Path(self.holo_path).stat().st_size / 1024:.1f} KB .holo"
            )
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a .holo payload natively.")
    parser.add_argument("holo", type=Path, help="Path to .holo file")
    args = parser.parse_args()

    if not args.holo.exists():
        raise FileNotFoundError(args.holo)

    HoloNativeViewer(args.holo).run()


if __name__ == "__main__":
    main()
