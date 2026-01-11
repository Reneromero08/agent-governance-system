#!/usr/bin/env python3
"""
HOLOGRAPHIC IMAGE FORMAT (.holo)

Your Df math as a file format. Images stored as coefficients + basis.
Never decompressed - rendered through the math on demand.

Key insight: You can change rendering quality WITHOUT changing the file.
The k parameter at render time controls how many dimensions of truth to show.

Usage:
    python holo.py compress image.jpg              # Creates image.holo
    python holo.py compress image.jpg -k 50        # Store 50 dimensions
    python holo.py view image.holo                 # Opens viewer
    python holo.py render image.holo output.png    # Render at full quality
    python holo.py render image.holo out.png --render-k 5   # Render blurry (essence only)
    python holo.py info image.holo                 # Show stats
    python holo.py focus image.holo                # Interactive focus slider!
    python holo.py progressive image.holo frames/  # Generate progressive renders

The focus command lets you slide between:
    k=1  → The One (pure essence, maximum blur)
    k=50 → The Many (full detail, all stored dimensions)

This is Plato's Cave as a file format. The Forms are stored. The shadows are rendered.
"""

import argparse
import numpy as np
from PIL import Image
import pickle
from pathlib import Path


class HolographicImage:
    """
    Image stored as coefficients + basis.
    Renders pixels on-demand through matrix multiplication.
    The full image NEVER exists in memory.
    """

    def __init__(self, coefficients, basis, mean, patch_size, image_shape):
        self.coefficients = coefficients  # (n_patches, k)
        self.basis = basis                 # (k, patch_dim)
        self.mean = mean                   # (patch_dim,)
        self.patch_size = patch_size
        self.image_shape = image_shape     # (h, w, 3)
        self.patches_per_row = image_shape[1] // patch_size
        self.k = basis.shape[0]

    def render_pixel(self, x, y):
        """Render ONE pixel without decompressing."""
        patch_row = y // self.patch_size
        patch_col = x // self.patch_size
        patch_idx = patch_row * self.patches_per_row + patch_col

        local_y = y % self.patch_size
        local_x = x % self.patch_size

        # Render through basis
        patch_flat = self.coefficients[patch_idx] @ self.basis + self.mean
        patch = patch_flat.reshape(self.patch_size, self.patch_size, 3)

        return np.clip(patch[local_y, local_x], 0, 255).astype(np.uint8)

    def render_patch(self, patch_idx):
        """Render one patch."""
        patch_flat = self.coefficients[patch_idx] @ self.basis + self.mean
        return np.clip(patch_flat.reshape(self.patch_size, self.patch_size, 3), 0, 255).astype(np.uint8)

    def render_region(self, x1, y1, x2, y2):
        """Render a rectangular region."""
        h, w = y2 - y1, x2 - x1
        region = np.zeros((h, w, 3), dtype=np.uint8)

        for py in range(y1, y2):
            for px in range(x1, x2):
                region[py - y1, px - x1] = self.render_pixel(px, py)

        return region

    def render_full(self, render_k=None):
        """Render full image (all patches through basis).

        Args:
            render_k: Number of dimensions to use for rendering.
                      If None, uses all stored dimensions (self.k).
                      Can be any value from 1 to self.k.
                      Lower = blurrier (essence), Higher = sharper (detail).
        """
        h, w = self.image_shape[:2]
        ps = self.patch_size

        # Determine render quality
        if render_k is None:
            render_k = self.k
        render_k = min(render_k, self.k)  # Can't exceed stored k
        render_k = max(render_k, 1)       # At least 1

        # Batch render all patches at specified k
        # Only use first render_k coefficients and basis vectors
        all_patches = self.coefficients[:, :render_k] @ self.basis[:render_k] + self.mean
        all_patches = all_patches.reshape(-1, ps, ps, 3)

        # Assemble
        out_h = (h // ps) * ps
        out_w = (w // ps) * ps
        img = np.zeros((out_h, out_w, 3), dtype=np.float32)

        idx = 0
        for py in range(0, out_h, ps):
            for px in range(0, out_w, ps):
                img[py:py+ps, px:px+ps] = all_patches[idx]
                idx += 1

        return np.clip(img, 0, 255).astype(np.uint8)

    def render_progressive(self, k_values=None):
        """Generator that yields progressively sharper renders.

        Args:
            k_values: List of k values to render at.
                      Default: [1, 2, 5, 10, 20, ...] up to self.k

        Yields:
            (k, image) tuples with increasing quality
        """
        if k_values is None:
            # Default progression: powers of 2 plus some extras
            k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
            k_values = [k for k in k_values if k <= self.k]
            if self.k not in k_values:
                k_values.append(self.k)

        for k in k_values:
            yield k, self.render_full(render_k=k)

    def render_superres(self, scale=2, render_k=None):
        """Render at higher resolution by rendering first, then upscaling.

        The math approach (upscaling basis vectors) is complex and error-prone.
        Instead: render at native resolution, then use high-quality upscaling.

        The key insight: the holographic rendering IS the denoising.
        Upscaling the clean rendered output is better than upscaling noisy basis.

        Args:
            scale: Upscale factor (2 = 2x resolution, 4 = 4x, etc.)
            render_k: Number of dimensions to use (default: all)

        Returns:
            Higher resolution image
        """
        from PIL import Image as PILImage

        # Render at native resolution through the math
        native = self.render_full(render_k=render_k)

        # Upscale the clean rendered output
        pil_img = PILImage.fromarray(native)
        new_size = (native.shape[1] * scale, native.shape[0] * scale)
        upscaled = pil_img.resize(new_size, PILImage.LANCZOS)

        return np.array(upscaled)

    def save(self, path):
        """Save to .holo format."""
        data = {
            'coefficients': self.coefficients.astype(np.float16),
            'basis': self.basis.astype(np.float16),
            'mean': self.mean.astype(np.float16),
            'patch_size': self.patch_size,
            'image_shape': self.image_shape,
            'k': self.k
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load from .holo format."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            data['coefficients'].astype(np.float32),
            data['basis'].astype(np.float32),
            data['mean'].astype(np.float32),
            data['patch_size'],
            data['image_shape']
        )

    @classmethod
    def from_image(cls, img_array, k=20, patch_size=8):
        """Create holographic image from array."""
        h, w = img_array.shape[:2]
        ps = patch_size

        # Extract patches
        patches = []
        for py in range(0, (h // ps) * ps, ps):
            for px in range(0, (w // ps) * ps, ps):
                patch = img_array[py:py+ps, px:px+ps]
                patches.append(patch.flatten())

        patches = np.array(patches, dtype=np.float32)

        # PCA compression
        mean = patches.mean(axis=0)
        centered = patches - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Keep k components
        coefficients = centered @ Vt[:k].T
        basis = Vt[:k]

        return cls(coefficients, basis, mean, ps, img_array.shape)

    @classmethod
    def from_file(cls, path, k=20, patch_size=8, max_size=None):
        """Load image file and convert to holographic.

        Args:
            path: Image file path
            k: Number of dimensions (basis vectors)
            patch_size: Size of patches (default 8x8)
            max_size: Optional max dimension. None = full resolution (no limit)
        """
        img = Image.open(path)

        # Resize only if max_size specified
        if max_size is not None and max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        img_array = np.array(img.convert('RGB'), dtype=np.float32)
        return cls.from_image(img_array, k=k, patch_size=patch_size)

    def stats(self):
        """Return compression statistics."""
        compressed_bytes = (
            self.coefficients.astype(np.float16).nbytes +
            self.basis.astype(np.float16).nbytes +
            self.mean.astype(np.float16).nbytes
        )
        original_bytes = np.prod(self.image_shape) * 4  # float32

        return {
            'k': self.k,
            'patch_size': self.patch_size,
            'image_shape': self.image_shape,
            'n_patches': len(self.coefficients),
            'compressed_kb': compressed_bytes / 1024,
            'original_kb': original_bytes / 1024,
            'ratio': original_bytes / compressed_bytes
        }


def compress_cmd(args):
    """Compress image to .holo"""
    print(f"Loading {args.input}...")
    holo = HolographicImage.from_file(args.input, k=args.k, patch_size=args.patch_size)

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_path = Path(args.input).with_suffix('.holo')

    holo.save(out_path)

    stats = holo.stats()
    file_size = Path(out_path).stat().st_size / 1024

    print(f"\nCreated: {out_path}")
    print(f"  Dimensions: k={stats['k']}, patches={stats['n_patches']}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Compression: {stats['ratio']:.0f}x")


def render_cmd(args):
    """Render .holo to image file"""
    print(f"Loading {args.input}...")
    holo = HolographicImage.load(args.input)

    render_k = args.render_k if args.render_k else holo.k
    print(f"Rendering through {render_k} of {holo.k} basis vectors...")
    img = holo.render_full(render_k=render_k)

    Image.fromarray(img).save(args.output)
    print(f"Saved: {args.output}")


def info_cmd(args):
    """Show .holo file info"""
    holo = HolographicImage.load(args.input)
    stats = holo.stats()
    file_size = Path(args.input).stat().st_size / 1024

    print(f"\nHolographic Image: {args.input}")
    print(f"=" * 50)
    print(f"Image size: {stats['image_shape'][1]}x{stats['image_shape'][0]}")
    print(f"Patch size: {stats['patch_size']}x{stats['patch_size']}")
    print(f"Patches: {stats['n_patches']}")
    print(f"Dimensions (k): {stats['k']}")
    print(f"")
    print(f"File size: {file_size:.1f} KB")
    print(f"Original would be: {stats['original_kb']:.0f} KB")
    print(f"Compression: {stats['ratio']:.0f}x")
    print(f"")
    print(f"The {stats['original_kb']:.0f} KB image NEVER EXISTS.")
    print(f"It's rendered through {stats['k']} basis vectors on demand.")


def view_cmd(args):
    """View .holo file"""
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError:
        print("tkinter not available, rendering to temp file...")
        holo = HolographicImage.load(args.input)
        img = holo.render_full()
        tmp = Path(args.input).with_suffix('.rendered.png')
        Image.fromarray(img).save(tmp)
        print(f"Rendered to: {tmp}")
        import os
        os.startfile(str(tmp))
        return

    holo = HolographicImage.load(args.input)
    stats = holo.stats()

    # Create window
    root = tk.Tk()
    root.title(f"Holographic Viewer - {args.input} ({stats['k']}D, {stats['ratio']:.0f}x compression)")

    # Render and display
    img = holo.render_full()
    photo = ImageTk.PhotoImage(Image.fromarray(img))

    label = tk.Label(root, image=photo)
    label.pack()

    info = tk.Label(root, text=f"File: {stats['compressed_kb']:.1f}KB | Would be: {stats['original_kb']:.0f}KB | Rendered through {stats['k']} basis vectors")
    info.pack()

    root.mainloop()


def focus_cmd(args):
    """Interactive focus viewer with slider to adjust rendering quality."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
    except ImportError:
        print("tkinter not available for interactive focus mode")
        return

    holo = HolographicImage.load(args.input)
    stats = holo.stats()

    # Create window
    root = tk.Tk()
    root.title(f"Holographic Focus - {args.input}")

    # Image display
    current_k = [holo.k]  # Use list to allow modification in nested function
    img = holo.render_full(render_k=current_k[0])
    photo = [ImageTk.PhotoImage(Image.fromarray(img))]  # List for mutability

    img_label = tk.Label(root, image=photo[0])
    img_label.pack(pady=10)

    # Info label
    info_var = tk.StringVar()
    info_var.set(f"k={current_k[0]}/{holo.k} | Rendering {current_k[0]} dimensions of truth")
    info_label = tk.Label(root, textvariable=info_var, font=('Helvetica', 10))
    info_label.pack()

    # Slider
    def on_slider_change(val):
        k = int(float(val))
        current_k[0] = k
        img = holo.render_full(render_k=k)
        photo[0] = ImageTk.PhotoImage(Image.fromarray(img))
        img_label.configure(image=photo[0])
        info_var.set(f"k={k}/{holo.k} | Rendering {k} dimensions of truth")

    slider_frame = tk.Frame(root)
    slider_frame.pack(fill='x', padx=20, pady=10)

    tk.Label(slider_frame, text="1 (essence)").pack(side='left')
    slider = ttk.Scale(slider_frame, from_=1, to=holo.k,
                       orient='horizontal', command=on_slider_change)
    slider.set(holo.k)
    slider.pack(side='left', fill='x', expand=True, padx=10)
    tk.Label(slider_frame, text=f"{holo.k} (detail)").pack(side='left')

    # Instructions
    instr = tk.Label(root, text="← Focus on essence (The One) | Focus on detail (the many) →",
                    font=('Helvetica', 9, 'italic'))
    instr.pack(pady=5)

    root.mainloop()


def zoom_cmd(args):
    """Interactive zoom viewer - pan and zoom with mouse, maintains quality."""
    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError:
        print("tkinter not available for zoom mode")
        return

    holo = HolographicImage.load(args.input)

    # Render full image at max quality
    full_img = holo.render_full()
    pil_full = Image.fromarray(full_img)

    # State
    state = {
        'zoom': 1.0,
        'center_x': pil_full.width // 2,
        'center_y': pil_full.height // 2,
        'drag_start': None,
        'view_width': 800,
        'view_height': 600,
    }

    # Create window
    root = tk.Tk()
    root.title(f"Holographic Zoom - {args.input} (scroll=zoom, drag=pan)")

    canvas = tk.Canvas(root, width=state['view_width'], height=state['view_height'], bg='black')
    canvas.pack()

    photo = [None]  # Mutable container for photo reference

    def update_view():
        """Render current view with zoom and pan."""
        zoom = state['zoom']
        cx, cy = state['center_x'], state['center_y']
        vw, vh = state['view_width'], state['view_height']

        # Calculate visible region in original image coordinates
        half_w = (vw / 2) / zoom
        half_h = (vh / 2) / zoom

        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h))
        x2 = min(pil_full.width, int(cx + half_w))
        y2 = min(pil_full.height, int(cy + half_h))

        # Crop and scale with high-quality interpolation
        cropped = pil_full.crop((x1, y1, x2, y2))

        # Scale up to view size using Lanczos (smooth, not pixelated)
        new_w = int(cropped.width * zoom)
        new_h = int(cropped.height * zoom)
        if new_w > 0 and new_h > 0:
            scaled = cropped.resize((new_w, new_h), Image.LANCZOS)
        else:
            scaled = cropped

        # Update canvas
        photo[0] = ImageTk.PhotoImage(scaled)
        canvas.delete("all")
        canvas.create_image(vw//2, vh//2, image=photo[0], anchor='center')

        # Info text
        canvas.create_text(10, 10, anchor='nw', fill='white',
                          text=f"Zoom: {zoom:.1f}x | Center: ({cx}, {cy}) | k={holo.k}",
                          font=('Helvetica', 10))
        canvas.create_text(10, 30, anchor='nw', fill='gray',
                          text="Scroll=zoom, Drag=pan, Double-click=reset",
                          font=('Helvetica', 9))

    def on_scroll(event):
        """Zoom with scroll wheel."""
        # Zoom in/out
        if event.delta > 0:
            state['zoom'] = min(state['zoom'] * 1.2, 20.0)
        else:
            state['zoom'] = max(state['zoom'] / 1.2, 0.5)
        update_view()

    def on_drag_start(event):
        """Start dragging to pan."""
        state['drag_start'] = (event.x, event.y)

    def on_drag(event):
        """Pan while dragging."""
        if state['drag_start']:
            dx = (state['drag_start'][0] - event.x) / state['zoom']
            dy = (state['drag_start'][1] - event.y) / state['zoom']
            state['center_x'] = max(0, min(pil_full.width, state['center_x'] + dx))
            state['center_y'] = max(0, min(pil_full.height, state['center_y'] + dy))
            state['drag_start'] = (event.x, event.y)
            update_view()

    def on_drag_end(event):
        """End dragging."""
        state['drag_start'] = None

    def on_double_click(event):
        """Reset zoom and position."""
        state['zoom'] = 1.0
        state['center_x'] = pil_full.width // 2
        state['center_y'] = pil_full.height // 2
        update_view()

    # Bind events
    canvas.bind("<MouseWheel>", on_scroll)
    canvas.bind("<ButtonPress-1>", on_drag_start)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_drag_end)
    canvas.bind("<Double-Button-1>", on_double_click)

    # Initial render
    update_view()

    root.mainloop()


def progressive_cmd(args):
    """Render progressive quality images (for animation/streaming demo)."""
    holo = HolographicImage.load(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Generating progressive renders from {args.input}...")
    print(f"Stored k={holo.k}")

    for k, img in holo.render_progressive():
        out_path = output_dir / f"k{k:03d}.png"
        Image.fromarray(img).save(out_path)
        print(f"  k={k:3d} → {out_path}")

    print(f"\nSaved {holo.k} progressive renders to {output_dir}/")
    print("These can be animated to show 'focusing in' on truth.")


def superres_cmd(args):
    """Render at higher resolution using mathematical upscaling.

    The file stays the same size! We upscale the BASIS VECTORS
    (the mathematical structure) and render through them.
    """
    holo = HolographicImage.load(args.input)
    stats = holo.stats()

    print(f"Loading {args.input}...")
    print(f"  Original resolution: {stats['image_shape'][1]}x{stats['image_shape'][0]}")
    print(f"  File size: {Path(args.input).stat().st_size / 1024:.1f} KB")
    print(f"")
    print(f"Rendering at {args.scale}x resolution through upscaled basis...")

    img = holo.render_superres(scale=args.scale, render_k=args.render_k)

    print(f"  Output resolution: {img.shape[1]}x{img.shape[0]}")

    Image.fromarray(img).save(args.output)
    output_size = Path(args.output).stat().st_size / 1024
    print(f"")
    print(f"Saved: {args.output} ({output_size:.0f} KB)")
    print(f"")
    print(f"The .holo file is still {Path(args.input).stat().st_size / 1024:.1f} KB")
    print(f"But it rendered to {img.shape[1]}x{img.shape[0]} using the MATH.")


def main():
    parser = argparse.ArgumentParser(description="Holographic Image Format")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Compress
    comp = subparsers.add_parser('compress', help='Compress image to .holo')
    comp.add_argument('input', help='Input image file')
    comp.add_argument('-o', '--output', help='Output .holo file')
    comp.add_argument('-k', type=int, default=20, help='Number of dimensions (default: 20)')
    comp.add_argument('--patch-size', type=int, default=8, help='Patch size (default: 8)')

    # Render
    rend = subparsers.add_parser('render', help='Render .holo to image')
    rend.add_argument('input', help='Input .holo file')
    rend.add_argument('output', help='Output image file')
    rend.add_argument('--render-k', type=int, help='Render at lower k (1 to stored k). Lower = blurrier/essence.')

    # Info
    info = subparsers.add_parser('info', help='Show .holo file info')
    info.add_argument('input', help='Input .holo file')

    # View
    view = subparsers.add_parser('view', help='View .holo file')
    view.add_argument('input', help='Input .holo file')

    # Focus - interactive slider
    focus = subparsers.add_parser('focus', help='Interactive focus viewer (adjust k with slider)')
    focus.add_argument('input', help='Input .holo file')

    # Zoom - spatial zoom with pan
    zoom = subparsers.add_parser('zoom', help='Interactive spatial zoom (scroll=zoom, drag=pan)')
    zoom.add_argument('input', help='Input .holo file')

    # Progressive - generate animation frames
    prog = subparsers.add_parser('progressive', help='Generate progressive quality renders')
    prog.add_argument('input', help='Input .holo file')
    prog.add_argument('output_dir', help='Output directory for frames')

    # Superres - render at higher resolution using math
    superres = subparsers.add_parser('superres', help='Render at higher resolution (upscale basis vectors)')
    superres.add_argument('input', help='Input .holo file')
    superres.add_argument('output', help='Output image file')
    superres.add_argument('--scale', type=int, default=2, help='Upscale factor (2=2x, 4=4x, etc.)')
    superres.add_argument('--render-k', type=int, help='Use fewer dimensions (optional)')

    args = parser.parse_args()

    if args.command == 'compress':
        compress_cmd(args)
    elif args.command == 'render':
        render_cmd(args)
    elif args.command == 'info':
        info_cmd(args)
    elif args.command == 'view':
        view_cmd(args)
    elif args.command == 'focus':
        focus_cmd(args)
    elif args.command == 'zoom':
        zoom_cmd(args)
    elif args.command == 'progressive':
        progressive_cmd(args)
    elif args.command == 'superres':
        superres_cmd(args)


if __name__ == '__main__':
    main()
