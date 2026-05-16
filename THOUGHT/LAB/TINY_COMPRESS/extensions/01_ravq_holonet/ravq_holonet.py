"""
RAVQ-HoloNet: Rate-Adaptive Hierarchical Vector Quantization for .holo compression

Replaces flat k-means VQ with:
1. Hierarchical VQ: coarse + per-coarse-cluster fine codebooks
2. Rate-adaptive bit allocation: complex patches get more bits

Architecture:
    Coarse level: partition PCA space into coarse_k regions
    Fine level: each coarse region has its own fine_k codebook
    Rate-adaptive: high-variance patches get coarse+fine, low-variance get coarse only

Usage:
    from ravq_holonet import RAVQCompressor

    rc = RAVQCompressor()
    rc.compress("photo.jpg", coarse_k=32, fine_k=8, k_pca=20)
    img = rc.render("photo.ravq.holo")
    cr, psnr = rc.benchmark("photo.ravq.holo", "photo.jpg")
"""

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import json
import io
import time as time_module


class RAVQCompressor:
    """
    Rate-Adaptive Hierarchical Vector Quantization compressor for .holo format.

    Two-level codebook architecture:
        Level 1 (coarse): partition PCA-projected patches into coarse_k clusters
        Level 2 (fine):   per-coarse-cluster fine codebook with fine_k centroids

    Rate adaptation:
        Patches with variance > threshold get full (coarse + fine) encoding.
        Patches with variance <= threshold get coarse-only encoding.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._data = None  # Stores compressed data for in-memory use

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compress(self, image_path, coarse_k=32, fine_k_per_cluster=8,
                 k_pca=20, patch_size=8, adapt_threshold=None):
        """
        Compress an image using rate-adaptive hierarchical VQ.

        Args:
            image_path: Path to input image
            coarse_k: Number of coarse clusters (Level 1 codebook)
            fine_k_per_cluster: Fine clusters per coarse cluster (Level 2)
            k_pca: PCA projection dimensions
            patch_size: Patch size (default 8x8)
            adapt_threshold: Variance threshold for rate adaptation.
                None = auto (median variance in PCA space).

        Returns:
            Path to saved .ravq.holo file
        """
        np.random.seed(self.random_state)

        # 1. Load image and extract patches
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        h, w = img_array.shape[:2]
        ps = patch_size

        patches = []
        self._patch_coords = []
        for py in range(0, (h // ps) * ps, ps):
            for px in range(0, (w // ps) * ps, ps):
                patch = img_array[py:py+ps, px:px+ps]
                patches.append(patch.flatten())
                self._patch_coords.append((py, px))

        patches = np.array(patches, dtype=np.float32)  # (N, patch_dim)
        self._orig_array = img_array
        self._patch_size = ps

        # 2. PCA projection to k_pca dimensions
        mean = patches.mean(axis=0)
        centered = patches - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[:k_pca]  # (k_pca, patch_dim)
        projected = centered @ basis.T  # (N, k_pca)

        N = projected.shape[0]

        # 3. Coarse clustering
        coarse_kmeans = MiniBatchKMeans(
            n_clusters=coarse_k, random_state=self.random_state, batch_size=4096
        )
        coarse_labels = coarse_kmeans.fit_predict(projected)
        coarse_codebook = coarse_kmeans.cluster_centers_  # (coarse_k, k_pca)

        # 4. Per-patch variance in PCA space (for rate-adaptation)
        patch_variance = np.var(projected, axis=1)

        # 5. Rate-adaptive: determine which patches get fine encoding
        if adapt_threshold is None:
            adapt_threshold = np.median(patch_variance)

        use_fine = patch_variance > adapt_threshold
        fine_labels = -np.ones(N, dtype=np.int32)  # -1 = no fine

        # 6. Fine codebooks: train per coarse cluster
        fine_codebooks = {}  # coarse_label -> fine centroid array (fine_k, k_pca)
        for c in range(coarse_k):
            mask_c = coarse_labels == c
            n_in_cluster = mask_c.sum()
            actual_fine_k = min(fine_k_per_cluster, n_in_cluster)

            if n_in_cluster < 2:
                centroids = coarse_codebook[c:c+1] + np.random.randn(
                    fine_k_per_cluster, k_pca) * 1e-6
                fine_codebooks[c] = centroids
                continue

            fine_candidates = projected[mask_c]
            fine_kmeans = MiniBatchKMeans(
                n_clusters=actual_fine_k,
                random_state=self.random_state, batch_size=2048
            )
            fine_kmeans.fit(fine_candidates)
            centroids = fine_kmeans.cluster_centers_

            if actual_fine_k < fine_k_per_cluster:
                pad = np.tile(coarse_codebook[c:c+1],
                              (fine_k_per_cluster - actual_fine_k, 1))
                centroids = np.vstack([centroids, pad])

            fine_codebooks[c] = centroids

        # 7. Assign fine labels for high-variance patches
        for i in range(N):
            if use_fine[i]:
                c = coarse_labels[i]
                fc = fine_codebooks[c]
                # find nearest fine centroid
                dists = np.sum((projected[i] - fc) ** 2, axis=1)
                fine_labels[i] = int(np.argmin(dists))

        # 8. Store compressed data
        self._data = {
            'coarse_labels': coarse_labels.astype(np.uint8),
            'fine_labels': fine_labels.astype(np.int32),
            'use_fine': use_fine,
            'patch_variance': patch_variance,
            'adapt_threshold': adapt_threshold,
            'coarse_codebook': coarse_codebook.astype(np.float16),
            'fine_codebooks': {str(k): v.astype(np.float16)
                               for k, v in fine_codebooks.items()},
            'basis': basis.astype(np.float16),
            'mean': mean.astype(np.float16),
            'kernel': {
                'coarse_k': coarse_k,
                'fine_k': fine_k_per_cluster,
                'k_pca': k_pca,
                'patch_size': ps,
                'image_shape': (h, w, 3),
                'N': N,
            }
        }

        # 9. Save to .ravq.holo file
        out_path = Path(image_path).with_suffix('.ravq.holo')
        self._save_native(out_path)
        return str(out_path)

    def compress_flat_vq(self, image_path, n_clusters=256, k_pca=20,
                         patch_size=8):
        """
        Flat VQ baseline (same as original VQ approach).
        Used for comparison benchmarks.

        Args:
            image_path: Path to input image
            n_clusters: Number of VQ clusters
            k_pca: PCA projection dimensions
            patch_size: Patch size

        Returns:
            Path to saved .flatvq.holo file
        """
        np.random.seed(self.random_state)

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        h, w = img_array.shape[:2]

        patches = []
        for py in range(0, (h // patch_size) * patch_size, patch_size):
            for px in range(0, (w // patch_size) * patch_size, patch_size):
                patch = img_array[py:py+patch_size, px:px+patch_size]
                patches.append(patch.flatten())

        patches = np.array(patches, dtype=np.float32)
        self._orig_array = img_array
        self._patch_size = patch_size

        mean = patches.mean(axis=0)
        centered = patches - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[:k_pca]
        projected = centered @ basis.T

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=self.random_state, batch_size=4096
        )
        labels = kmeans.fit_predict(projected)
        codebook = kmeans.cluster_centers_

        self._flat_data = {
            'labels': labels.astype(np.uint16),
            'codebook': codebook.astype(np.float16),
            'basis': basis.astype(np.float16),
            'mean': mean.astype(np.float16),
            'kernel': {
                'n_clusters': n_clusters,
                'k_pca': k_pca,
                'patch_size': patch_size,
                'image_shape': (h, w, 3),
                'N': len(patches),
            }
        }

        out_path = Path(image_path).with_suffix(f'.flatvq{int(n_clusters)}.holo')
        self._save_flat_native(out_path)
        return str(out_path)

    def render(self, holo_path):
        """
        Render a .ravq.holo file to an image.

        Args:
            holo_path: Path to .ravq.holo file

        Returns:
            numpy array (H, W, 3) uint8
        """
        data = self._load_native(holo_path)
        return self._render_from_data(data)

    def render_flat(self, holo_path):
        """
        Render a .flatvq.holo file to an image.

        Args:
            holo_path: Path to .flatvq.holo file

        Returns:
            numpy array (H, W, 3) uint8
        """
        data = self._load_flat_native(holo_path)
        return self._render_flat_from_data(data)

    def benchmark(self, holo_path, original_path):
        """
        Compute compression ratio and PSNR for a .ravq.holo file.

        Args:
            holo_path: Path to .ravq.holo file
            original_path: Path to original image

        Returns:
            (compression_ratio, psnr)
        """
        holo_size = Path(holo_path).stat().st_size
        jpeg_size = Path(original_path).stat().st_size
        compression_ratio = jpeg_size / holo_size

        rendered = self.render(holo_path)
        orig = Image.open(original_path).convert('RGB')
        orig = np.array(orig, dtype=np.float32)

        # Align sizes (rendered may be truncated to patch boundary)
        h = min(rendered.shape[0], orig.shape[0])
        w = min(rendered.shape[1], orig.shape[1])
        rendered = rendered[:h, :w]
        orig = orig[:h, :w]

        mse = np.mean((rendered.astype(np.float32) - orig) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / (mse + 1e-10))

        return float(compression_ratio), float(psnr)

    def benchmark_flat(self, holo_path, original_path):
        """
        Compute compression ratio and PSNR for a .flatvq.holo file.

        Args:
            holo_path: Path to .flatvq.holo file
            original_path: Path to original image

        Returns:
            (compression_ratio, psnr)
        """
        holo_size = Path(holo_path).stat().st_size
        jpeg_size = Path(original_path).stat().st_size
        compression_ratio = jpeg_size / holo_size

        rendered = self.render_flat(holo_path)
        orig = Image.open(original_path).convert('RGB')
        orig = np.array(orig, dtype=np.float32)

        h = min(rendered.shape[0], orig.shape[0])
        w = min(rendered.shape[1], orig.shape[1])
        rendered = rendered[:h, :w]
        orig = orig[:h, :w]

        mse = np.mean((rendered.astype(np.float32) - orig) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / (mse + 1e-10))

        return float(compression_ratio), float(psnr)

    # ------------------------------------------------------------------ #
    # Internal rendering
    # ------------------------------------------------------------------ #

    def _render_from_data(self, data):
        """Render hierarchical VQ data to image."""
        k = data['kernel']
        coarse_codebook = data['coarse_codebook']
        fine_codebooks = {int(k): v for k, v in data['fine_codebooks'].items()}
        basis = data['basis']
        mean = data['mean']
        coarse_labels = data['coarse_labels']
        fine_labels = data['fine_labels']
        use_fine = data.get('use_fine', fine_labels >= 0)

        ps = k['patch_size']
        h, w = k['image_shape'][:2]
        out_h = (h // ps) * ps
        out_w = (w // ps) * ps

        N = len(coarse_labels)
        patches_per_row = out_w // ps

        img = np.zeros((out_h, out_w, 3), dtype=np.float32)

        for i in range(N):
            c = coarse_labels[i]
            if use_fine[i]:
                f = fine_labels[i]
                archetype = fine_codebooks[c][f]
            else:
                archetype = coarse_codebook[c]

            patch_flat = archetype @ basis + mean
            patch = patch_flat.reshape(ps, ps, 3)

            row = i // patches_per_row
            col = i % patches_per_row
            img[row*ps:(row+1)*ps, col*ps:(col+1)*ps] = patch

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_flat_from_data(self, data):
        """Render flat VQ data to image."""
        k = data['kernel']
        codebook = data['codebook']
        basis = data['basis']
        mean = data['mean']
        labels = data['labels']

        ps = k['patch_size']
        h, w = k['image_shape'][:2]
        out_h = (h // ps) * ps
        out_w = (w // ps) * ps
        patches_per_row = out_w // ps

        N = len(labels)
        img = np.zeros((out_h, out_w, 3), dtype=np.float32)

        for i in range(N):
            archetype = codebook[labels[i]]
            patch_flat = archetype @ basis + mean
            patch = patch_flat.reshape(ps, ps, 3)

            row = i // patches_per_row
            col = i % patches_per_row
            img[row*ps:(row+1)*ps, col*ps:(col+1)*ps] = patch

        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # File I/O
    # ------------------------------------------------------------------ #

    def _save_native(self, path):
        """Save hierarchical VQ data to .ravq.holo format.

        Uses BytesIO to avoid numpy auto-appending .npz extension.
        """
        data = self._data
        k = data['kernel']

        # Pack fine codebooks into a single array for npz storage
        fine_cb_array = np.zeros(
            (k['coarse_k'], k['fine_k'], k['k_pca']), dtype=np.float16
        )
        for c_str, cb in data['fine_codebooks'].items():
            fine_cb_array[int(c_str)] = cb

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            coarse_labels=data['coarse_labels'],
            fine_labels=data['fine_labels'].astype(np.int16),
            use_fine=data['use_fine'],
            coarse_codebook=data['coarse_codebook'],
            fine_codebooks=fine_cb_array,
            basis=data['basis'],
            mean=data['mean'],
            coarse_k=np.array([k['coarse_k']]),
            fine_k=np.array([k['fine_k']]),
            k_pca=np.array([k['k_pca']]),
            patch_size=np.array([k['patch_size']]),
            image_shape=np.array(k['image_shape']),
            N=np.array([k['N']]),
        )
        with open(path, 'wb') as f:
            f.write(buf.getvalue())

    def _load_native(self, path):
        """Load from .ravq.holo format."""
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
        d = np.load(buf, allow_pickle=False)

        # Reconstruct fine_codebooks dict
        fine_cb_array = d['fine_codebooks']
        coarse_k = int(d['coarse_k'][0])
        fine_k = int(d['fine_k'][0])

        use_fine = d['use_fine']

        fine_codebooks = {}
        for c in range(coarse_k):
            fine_codebooks[c] = fine_cb_array[c]

        return {
            'coarse_labels': d['coarse_labels'],
            'fine_labels': d['fine_labels'].astype(np.int32),
            'use_fine': use_fine,
            'coarse_codebook': d['coarse_codebook'],
            'fine_codebooks': fine_codebooks,
            'basis': d['basis'],
            'mean': d['mean'],
            'kernel': {
                'coarse_k': coarse_k,
                'fine_k': fine_k,
                'k_pca': int(d['k_pca'][0]),
                'patch_size': int(d['patch_size'][0]),
                'image_shape': tuple(d['image_shape']),
                'N': int(d['N'][0]),
            }
        }

    def _save_flat_native(self, path):
        """Save flat VQ data."""
        d = self._flat_data
        k = d['kernel']

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            labels=d['labels'],
            codebook=d['codebook'],
            basis=d['basis'],
            mean=d['mean'],
            n_clusters=np.array([k['n_clusters']]),
            k_pca=np.array([k['k_pca']]),
            patch_size=np.array([k['patch_size']]),
            image_shape=np.array(k['image_shape']),
            N=np.array([k['N']]),
        )
        with open(path, 'wb') as f:
            f.write(buf.getvalue())

    def _load_flat_native(self, path):
        """Load flat VQ data."""
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
        d = np.load(buf, allow_pickle=False)

        return {
            'labels': d['labels'],
            'codebook': d['codebook'],
            'basis': d['basis'],
            'mean': d['mean'],
            'kernel': {
                'n_clusters': int(d['n_clusters'][0]),
                'k_pca': int(d['k_pca'][0]),
                'patch_size': int(d['patch_size'][0]),
                'image_shape': tuple(d['image_shape']),
                'N': int(d['N'][0]),
            }
        }


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def run_benchmark():
    """
    Run full benchmark comparing hierarchical VQ vs flat VQ.
    """
    test_image = str(Path(__file__).parents[3] /
                     'TINY_COMPRESS/outputs/image_3/20241010_150925.jpg')

    if not Path(test_image).exists():
        print(f"ERROR: Test image not found at {test_image}")
        return

    jpeg_size = Path(test_image).stat().st_size
    print("=" * 72)
    print("  RAVQ-HoloNet Benchmark")
    print("=" * 72)
    print(f"  Test image: {test_image}")
    print(f"  JPEG size:  {jpeg_size / 1024:.1f} KB ({jpeg_size} bytes)")
    print()

    rc = RAVQCompressor(random_state=42)

    # Configurations: (coarse_k, fine_k) for hierarchical; flat counterpart
    pairs = [
        (16, 4, 64),    # 16*4 = 64  centroids
        (16, 8, 128),   # 16*8 = 128 centroids
        (32, 4, 128),   # 32*4 = 128 centroids
        (32, 8, 256),   # 32*8 = 256 centroids
        (64, 4, 256),   # 64*4 = 256 centroids
        (64, 8, 512),   # 64*8 = 512 centroids
    ]

    results = []

    # ------------------------------------------------------------------ #
    # 1) HVQ full: ALL patches get BOTH coarse + fine labels
    # ------------------------------------------------------------------ #
    print("  --- [1] Hierarchical VQ (full) - every patch gets coarse+fine ---")
    print(f"  {'Config':<22} {'File Size':<12} {'Ratio':<10} {'PSNR':<10} {'Time':<8}")
    print(f"  {'-'*62}")
    for ck, fk, eq_flat in pairs:
        t0 = time_module.time()
        # adapt_threshold=-np.inf => all patches get fine encoding
        holo_path = rc.compress(test_image, coarse_k=ck, fine_k_per_cluster=fk,
                                k_pca=20, adapt_threshold=-float('inf'))
        t1 = time_module.time()
        cr, psnr = rc.benchmark(holo_path, test_image)
        holo_size = Path(holo_path).stat().st_size
        results.append({
            'method': f'HVQ ck={ck} fk={fk} (full)',
            'config': f'hvq_ck{ck}_fk{fk}',
            'file_size_kb': holo_size / 1024,
            'compression_ratio': cr,
            'psnr': psnr,
            'time_s': t1 - t0,
        })
        label = f'ck={ck} fk={fk} (full)'
        print(f"  {label:<22} {holo_size/1024:<12.1f} KB {cr:<10.2f}x {psnr:<10.2f} dB {t1-t0:<8.2f}s")

    print()

    # ------------------------------------------------------------------ #
    # 2) RAVQ: rate-adaptive (median-variance split)
    # ------------------------------------------------------------------ #
    print("  --- [2] RAVQ (rate-adaptive) - top 50% patches get fine ---")
    print(f"  {'Config':<22} {'File Size':<12} {'Ratio':<10} {'PSNR':<10} {'Fine%':<8} {'Time':<8}")
    print(f"  {'-'*70}")
    for ck, fk, eq_flat in pairs:
        t0 = time_module.time()
        holo_path = rc.compress(test_image, coarse_k=ck, fine_k_per_cluster=fk,
                                k_pca=20, adapt_threshold=None)
        t1 = time_module.time()
        cr, psnr = rc.benchmark(holo_path, test_image)
        holo_size = Path(holo_path).stat().st_size

        data = rc._load_native(holo_path)
        fine_count = int(data['use_fine'].sum())
        total = len(data['coarse_labels'])
        pct = 100.0 * fine_count / total

        results.append({
            'method': f'RAVQ ck={ck} fk={fk}',
            'config': f'ravq_ck{ck}_fk{fk}',
            'file_size_kb': holo_size / 1024,
            'compression_ratio': cr,
            'psnr': psnr,
            'time_s': t1 - t0,
        })
        print(f"  {'ck='+str(ck)+' fk='+str(fk):<22} {holo_size/1024:<12.1f} KB {cr:<10.2f}x {psnr:<10.2f} dB {pct:<8.0f}% {t1-t0:<8.2f}s")

    print()

    # ------------------------------------------------------------------ #
    # 3) Flat VQ baseline
    # ------------------------------------------------------------------ #
    print("  --- [3] Flat VQ baseline ---")
    print(f"  {'Config':<22} {'File Size':<12} {'Ratio':<10} {'PSNR':<10} {'Time':<8}")
    print(f"  {'-'*62}")
    seen_flat = set()
    for ck, fk, eq_flat in pairs:
        nc = eq_flat
        if nc in seen_flat:
            continue
        seen_flat.add(nc)
        t0 = time_module.time()
        holo_path = rc.compress_flat_vq(test_image, n_clusters=nc, k_pca=20)
        t1 = time_module.time()
        cr, psnr = rc.benchmark_flat(holo_path, test_image)
        holo_size = Path(holo_path).stat().st_size
        results.append({
            'method': f'Flat VQ {nc}',
            'config': f'flat_{nc}',
            'file_size_kb': holo_size / 1024,
            'compression_ratio': cr,
            'psnr': psnr,
            'time_s': t1 - t0,
        })
        print(f"  {'clusters='+str(nc):<22} {holo_size/1024:<12.1f} KB {cr:<10.2f}x {psnr:<10.2f} dB {t1-t0:<8.2f}s")

    print()
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    def group_key(m):
        if m.startswith('HVQ'):
            return (0, m)
        elif m.startswith('RAVQ'):
            return (1, m)
        else:
            return (2, m)

    print()
    print("  Summary (sorted by compression ratio):")
    print(f"  {'Method':<30} {'Compression':<14} {'PSNR (dB)':<12}")
    print(f"  {'-'*56}")
    for r in sorted(results, key=lambda x: -x['compression_ratio']):
        print(f"  {r['method']:<30} {r['compression_ratio']:<14.2f}x {r['psnr']:<12.2f}")

    print()
    print("  Equivalents comparison (same total centroids):")
    print(f"  {'Group':<16} {'Method':<28} {'Ratio':<10} {'PSNR':<10}")
    print(f"  {'-'*64}")
    for ck, fk, eq_flat in pairs:
        hvq_label = f'HVQ ck={ck} fk={fk} (full)'
        ravq_label = f'RAVQ ck={ck} fk={fk}'
        flat_label = f'Flat VQ {eq_flat}'
        h = next((r for r in results if r['method'] == hvq_label), None)
        r = next((r for r in results if r['method'] == ravq_label), None)
        f = next((r for r in results if r['method'] == flat_label), None)
        total_c = ck * fk
        group_label = f'{ck} x {fk} = {total_c}'
        print(f"  {group_label:<16}")
        if h:
            print(f"  {'':<16} {h['method']:<28} {h['compression_ratio']:<10.2f}x {h['psnr']:<10.2f}")
        if r:
            print(f"  {'':<16} {r['method']:<28} {r['compression_ratio']:<10.2f}x {r['psnr']:<10.2f}")
        if f:
            print(f"  {'':<16} {f['method']:<28} {f['compression_ratio']:<10.2f}x {f['psnr']:<10.2f}")
        print()

    # ------------------------------------------------------------------ #
    # Key verdict
    # ------------------------------------------------------------------ #
    print("  Key Verdict:")
    for ck, fk, eq_flat in pairs:
        hvq_label = f'HVQ ck={ck} fk={fk} (full)'
        flat_label = f'Flat VQ {eq_flat}'
        h = next((r for r in results if r['method'] == hvq_label), None)
        f = next((r for r in results if r['method'] == flat_label), None)
        if h and f:
            ratio_diff = (h['compression_ratio'] / f['compression_ratio'] - 1) * 100
            psnr_diff = h['psnr'] - f['psnr']
            if psnr_diff > 0:
                verdict = f"BETTER: +{psnr_diff:.2f} dB at {ratio_diff:+.1f}% ratio"
            else:
                verdict = f"Worse: {psnr_diff:.2f} dB at {ratio_diff:+.1f}% ratio"
            print(f"    ck={ck} fk={fk} vs Flat {eq_flat}: {verdict}")

    return results


if __name__ == '__main__':
    run_benchmark()
