# CAT_CAS Topology `.holo` Image Report

## Scope

This report covers the image `.holo` work performed against:

- `THOUGHT/LAB/CAT_CAS/README.md`
- `THOUGHT/LAB/HOLO`
- `THOUGHT/LAB/TINY_COMPRESS/holographic-image`
- `THOUGHT/LAB/TINY_COMPRESS/outputs/NEW_RUN/FULLRES_rendered.png`

The goal was to stop treating `.holo` as a bitmap-like compression container and rebuild the image path around the CAT_CAS idea: `.holo` captures topology, and the viewer illuminates that topology at a requested resolution.

## What Was Corrected

The first implementation attempts were wrong for the actual assignment.

1. The legacy image codec used real-valued patch PCA.
2. A second attempt used complex Hermitian phase patches, but it still fixed the image to the source raster grid.
3. The final implementation removes patch storage from the active path and stores sparse complex 2D spectral topology instead.

The final `.holo` file is therefore not "a 2944x2208 image compressed into basis patches." It is a set of dominant phase-frequency modes that can be sampled onto a fresh lattice at native resolution, 4K, 8K, or any custom width and height.

## CAT_CAS Basis

The implementation follows the relevant CAT_CAS patterns:

- Complex-native phase representation from the contained `.holo` phase cavity work.
- Dominant eigenmode / spectral topology as the stored object, not a decoded bitmap.
- Illumination as the measurement step: the file remains a topology payload until the viewer samples it.
- Native viewer behavior: no PNG export is required for inspection.

The important CAT_CAS readme references were:

- Exp 20.11: contained `.holo` phase cavity, where truth emerges when illuminated through the stored basis.
- Exp 21: phase cavity sieve and eigenmode selection.
- Exp 31: `.holo` spectral signatures.
- Exp 33: `.holo` files as native compressed spectral payloads.

## Implemented Files

### `holographic-image/catcas_holo_image.py`

New topology-first codec.

The codec:

1. Converts RGB into a complex phase field on `S1`.
2. Computes a 2D FFT per channel.
3. Selects the top `k` frequency modes by cross-channel spectral energy.
4. Stores signed frequency coordinates plus complex coefficients.
5. Illuminates by placing those modes onto a requested output frequency lattice and applying inverse FFT.

The active format signature is:

```text
CAT_CAS_TOPO_IMAGE_HOLO_V2
```

Older fixed-grid payloads are intentionally rejected.

### `holographic-image/holo_native_viewer.py`

Rewritten native viewer.

The viewer:

- Opens only the new topology `.holo` format.
- Provides `k` illumination control.
- Provides explicit width and height controls.
- Includes Native, 2x, 4K, and 8K sampling presets.
- Renders in memory for display.
- Does not write PNG outputs.

### `holographic-image/holo_open.pyw`

Console-free Windows opener.

This lets `.holo` files open through `pythonw.exe` without spawning a terminal window.

## Generated `.holo` Outputs

Source image:

```text
THOUGHT/LAB/TINY_COMPRESS/outputs/NEW_RUN/FULLRES_rendered.png
```

Source dimensions:

```text
2208x2944 RGB
```

Generated topology payloads:

| File | k | Size | Ratio vs raw RGB |
|---|---:|---:|---:|
| `FULLRES_rendered_catcas_topology_k2048.holo` | 2,048 | 59,381 bytes | 328.4x |
| `FULLRES_rendered_catcas_topology_k8192.holo` | 8,192 | 233,202 bytes | 83.6x |
| `FULLRES_rendered_catcas_topology_k65536.holo` | 65,536 | 1,837,488 bytes | 10.6x |

The k65536 file is the current quality default.

## Verification

Commands run:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  THOUGHT\LAB\TINY_COMPRESS\holographic-image\catcas_holo_image.py `
  THOUGHT\LAB\TINY_COMPRESS\holographic-image\holo_native_viewer.py `
  THOUGHT\LAB\TINY_COMPRESS\holographic-image\holo_open.pyw
```

```powershell
.\.venv\Scripts\python.exe THOUGHT\LAB\TINY_COMPRESS\holographic-image\catcas_holo_image.py `
  info THOUGHT\LAB\TINY_COMPRESS\outputs\NEW_RUN\FULLRES_rendered_catcas_topology_k65536.holo
```

Observed k65536 info:

```text
codec=CAT_CAS_TOPO_IMAGE_HOLO_V2
source_shape=(2944, 2208, 3)
native_resolution=2208x2944
k=65536
raw_rgb_bytes=19501056
holo_bytes=1837488
compression_ratio_vs_rgb=10.612888900498943
d_pr=6.053352487423363
d_shannon=28.468168258666992
k95=491
```

Sampling checks confirmed that the same `.holo` illuminates at:

```text
2208x2944
3840x2160
7680x4320
```

## Current Limitations

This is now structurally aligned with the CAT_CAS `.holo` theory, but visual quality is still governed by the number and type of spectral modes. The current topology stores global Fourier modes. That proves resolution-independent illumination, but it can look soft or stylized at low `k`.

The next research step is a better topology basis:

- multi-scale wavelet or steerable pyramid modes,
- local phase cavities,
- learned spectral atoms,
- or a CAT_CAS-style phase-cavity sieve that separates structural modes from texture dispersion.

That would preserve the resolution-independent `.holo` contract while improving visual fidelity per byte.
