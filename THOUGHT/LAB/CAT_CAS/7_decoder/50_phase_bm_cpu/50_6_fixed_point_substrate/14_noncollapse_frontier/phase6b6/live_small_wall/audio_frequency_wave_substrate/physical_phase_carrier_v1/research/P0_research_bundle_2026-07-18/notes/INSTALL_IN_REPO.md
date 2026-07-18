# Installing the bundle beside P0

This repository copy is installed at:

```text
THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/physical_phase_carrier_v1/research/P0_research_bundle_2026-07-18/
```

## Preferred workflow

1. Run the downloader and verifier from this directory.
2. Import the current P0 authored context into `repo_context/` only when a separate custody copy is useful.
3. Make a completed private archive after source downloads.
4. Do not edit candidate-bound P0 files until you intentionally create a new candidate root.

## Windows commands

```powershell
py scripts/download_sources.py --all
py scripts/verify_downloads.py
py scripts/import_repo_context.py --p0-dir "D:\CCC 2.0\AI\agent-governance-system-audio-recursive\THOUGHT\LAB\CAT_CAS\7_decoder\50_phase_bm_cpu\50_6_fixed_point_substrate\14_noncollapse_frontier\phase6b6\live_small_wall\audio_frequency_wave_substrate\physical_phase_carrier_v1"
py scripts/make_complete_archive.py
```

The downloader creates private ignored `DOWNLOAD_RECEIPT.json`. Manually downloaded files should use the `local_filename` from `MANIFEST.json`; rerun the verifier and then `scripts/build_custody_snapshot.py` afterward so repository-safe outcomes are preserved in `SOURCE_CUSTODY.json`.

## Git policy

Commit the small authored files, manifests, scripts and repository-safe `SOURCE_CUSTODY.json`. Keep raw downloader/verifier receipts and third-party binaries in a private local archive or dedicated artifact storage; they are deliberately ignored here. Do not silently replace the current P0 component registry. Compare actual hashes with legacy expected hashes first, then regenerate and review the P0 candidate if normative files change.
