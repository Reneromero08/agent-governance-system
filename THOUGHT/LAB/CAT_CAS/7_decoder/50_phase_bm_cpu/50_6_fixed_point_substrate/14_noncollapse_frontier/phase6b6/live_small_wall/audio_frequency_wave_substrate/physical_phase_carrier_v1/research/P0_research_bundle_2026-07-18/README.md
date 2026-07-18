# P0 research bundle

**Bundle version:** `1.0.0`

**Research/access audit:** `2026-07-18`
**Scope:** P0 physical phase-carrier WIP, non-executing research only

## What is here

This bundle converts the P0 research trail into a reproducible source archive plan:

- `MANIFEST.json`: 24 core component/instrument records plus scientific and simulation resources.
- `DOWNLOAD_LINKS.md`: clickable catalog for official downloads and product pages.
- `scripts/download_sources.py`: downloads direct official/open-access PDFs and writes a hash receipt.
- `scripts/verify_downloads.py`: inventories all retained bytes and compares them to legacy expected hashes.
- `scripts/import_repo_context.py`: copies the current authored P0 files into `repo_context/` without changing the repository.
- `RESEARCH_SYNTHESIS.md`: what the research says about the non-hardware simulation and the next model.
- `OPEN_GAPS.md`: what still requires modeling, calibration or implementation.
- `REGISTRY_RECONCILIATION.md`: how to repair the misleading “captured” status safely.

## Archival source state and later private refresh

At source commit `cb53976612cbe83bec82df826a9889418f7e0b89`, direct third-party downloads were unavailable in the originating artifact container, so the imported authored bundle contained scripts, official links, target filenames and legacy comparisons rather than source bytes. A later private refresh in this worktree made automated public-source requests and captured 11 hash-verified files in ignored local storage. `SOURCE_CUSTODY.json` is the repository-safe snapshot of that later refresh; raw receipts and third-party binaries remain private, ignored and outside the candidate root.

## Fastest path on Windows

From this repository directory, open PowerShell and run:

```powershell
py scripts/download_sources.py --all
py scripts/verify_downloads.py
```

Then open `DOWNLOAD_LINKS.md` and manually download any record reported as `MANUAL_REQUIRED`. Save it under the exact `local_filename` in `MANIFEST.json`, in either:

```text
sources/official/
sources/supplemental/
```

Rerun:

```powershell
py scripts/verify_downloads.py --require-all-core
```

To add the authored local P0 context:

```powershell
py scripts/import_repo_context.py --p0-dir "D:/CCC 2.0/AI/agent-governance-system-audio-recursive/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/live_small_wall/audio_frequency_wave_substrate/physical_phase_carrier_v1"
```

To make a final private archive containing the downloaded documents:

```powershell
py scripts/make_complete_archive.py
```

## Most important research result

The current simulator is real signal-processing software and internally consistent, but it directly constructs the desired ringdown waveform. The strongest next scientific step is a source-cited BVD plus circuit transient model using the available ADG1419 and OPA810 vendor models, with the existing analyzer left unchanged. See `RESEARCH_SYNTHESIS.md`.

## Do not silently overwrite the current P0 candidate

The P0 validator binds candidate bytes into a root. Keep this bundle separate first. Updating `P0_COMPONENT_DOCUMENTS.json` or other candidate-bound files should create a deliberate new candidate root followed by review.
