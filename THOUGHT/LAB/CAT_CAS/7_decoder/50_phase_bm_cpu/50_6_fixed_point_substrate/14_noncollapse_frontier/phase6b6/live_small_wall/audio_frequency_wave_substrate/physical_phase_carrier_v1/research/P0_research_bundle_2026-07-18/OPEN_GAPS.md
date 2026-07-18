# Open research and engineering gaps

## A. Document custody gaps that this bundle makes actionable

1. Run the downloader and retain the actual vendor bytes.
2. Manually capture dynamic/product-page-only documents and place them under `sources/official/` using the manifest filename.
3. Record actual byte count, SHA-256, final resolved URL and retrieval timestamp in `DOWNLOAD_RECEIPT.json`.
4. Preserve legacy expected hashes even when current documents differ.
5. Reconcile the local P0 registry only after actual files exist.

## B. Revision discrepancies found

- `ADR45XX_REV_F` is stale as a label. The current official family data sheet is Rev. G.
- ST UM2591 is currently Rev. 2, April 2026, and the current NUCLEO-G031K8 resource set references MB1455-G031K8-D01.
- The Sensirion product page labels the SHT4x document 04/2025, while the PDF cover says Version 7.1, March 2025.
- Omron's official G6K page lists a data-sheet update of 06/01/2026. Capture the exact current asset rather than retaining only the older static URL.

## C. Missing science model

The current synthetic generator inserts a desired decaying sinusoid. The next model must derive it from:

- BVD carrier parameters and uncertainty;
- source impedance and limiter;
- analog-switch dynamics and parasitics;
- relay state sequence and bounce/leakage;
- OPA810 dynamics and input/output loading;
- digitizer input admittance;
- cable/PCB/ground parasitics;
- noise and quantization.

## D. Missing software integrations

- Spectrum SDK lossless export adapter with exact source custody.
- Nucleo gate/relay/environment firmware and timing proof.
- SPICE-to-raw-bundle adapter that produces the same canonical four-channel bytes consumed by `p0_scientific_analyzer.py`.
- Independent cross-check implementation of the primary phase/decay metrics.

## E. Parameters that should remain sweeps until measured

Do not replace these with one asserted value:

- FC-135 motional R/L/C and C0;
- Q and frequency versus mounting, temperature and drive;
- total sense-node capacitance and leakage;
- switch charge injection/off capacitance;
- relay timing and bounce;
- digitizer input admittance;
- source phase error;
- vibration coupling and enclosure microphonics.
