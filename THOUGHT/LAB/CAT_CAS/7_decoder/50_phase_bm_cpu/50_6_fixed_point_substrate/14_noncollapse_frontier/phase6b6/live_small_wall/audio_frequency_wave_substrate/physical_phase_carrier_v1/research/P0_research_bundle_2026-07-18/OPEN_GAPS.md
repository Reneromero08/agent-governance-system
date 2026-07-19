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
- Omron's official G6K page lists a data-sheet update of 2026-06-01. Capture the exact current asset rather than retaining only the older static URL.
- Nexperia marks the frozen `2N7002PW,115` relay driver Not for Design In. Production type `NX6008NBKW` is a same-package-family candidate, not an authorized or pin-compatible substitution; its exact ordering suffix, pin map, 3.3 V gate margin, on resistance, relay-coil current, thermal behavior and switching timing remain to be reviewed.
- The exact current first-party source for frozen Nexperia `1N4148W,115` could not be verified. Vishay's current `1N4148W` product family is an alternative candidate, not an authorized substitution; exact suffix, source revision, electrical limits, capacitance, polarity, footprint, lifecycle and procurement identity remain to be reviewed.
- SIGLENT North America marks the SDG1032X obsolete and recommends the SDG1032X Plus, while SIGLENT's global SDG1000X page still lists the original model. Treat lifecycle and availability as region-dependent. Current SDG1000X source documents are EN01I, EN01J and E05C; capture those bytes while retaining the old hashes as historical records. The existing-lab SDG1032X assumption remains distinct from any SDG1032X Plus substitution, and ownership is not asserted.
- Spectrum lists the DN2.59x data sheet/manual and driver 7.010 at 2026-05-19. The official data-sheet URL currently yields bytes matching the retained legacy hash, so no distinct new data-sheet byte revision is asserted; the current manual still requires capture.
- Four attempted PMC filename endpoints, including the passive-damping paper, currently resolve to HTML rather than a PDF in automated retrieval. The stable article pages remain authoritative; manual PDF capture and hashing are still required.

## C. Missing science model

The current synthetic generator inserts a desired decaying sinusoid. The next model must derive it from:

- BVD carrier parameters and uncertainty;
- source impedance and limiter;
- analog-switch dynamics and parasitics, with exact ADG1419 model and simulator versions pinned and data-sheet conformance checked;
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
- source phase error for the actual existing SDG1032X and firmware; any SDG1032X Plus substitution must be modeled and qualified separately;
- vibration coupling and enclosure microphonics.
