# Source storage

- `official/`: vendor manuals, data sheets and exact product documents.
- `supplemental/`: papers and simulation/design resources.

At source commit `cb53976612cbe83bec82df826a9889418f7e0b89`, these binary directories were empty because source documents could not be exported from the originating environment. A later private refresh captured a subset locally; all such bytes remain ignored. `SOURCE_CUSTODY.json` records the repository-safe current state. Run `scripts/download_sources.py --all`, place manual downloads under the exact manifest filenames, verify, and rebuild the custody snapshot. The repository edition uses `DOWNLOAD_LINKS.md` and `MANIFEST.json` instead of ZIP-only shortcuts or generated views.
