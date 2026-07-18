# Third-party rights and repository policy

This bundle contains metadata, links, original research notes and downloader scripts. It does not redistribute vendor manuals or papers downloaded from third parties.

When you populate `sources/`:

- keep each document's copyright/license notice intact;
- use official or author-provided sources;
- do not publish paywalled copies;
- retain open-access license metadata with papers;
- keep vendor PDFs and binary models in a private local artifact cache outside Git history;
- commit the manifest and the repository-safe `SOURCE_CUSTODY.json` retrieval summary even when binary documents remain local-only;
- keep the raw `DOWNLOAD_RECEIPT.json` and `VERIFICATION_REPORT.json` private and ignored because they are operational cache receipts, while preserving their non-sensitive outcomes in `SOURCE_CUSTODY.json`.

The optional `.url` files in the downloadable ZIP are bookmarks, not copies of the underlying works.
