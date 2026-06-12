# VERDICT.md Schema Reference (verdict/v2)

Normative reference for v2_3 verdict files. The validator implements exactly
this document. ASCII only in all files.

---

## 1. Status enum

Ascending total order:

    FALSIFIED < UNSUPPORTED < PARTIALLY_VERIFIED < VERIFIED

- A verdict's overall `status` MUST equal the MIN over `claims[].status`.
  The validator recomputes this; a mismatch is `E_STATUS_NOT_MIN`.
- `OPEN` exists ONLY at index level. It means: no VERDICT.md file exists for
  that question directory. `OPEN` never appears inside a VERDICT.md.
- Legacy vocabulary is invalid in v2_3: `CONFIRMED`, `PARTIALLY VERIFIED`
  (with a space), `CONFIRMED (boundary)`, etc.

## 2. File layout

A VERDICT.md is YAML frontmatter delimited by `---` lines, followed by a
markdown body.

## 3. Frontmatter fields

STRICT: unknown keys are validation errors. All keys required unless noted.

| Field | Type | Rules |
|---|---|---|
| `schema` | string | Must be exactly `verdict/v2`. |
| `question` | string | `Q<N>`, N = 1..57, no zero padding. |
| `slug` | string | `q<NN>_<name>`, zero-padded 2-digit; MUST match the directory name containing the VERDICT.md. |
| `date` | string | `YYYY-MM-DD`. |
| `status` | enum | One of the 4 statuses; must equal MIN over claim statuses. |
| `verification` | enum | `blind` or `primed`. |
| `packet_sha256` | string or null | 64 hex chars or null. MUST be non-null when `verification: blind`, and `q<NN>_<slug>/_packets/` must contain a packet file whose sha256 matches. |
| `predecessor` | string or null | Repo-relative path. If non-null, the path must exist on disk (points at the v2_2 question dir). |
| `method_summary` | string | One line. |
| `registry_ids` | list | Every ID must exist as a row in VARIABLES.md. Empty list allowed only for desk-review verdicts. |
| `prediction_ids` | list | `P-NNN` refs; every ref must exist in PREDICTIONS.md. MUST be non-empty when `verification: blind`. |
| `claims` | list of maps | See section 4. |
| `evidence_manifest` | list of maps | See section 5. |
| `verifications` | list of maps | See section 6. |

## 4. Claims

Each entry in `claims` is a map with keys:

| Key | Type | Rules |
|---|---|---|
| `id` | string | `C<N>`. |
| `text` | string | The claim statement. |
| `status` | enum | One of the 4 statuses. |
| `falsifier` | string | Non-empty; states what observation falsifies this claim. Empty/missing = `E_FALSIFIER`. |
| `key_results` | list of strings | Each string must appear VERBATIM somewhere in the body `## Results` section. Miss = `E_KEYNUM`. |
| `evidence` | list of paths | Must be a subset of `evidence_manifest` paths. MUST be non-empty if claim status is `PARTIALLY_VERIFIED` or `VERIFIED`. |

## 5. Evidence manifest

`evidence_manifest` is a list of maps:

    - path: <relative path inside the question dir>
      sha256: <64 hex>

- Every path must exist on disk, be relative, and stay inside the question
  dir (no absolute paths, no `..` escapes). Violation = `E_MANIFEST_MISSING`.
- The file's actual sha256 must match. Mismatch = `E_HASH`.

## 6. Verifications

`verifications` is a list of maps:

    - date: YYYY-MM-DD
      mode: blind | primed | refute
      result: <STATUS or UNREFUTED>

## 7. Floor rules (validator-enforced, `E_FLOOR`)

1. If `evidence_manifest` is empty, every claim status must be
   `<= UNSUPPORTED` (i.e. `FALSIFIED` or `UNSUPPORTED`).
2. Any claim with status `VERIFIED` or `PARTIALLY_VERIFIED` must list at
   least 1 evidence path.

## 8. Required body sections

Exact H2 headings, in this exact order:

    ## Hypothesis
    ## Claims
    ## Method
    ## Results
    ## Status
    ## Provenance

- `## Status` contains exactly one line of the form:

      **Status:** <STATUS>

  and that token must equal the frontmatter `status`. No other line anywhere
  in the body may begin with `**Status:**`. Violation = `E_BODY_MISMATCH`.
- `## Results` must contain verbatim command lines, verbatim output
  excerpts, and exit codes for every run claimed.

## 9. Validator error codes

Any hit = exit 1. Print `ERROR <CODE> <file>: <detail>` to stderr. There is
no warn-and-continue mode.

| Code | Meaning |
|---|---|
| `E_YAML` | Frontmatter missing or unparseable. |
| `E_SCHEMA` | Missing/unknown field, bad enum, bad date, slug-dir mismatch. |
| `E_STATUS_NOT_MIN` | Frontmatter status != MIN(claim statuses). |
| `E_BODY_MISMATCH` | Body Status line absent, duplicated, or != frontmatter. |
| `E_FLOOR` | Floor rules violated. |
| `E_MANIFEST_MISSING` | Manifest path absent on disk, absolute, or escapes the question dir. |
| `E_HASH` | sha256 mismatch on any manifest file. |
| `E_REGISTRY` | `registry_id` not found in VARIABLES.md. |
| `E_PREDICTION` | `prediction_id` not found in PREDICTIONS.md, or blind verdict with empty `prediction_ids`. |
| `E_KEYNUM` | A `key_results` string not found verbatim in body Results section. |
| `E_FALSIFIER` | Empty/missing falsifier on any claim. |
| `E_PREDECESSOR` | Predecessor non-null and path does not exist. |
| `E_CATALOG` | Question dir slug not present in `_meta/questions.yaml`, or duplicate question id. |
| `E_BLIND` | Verification blind without valid packet (null hash, missing packet file, or hash mismatch). |

## 10. Index generation (summary)

`python _meta/generate_index.py [--root <v2_3 dir>] [--check]`

- Validates EVERY verdict first; any error = exit 1 and INDEX.md not written.
- A question dir with no VERDICT.md renders as `OPEN` (legitimate staged
  state).
- INDEX.md is UTF-8, LF line endings, no BOM, byte-deterministic (INV-005):
  no wall-clock, no randomness, sorted iteration everywhere.
- Line 1: `<!-- GENERATED FILE. DO NOT EDIT. Regenerate: python _meta/generate_index.py -->`
- Line 2: `<!-- INPUTS_DIGEST: <sha256 over sorted (relpath, sha256) pairs of all input files> -->`
- Then: `# Living Formula v2.3 - Index`, `## Status Legend` (max 6 lines),
  `## Summary` (counts by status; blind vs primed verdict counts), then
  `## Tier 0` .. `## Tier 5` tables.
- Tier table columns: `Q | Hypothesis | Status | Ver | Df(ver) | Claims | Evidence | Dir`
  - Hypothesis comes ONLY from questions.yaml.
  - Ver = `blind` | `primed` | `-` (dash when OPEN).
  - Df(ver) = count of `verifications[]` entries with mode `blind` or `refute`.
  - Claims = summary like `2V/1F`, or `-`.
  - Evidence = manifest file count, or `0`.
- `--check`: regenerate in memory, byte-compare with existing INDEX.md,
  exit 1 on any difference.

## 11. Questions catalog

`_meta/questions.yaml` is a YAML list of entries:

    - id: Q<N>
      slug: q<NN>_<name>
      tier: 0-5
      hypothesis: <one-line STATUS-FREE falsifiable statement>
      predecessor: <repo-relative v2_2 path> or null

## 12. Fixture format

`_meta/fixtures/<name>/` contains:

- `tree/` - a miniature v2_3 root with `_meta/questions.yaml`,
  `VARIABLES.md`, `PREDICTIONS.md`, and optional `qNN` dirs.
- `expect.json` with either:
  - `{"exit_code": 0, "expected_index": "expected_INDEX.md"}` (the expected
    file sits next to expect.json), or
  - `{"exit_code": 1, "error_code": "E_..."}`

`_meta/test_generator.py` iterates ALL fixture dirs sorted, runs
`generate_index.py --root <fixture>/tree`, asserts exit code, asserts the
error code appears in stderr when expected, byte-compares generated INDEX.md
against expected_INDEX.md when exit 0. Prints PASS/FAIL per fixture, exits
non-zero if any FAIL.

## 13. Packet linter

`python _meta/lint_packet.py <packet-file>`

Greps for forbidden substrings (case-insensitive): `/v4/`, `INDEX.md`,
`AGENT_HANDOFF`, `VERDICT`, `DISCOVERY_REPORT`, `RECONCILIATION`,
`VERIFIED`, `FALSIFIED`, `CONFIRMED`, `UNSUPPORTED`, `Tier `.
Any hit: print the offending line, exit 1. Clean: print the sha256 of the
file, exit 0.

## 14. Clarifications

1. Verification modes (2026-06-12, owner ruling): the STANDARD protocol is
   the INFORMED run - `verification: primed`, `packet_sha256: null`. The
   verifier receives a context brief (see
   `_meta/prompts/context_brief_template.md`) that may cite v2_3-VERIFIED
   results as premises, and may read the entire v2_3 lab. `verification:
   blind` is reserved for the OPTIONAL Stage 9 comparison run (RUNBOOK),
   which never determines a question's status; for those runs the packet
   under `qNN_slug/_packets/` is linted by `lint_packet.py` and its sha256
   goes into `packet_sha256`. The lint forbidden-token list applies to
   blind packets only - context briefs are gated by the Stage 3 checks
   instead (no FILL_ME leftovers, no v2_2 references).
2. Read access (2026-06-12): an INFORMED verifier may read the entire v2_3
   lab. FORBIDDEN to every agent in every mode: anything under `v2_2/` or
   any other FORMULA version (including the predecessor directory - its
   path is copied into frontmatter verbatim without being opened), and any
   pre-existing implementation of the measurement. A Stage 9 BLIND agent
   additionally reads only: its packet, its question directory,
   `_meta/VERDICT_SCHEMA.md`, `VARIABLES.md`, `PREDICTIONS.md`.
3. PRIME RULE - fresh code only (owner ruling, 2026-06-12): no file from
   v2_2 or any other FORMULA version is ever copied into v2_3; no v2_3 code
   imports, executes, or reads v2_2 code. The blind verifier WRITES ITS OWN
   implementation from the packet spec - it never runs pre-existing
   experiment code, and a question directory must contain no executable
   experiment code before the verifier starts. Evidence produced by copied
   or pre-existing code is invalid regardless of result; a verdict citing
   such evidence is void. Questions are answered SEPARATELY: no experiment
   code is shared between questions. Verdicts must state under Provenance
   that the verifier authored the implementation itself.
