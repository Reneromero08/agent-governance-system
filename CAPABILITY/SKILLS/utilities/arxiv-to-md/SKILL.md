# Skill: arxiv-to-md

**Version:** 1.0.0

**Status:** Active

**required_canon_version:** ">=1.0.0"

## Trigger

When the user asks to download, convert, or fetch an arXiv paper as markdown. Examples:
- "Download arxiv 1706.03762"
- "Get this paper as markdown: https://arxiv.org/abs/2401.12345"
- "Convert arxiv paper 1510.00149 to md"

## Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| arxiv_id | string | Yes | - | arXiv paper ID (e.g., `1706.03762`) or full URL |
| output_path | string | No | `THOUGHT/LAB/FERAL_RESIDENT/research/papers/markdown/{id}.md` | Where to save the markdown file |
| method | string | No | `auto` | Conversion method: `html`, `latex`, or `auto` |

## Outputs

- Markdown file with proper `#`, `##`, `###` heading structure
- Prints confirmation with output path

## Methods

| Method | Description | Requirements |
|--------|-------------|--------------|
| `latex` | Downloads LaTeX source, converts via pandoc. Best heading structure. | pandoc installed |
| `html` | Fetches ar5iv.org HTML, converts to markdown. Fast fallback. | None (requests, markdownify, bs4) |
| `auto` | Tries latex first, falls back to html on failure. | Best of both |

## Execution

```bash
# Run via venv python
.venv/Scripts/python.exe CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py <arxiv_id> <output_path> [method]
```

### Examples

```bash
# Auto method (recommended)
.venv/Scripts/python.exe CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py 1706.03762 output.md

# Force HTML (no pandoc needed)
.venv/Scripts/python.exe CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py 1706.03762 output.md html

# Force LaTeX (best quality)
.venv/Scripts/python.exe CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py 1706.03762 output.md latex

# From URL
.venv/Scripts/python.exe CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py https://arxiv.org/abs/1706.03762 output.md
```

## Constraints

- Network access required to fetch from arxiv.org / ar5iv.labs.arxiv.org
- LaTeX method requires pandoc (`C:\Users\<user>\AppData\Local\Pandoc\pandoc.exe`)
- Some papers have non-standard LaTeX that pandoc can't parse (auto falls back to HTML)
- Output is UTF-8 encoded markdown

## Dependencies

**Python packages** (in `.venv`):
- `requests`
- `markdownify`
- `beautifulsoup4`

**System** (for latex method):
- `pandoc` - installed via `winget install JohnMacFarlane.Pandoc`

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `Could not parse arXiv ID` | Invalid ID format | Use format `YYMM.NNNNN` or full arxiv.org URL |
| `Pandoc failed` | Non-standard LaTeX | Use `html` method instead |
| `404 Not Found` | Paper doesn't exist on ar5iv | Try `latex` method or verify paper ID |

## Implementation

Script: `CAPABILITY/SKILLS/utilities/arxiv-to-md/pdf_converter.py`

Key functions:
- `convert_arxiv(arxiv_input, output_path, method)` - Main entry point
- `convert_arxiv_latex(arxiv_id)` - LaTeX + pandoc method
- `convert_arxiv_html(arxiv_id)` - ar5iv HTML method
- `parse_arxiv_id(arxiv_input)` - Extract ID from URL or raw input
