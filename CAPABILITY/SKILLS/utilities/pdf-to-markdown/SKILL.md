---
name: pdf-to-markdown
description: "Use when converting PDF documents to Markdown format for documentation or content processing."
---

<!-- CONTENT_HASH: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2 -->

**required_canon_version:** >=3.0.0


# Skill: pdf-to-markdown

**Version:** 0.1.0

**Status:** Draft



## Trigger

Use when converting PDF documents to Markdown format, typically for documentation purposes or to make PDF content more accessible and editable.

## Inputs

- `input.json` with the following structure:
  ```json
  {
    "pdf_path": "path/to/document.pdf",
    "output_path": "path/to/output.md",
    "options": {
      "extract_images": false,
      "preserve_formatting": true,
      "page_breaks": "---"
    }
  }
  ```

### Fields:
- `pdf_path` (required, string): Absolute or relative path to input PDF file
- `output_path` (required, string): Path where Markdown file will be written
- `options.extract_images` (optional, boolean): Whether to extract embedded images (default: false)
- `options.preserve_formatting` (optional, boolean): Attempt to preserve text formatting (default: true)
- `options.page_breaks` (optional, string): String to insert between pages (default: "---")

## Outputs

- Creates a Markdown file at the specified `output_path` containing:
  - Extracted text from the PDF
  - Headers converted from document structure
  - Tables converted to Markdown tables
  - Optional page break markers between pages
  - Preserved whitespace and basic formatting

### Output Format:
```markdown
# Document Title

Section header

Paragraph text with **bold** and *italic* formatting.

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |

---

Page 2 content continues...
```

## Constraints

- Input PDF must be readable and not password-protected
- Output path must be within project root (enforced by GuardedWriter)
- Cannot write outside allowed locations (BUILD/, CONTRACTS/_runs/, etc.)
- Deterministic output: same input PDF always produces same Markdown
- Must use GuardedWriter for all file writes (write firewall enforcement)
- Images are extracted only when explicitly requested

## Dependencies

- `pdfplumber>=0.9.0` - PDF text and structure extraction
- Standard library only (no additional dependencies for basic operation)

## Fixtures

- `fixtures/basic/` - Simple PDF conversion test
- `fixtures/multi-page/` - Multi-page document with page breaks
- `fixtures/tables/` - PDF containing tables for table extraction

## Error Handling

- Returns exit code 1 on errors with descriptive message
- Handles common PDF errors:
  - File not found
  - Invalid PDF format
  - Password-protected PDF (not supported)
  - Encoding issues in text extraction

**required_canon_version:** >=3.0.0

