# PDF to Markdown Skill

Converts PDF documents to Markdown format for easier editing and documentation processing.

## Installation

Install the required dependency:
```bash
pip install pdfplumber>=0.9.0
```

## Usage

### Direct execution
```bash
python CAPABILITY/SKILLS/utilities/pdf-to-markdown/run.py input.json output.json
```

### Input format (input.json)
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

### Output
- Creates Markdown file at specified `output_path`
- Writes status JSON to `output.json` with conversion details

## Features

- Text extraction from PDFs
- Header detection and conversion
- Basic formatting preservation
- Page break markers
- Deterministic output

## Limitations

- No image extraction (unless explicitly enabled)
- Tables may not be perfectly formatted
- Password-protected PDFs not supported
- Complex layouts may not preserve exact formatting
