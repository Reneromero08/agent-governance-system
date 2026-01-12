#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        import pdfplumber
    except ImportError:
        print("Error: pdfplumber is required. Install with: pip install pdfplumber>=0.9.0")
        return "", []
    
    pages = []
    full_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                full_text.append(text)
            pages.append({
                "page_num": i,
                "text": text or ""
            })
    
    return "\n\n".join(full_text), pages


def convert_to_markdown(text: str, options: Dict[str, Any]) -> str:
    page_break = options.get("page_breaks", "---")
    preserve_formatting = options.get("preserve_formatting", True)
    
    if preserve_formatting:
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.isupper() and len(line) < 60:
                lines.append(f"\n# {line}\n")
            elif line[0].isdigit() and ". " in line:
                lines.append(f"{line}")
            else:
                lines.append(f"{line}\n")
        return "\n".join(lines)
    
    return text


def main(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1
    
    pdf_path_str = payload.get("pdf_path")
    output_path_str = payload.get("output_path")
    options = payload.get("options", {})
    
    if not pdf_path_str:
        print("Error: 'pdf_path' is required in input JSON")
        return 1
    
    if not output_path_str:
        print("Error: 'output_path' is required in input JSON")
        return 1
    
    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return 1
    
    text, pages = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"Warning: No text extracted from PDF: {pdf_path}")
    
    markdown = convert_to_markdown(text, options)
    
    if GuardedWriter is None:
        print("Error: GuardedWriter not available")
        return 1
    
    writer = writer or GuardedWriter(project_root=PROJECT_ROOT)
    
    try:
        markdown_output_path = Path(output_path_str)
        rel_output_path = str(markdown_output_path.resolve().relative_to(PROJECT_ROOT))
        writer.mkdir_auto(str(Path(rel_output_path).parent))
        writer.write_auto(rel_output_path, markdown)
    except ValueError:
        print(f"Error: Output path {output_path_str} is outside project root")
        return 1
    
    result = {
        "status": "success",
        "pages_extracted": len(pages),
        "output_path": str(output_path_str),
        "message": f"Successfully converted {pdf_path} to {output_path_str}"
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    
    print(f"Successfully converted {pdf_path} to {output_path_str}")
    print(f"Extracted {len(pages)} pages")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
