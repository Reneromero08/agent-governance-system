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


def extract_standard(pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        import pdfplumber
    except ImportError:
        print("Error: pdfplumber is required for standard mode. Install with: pip install pdfplumber>=0.9.0")
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


def extract_high_fidelity(pdf_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        import fitz
    except ImportError:
        print("Error: pymupdf is required for high-fidelity mode. Install with: pip install pymupdf>=1.27.2")
        return "", []

    pages = []
    full_parts = []

    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        page_num = i + 1
        parts = []

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:
                lines = block["lines"]
                if not lines:
                    continue
                first_span = lines[0]["spans"][0]
                font_size = first_span["size"]
                body_size = _detect_body_size(blocks)

                is_bold = bool(first_span["flags"] & 2)
                is_italic = bool(first_span["flags"] & 1)

                text_parts = []
                for line in lines:
                    for span in line["spans"]:
                        t = span["text"]
                        if not t.strip():
                            continue
                        if span["flags"] & 2 and span["flags"] & 1:
                            t = f"***{t}***"
                        elif span["flags"] & 2:
                            t = f"**{t}**"
                        elif span["flags"] & 1:
                            t = f"*{t}*"
                        text_parts.append(t)

                line_text = " ".join(text_parts).strip()
                if not line_text:
                    continue

                if body_size and font_size > body_size * 1.2:
                    level = 1 if font_size > body_size * 1.8 else 2 if font_size > body_size * 1.4 else 3
                    level = min(level, 4)
                    parts.append(f"\n{'#' * level} {line_text}\n")
                else:
                    parts.append(line_text)

                non_text = ["EDITED BY", "REVIEWED BY", "CORRESPONDENCE", "CITATION",
                            "COPYRIGHT", "OPEN ACCESS", "PUBLISHED", "RECEIVED", "ACCEPTED",
                            "Frontiers in", "doi:", "Creative Commons", "CC BY"]
                skip = any(line_text.startswith(x) for x in non_text)
                if not skip and body_size and abs(font_size - body_size) <= 0.5:
                    parts.append("\n")

            elif block["type"] == 1:
                parts.append("")

        try:
            tables = page.find_tables()
            for t in tables.tables:
                md_table = _table_to_markdown(t)
                if md_table:
                    parts.append("\n" + md_table + "\n")
        except Exception:
            pass

        text = _merge_paragraphs(parts)
        pages.append({"page_num": page_num, "text": text})
        full_parts.append(f"\n{text}\n")

    doc.close()
    return "\n".join(full_parts), pages


def _detect_body_size(blocks: List[Dict]) -> Optional[float]:
    sizes = []
    for block in blocks:
        if block["type"] == 0:
            for line in block["lines"]:
                for span in line["spans"]:
                    s = span["size"]
                    if 8 <= s <= 14:
                        sizes.append(s)
    if not sizes:
        return None
    counts = {}
    for s in sizes:
        rounded = round(s, 1)
        counts[rounded] = counts.get(rounded, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _table_to_markdown(table) -> str:
    rows = []
    for row in table.extract():
        cleaned = [str(cell).replace("\n", " ") if cell else "" for cell in row]
        rows.append(cleaned)
    if not rows:
        return ""
    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _merge_paragraphs(parts: List[str]) -> str:
    merged = []
    buf = []
    for p in parts:
        if p.startswith("\n#") or p.startswith("\n##") or p.startswith("\n###") or p.startswith("\n####"):
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append(p.strip())
        elif p.startswith("|"):
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append(p)
        elif p == "":
            if buf:
                merged.append(" ".join(buf))
                buf = []
            merged.append("")
        else:
            buf.append(p)
    if buf:
        merged.append(" ".join(buf))
    return "\n\n".join(merged)


def convert_standard_markdown(text: str, options: Dict[str, Any]) -> str:
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
    mode = options.get("mode", "standard")

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

    if mode == "high":
        text, pages = extract_high_fidelity(pdf_path)
        if not text:
            print(f"Warning: No text extracted from PDF: {pdf_path}")
        markdown = text
    else:
        text, pages = extract_standard(pdf_path)
        if not text:
            print(f"Warning: No text extracted from PDF: {pdf_path}")
        markdown = convert_standard_markdown(text, options)

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
        "mode": mode,
        "pages_extracted": len(pages),
        "output_path": str(output_path_str),
        "message": f"Successfully converted {pdf_path} to {output_path_str}"
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(f"Successfully converted {pdf_path} to {output_path_str}")
    print(f"Extracted {len(pages)} pages (mode: {mode})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
