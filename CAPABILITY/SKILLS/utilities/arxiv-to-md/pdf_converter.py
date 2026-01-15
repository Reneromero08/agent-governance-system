#!/usr/bin/env python3
"""
Enhanced PDF to Markdown Converter for Feral Resident.

Supports multiple backends:
1. marker-pdf (best for academic papers with equations)
2. docling (IBM, best for structured documents)
3. pymupdf4llm (lightweight, offline)
4. Legacy PyMuPDF font-size heuristics (fallback)

Also supports direct arXiv paper conversion (much easier than PDF!):
- HTML method: Fetches ar5iv.org HTML version, converts to markdown
- LaTeX method: Downloads source .tex, uses pandoc (best heading structure)

Usage:
    # PDF conversion
    from pdf_converter import convert_pdf
    convert_pdf("input.pdf", "output.md", backend="marker")

    # arXiv conversion (recommended for arXiv papers)
    from pdf_converter import convert_arxiv
    convert_arxiv("2401.12345", "paper.md")  # Just the ID
    convert_arxiv("https://arxiv.org/abs/2401.12345", "paper.md")  # Full URL

CLI:
    python pdf_converter.py 2401.12345 paper.md        # arXiv (auto method)
    python pdf_converter.py 2401.12345 paper.md html   # arXiv HTML only
    python pdf_converter.py paper.pdf output.md        # PDF conversion
"""

import re
import sys
import subprocess
import tempfile
import tarfile
from pathlib import Path
from typing import Optional, Literal

import requests
from markdownify import markdownify as md

# Backend availability flags
_HAS_MARKER = False
_HAS_DOCLING = False
_HAS_PYMUPDF4LLM = False
_HAS_PYMUPDF = False

try:
    import marker
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    _HAS_MARKER = True
except ImportError:
    pass

try:
    from docling.document_converter import DocumentConverter
    _HAS_DOCLING = True
except ImportError:
    pass

try:
    import pymupdf4llm
    _HAS_PYMUPDF4LLM = True
except ImportError:
    pass

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    pass


Backend = Literal["marker", "docling", "pymupdf4llm", "pymupdf", "auto"]


def get_available_backends() -> list[str]:
    """Return list of available backends."""
    available = []
    if _HAS_MARKER:
        available.append("marker")
    if _HAS_DOCLING:
        available.append("docling")
    if _HAS_PYMUPDF4LLM:
        available.append("pymupdf4llm")
    if _HAS_PYMUPDF:
        available.append("pymupdf")
    return available


def convert_with_marker(pdf_path: Path, use_llm: bool = False) -> str:
    """
    Convert PDF using marker-pdf.

    Args:
        pdf_path: Path to PDF file
        use_llm: Use LLM for highest accuracy (requires API key)

    Returns:
        Markdown string
    """
    if not _HAS_MARKER:
        raise ImportError("marker-pdf not installed. Run: pip install marker-pdf")

    # Create converter with models
    converter = PdfConverter(artifact_dict=create_model_dict())

    # Convert
    result = converter(str(pdf_path))

    # result.markdown contains the markdown text
    return result.markdown


def convert_with_docling(pdf_path: Path) -> str:
    """
    Convert PDF using IBM Docling.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown string
    """
    if not _HAS_DOCLING:
        raise ImportError("docling not installed. Run: pip install docling")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    # Export to markdown
    return result.document.export_to_markdown()


def convert_with_pymupdf4llm(pdf_path: Path) -> str:
    """
    Convert PDF using PyMuPDF4LLM.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown string
    """
    if not _HAS_PYMUPDF4LLM:
        raise ImportError("pymupdf4llm not installed. Run: pip install pymupdf4llm")

    return pymupdf4llm.to_markdown(str(pdf_path))


def convert_with_pymupdf_legacy(pdf_path: Path) -> str:
    """
    Convert PDF using legacy PyMuPDF font-size heuristics.

    This is the original method from paper_pipeline.py.
    Less accurate but works offline with minimal deps.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Markdown string
    """
    if not _HAS_PYMUPDF:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))

    # First pass: collect all font sizes
    all_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = round(span["size"], 1)
                        if span["text"].strip():
                            all_sizes.append(size)

    if not all_sizes:
        doc.close()
        return ""

    # Determine thresholds
    unique_sizes = sorted(set(all_sizes), reverse=True)
    body_size = max(set(all_sizes), key=all_sizes.count)

    h1_min = unique_sizes[0] if len(unique_sizes) > 0 else 999
    h2_min = unique_sizes[1] if len(unique_sizes) > 1 else 999
    h3_min = unique_sizes[2] if len(unique_sizes) > 2 else 999

    # Second pass: extract with structure
    lines = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = []
            block_size = body_size

            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    text = span["text"]
                    size = round(span["size"], 1)
                    if size > block_size:
                        block_size = size
                    line_text += text
                block_text.append(line_text)

            full_text = " ".join(block_text).strip()
            if not full_text:
                continue

            full_text = re.sub(r'\s+', ' ', full_text)

            if block_size >= h1_min - 0.5 and len(full_text) < 200:
                lines.append(f"\n# {full_text}\n")
            elif block_size >= h2_min - 0.5 and block_size < h1_min - 0.5 and len(full_text) < 150:
                lines.append(f"\n## {full_text}\n")
            elif block_size >= h3_min - 0.5 and block_size < h2_min - 0.5 and len(full_text) < 100:
                lines.append(f"\n### {full_text}\n")
            else:
                lines.append(full_text + "\n")

        if page_num < len(doc) - 1:
            lines.append("\n---\n")

    doc.close()
    return "\n".join(lines)


def convert_pdf(
    pdf_path: str | Path,
    output_path: Optional[str | Path] = None,
    backend: Backend = "auto",
    use_llm: bool = False
) -> str:
    """
    Convert PDF to Markdown using best available backend.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to write markdown file
        backend: Which backend to use:
            - "marker": Best for academic papers (equations, structure)
            - "docling": Best for enterprise docs (tables, forms)
            - "pymupdf4llm": Lightweight, offline
            - "pymupdf": Legacy font-size heuristics
            - "auto": Use best available
        use_llm: For marker, use LLM for highest accuracy

    Returns:
        Markdown string
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Auto-select backend
    if backend == "auto":
        available = get_available_backends()
        if not available:
            raise ImportError("No PDF conversion backends available!")

        # Preference order
        for preferred in ["marker", "docling", "pymupdf4llm", "pymupdf"]:
            if preferred in available:
                backend = preferred
                break

    # Convert
    print(f"[PDF] Converting with {backend}...")

    if backend == "marker":
        markdown = convert_with_marker(pdf_path, use_llm=use_llm)
    elif backend == "docling":
        markdown = convert_with_docling(pdf_path)
    elif backend == "pymupdf4llm":
        markdown = convert_with_pymupdf4llm(pdf_path)
    elif backend == "pymupdf":
        markdown = convert_with_pymupdf_legacy(pdf_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Write output if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding='utf-8')
        print(f"[PDF] Written to {output_path}")

    return markdown


def parse_arxiv_id(arxiv_input: str) -> str:
    """Extract arXiv ID from URL or raw ID."""
    # Handle full URLs
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'ar5iv\.labs\.arxiv\.org/html/(\d+\.\d+)',
        r'^(\d+\.\d+)$',  # Raw ID like 2401.12345
    ]
    for pattern in patterns:
        match = re.search(pattern, arxiv_input)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse arXiv ID from: {arxiv_input}")


def convert_arxiv_html(arxiv_id: str) -> str:
    """
    Convert arXiv paper to markdown via ar5iv HTML.

    Fast and simple, works for most papers.
    """
    from bs4 import BeautifulSoup

    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    print(f"[arXiv] Fetching {url}...")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Parse and clean HTML before converting
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Remove script, style, nav, footer elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'noscript']):
        tag.decompose()

    # Get just the article content if available
    article = soup.find('article') or soup.find('main') or soup.find('body')
    html_content = str(article) if article else str(soup)

    # Convert HTML to markdown with ATX-style headings (# ## ###)
    markdown = md(
        html_content,
        heading_style="ATX",
        strip=['script', 'style', 'nav', 'footer', 'noscript'],
    )

    # Clean up excessive whitespace
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown


def convert_arxiv_latex(arxiv_id: str) -> str:
    """
    Convert arXiv paper to markdown via LaTeX source + pandoc.

    Best quality - preserves original heading structure perfectly.
    Requires pandoc to be installed.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"[arXiv] Downloading LaTeX source from {url}...")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive_path = tmpdir / "source.tar.gz"

        # Write the downloaded content
        archive_path.write_bytes(resp.content)

        # Extract (arXiv sources are usually tar.gz or plain tex)
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(tmpdir)
        except tarfile.ReadError:
            # Might be a single .tex file, not an archive
            tex_path = tmpdir / "main.tex"
            tex_path.write_bytes(resp.content)

        # Find the main .tex file
        tex_files = list(tmpdir.glob("*.tex"))
        if not tex_files:
            raise FileNotFoundError("No .tex files found in arXiv source")

        # Prefer main.tex, paper.tex, or the largest .tex file
        main_tex = None
        for name in ["main.tex", "paper.tex", "manuscript.tex"]:
            candidate = tmpdir / name
            if candidate.exists():
                main_tex = candidate
                break

        if not main_tex:
            # Use largest tex file (usually the main one)
            main_tex = max(tex_files, key=lambda f: f.stat().st_size)

        print(f"[arXiv] Converting {main_tex.name} with pandoc...")

        # Run pandoc (check common install locations)
        pandoc_cmd = "pandoc"
        for pandoc_path in [
            Path.home() / "AppData/Local/Pandoc/pandoc.exe",
            Path("C:/Program Files/Pandoc/pandoc.exe"),
        ]:
            if pandoc_path.exists():
                pandoc_cmd = str(pandoc_path)
                break

        result = subprocess.run(
            [pandoc_cmd, str(main_tex), "-o", "-", "--wrap=none", "-t", "markdown"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            encoding='utf-8',
            errors='replace',
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.returncode != 0 or not stdout.strip():
            # Check if it's just warnings vs actual error
            if "Error at" in stderr or not stdout.strip():
                raise RuntimeError(f"Pandoc failed: {stderr}")

        return stdout


def convert_arxiv(
    arxiv_input: str,
    output_path: Optional[str | Path] = None,
    method: Literal["html", "latex", "auto"] = "auto"
) -> str:
    """
    Convert arXiv paper to markdown.

    Args:
        arxiv_input: arXiv ID (2401.12345) or URL
        output_path: Optional path to write markdown file
        method:
            - "html": Use ar5iv HTML (fast, works offline after fetch)
            - "latex": Use LaTeX source + pandoc (best quality)
            - "auto": Try latex first, fall back to html

    Returns:
        Markdown string with proper # ## ### heading structure
    """
    arxiv_id = parse_arxiv_id(arxiv_input)
    print(f"[arXiv] Processing paper {arxiv_id}...")

    if method == "auto":
        # Try LaTeX first (better quality), fall back to HTML
        try:
            markdown = convert_arxiv_latex(arxiv_id)
        except Exception as e:
            print(f"[arXiv] LaTeX method failed ({e}), trying HTML...")
            markdown = convert_arxiv_html(arxiv_id)
    elif method == "latex":
        markdown = convert_arxiv_latex(arxiv_id)
    else:
        markdown = convert_arxiv_html(arxiv_id)

    # Write output if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding='utf-8')
        print(f"[arXiv] Written to {output_path}")

    return markdown


def install_backend(backend: str) -> None:
    """Print installation instructions for a backend."""
    instructions = {
        "marker": "pip install marker-pdf torch",
        "docling": "pip install docling",
        "pymupdf4llm": "pip install pymupdf4llm",
        "pymupdf": "pip install pymupdf"
    }
    if backend in instructions:
        print(f"To install {backend}: {instructions[backend]}")
    else:
        print(f"Unknown backend: {backend}")


if __name__ == "__main__":
    print("Available backends:", get_available_backends())

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

        # Check if it's an arXiv ID or URL
        if "arxiv" in input_file.lower() or re.match(r'^\d+\.\d+$', input_file):
            arxiv_id = parse_arxiv_id(input_file)
            md_file = sys.argv[2] if len(sys.argv) > 2 else f"{arxiv_id}.md"
            method = sys.argv[3] if len(sys.argv) > 3 else "auto"
            convert_arxiv(input_file, md_file, method=method)
        else:
            # PDF conversion
            md_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.pdf', '.md')
            backend = sys.argv[3] if len(sys.argv) > 3 else "auto"
            convert_pdf(input_file, md_file, backend=backend)
    else:
        print("\nUsage:")
        print("  python pdf_converter.py input.pdf [output.md] [backend]")
        print("  python pdf_converter.py 2401.12345 [output.md] [html|latex|auto]")
        print("  python pdf_converter.py https://arxiv.org/abs/2401.12345 [output.md]")
        print("\nPDF backends: marker, docling, pymupdf4llm, pymupdf, auto")
        print("arXiv methods: html (fast), latex (best quality), auto")

        for b in ["marker", "docling", "pymupdf4llm", "pymupdf"]:
            install_backend(b)
        print("\nFor arXiv HTML method: pip install requests markdownify")
        print("For arXiv LaTeX method: install pandoc (https://pandoc.org)")
