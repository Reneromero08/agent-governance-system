#!/usr/bin/env python3
"""
Paper Pipeline for Feral Resident Beta (B.1)

Downloads arxiv papers, converts PDFs to markdown, indexes into geometric memory.

Just run: python paper_pipeline.py
"""

import urllib.request
import fitz  # PyMuPDF - PROPER PDF parsing with structure
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import json
import time
import re

FERAL_DIR = Path(__file__).parent
RESEARCH_DIR = FERAL_DIR / "research"
PAPERS_DIR = RESEARCH_DIR / "papers"
RAW_DIR = PAPERS_DIR / "raw"
MD_DIR = PAPERS_DIR / "markdown"

# === THE DEFINITIVE PAPER CORPUS (100+ PAPERS) ===

PAPERS = [
    # ========== BRAIN PROMPTS (already markdown) ==========
    {"id": "Grok42", "name": "Grok 4.2 Thinking", "category": "brain_prompts",
     "source": "research/Grok 4.2 Thinking - Brain Prompt 1.md", "type": "markdown"},
    {"id": "GPT52", "name": "GPT 5.2 Thinking", "category": "brain_prompts",
     "source": "research/GPT 5.2 Thinking - Brain Prompt.md", "type": "markdown"},
    {"id": "Claude45", "name": "Claude 4.5 Sonnet Brain Prompt", "category": "brain_prompts",
     "source": "research/Claude 4.5 Sonnet Brain Prompt 1.md", "type": "markdown"},

    # ========== HDC/VSA (Hyperdimensional Computing) ==========
    {"id": "2111.06077", "name": "HDC-Survey-I", "category": "hdc_vsa", "type": "arxiv"},
    {"id": "2112.15424", "name": "HDC-Survey-II", "category": "hdc_vsa", "type": "arxiv"},
    {"id": "2301.10352", "name": "VSA-Capacity", "category": "hdc_vsa", "type": "arxiv"},
    {"id": "2106.05268", "name": "VSA-Hardware", "category": "hdc_vsa", "type": "arxiv"},
    {"id": "2501.16863", "name": "HD-CB", "category": "hdc_vsa", "type": "arxiv"},

    # ========== Vec2Text / Embedding Inversion ==========
    {"id": "2310.06816", "name": "Vec2Text", "category": "vec2text", "type": "arxiv"},
    {"id": "2311.13647", "name": "LM-Inversion", "category": "vec2text", "type": "arxiv"},
    {"id": "2402.12784", "name": "Vec2Text-Threat", "category": "vec2text", "type": "arxiv"},
    {"id": "2401.12192", "name": "Multilingual-Invert", "category": "vec2text", "type": "arxiv"},
    {"id": "2504.00147", "name": "ZS-Invert", "category": "vec2text", "type": "arxiv"},

    # ========== Latent Reasoning ==========
    {"id": "2412.06769", "name": "Coconut", "category": "latent_reasoning", "type": "arxiv"},
    {"id": "2510.04573", "name": "LaDiR", "category": "latent_reasoning", "type": "arxiv"},
    {"id": "2510.11052", "name": "LRD", "category": "latent_reasoning", "type": "arxiv"},
    {"id": "2510.13117", "name": "MDL-Reason", "category": "latent_reasoning", "type": "arxiv"},
    {"id": "2410.17891", "name": "DiffuLLaMA", "category": "latent_reasoning", "type": "arxiv"},
    {"id": "2508.10875", "name": "DLM-Survey", "category": "latent_reasoning", "type": "arxiv"},

    # ========== Semantic Compression ==========
    {"id": "2304.12512", "name": "SemanticComp", "category": "compression", "type": "arxiv"},
    {"id": "2412.08821", "name": "LCM", "category": "compression", "type": "arxiv"},
    {"id": "2508.00220", "name": "DWT-Compress", "category": "compression", "type": "arxiv"},
    {"id": "2601.05075", "name": "SemPA", "category": "compression", "type": "arxiv"},
    {"id": "2403.15362", "name": "CoLLEGe", "category": "compression", "type": "arxiv"},
    {"id": "2312.09571", "name": "Context-Compress", "category": "compression", "type": "arxiv"},
    {"id": "2409.13385", "name": "RAG-Compress", "category": "compression", "type": "arxiv"},
    {"id": "2512.24617", "name": "DLCM", "category": "compression", "type": "arxiv"},

    # ========== Representation / Platonic ==========
    {"id": "2405.07987", "name": "Platonic", "category": "representation", "type": "arxiv"},
    {"id": "2505.12540", "name": "vec2vec", "category": "representation", "type": "arxiv"},
    {"id": "2507.01098", "name": "Platonic-Proof", "category": "representation", "type": "arxiv"},
    {"id": "2509.19453", "name": "Platonic-Universe", "category": "representation", "type": "arxiv"},
    {"id": "2510.17833", "name": "Brain-LM", "category": "representation", "type": "arxiv"},

    # ========== Sentence Embeddings / Transformers ==========
    {"id": "1908.10084", "name": "SBERT", "category": "sentence_embed", "type": "arxiv"},
    {"id": "2408.08073", "name": "Extract-Embed", "category": "sentence_embed", "type": "arxiv"},
    {"id": "2402.13130", "name": "ELECTRA-Embed", "category": "sentence_embed", "type": "arxiv"},
    {"id": "2402.14776", "name": "Matryoshka", "category": "sentence_embed", "type": "arxiv"},

    # ========== Memory / Attention ==========
    {"id": "2508.10824", "name": "Mem-Trans-Survey", "category": "memory", "type": "arxiv"},
    {"id": "2410.13166", "name": "Evolved-Memory", "category": "memory", "type": "arxiv"},
    {"id": "2207.06881", "name": "RMT", "category": "memory", "type": "arxiv"},

    # ========== Neural-Symbolic / Knowledge Graphs ==========
    {"id": "2412.10390", "name": "NS-KG-Survey", "category": "neural_symbolic", "type": "arxiv"},
    {"id": "2302.07200", "name": "NeuroSymb-KG", "category": "neural_symbolic", "type": "arxiv"},
    {"id": "2405.03524", "name": "NS-KG-Apps", "category": "neural_symbolic", "type": "arxiv"},

    # ========== Dense Retrieval ==========
    {"id": "2410.21242", "name": "ZS-Dense", "category": "dense_retrieval", "type": "arxiv"},
    {"id": "2312.06648", "name": "DenseX", "category": "dense_retrieval", "type": "arxiv"},
    {"id": "2503.05037", "name": "Dense-Collapse", "category": "dense_retrieval", "type": "arxiv"},
    {"id": "2405.05200", "name": "Dense-Essay", "category": "dense_retrieval", "type": "arxiv"},

    # ========== Sparse Retrieval ==========
    {"id": "2404.13950", "name": "SPLATE", "category": "sparse_retrieval", "type": "arxiv"},
    {"id": "2408.11119", "name": "Mistral-SPLADE", "category": "sparse_retrieval", "type": "arxiv"},
    {"id": "2107.05720", "name": "SPLADE", "category": "sparse_retrieval", "type": "arxiv"},
    {"id": "2109.10086", "name": "SPLADE-v2", "category": "sparse_retrieval", "type": "arxiv"},
    {"id": "2404.18812", "name": "Efficient-Sparse", "category": "sparse_retrieval", "type": "arxiv"},

    # ========== Multimodal / CLIP ==========
    {"id": "2411.17040", "name": "MM-Align-Survey", "category": "multimodal", "type": "arxiv"},
    {"id": "2411.05195", "name": "CLIP-Errors", "category": "multimodal", "type": "arxiv"},
    {"id": "2412.08802", "name": "jina-clip-v2", "category": "multimodal", "type": "arxiv"},
    {"id": "2406.17639", "name": "CLIP-Gap", "category": "multimodal", "type": "arxiv"},
    {"id": "2409.19425", "name": "Unimodal-MM", "category": "multimodal", "type": "arxiv"},

    # ========== Text Embedding Models ==========
    {"id": "2402.03216", "name": "BGE-M3", "category": "text_embed", "type": "arxiv"},
    {"id": "2402.05672", "name": "mE5", "category": "text_embed", "type": "arxiv"},
    {"id": "2401.00368", "name": "E5-LLM", "category": "text_embed", "type": "arxiv"},
    {"id": "2406.01607", "name": "MTEB-Review", "category": "text_embed", "type": "arxiv"},
    {"id": "2506.05176", "name": "Qwen3-Embed", "category": "text_embed", "type": "arxiv"},
    {"id": "2502.07972", "name": "MoE-Embed", "category": "text_embed", "type": "arxiv"},

    # ========== Chain of Thought ==========
    {"id": "2402.10200", "name": "CoT-NoProm", "category": "chain_of_thought", "type": "arxiv"},
    {"id": "2201.11903", "name": "CoT-Original", "category": "chain_of_thought", "type": "arxiv"},
    {"id": "2210.03493", "name": "Auto-CoT", "category": "chain_of_thought", "type": "arxiv"},

    # ========== Vector Database / ANN ==========
    {"id": "2310.11703", "name": "VecDB-Survey", "category": "vector_db", "type": "arxiv"},
    {"id": "2505.11783", "name": "d-HNSW", "category": "vector_db", "type": "arxiv"},
    {"id": "2412.01940", "name": "HNSW-Hubs", "category": "vector_db", "type": "arxiv"},
    {"id": "2403.04871", "name": "ACORN", "category": "vector_db", "type": "arxiv"},
    {"id": "2401.08281", "name": "FAISS", "category": "vector_db", "type": "arxiv"},
    {"id": "2601.01291", "name": "Curator", "category": "vector_db", "type": "arxiv"},

    # ========== RAG (Retrieval-Augmented Generation) ==========
    {"id": "2312.10997", "name": "RAG-Survey", "category": "rag", "type": "arxiv"},
    {"id": "2410.12837", "name": "RAG-Evolution", "category": "rag", "type": "arxiv"},
    {"id": "2407.13193", "name": "RAG-NLP", "category": "rag", "type": "arxiv"},
    {"id": "2501.09136", "name": "Agentic-RAG", "category": "rag", "type": "arxiv"},
    {"id": "2408.08921", "name": "GraphRAG", "category": "rag", "type": "arxiv"},
    {"id": "2409.14924", "name": "RAG-Beyond", "category": "rag", "type": "arxiv"},

    # ========== Contrastive Learning ==========
    {"id": "2002.05709", "name": "SimCLR", "category": "contrastive", "type": "arxiv"},
    {"id": "2402.10150", "name": "f-MICL", "category": "contrastive", "type": "arxiv"},
    {"id": "2503.17538", "name": "CL-Theory", "category": "contrastive", "type": "arxiv"},
    {"id": "2506.10159", "name": "VCL", "category": "contrastive", "type": "arxiv"},

    # ========== Tokenization ==========
    {"id": "2410.03258", "name": "AdaptBPE", "category": "tokenization", "type": "arxiv"},
    {"id": "2409.04599", "name": "BPE-Picky", "category": "tokenization", "type": "arxiv"},
    {"id": "2402.18376", "name": "Token-Compress", "category": "tokenization", "type": "arxiv"},
    {"id": "2406.11687", "name": "Token-Robust", "category": "tokenization", "type": "arxiv"},

    # ========== Model Compression ==========
    {"id": "1510.00149", "name": "DeepCompress", "category": "model_compress", "type": "arxiv"},
    {"id": "2101.09671", "name": "PQ-Survey", "category": "model_compress", "type": "arxiv"},
    {"id": "2307.02973", "name": "PvQ", "category": "model_compress", "type": "arxiv"},
    {"id": "2509.04244", "name": "PQ-Integrate", "category": "model_compress", "type": "arxiv"},
    {"id": "2407.04803", "name": "PQ-DRL", "category": "model_compress", "type": "arxiv"},

    # ========== Attention / Transformers ==========
    {"id": "1706.03762", "name": "Attention-Is-All", "category": "attention", "type": "arxiv"},
    {"id": "2009.06732", "name": "Longformer", "category": "attention", "type": "arxiv"},
    {"id": "2004.05150", "name": "Linformer", "category": "attention", "type": "arxiv"},
    {"id": "2006.04768", "name": "BigBird", "category": "attention", "type": "arxiv"},

    # ========== Information Retrieval ==========
    {"id": "2004.04906", "name": "DPR", "category": "retrieval", "type": "arxiv"},
    {"id": "2112.09118", "name": "GTR", "category": "retrieval", "type": "arxiv"},
    {"id": "2104.08663", "name": "Contriever", "category": "retrieval", "type": "arxiv"},

    # ========== Cross-encoder / Rerankers ==========
    {"id": "2010.11386", "name": "ColBERT", "category": "reranker", "type": "arxiv"},
    {"id": "2112.01488", "name": "ColBERTv2", "category": "reranker", "type": "arxiv"},
    {"id": "2204.02311", "name": "Promptagator", "category": "reranker", "type": "arxiv"},

    # ========== KV Cache / Efficient Inference ==========
    {"id": "2405.04532", "name": "KV-Compress", "category": "kv_cache", "type": "arxiv"},
    {"id": "2403.09636", "name": "StreamingLLM", "category": "kv_cache", "type": "arxiv"},
]

# Total: 103 papers (3 brain prompts + 100 arxiv)


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> bool:
    """Download PDF from arxiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"    Downloading {url}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def pdf_to_markdown(pdf_path: Path, output_path: Path) -> bool:
    """
    Convert PDF to Markdown using PyMuPDF with PROPER heading detection.

    Detects headings based on font size:
    - Largest font = # (title)
    - Second largest = ## (section)
    - Third largest = ### (subsection)
    - Everything else = body text
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)

        # First pass: collect all font sizes to determine heading thresholds
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
            output_path.write_text("", encoding='utf-8')
            return True

        # Determine thresholds: top 3 unique sizes are headings
        unique_sizes = sorted(set(all_sizes), reverse=True)
        body_size = max(set(all_sizes), key=all_sizes.count)  # Most common = body

        # Heading thresholds (sizes larger than body text)
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

                # Clean up common PDF artifacts
                full_text = re.sub(r'\s+', ' ', full_text)

                # Determine heading level based on font size
                if block_size >= h1_min - 0.5 and len(full_text) < 200:
                    lines.append(f"\n# {full_text}\n")
                elif block_size >= h2_min - 0.5 and block_size < h1_min - 0.5 and len(full_text) < 150:
                    lines.append(f"\n## {full_text}\n")
                elif block_size >= h3_min - 0.5 and block_size < h2_min - 0.5 and len(full_text) < 100:
                    lines.append(f"\n### {full_text}\n")
                else:
                    lines.append(full_text + "\n")

            # Page break
            if page_num < len(doc) - 1:
                lines.append("\n---\n")

        content = "\n".join(lines)
        output_path.write_text(content, encoding='utf-8')
        doc.close()
        return True

    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def process_paper(paper: Dict) -> Dict:
    """Process a single paper."""
    result = {"paper": paper, "status": "unknown", "markdown_path": None}

    if paper["type"] == "markdown":
        # Already markdown - use directly
        source_path = FERAL_DIR / paper["source"]
        if source_path.exists():
            result["status"] = "ready"
            result["markdown_path"] = str(source_path)
            print(f"  [OK] Already markdown")
        else:
            result["status"] = "missing"
            print(f"  [SKIP] Source not found: {source_path}")

    elif paper["type"] == "arxiv":
        arxiv_id = paper["id"]
        category = paper["category"]

        pdf_path = RAW_DIR / category / f"{arxiv_id}.pdf"
        md_path = MD_DIR / f"{arxiv_id}.md"

        # Check if already converted
        if md_path.exists():
            result["status"] = "ready"
            result["markdown_path"] = str(md_path)
            print(f"  [OK] Already converted")
            return result

        # Check if PDF exists
        if not pdf_path.exists():
            print(f"  [DOWNLOAD] Fetching from arxiv...")
            if not download_arxiv_pdf(arxiv_id, pdf_path):
                result["status"] = "download_failed"
                return result
            time.sleep(1)  # Be nice to arxiv

        # Convert PDF to markdown
        print(f"  [CONVERT] PDF -> Markdown...")
        if pdf_to_markdown(pdf_path, md_path):
            result["status"] = "converted"
            result["markdown_path"] = str(md_path)
        else:
            result["status"] = "convert_failed"

    return result


def run_pipeline(limit: int = None):
    """Run the full paper pipeline."""
    print("=" * 60)
    print("B.1 PAPER PIPELINE - FERAL RESIDENT BETA")
    print("=" * 60)
    print(f"Total papers in corpus: {len(PAPERS)}")
    print()

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    papers_to_process = PAPERS[:limit] if limit else PAPERS
    results = []

    for i, paper in enumerate(papers_to_process, 1):
        print(f"[{i}/{len(papers_to_process)}] {paper['id']} - {paper['name']}")
        result = process_paper(paper)
        results.append(result)
        print()

    # Summary
    ready = [r for r in results if r["status"] in ("ready", "converted")]
    failed = [r for r in results if r["status"] in ("download_failed", "convert_failed", "missing")]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ready for indexing: {len(ready)}/{len(results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed papers:")
        for f in failed:
            print(f"  - {f['paper']['id']}: {f['status']}")

    # Write results
    results_path = PAPERS_DIR / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "total": len(results),
            "ready": len(ready),
            "failed": len(failed),
            "papers": [{"id": r["paper"]["id"], "status": r["status"], "md": r["markdown_path"]} for r in results]
        }, f, indent=2)
    print(f"\nResults written to: {results_path}")

    return results


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_pipeline(limit)
