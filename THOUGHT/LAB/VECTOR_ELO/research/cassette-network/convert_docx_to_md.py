#!/usr/bin/env python3
"""Convert DOCX to Markdown using python-docx"""

from docx import Document
import os

def convert_docx_to_markdown(docx_path, md_path):
    """Convert a DOCX file to Markdown format"""
    doc = Document(docx_path)
    
    markdown_lines = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            markdown_lines.append("")
            continue
            
        # Handle headings
        if para.style.name.startswith('Heading'):
            level = para.style.name.replace('Heading ', '')
            try:
                level_num = int(level)
                markdown_lines.append(f"{'#' * level_num} {text}")
            except ValueError:
                markdown_lines.append(text)
        # Handle list items
        elif para.style.name.startswith('List'):
            markdown_lines.append(f"- {text}")
        # Regular paragraphs
        else:
            markdown_lines.append(text)
    
    # Write to markdown file
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"Converted: {docx_path}")
    print(f"Output: {md_path}")

if __name__ == "__main__":
    docx_file = "Overview of the Universal Semantic Anchor Hypothesis.docx"
    md_file = "Overview of the Universal Semantic Anchor Hypothesis.md"
    
    convert_docx_to_markdown(docx_file, md_file)
