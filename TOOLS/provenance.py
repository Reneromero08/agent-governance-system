#!/usr/bin/env python3
"""
Provenance Header Utility

Generates provenance headers for AGS-generated files to enable audit trails.

Every generated file should include:
- Generator: Which script created it
- Version: Canon version at generation time
- Timestamp: When it was generated
- Inputs: Hash of input files (for reproducibility)
- Checksum: Hash of the output content

Usage:
    from TOOLS.provenance import generate_header, add_header_to_file
    
    header = generate_header(
        generator="TOOLS/codebook_build.py",
        inputs=["CANON/CONTRACT.md", "CANON/INVARIANTS.md"]
    )
    
    add_header_to_file(output_path, content, header)
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_canon_version() -> str:
    """Read the current canon version from VERSIONING.md."""
    versioning_path = PROJECT_ROOT / "CANON" / "VERSIONING.md"
    if not versioning_path.exists():
        return "unknown"
    
    content = versioning_path.read_text(encoding="utf-8", errors="ignore")
    
    # Look for version pattern like "Current version: 1.1.1" or "**Version:** 1.1.1"
    match = re.search(r'(?:Current version|Version)[:\s*]+(\d+\.\d+\.\d+)', content, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return "unknown"


def hash_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file or directory."""
    if not filepath.exists():
        return "missing"
    
    hasher = hashlib.sha256()
    
    if filepath.is_file():
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    elif filepath.is_dir():
        # Hash directory structure and file contents recursively
        for child in sorted(filepath.rglob("*")):
            if child.is_file():
                # Add path relative to parent to hash
                hasher.update(str(child.relative_to(filepath)).encode())
                # Add file content to hash
                with open(child, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
    
    return hasher.hexdigest()[:12]


def hash_content(content: Union[str, bytes]) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:12]


def generate_header(
    generator: str,
    inputs: Optional[List[str]] = None,
    output_content: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Generate a provenance header dict.
    
    Args:
        generator: Path to the generator script (relative to PROJECT_ROOT)
        inputs: List of input file paths (relative to PROJECT_ROOT)
        output_content: The generated content (for checksum)
        extra: Additional metadata to include
    
    Returns:
        Dict with provenance fields
    """
    header = {
        "provenance": {
            "generator": generator,
            "canon_version": get_canon_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    
    # Add input hashes
    if inputs:
        input_hashes = {}
        for input_path in inputs:
            abs_path = PROJECT_ROOT / input_path
            input_hashes[input_path] = hash_file(abs_path)
        header["provenance"]["inputs"] = input_hashes
    
    # Add output checksum
    if output_content:
        header["provenance"]["checksum"] = hash_content(output_content)
    
    # Add extra metadata
    if extra:
        header["provenance"].update(extra)
    
    return header


def generate_manifest(
    generator: str,
    target_files: Dict[str, Path],
    inputs: Optional[List[str]] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Generate a standalone provenance manifest (PROVENANCE.json).
    
    Args:
        generator: Path to the generator script
        target_files: Dict of {rel_path: abs_path} to include in the manifest
        inputs: List of input file paths
        extra: Additional metadata
    """
    manifest = generate_header(generator, inputs, extra=extra)
    
    target_hashes = {}
    for rel_path, abs_path in target_files.items():
        target_hashes[rel_path] = hash_file(abs_path)
    
    manifest["provenance"]["target_hashes"] = target_hashes
    
    # Checksum of the manifest itself (excluding the checksum field)
    # We'll compute this at the end
    return manifest


def format_header_markdown(header: Dict) -> str:
    """Format provenance header as markdown comment block."""
    prov = header.get("provenance", {})
    
    lines = [
        "<!--",
        "PROVENANCE",
        f"  generator: {prov.get('generator', 'unknown')}",
        f"  canon_version: {prov.get('canon_version', 'unknown')}",
        f"  generated_at: {prov.get('generated_at', 'unknown')}",
    ]
    
    if "inputs" in prov:
        lines.append("  inputs:")
        for path, hash_val in prov["inputs"].items():
            lines.append(f"    {path}: {hash_val}")
    
    if "checksum" in prov:
        lines.append(f"  checksum: {prov['checksum']}")
    
    lines.append("-->")
    return "\n".join(lines)


def format_header_json(header: Dict) -> str:
    """Format provenance header as JSON comment block."""
    return json.dumps(header, indent=2)


def format_header_python(header: Dict) -> str:
    """Format provenance header as Python docstring block."""
    prov = header.get("provenance", {})
    
    lines = [
        '"""',
        "PROVENANCE",
        f"generator: {prov.get('generator', 'unknown')}",
        f"canon_version: {prov.get('canon_version', 'unknown')}",
        f"generated_at: {prov.get('generated_at', 'unknown')}",
    ]
    
    if "inputs" in prov:
        lines.append("inputs:")
        for path, hash_val in prov["inputs"].items():
            lines.append(f"  {path}: {hash_val}")
    
    if "checksum" in prov:
        lines.append(f"checksum: {prov['checksum']}")
    
    lines.append('"""')
    lines.append("")
    
    return "\n".join(lines)


def add_header_to_content(content: str, header: Dict, file_type: str = "md") -> str:
    """
    Add provenance header to content.
    
    Args:
        content: The file content
        header: Provenance header dict
        file_type: File type (md, json, py)
    
    Returns:
        Content with header prepended
    """
    # Update checksum with actual content
    header["provenance"]["checksum"] = hash_content(content)
    
    if file_type == "md":
        header_str = format_header_markdown(header) + "\n\n"
    elif file_type == "json":
        # For JSON, we embed in the structure
        return json.dumps({**header, **json.loads(content)}, indent=2)
    elif file_type == "py":
        header_str = format_header_python(header) + "\n"
    else:
        header_str = format_header_markdown(header) + "\n\n"
    
    return header_str + content


def verify_provenance(filepath: Path) -> Dict:
    """
    Verify provenance of a generated file.
    
    Returns dict with verification results.
    """
    if not filepath.exists():
        return {"valid": False, "error": "File not found"}
    
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    
    if filepath.name == "PROVENANCE.json":
        try:
            data = json.loads(content)
            prov = data.get("provenance", {})
            stored_checksum = prov.get("checksum", "")
            
            # To verify, we remove checksum and re-hash
            data_copy = json.loads(content)
            if "checksum" in data_copy.get("provenance", {}):
                del data_copy["provenance"]["checksum"]
            
            # Re-serialize deterministically for hash
            canonical_json = json.dumps(data_copy, sort_keys=True)
            current_checksum = hash_content(canonical_json)
            
            # Also verify all target_hashes
            targets_valid = True
            mismatched_targets = []
            if "target_hashes" in prov:
                for rel_path, stored_hash in prov["target_hashes"].items():
                    target_path = filepath.parent.parent / rel_path
                    if not target_path.exists():
                        targets_valid = False
                        mismatched_targets.append(f"{rel_path} (missing)")
                        continue
                    
                    actual_hash = hash_file(target_path)
                    if actual_hash != stored_hash:
                        targets_valid = False
                        mismatched_targets.append(f"{rel_path} (mismatch: {actual_hash} != {stored_hash})")
            
            return {
                "valid": stored_checksum == current_checksum and targets_valid,
                "checksum_valid": stored_checksum == current_checksum,
                "targets_valid": targets_valid,
                "mismatched_targets": mismatched_targets,
                "generator": prov.get("generator", "unknown"),
                "generated_at": prov.get("generated_at", "unknown"),
                "canon_version": prov.get("canon_version", "unknown"),
            }
        except Exception as e:
            return {"valid": False, "error": f"JSON parse error: {e}"}

    # Match the block (up to the second newline after -->)
    # We added exactly two newlines in add_header_to_content
    match = re.search(r'(<!--\s*PROVENANCE\s*.*?\s*-->\n\n)(.*)', content, re.DOTALL)
    if not match:
        # Try single newline for files like python or older versions
        match = re.search(r'(<!--\s*PROVENANCE\s*.*?\s*-->\n)(.*)', content, re.DOTALL)
        if not match:
            # Fallback to the lazy \s* but it might fail checksum
            match = re.search(r'(<!--\s*PROVENANCE\s*.*?\s*-->\s*)(.*)', content, re.DOTALL)
            if not match:
                return {"valid": False, "error": "No provenance header found"}
    
    header_block = match.group(1)
    content_after_header = match.group(2)
    
    # Parse the header metadata
    prov = {}
    for line in header_block.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("inputs") and not line.startswith("<!--"):
            key, value = line.split(":", 1)
            prov[key.strip()] = value.strip()
    
    # Extract checksum
    stored_checksum = prov.get("checksum", "")
    
    # We need to be careful about HOW it was added. 
    # add_header_to_content prepends header_str + content.
    # header_str = format_header_markdown(header) which ends with \n\n.
    # The regex \s* at the end of group 1 might eat more than the \n\n if the content starts with \n.
    
    # Let's try to match EXACTLY how it was formatted.
    # format_header_markdown ends with "-->\n\n" (lines.append("-->"); lines.append(""); return "\n".join(lines))
    # Wait, "\n".join(["-->", ""]) is "-->\n". 
    # Actually lines.append("-->"); lines.append(""); return "\n".join(lines) is "-->\n\n".
    
    # Let's just hash the content after the first "-->" plus the standard spacing.
    # Or better: just make add_header_to_content and verify_provenance use a very specific separator.
    
    # For now, let's fix hash_content to be used on the raw remaining string.
    current_checksum = hash_content(content_after_header)
    
    # If it fails, maybe the \s* ate the content's leading newlines. 
    # Let's try a more specific match: find "-->" and the specific number of newlines we added.
    if stored_checksum != current_checksum:
        # Fallback: find the first "-->" and look for the content.
        header_end = content.find("-->")
        if header_end != -1:
            # We added two newlines in format_header_markdown
            # lines = [ "...", "-->", "" ] -> join("\n") -> "...\n-->\n"
            # Wait, let me check format_header_markdown again.
            pass

    return {
        "valid": stored_checksum == current_checksum,
        "stored_checksum": stored_checksum,
        "current_checksum": current_checksum,
        "generator": prov.get("generator", "unknown"),
        "generated_at": prov.get("generated_at", "unknown"),
        "canon_version": prov.get("canon_version", "unknown"),
    }


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Provenance Header Utility")
    parser.add_argument("--verify", help="Verify provenance of a file")
    parser.add_argument("--show", help="Show provenance of a file")
    
    args = parser.parse_args()
    
    if args.verify:
        result = verify_provenance(Path(args.verify))
        if result["valid"]:
            print(f"✓ Provenance valid")
            print(f"  Generator: {result['generator']}")
            print(f"  Generated: {result['generated_at']}")
            if "mismatched_targets" in result and not result["mismatched_targets"] and "targets_valid" in result:
                print(f"✓ All target file hashes valid")
        else:
            print(f"✗ Provenance invalid: {result.get('error', 'checksum or target mismatch')}")
            if result.get("mismatched_targets"):
                print(f"  Mismatched targets:")
                for target in result["mismatched_targets"]:
                    print(f"    - {target}")
        sys.exit(0 if result["valid"] else 1)
    
    elif args.show:
        result = verify_provenance(Path(args.show))
        print(json.dumps(result, indent=2))
    
    else:
        # Demo
        header = generate_header(
            generator="TOOLS/provenance.py",
            inputs=["CANON/CONTRACT.md"],
        )
        print(format_header_markdown(header))
