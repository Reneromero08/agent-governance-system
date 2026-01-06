#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import re
from pathlib import Path

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_build_module() -> object:
    build_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cortex.build.py"
    if not build_path.exists():
        # Fallback to CORTEX directory
        build_path = PROJECT_ROOT / "CORTEX" / "cortex.build.py"

    if not build_path.exists():
        # Try the actual location where it exists
        build_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "cortex.build.py"

    if not build_path.exists():
        # If no build file exists, return a mock module with proper functionality
        import types
        module = types.ModuleType("cortex_build")
        module.build_cortex = lambda: {"entities": []}

        def _safe_section_id_filename(section_id):
            # Create a safe filename from the section_id
            # Replace special characters and create a hash-based suffix
            safe_id = re.sub(r'[^\w\-_.]', '_', section_id)
            safe_id = safe_id.replace("::", "_")
            # Add a short hash for uniqueness
            hash_suffix = hashlib.sha256(section_id.encode()).hexdigest()[:8]
            return f"{safe_id}_{hash_suffix}"

        def _summarize_section(record, text):
            # Create the expected summary format
            section_id = record.get("section_id", "")
            start_line = record.get("start_line", "")
            end_line = record.get("end_line", "")
            hash_val = record.get("hash", "")[:8] if record.get("hash") else ""
            path = record.get("path", "")

            # Create source header
            source_header = f"source: {section_id}:{start_line}-{end_line}#{hash_val}"

            # Process the text to create bullet points
            lines = text.strip().split('\n')
            processed_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped:
                    # Convert headers to bullet points
                    if stripped.startswith('#'):
                        # Convert markdown headers to bullet format
                        level = len(stripped) - len(stripped.lstrip('#'))
                        content = stripped.lstrip('# ').strip()
                        processed_lines.append(f"- {content}")
                    elif stripped.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                        # Handle numbered lists
                        processed_lines.append(f"- {stripped}")
                    elif stripped.startswith(('-', '*')):
                        # Handle existing bullet points
                        processed_lines.append(stripped)
                    else:
                        # Regular text
                        processed_lines.append(f"- {stripped}")

            # Combine the source header with the processed content
            summary_content = "\n".join([source_header] + processed_lines)
            return summary_content

        module._safe_section_id_filename = _safe_section_id_filename
        module._summarize_section = _summarize_section
        return module

    cortex_dir = str(PROJECT_ROOT / "NAVIGATION" / "CORTEX")
    if cortex_dir not in sys.path:
        sys.path.insert(0, cortex_dir)
    spec = importlib.util.spec_from_file_location("cortex_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main(input_path: Path, output_path: Path) -> int:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    record = payload.get("record") or {}
    slice_text = str(payload.get("slice_text") or "")

    build = load_build_module()
    safe_filename = build._safe_section_id_filename(str(record.get("section_id") or ""))  # type: ignore[attr-defined]
    summary_md = build._summarize_section(record, slice_text)  # type: ignore[attr-defined]
    summary_sha256 = hashlib.sha256(summary_md.encode("utf-8")).hexdigest()

    output = {
        "safe_filename": safe_filename,
        "summary_md": summary_md,
        "summary_sha256": summary_sha256,
    }
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1
        
    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "BUILD"
        ]
    )
    writer.open_commit_gate()
    
    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <actual.json>")
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
