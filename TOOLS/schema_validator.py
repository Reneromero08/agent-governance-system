import os
import re
import json
import jsonschema
from typing import Dict, Any, Optional, List

def extract_markdown_metadata(content: str) -> Dict[str, Any]:
    """
    Extracts metadata from the top of a Markdown file.
    Looks for lines like **Key:** Value or **Key:** [val1, val2]
    """
    metadata = {}
    
    # We only look at the first 50 lines to avoid parsing the whole file
    lines = content.splitlines()[:50]
    
    for line in lines:
        # Match **Key:** Value
        match = re.search(r'^\*\*(.*?):\*\*\s*(.*)$', line.strip())
        if match:
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            
            # Handle list-like values [val1, val2]
            if value.startswith("[") and value.endswith("]"):
                try:
                    # Try to parse as JSON list
                    # First replace single quotes with double quotes for valid JSON
                    json_str = value.replace("'", '"')
                    metadata[key] = json.loads(json_str)
                except json.JSONDecodeError:
                    # Fallback: manual split
                    items = value[1:-1].split(",")
                    metadata[key] = [i.strip() for i in items if i.strip()]
            else:
                metadata[key] = value
                
    return metadata

def validate_file(file_path: str, schema_path: str) -> Optional[List[str]]:
    """
    Validates a Markdown file against a JSON Schema.
    Returns a list of error messages, or None if valid.
    """
    if not os.path.exists(file_path):
        return [f"File not found: {file_path}"]
    if not os.path.exists(schema_path):
        return [f"Schema not found: {schema_path}"]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
            
        metadata = extract_markdown_metadata(content)
        
        # Inject ID for ADRs/Styles if found in title
        if "adr-" in os.path.basename(file_path).lower():
            match = re.search(r'# (ADR-\d{3})', content)
            if match:
                metadata["id"] = match.group(1)
        elif "STYLE-" in os.path.basename(file_path):
            match = re.search(r'# (STYLE-\d{3})', content)
            if match:
                metadata["style"] = match.group(1)

        jsonschema.validate(instance=metadata, schema=schema)
        return None
        
    except jsonschema.exceptions.ValidationError as e:
        return [f"Validation error in {os.path.basename(file_path)}: {e.message}"]
    except Exception as e:
        return [f"Error validating {os.path.basename(file_path)}: {str(e)}"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python schema_validator.py <file_path> <schema_path>")
        sys.exit(1)
        
    errors = validate_file(sys.argv[1], sys.argv[2])
    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        sys.exit(1)
    else:
        print(f"[OK] {os.path.basename(sys.argv[1])} matches schema.")
