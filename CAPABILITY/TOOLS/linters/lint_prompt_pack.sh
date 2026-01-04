#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AGS Prompt Pack Linter
# Enforces NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md mechanically
# Exit codes: 0=PASS, 1=POLICY_VIOLATION (block), 2=WARNING (non-block)
# ==============================================================================

PROMPTS_DIR="${1:-}"
if [[ -z "$PROMPTS_DIR" ]]; then
  echo "Usage: $0 <PROMPTS_DIR>" >&2
  exit 1
fi

if [[ ! -d "$PROMPTS_DIR" ]]; then
  echo "ERROR: Directory not found: $PROMPTS_DIR" >&2
  exit 1
fi

# Resolve to absolute path
PROMPTS_DIR="$(cd "$PROMPTS_DIR" && pwd)"

# ==============================================================================
# State tracking
# ==============================================================================
VIOLATIONS=()
WARNINGS=()
EXIT_CODE=0

# ==============================================================================
# Helper: Compute SHA256
# ==============================================================================
compute_sha256() {
  local file="$1"
  if command -v sha256sum &>/dev/null; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum &>/dev/null; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$file"
  fi
}

# ==============================================================================
# A) Manifest validity
# ==============================================================================
check_manifest() {
  local manifest="$PROMPTS_DIR/PROMPT_PACK_MANIFEST.json"
  
  if [[ ! -f "$manifest" ]]; then
    VIOLATIONS+=("MANIFEST_MISSING: $manifest not found")
    return
  fi
  
  # Parse JSON and check structure
  python3 - "$manifest" "$PROMPTS_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
prompts_dir = Path(sys.argv[2])

try:
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
except json.JSONDecodeError as e:
    print(f"MANIFEST_INVALID_JSON: {e}", file=sys.stderr)
    sys.exit(1)

if 'tasks' not in data:
    print("MANIFEST_MISSING_TASKS_ARRAY", file=sys.stderr)
    sys.exit(1)

tasks = data['tasks']
if not isinstance(tasks, list):
    print("MANIFEST_TASKS_NOT_ARRAY", file=sys.stderr)
    sys.exit(1)

required_fields = ['task_id', 'prompt_path', 'receipt_path', 'report_path', 'depends_on']
errors = []

for i, task in enumerate(tasks):
    for field in required_fields:
        if field not in task:
            errors.append(f"Task {i} (id={task.get('task_id','?')}): missing field '{field}'")
    
    # Check prompt_path exists (relative to NAVIGATION/)
    if 'prompt_path' in task:
        # Paths in manifest are relative to repo root, starting with PROMPTS/
        # We need to resolve them relative to PROMPTS_DIR parent
        prompt_rel = task['prompt_path']
        if prompt_rel.startswith('PROMPTS/'):
            prompt_rel = prompt_rel[len('PROMPTS/'):]
        prompt_full = prompts_dir / prompt_rel
        if not prompt_full.exists():
            errors.append(f"Task {task.get('task_id','?')}: prompt_path does not exist: {task['prompt_path']}")

if errors:
    for err in errors:
        print(f"MANIFEST_TASK_INVALID: {err}", file=sys.stderr)
    sys.exit(1)
PYEOF
  
  if [[ $? -ne 0 ]]; then
    VIOLATIONS+=("MANIFEST_VALIDATION_FAILED")
  fi
}

# ==============================================================================
# B) INDEX link validity
# ==============================================================================
check_index() {
  local index="$PROMPTS_DIR/INDEX.md"
  
  if [[ ! -f "$index" ]]; then
    VIOLATIONS+=("INDEX_MISSING: $index not found")
    return
  fi
  
  # Extract markdown links and check they exist
  python3 - "$index" "$PROMPTS_DIR" <<'PYEOF'
import re, sys
from pathlib import Path

index_path = Path(sys.argv[1])
prompts_dir = Path(sys.argv[2])

content = index_path.read_text(encoding='utf-8')
# Pattern: `NAVIGATION/PROMPTS/...` or just PHASE_XX/...
# From INDEX.md we see format: `NAVIGATION/PROMPTS/PHASE_XX/X.X_slug.md`
pattern = r'`(NAVIGATION/PROMPTS/[^`]+\.md|PHASE_\d+/[^`]+\.md)`'
links = re.findall(pattern, content)

errors = []
for link in links:
    # Resolve relative to PROMPTS_DIR
    if link.startswith('NAVIGATION/PROMPTS/'):
        rel_path = link[len('NAVIGATION/PROMPTS/'):]
    else:
        rel_path = link
    
    target = prompts_dir / rel_path
    if not target.exists():
        errors.append(f"INDEX_BROKEN_LINK: {link} -> {target}")

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    sys.exit(1)
PYEOF
  
  if [[ $? -ne 0 ]]; then
    VIOLATIONS+=("INDEX_LINK_CHECK_FAILED")
  fi
}

# ==============================================================================
# C) YAML front matter + D) Canon hash consistency + E) Forbidden terms
# F) Empty bullet lines + G) FILL token containment
# ==============================================================================
check_prompts() {
  # Compute canon hashes
  local policy_canon="$PROMPTS_DIR/1_PROMPT_POLICY_CANON.md"
  local guide_canon="$PROMPTS_DIR/2_PROMPT_GENERATOR_GUIDE_FINAL.md"
  
  if [[ ! -f "$policy_canon" ]]; then
    VIOLATIONS+=("CANON_MISSING: $policy_canon not found")
    return
  fi
  
  if [[ ! -f "$guide_canon" ]]; then
    VIOLATIONS+=("CANON_MISSING: $guide_canon not found")
    return
  fi
  
  local expected_policy_hash
  local expected_guide_hash
  expected_policy_hash="$(compute_sha256 "$policy_canon")"
  expected_guide_hash="$(compute_sha256 "$guide_canon")"
  
  # Find all prompt files under PHASE_*
  local prompt_files=()
  while IFS= read -r -d '' file; do
    prompt_files+=("$file")
  done < <(find "$PROMPTS_DIR" -type f -path "*/PHASE_*/*.md" -print0 | sort -z)
  
  if [[ ${#prompt_files[@]} -eq 0 ]]; then
    VIOLATIONS+=("NO_PROMPT_FILES_FOUND: No .md files under PHASE_* directories")
    return
  fi
  
  # Check each prompt
  for prompt in "${prompt_files[@]}"; do
    local rel_path="${prompt#$PROMPTS_DIR/}"
    
    python3 - "$prompt" "$expected_policy_hash" "$expected_guide_hash" "$rel_path" <<'PYEOF'
import re, sys
from pathlib import Path

prompt_path = Path(sys.argv[1])
expected_policy = sys.argv[2]
expected_guide = sys.argv[3]
rel_path = sys.argv[4]

content = prompt_path.read_text(encoding='utf-8')
lines = content.splitlines()

errors = []
warnings = []

# C) YAML front matter check
if not lines or not lines[0].strip().startswith('---'):
    errors.append(f"{rel_path}: Missing YAML front matter (must start with ---)")
    sys.exit(1)

# Extract YAML front matter
yaml_end = -1
for i in range(1, min(len(lines), 50)):
    if lines[i].strip() == '---':
        yaml_end = i
        break

if yaml_end == -1:
    errors.append(f"{rel_path}: YAML front matter not closed (missing second ---)")
    sys.exit(1)

yaml_block = '\n'.join(lines[1:yaml_end])

# Required YAML fields
required_yaml = [
    'phase', 'task_id', 'slug', 'policy_canon_sha256', 'guide_canon_sha256',
    'depends_on', 'primary_model', 'fallback_chain', 'receipt_path', 
    'report_path', 'max_report_lines'
]

yaml_dict = {}
for line in lines[1:yaml_end]:
    if ':' in line:
        key = line.split(':', 1)[0].strip()
        value = line.split(':', 1)[1].strip()
        yaml_dict[key] = value

for field in required_yaml:
    if field not in yaml_dict:
        errors.append(f"{rel_path}: Missing YAML field: {field}")

# Validate phase is integer
if 'phase' in yaml_dict:
    try:
        int(yaml_dict['phase'])
    except ValueError:
        errors.append(f"{rel_path}: phase must be an integer, got: {yaml_dict['phase']}")

# Validate task_id format (N.M or N.M.K)
if 'task_id' in yaml_dict:
    task_id = yaml_dict['task_id'].strip('"').strip("'")
    if not re.match(r'^\d+\.\d+(\.\d+)?$', task_id):
        errors.append(f"{rel_path}: task_id must match N.M or N.M.K format, got: {task_id}")

# Validate slug is kebab-case
if 'slug' in yaml_dict:
    slug = yaml_dict['slug'].strip('"').strip("'")
    if not re.match(r'^[a-z0-9-]+$', slug):
        errors.append(f"{rel_path}: slug must be kebab-case (lowercase, digits, hyphens), got: {slug}")

# D) Canon hash consistency
if 'policy_canon_sha256' in yaml_dict:
    actual_policy = yaml_dict['policy_canon_sha256'].strip('"').strip("'")
    if actual_policy != expected_policy:
        errors.append(f"{rel_path}: policy_canon_sha256 mismatch (expected {expected_policy}, got {actual_policy})")

if 'guide_canon_sha256' in yaml_dict:
    actual_guide = yaml_dict['guide_canon_sha256'].strip('"').strip("'")
    if actual_guide != expected_guide:
        errors.append(f"{rel_path}: guide_canon_sha256 mismatch (expected {expected_guide}, got {actual_guide})")

# E) Forbidden inference terms (hex-escaped patterns)
# Pattern: \b\x61\x73\x73\x75\x6d\x65\b (assume)
# Pattern: \b\x61\x73\x73\x75\x6d\x70\x74\x69\x6f\x6e(s)?\b (assumption/assumptions)
forbidden_patterns = [
    rb'\bassume\b',
    rb'\bassumption(s)?\b',
]

content_bytes = content.encode('utf-8')
for pattern in forbidden_patterns:
    if re.search(pattern, content_bytes, re.IGNORECASE):
        errors.append(f"{rel_path}: Contains forbidden inference term")
        break

# F) Empty bullet lines (warning only)
for i, line in enumerate(lines, 1):
    if re.match(r'^\s*-\s*$', line):
        warnings.append(f"{rel_path}:{i}: Empty bullet line (- with no content)")

# G) FILL_ME__ token containment
fill_tokens = re.findall(r'FILL_ME__\w+', content)
if fill_tokens:
    # Check if inside REQUIRED FACTS section
    in_required_facts = False
    for i, line in enumerate(lines):
        if re.match(r'^##\s+REQUIRED FACTS', line, re.IGNORECASE):
            in_required_facts = True
        elif in_required_facts and re.match(r'^##\s+', line):
            in_required_facts = False
        
        if not in_required_facts:
            for token in fill_tokens:
                if token in line:
                    errors.append(f"{rel_path}:{i+1}: FILL_ME__ token outside REQUIRED FACTS section: {token}")
                    break

# Output
for err in errors:
    print(f"VIOLATION: {err}", file=sys.stderr)
for warn in warnings:
    print(f"WARNING: {warn}", file=sys.stderr)

if errors:
    sys.exit(1)
elif warnings:
    sys.exit(2)
PYEOF
    
    local result=$?
    if [[ $result -eq 1 ]]; then
      VIOLATIONS+=("PROMPT_CHECK_FAILED: $rel_path")
    elif [[ $result -eq 2 ]]; then
      WARNINGS+=("PROMPT_WARNING: $rel_path")
    fi
  done
}

# ==============================================================================
# Main execution
# ==============================================================================
echo "=== AGS Prompt Pack Linter ==="
echo "Checking: $PROMPTS_DIR"
echo ""

check_manifest
check_index
check_prompts

# ==============================================================================
# Summary output
# ==============================================================================
echo ""
echo "=== LINT SUMMARY ==="
echo "Violations: ${#VIOLATIONS[@]}"
echo "Warnings: ${#WARNINGS[@]}"
echo ""

if [[ ${#VIOLATIONS[@]} -gt 0 ]]; then
  echo "POLICY VIOLATIONS (blocking):"
  printf '%s\n' "${VIOLATIONS[@]}" | sort
  EXIT_CODE=1
fi

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  echo ""
  echo "WARNINGS (non-blocking):"
  printf '%s\n' "${WARNINGS[@]}" | sort
  if [[ $EXIT_CODE -eq 0 ]]; then
    EXIT_CODE=2
  fi
fi

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ PASS: All checks passed"
fi

exit $EXIT_CODE
