---
uuid: "a7f3c8e1-9d2b-4f5a-8c6e-1a2b3c4d5e6f"
title: "Document Merge Tooling Strategy Report"
section: "report"
bucket: "tooling/document-management"
author: "Antigravity"
priority: "High"
created: "2026-01-01 13:36"
modified: "2026-01-01 13:36"
status: "Complete"
summary: "Technical report on implementing 100% fidelity document comparison and batch merge tooling for CAT_CHAT consolidation"
tags: ["tooling", "document-merge", "efficiency"]
hashtags: ["#tooling", "#doc-merge", "#cat-chat"]
---
<!-- CONTENT_HASH: d2949091d4c4bf0e89df5b1cad8244b39e5fb70baff94f6fcd669021b738e578 -->

# Document Merge Tooling Strategy Report

## Executive Summary

User requires 100% fidelity document comparison for merging CAT_CHAT documents. LLM-based approaches are insufficient. Solution: Batch wrapper around existing deterministic `doc-diff` skill.

## Requirements

1. **100% Fidelity**: Exact line-by-line differences, no approximations
2. **Token Efficiency**: Minimal context usage
3. **Batch Processing**: Handle multiple file pairs
4. **No LLM**: Deterministic Python only

## Proposed Solution: `doc-merge-batch`

### Architecture

```
Input: Directory or file list
  ↓
For each file pair:
  ↓
Call doc-diff (Python difflib)
  ↓
Aggregate results
  ↓
Output: Structured diff report
```

### Input Format
```json
{
  "files": [
    "THOUGHT/LAB/CAT_CHAT/CHANGELOG.md",
    "THOUGHT/LAB/CAT_CHAT/archive/CAT_CHAT_CHANGELOG_old.md"
  ],
  "max_diff_lines": 100
}
```

### Output Format
```json
{
  "comparisons": [
    {
      "file_a": "CHANGELOG.md",
      "file_b": "CAT_CHAT_CHANGELOG_old.md",
      "identical": false,
      "similarity": 0.73,
      "unique_to_a": ["Line 45-67", "Line 203-210"],
      "unique_to_b": ["Line 12-15"],
      "diff_summary": "78 lines differ"
    }
  ]
}
```

## Implementation

### Location
`CAPABILITY/SKILLS/doc-merge-batch/`

### Dependencies
- Existing `doc-diff` skill (already implemented)
- Python `pathlib`, `json`

### Key Features
1. **Pairwise comparison**: All combinations or specified pairs
2. **Truncation**: Configurable diff line limits
3. **Similarity scoring**: 0.0-1.0 for quick triage
4. **Section extraction**: Identify unique blocks

## Usage Workflow

```bash
# 1. Generate file list
ls THOUGHT/LAB/CAT_CHAT/*.md > files.txt

# 2. Run batch diff
python CAPABILITY/SKILLS/doc-merge-batch/run.py < input.json > report.json

# 3. Review exact differences
# Report shows which sections are unique to each file
```

## Token Savings

- **Without tool**: Load both files into context (~10K tokens)
- **With tool**: Receive structured diff report (~500 tokens)
- **Savings**: ~95% reduction

## Next Steps

1. Implement `doc-merge-batch/run.py`
2. Create fixtures
3. Test on CAT_CHAT documents
4. Iterate based on user feedback