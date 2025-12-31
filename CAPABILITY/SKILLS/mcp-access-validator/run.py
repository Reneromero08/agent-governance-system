#!/usr/bin/env python3
"""
MCP Access Validator - Run Script

Validates that agents use MCP tools instead of manual database/file operations
to prevent token waste and enforce governance compliance.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# MCP tool mappings
MCP_TOOLS = {
    # Database access patterns
    r"(?i)sqlite3.*connect.*system[13]\.db": "cortex_query",
    r"(?i)SELECT.*FROM.*symbols": "cortex_query",
    r"(?i)SELECT.*FROM.*vectors": "cortex_query",
    r"(?i)SELECT.*FROM.*cassettes": "context_search",
    r"(?i)\.db.*cursor.*execute": "cortex_query",
    
    # File system access patterns
    r"(?i)open\(.*\.md\)\.read\(\)": "canon_read",
    r"(?i)Path\(.*\)\.read_text\(\)": "canon_read",
    r"(?i)read_file.*LAW/CANON": "canon_read",
    r"(?i)read_file.*LAW/CONTEXT": "context_search",
    r"(?i)read_file.*NAVIGATION/CORTEX": "cortex_query",
    
    # Manual search patterns
    r"(?i)os\.walk.*\.md": "cortex_query",
    r"(?i)glob.*\.md": "cortex_query",
    r"(?i)find.*-name.*\.md": "cortex_query",
    
    # Semantic core patterns
    r"(?i)embeddings\.py": "cortex_query",
    r"(?i)vector.*search": "cortex_query",
    r"(?i)semantic.*search": "cortex_query",
    
    # Context patterns
    r"(?i)ADR-\d+": "context_search",
    r"(?i)LAW/CONTEXT/decisions": "context_search",
    r"(?i)LAW/CONTEXT/preferences": "context_search",
    
    # Session/identity patterns
    r"(?i)session.*id": "session_info",
    r"(?i)uuid.*generate": "session_info",
    r"(?i)ADR-021": "session_info",
}

# Tool recommendations with examples
TOOL_EXAMPLES = {
    "cortex_query": {
        "description": "Search the cortex index for content",
        "example": {"query": "catalytic", "limit": 10},
        "token_savings": 0.95,
        "governance": "ADR-004, ADR-021"
    },
    "context_search": {
        "description": "Search context records (ADRs, preferences, etc.)",
        "example": {"type": "decisions", "query": "catalytic"},
        "token_savings": 0.90,
        "governance": "ADR-004"
    },
    "canon_read": {
        "description": "Read canon governance documents",
        "example": {"file": "CONTRACT"},
        "token_savings": 0.85,
        "governance": "ADR-004"
    },
    "session_info": {
        "description": "Get session information including UUID for ADR-021 compliance",
        "example": {"include_audit_log": True},
        "token_savings": 0.80,
        "governance": "ADR-021"
    },
    "codebook_lookup": {
        "description": "Look up codebook entries",
        "example": {"id": "CATALYTIC", "expand": False},
        "token_savings": 0.75,
        "governance": "ADR-004"
    }
}

def detect_mcp_tool_needed(agent_action: str, agent_code: str = "") -> Optional[str]:
    """
    Detect which MCP tool should be used based on agent action and code.
    Returns the tool name or None if no match found.
    """
    # Check action description first
    text_to_check = f"{agent_action} {agent_code}".lower()
    
    for pattern, tool in MCP_TOOLS.items():
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return tool
    
    # Check specific keywords in action
    action_keywords = {
        "database": "cortex_query",
        "sqlite": "cortex_query",
        "query": "cortex_query",
        "search": "cortex_query",
        "read": "canon_read",
        "file": "canon_read",
        "document": "canon_read",
        "context": "context_search",
        "adr": "context_search",
        "decision": "context_search",
        "session": "session_info",
        "uuid": "session_info",
        "identity": "session_info",
        "codebook": "codebook_lookup",
        "symbol": "codebook_lookup"
    }
    
    for keyword, tool in action_keywords.items():
        if keyword in text_to_check:
            return tool
    
    return None

def calculate_token_waste(agent_code: str, recommended_tool: str) -> Dict[str, Any]:
    """
    Calculate token waste metrics.
    """
    if not agent_code:
        return {
            "token_waste_detected": False,
            "estimated_token_savings": 0.0,
            "reason": "No code provided for analysis"
        }
    
    # Simple heuristic: code length vs tool call length
    code_tokens = len(agent_code.split())  # rough estimate
    tool_info = TOOL_EXAMPLES.get(recommended_tool, {})
    tool_example = json.dumps(tool_info.get("example", {}))
    tool_tokens = len(tool_example.split())
    
    if code_tokens == 0:
        savings = 0.0
    else:
        savings = max(0.0, (code_tokens - tool_tokens) / code_tokens)
    
    # Apply baseline savings from tool info
    baseline_savings = tool_info.get("token_savings", 0.7)
    total_savings = min(0.99, savings + (1 - savings) * baseline_savings)
    
    waste_detected = savings > 0.1 or code_tokens > 50
    
    return {
        "token_waste_detected": waste_detected,
        "estimated_token_savings": round(total_savings, 3),
        "code_tokens": code_tokens,
        "tool_tokens": tool_tokens,
        "raw_savings": round(savings, 3),
        "baseline_savings": baseline_savings
    }

def create_audit_entry(validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an audit log entry for governance tracking.
    """
    from datetime import datetime
    import uuid
    
    return {
        "audit_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "skill": "mcp-access-validator",
        "validation_passed": validation_result.get("validation_passed", False),
        "token_waste_detected": validation_result.get("token_waste_detected", False),
        "recommended_tool": validation_result.get("recommended_mcp_tool"),
        "estimated_savings": validation_result.get("estimated_token_savings", 0.0),
        "action_taken": "validation_only"  # Could be "blocked", "warned", etc.
    }

def main(input_path: str, output_path: str) -> None:
    """
    Main validation function.
    """
    try:
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        agent_action = input_data.get("agent_action", "")
        agent_code = input_data.get("agent_code_snippet", "")
        files_accessed = input_data.get("files_accessed", [])
        databases_queried = input_data.get("databases_queried", [])
        
        # Detect MCP tool needed
        recommended_tool = detect_mcp_tool_needed(agent_action, agent_code)
        
        # Calculate token waste
        waste_metrics = calculate_token_waste(agent_code, recommended_tool if recommended_tool else "")
        
        # Determine validation result
        validation_passed = not waste_metrics["token_waste_detected"]
        
        # Prepare output
        output = {
            "validation_passed": validation_passed,
            "token_waste_detected": waste_metrics["token_waste_detected"],
            "agent_action": agent_action,
            "code_snippet_length": len(agent_code),
            "files_accessed_count": len(files_accessed),
            "databases_queried_count": len(databases_queried)
        }
        
        # Add recommendations if waste detected
        if recommended_tool and waste_metrics["token_waste_detected"]:
            tool_info = TOOL_EXAMPLES.get(recommended_tool, {})
            output.update({
                "recommended_mcp_tool": recommended_tool,
                "tool_description": tool_info.get("description", ""),
                "tool_usage_example": tool_info.get("example", {}),
                "estimated_token_savings": waste_metrics["estimated_token_savings"],
                "governance_compliance": tool_info.get("governance", ""),
                "action_required": "Use MCP tool instead of manual operations"
            })
        elif recommended_tool:
            # Tool available but no significant waste
            output.update({
                "recommended_mcp_tool": recommended_tool,
                "note": "MCP tool available but token waste minimal"
            })
        else:
            # No MCP tool match found
            output.update({
                "recommended_mcp_tool": None,
                "note": "No specific MCP tool match found. Consider if task could be done via existing skills."
            })
        
        # Add waste metrics
        output.update({
            "token_waste_metrics": {
                "code_tokens": waste_metrics["code_tokens"],
                "tool_tokens": waste_metrics["tool_tokens"],
                "raw_savings": waste_metrics["raw_savings"],
                "baseline_savings": waste_metrics["baseline_savings"]
            }
        })
        
        # Create audit entry
        audit_entry = create_audit_entry(output)
        output["audit_entry"] = audit_entry
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        # Exit code: 0 for passed, 1 for failed
        sys.exit(0 if validation_passed else 1)
        
    except Exception as e:
        # Error handling
        error_output = {
            "validation_passed": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "note": "Skill execution failed"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])