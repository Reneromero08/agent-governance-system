---
uuid: 00000000-0000-0000-0000-000000000000
title: Agent Bootstrap Test Design
section: report
bucket: 2025-12/Week-52
author: System
priority: Medium
created: 2025-12-28 12:00
modified: 2026-01-06 13:09
status: Complete
summary: Design for agent bootstrap testing (Restored)
tags:
- bootstrap
- test
- design
hashtags: []
---
<!-- CONTENT_HASH: 052be09a95a5e2a09705f4fac51c6f7939e7480d19082f85f4f6ff2d4cd5dc12 -->

# Simulated Agent Bootstrap Test Design

**Purpose**: Design a test to validate that new agents can successfully bootstrap themselves using the updated AGENTS.md guidance.

## 1. Test Objectives

### Primary Objectives
1. **Validate AGENTS.md Section 0**: Test that connection guidance works for new agents
2. **Verify ADR-021 Compliance**: Ensure agents can discover their session_id
3. **Test Complete Bootstrap Sequence**: Validate agents can go from zero to operational
4. **Identify Remaining Gaps**: Find any undocumented steps or assumptions

### Secondary Objectives
1. **Measure Time to First Query**: Establish baseline for onboarding efficiency
2. **Test Error Recovery**: Validate troubleshooting guidance works
3. **Verify Tool Accessibility**: Ensure all essential MCP tools are accessible

## 2. Test Methodology

### Simulated Agent Approach
- **Manual Simulation**: Human operator following agent instructions
- **Automated Script**: Python script that mimics agent behavior
- **Hybrid Approach**: Script with manual validation steps

### Test Environment
- **Clean Repository**: Fresh clone or reset state
- **Standard Setup**: Python 3.8+, dependencies installed
- **No Prior Knowledge**: Operator assumes no cortex/MCP experience

## 3. Test Scenarios

### Scenario 1: Happy Path (Primary Validation)
**Goal**: Test complete successful bootstrap

**Steps**:
1. Start with only AGENTS.md as reference
2. Follow Section 0 connection guidance
3. Execute Phase 1-4 from Cortex Quick Reference
4. Complete a simple task (e.g., read CONTRACT.md)

**Success Criteria**:
- MCP server starts successfully
- Agent obtains session_id via session_info()
- Agent reads all required canon files
- Agent can execute cortex queries
- Total time < 10 minutes

### Scenario 2: Error Recovery
**Goal**: Test troubleshooting guidance

**Steps**:
1. Introduce common errors:
   - Server not running
   - Python version mismatch
   - Missing dependencies
   - Dirty repository
2. Follow troubleshooting guidance
3. Recover and complete bootstrap

**Success Criteria**:
- Error symptoms match documented issues
- Troubleshooting steps resolve issues
- Agent recovers without external help

### Scenario 3: Tool Accessibility
**Goal**: Verify all essential tools work

**Steps**:
1. Test each essential MCP tool:
   - cortex_query
   - canon_read
   - context_search
   - session_info
   - critic_run
   - agent_inbox_list
2. Verify tool responses are valid
3. Check error handling for invalid inputs

**Success Criteria**:
- All tools return expected responses
- Error handling provides useful information
- No tool requires undocumented parameters

## 4. Test Implementation

### Manual Test Script
```python
#!/usr/bin/env python3
"""
Manual Agent Bootstrap Test Script
Run this script and follow the prompts to simulate a new agent.
"""

import subprocess
import json
import time
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return output."""
    print(f"\n=== {description} ===")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"Stdout: {result.stdout[:500]}...")
    if result.stderr:
        print(f"Stderr: {result.stderr[:500]}...")
    return result

def test_step(step_num, description, command, expected_success=True):
    """Execute a test step and validate."""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*60}")
    
    result = run_command(command, description)
    
    if expected_success and result.returncode != 0:
        print(f"❌ Step {step_num} FAILED")
        return False
    elif not expected_success and result.returncode == 0:
        print(f"⚠️  Step {step_num} unexpectedly succeeded")
        return True
    else:
        print(f"✅ Step {step_num} passed")
        return True

def main():
    print("Agent Bootstrap Test - Starting...")
    start_time = time.time()
    
    # Step 1: Check Python version
    test_step(1, "Check Python version", "python --version")
    
    # Step 2: Start MCP server (background)
    test_step(2, "Start MCP server", "python LAW/CONTRACTS/ags_mcp_entrypoint.py --test", expected_success=True)
    
    # Step 3: Test basic connection (simulated)
    print("\n=== Simulated MCP Connection ===")
    print("At this point, the agent would connect via MCP protocol")
    print("For manual testing, we'll simulate the essential steps:")
    
    # Simulated steps that would normally happen via MCP
    steps = [
        ("Connect to MCP server", "MCP connection established"),
        ("Call session_info()", "session_id: test-uuid-1234"),
        ("Call cortex_query('test')", "Results returned"),
        ("Call canon_read('CONTRACT')", "CONTRACT.md content"),
        ("Call critic_run()", "Governance checks passed"),
    ]
    
    for i, (action, expected) in enumerate(steps, 3):
        print(f"\nStep {i}: {action}")
        print(f"Expected: {expected}")
        response = input(f"Did this succeed? (y/n): ")
        if response.lower() != 'y':
            print(f"❌ Step {i} failed")
            break
        print(f"✅ Step {i} passed")
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Test completed in {total_time:.1f} seconds")
    print(f"{'='*60}")
    
    if total_time < 600:  # 10 minutes
        print("✅ Bootstrap test PASSED - Agent can bootstrap within 10 minutes")
    else:
        print("⚠️  Bootstrap test WARNING - Bootstrap took longer than 10 minutes")

if __name__ == "__main__":
    main()
```

### Automated Test Design (Future)
```python
# Future automated test using MCP client
# This would require a full MCP client implementation

class AgentBootstrapTest:
    def __init__(self):
        self.mcp_client = None
        self.session_id = None
        
    def connect_to_mcp(self):
        """Establish MCP connection."""
        # Implementation would use MCP stdio protocol
        pass
        
    def test_session_info(self):
        """Test session_info tool."""
        response = self.mcp_client.call_tool("session_info", {})
        self.session_id = response.get("session_id")
        return self.session_id is not None
        
    def test_cortex_query(self):
        """Test cortex_query tool."""
        response = self.mcp_client.call_tool("cortex_query", {"query": "test"})
        return "content" in response
        
    # Additional test methods...
```

## 5. Success Metrics

### Quantitative Metrics
1. **Time to First Successful Query**: < 5 minutes
2. **Time to Complete Bootstrap**: < 10 minutes
3. **Tool Success Rate**: 100% for essential tools
4. **Error Recovery Time**: < 3 minutes per error

### Qualitative Metrics
1. **Clarity of Instructions**: All steps unambiguous
2. **Completeness of Guidance**: No missing steps
3. **Usefulness of Troubleshooting**: Solutions match problems
4. **Agent Confidence**: Operator feels capable of proceeding

## 6. Test Execution Plan

### Phase 1: Manual Validation (Immediate)
- Execute manual test script
- Document any issues or ambiguities
- Update documentation based on findings

### Phase 2: Peer Review (Next)
- Have another team member attempt bootstrap
- Compare experiences and identify differences
- Refine guidance based on multiple perspectives

### Phase 3: Automated Testing (Future)
- Implement MCP client for automated testing
- Create CI/CD pipeline for bootstrap validation
- Run tests on repository changes to prevent regressions

## 7. Expected Outcomes

### Immediate Outcomes
1. **Validation Report**: Document test results and findings
2. **Documentation Updates**: Refine AGENTS.md based on test feedback
3. **Bug Fixes**: Address any discovered issues in MCP server or tools

### Long-term Outcomes
1. **Standardized Onboarding**: Repeatable, reliable bootstrap process
2. **Quality Metrics**: Baseline measurements for onboarding efficiency
3. **Prevention of Regressions**: Automated tests catch breaking changes

## 8. Risk Mitigation

### Technical Risks
- **MCP Server Changes**: Test may break if MCP interface changes
  - **Mitigation**: Version pinning and compatibility checks
- **Dependency Issues**: Test environment differences
  - **Mitigation**: Clear environment requirements in test documentation

### Process Risks
- **False Positives**: Test passes but real agents struggle
  - **Mitigation**: Include qualitative assessment and peer review
- **Documentation Drift**: Test becomes outdated
  - **Mitigation**: Regular test execution and documentation updates

## 9. Conclusion

This test design provides a comprehensive approach to validating the agent bootstrap process. By combining manual validation with future automated testing, we can ensure that new agents can successfully onboard to the Agent Governance System using the updated AGENTS.md guidance.

The test will validate not only the technical functionality but also the clarity and completeness of the documentation, ensuring that agents can bootstrap themselves without external assistance.

---

**Test Owner**: Kilo Code (Agent ID: 17cb4e78-ae76-49df-b336-c0cccbf5878d)
**Test Status**: Design Complete
**Next Steps**: Execute manual test, document findings, update documentation
**Related Documents**: AGENTS.md, Cortex Quick Reference, ADR-021