<!-- CONTENT_HASH: a1cb23e142abc2a8bdff0d02a24f8e211fb867f0f31395e0992658ea5efd7c05 -->

# Implementation Reports

**Canonical Requirement:** All implementations must produce a signed report.

## Purpose

This document establishes the requirement that **every new implementation** in the Agent Governance System must be documented with a signed report containing:

1. **Agent identity** - Model name and session identifier
2. **Date** - When the implementation was completed
3. **What was built** - Technical summary of the implementation
4. **What was demonstrated** - Verification results and testing
5. **Real vs simulated** - Confirmation that actual data was used
6. **Next steps** - Roadmap progression or follow-up tasks

## Report Format

All implementation reports must follow this standard format:

```markdown
# [Feature Name] Implementation Report

**Date:** YYYY-MM-DD
**Status:** COMPLETE | PARTIAL | FAILED
**Agent:** [model-name]@[system-identifier] | YYYY-MM-DD

---

## Executive Summary

[One-paragraph summary of what was implemented and why it matters]

---

## What Was Built

[Technical details of files, functions, classes, databases, protocols created]

### Files Created
- `path/to/file1.py` - [description]
- `path/to/file2.py` - [description]

### Architecture
```
[Diagrams or code structure if applicable]
```

### Key Features
- [Feature 1]: [description]
- [Feature 2]: [description]

---

## What Was Demonstrated

[Testing results, queries executed, verification performed]

### Test Results
- [Test 1]: ✅ PASS / ❌ FAIL - [details]
- [Test 2]: ✅ PASS / ❌ FAIL - [details]

### Output Examples
```
[Sample output or query results]
```

---

## Real vs Simulated

### Real Data Processing
- Database connections: Direct SQLite / API calls
- Data retrieved: Actual content from [databases/systems]
- Query matching: Content-based / vector-based
- Results displayed: Chunk IDs, headings, content previews

### What's Not Simulation
- No synthetic data generation
- No mocked responses
- No hardcoded test fixtures in production

---

## Metrics

### Code Statistics
- Files created: [number]
- Lines of code: [number]
- Database coverage: [number] chunks/docs

### Performance
- Query latency: [time]
- Indexing throughput: [chunks/sec]
- Token savings: [percentage]

---

## Conclusion

[Summary of success/failure, road to next phase, open questions]

---

**Report Generated:** YYYY-MM-DD
**Implementation Status:** [status]
```

## Required Sections

All reports MUST include:

1. ✅ **Signature Block**
   - Agent identity: `[model-name]@[system-identifier] | YYYY-MM-DD`
   - Format: Markdown heading at top of report

2. ✅ **Executive Summary**
   - High-level overview
   - Why this implementation matters

3. ✅ **What Was Built**
   - Files created with descriptions
   - Architecture diagrams (if applicable)
   - Key features with bullet points

4. ✅ **What Was Demonstrated**
   - Test results with pass/fail status
   - Output examples (code blocks, query results)
   - Verification performed

5. ✅ **Real vs Simulated**
   - Explicit confirmation of real data usage
   - List what's NOT simulation
   - Database/API connections documented

6. ✅ **Metrics**
   - Code statistics
   - Performance numbers
   - Quantitative results

7. ✅ **Conclusion**
   - Status summary
   - Next steps
   - Roadmap progression

## Report Storage

All implementation reports must be stored under:

```
LAW/CONTRACTS/_runs/<feature-name>-implementation-report.md
```

## Examples

- `SEMANTIC_DATABASE_NETWORK_REPORT.md` - Cassette Network Phase 0
- `semantic-core-phase1-final-report.md` - Semantic Core implementation
- `session-report-YYYY-MM-DD.md` - Session documentation

## Enforcement

The `critic.py` tool and governance checks will verify:

1. Reports exist for all implementations
2. Reports are signed (agent + date)
3. Reports contain all required sections
4. Reports are stored in `LAW/CONTRACTS/_runs/`

## Rationale

**Why Signed Reports?**

- **Provenance:** Verifies which agent implemented what and when
- **Auditability:** Enables reconstruction of implementation history
- **Attribution:** Credits the work to the implementing agent
- **Reproducibility:** Records the state of system at implementation time

**Why Implementation Reports?**

- **Documentation:** Captures intent, decisions, and results
- **Governance:** Ensures all work is accounted for and audited
- **Learning:** Provides record of what worked and what didn't
- **Trust:** Demonstrates real data was used, not simulations

---

**Canon Version:** 2.15.1
**Required Canon Version:** >=2.15.1
