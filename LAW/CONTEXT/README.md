<!-- CONTENT_HASH: aa8752dad68e92e8099a757d34638b5d4441ce51ae3435cbe85b242574b8db9a -->

# Context

**Essence ($E$):** The long-term memory and decision record of the Agent Governance System.

| Directory | Content Type | Entropy ($\nabla S$) | Handling |
|-----------|--------------|----------------------|----------|
| `decisions/` | ADRs (Law) | Low | Append-only. Reference `@D{N}`. |
| `preferences/` | Styles/Guides | Low | Append-only. |
| `research/` | Active Inquiries | Medium | Move to `archive/` when done. |
| `archive/` | History | High | Compressed storage. |
| `feedback/` | Driver Reports | Medium | Feedback loops. |

## Usage
- **Read**: Use `cortex search` or `review_context.py` (Tool) to query.
- **Write**: Only append new records. Do not rewrite history unless correcting factual errors in metadata.
