#!/usr/bin/env python3
"""
Practical Catalytic Chat Demo
Shows how to make this conversation catalytic.
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

class CatalyticChat:
    """Simple catalytic chat implementation for this conversation."""
    
    def __init__(self, chat_id="current_chat"):
        self.chat_id = chat_id
        self.catalytic_dir = Path("LAW/CONTRACTS/_runs/_tmp/catalytic_chat")
        self.catalytic_dir.mkdir(parents=True, exist_ok=True)
        
        # Content-addressed storage
        self.content_store = self.catalytic_dir / "content_store"
        self.content_store.mkdir(exist_ok=True)
        
        # Receipt ledger
        self.receipt_file = self.catalytic_dir / "receipts.jsonl"
        
        print(f"[Catalytic Chat] Initialized: {self.chat_id}")
        print(f"[Catalytic Chat] Catalytic domain: {self.catalytic_dir}")
        print(f"[Catalytic Chat] Content store: {self.content_store}")
        print()
    
    def store_content(self, content: str) -> str:
        """Store content in catalytic space, return hash reference."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        content_path = self.content_store / content_hash
        
        # Only write if not already stored (content-addressed)
        if not content_path.exists():
            content_path.write_text(content, encoding='utf-8')
            print(f"[Catalytic] Stored {len(content)} chars → hash://{content_hash[:16]}...")
        
        return content_hash
    
    def get_slice(self, content_hash: str, slice_spec: str = "lines[0:100]") -> str:
        """Retrieve slice from content-addressed store."""
        content_path = self.content_store / content_hash
        if not content_path.exists():
            return f"[Error] Content not found: {content_hash}"
        
        content = content_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # Parse slice spec (simplified)
        if slice_spec.startswith("lines["):
            try:
                # Parse "lines[start:end]"
                range_str = slice_spec[6:-1]  # Remove "lines[" and "]"
                start, end = map(int, range_str.split(':'))
                sliced = '\n'.join(lines[start:end])
                return sliced
            except:
                return content[:500] + "..."  # Fallback
        
        return content[:500] + "..."
    
    def generate_receipt(self, operation: str, content_hash: str, slice_used: str = "") -> dict:
        """Generate verifiable receipt for catalytic operation."""
        receipt = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "chat_id": self.chat_id,
            "operation": operation,
            "content_hash": content_hash,
            "slice_used": slice_used,
            "receipt_hash": None  # Will be computed
        }
        
        # Compute receipt hash
        receipt_json = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_json.encode('utf-8')).hexdigest()
        receipt["receipt_hash"] = receipt_hash
        
        # Append to ledger
        with open(self.receipt_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(receipt) + '\n')
        
        return receipt
    
    def catalytic_response(self, query: str, full_content: str) -> str:
        """Generate catalytic response instead of pasting full content."""
        print(f"[User Query]: {query}")
        print()
        
        # 1. Store content catalytically
        content_hash = self.store_content(full_content)
        
        # 2. Determine appropriate slice based on query
        if "principle" in query.lower() or "what is" in query.lower():
            slice_spec = "lines[0:10]"  # Introduction
        elif "example" in query.lower():
            slice_spec = "lines[20:30]"  # Example section
        else:
            slice_spec = "lines[0:5]"  # Default
        
        # 3. Get sliced content
        sliced_content = self.get_slice(content_hash, slice_spec)
        
        # 4. Generate receipt
        receipt = self.generate_receipt(
            operation="query_response",
            content_hash=content_hash,
            slice_used=slice_spec
        )
        
        # 5. Construct catalytic response
        response = f"""## Catalytic Response

**Relevant excerpt ({slice_spec}):**
```
{sliced_content}
```

**Full content reference:**
- Hash: `{content_hash[:32]}...`
- Location: `{self.content_store / content_hash}`

**Verification:**
- Receipt hash: `{receipt['receipt_hash'][:32]}...`
- Timestamp: {receipt['timestamp']}

**Catalytic savings:**
- Full content: {len(full_content)} chars
- Context used: {len(sliced_content)} chars
- Reduction: {100 - (len(sliced_content) / len(full_content) * 100):.0f}%"""
        
        return response

# Demo: Making this chat catalytic
if __name__ == "__main__":
    print("=" * 60)
    print("MAKING THIS CHAT CATALYTIC - PRACTICAL DEMO")
    print("=" * 60)
    print()
    
    # Initialize catalytic chat
    chat = CatalyticChat()
    
    # Example: User asks about catalytic computing
    query = "What is catalytic computing?"
    
    # Instead of pasting the full CANON document, we use catalytic approach
    full_content = """# Catalytic Computing

This document defines catalytic computing for the Agent Governance System. It separates formal theory from engineering translation so agents do not hallucinate implementation details.

## Formal Model (Complexity Theory)

Catalytic space computation uses two kinds of memory:

1. **Clean space** - A small amount of blank working memory the algorithm can use freely.
2. **Catalytic space** - A much larger memory that starts in an arbitrary state and must be returned to exactly that state at the end.

The key constraint: the algorithm must work for any initial catalytic content (possibly incompressible) and cannot permanently encode new information. The catalytic bits act like a catalyst in chemistry - they enable the computation but remain unchanged afterward.

**Key results** (Buhrman, Cleve, Koucky, Loff, Speelman, 2014):
- Catalytic logspace can compute uniform TC^1 circuits (includes matrix determinant)
- Upper bound: catalytic logspace is contained in ZPP
- The restoration constraint does not kill usefulness; it forces reversible, structured transformations

## AGS Translation

For AGS, catalytic computing provides a memory model:

| Formal Concept | AGS Analog | Examples |
|----------------|------------|----------|
| Clean space | Context tokens | LITE pack contents, working memory |
| Catalytic space | Disk state | Indexes, caches, generated artifacts |
| Restoration | Repo returns identical | Git worktree, content-addressed cache |

**Core insight**: Large disk state can be used as powerful scratch space if you guarantee restoration. This enables high-impact operations (index builds, pack generation, refactors) while keeping context minimal.

## Five Engineering Patterns

### Pattern 1: Clean Context vs Catalytic Store

Keep context tokens minimal. Use disk as the large, addressable store.

- **Clean context (LITE pack)**: laws, maps, contracts, symbolic indexes, short summaries, retrieval instructions
- **Catalytic store**: full file bodies, generated indexes, caches - addressable by hash or path

### Pattern 2: Restore Guarantee as First-Class Artifact

Every operation that uses "big scratch" must produce:
- A **patch** (what changed)
- A **restore plan** (how to undo)
- A **verification check** (prove restoration happened)

Practical mechanisms:
- Git worktree or temporary checkout
- Overlay filesystem (copy-on-write)
- Content-addressed cache for generated artifacts

[Content continues for 1000+ more lines...]"""
    
    print("NON-CATALYTIC APPROACH:")
    print("-" * 40)
    print(f"Agent would paste {len(full_content)} chars into context")
    print("Result: Token waste, no verifiability, no restoration guarantee")
    print()
    
    print("CATALYTIC APPROACH:")
    print("-" * 40)
    response = chat.catalytic_response(query, full_content)
    print(response)
    print()
    
    print("=" * 60)
    print("THIS CHAT IS NOW CATALYTIC")
    print(f"- Content stored at: {chat.content_store}")
    print(f"- Receipts logged at: {chat.receipt_file}")
    print(f"- Catalytic domain: {chat.catalytic_dir}")
    print("=" * 60)