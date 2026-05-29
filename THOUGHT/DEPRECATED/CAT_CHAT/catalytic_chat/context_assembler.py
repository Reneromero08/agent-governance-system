"""
Context Assembler

Deterministic context-assembly pipeline that selects, truncates, and orders 
conversational inputs and symbol expansions to fit within a fixed model context budget.

Pure logic, in-memory only.

Phase 3.2.1 Truncation Behavior:
- Only HEAD truncation is supported (preserves start of content, discards end).
- Truncation is character-based via binary search approximation.
- Token estimates are approximate; no token-exact guarantees.
- Deterministic for identical inputs and token_estimator functions.

Phase 3.2.1 Expansion Role Assignment:
- All expansion items are assigned role="SYSTEM" by design.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any, Union, Tuple

@dataclass
class ContextBudget:
    max_total_tokens: int
    reserve_response_tokens: int
    max_messages: int
    max_expansions: int
    max_tokens_per_message: int
    max_tokens_per_expansion: int

@dataclass
class ContextMessage:
    id: str
    source: str  # SYSTEM, USER, ASSISTANT, TOOL
    content: str
    created_at: str  # ISO timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextExpansion:
    symbol_id: str
    content: str
    is_explicit_reference: bool  # Referenced by latest user message?
    priority: int = 0  # For optional extras tie-breaking

@dataclass
class AssembledItem:
    """
    Assembled context item with truncation metadata.
    
    Truncation Metadata (Phase 3.2.1):
    - start_trimmed: Always False (HEAD truncation preserves start).
    - end_trimmed: True if content was truncated, False otherwise.
    """
    role: str
    content: str
    start_trimmed: bool
    end_trimmed: bool
    original_id: str
    type: str # "message" or "expansion"
    token_estimate: int

@dataclass
class AssemblyReceipt:
    budget_used: Dict[str, int]
    items_included: List[str]  # IDs
    items_excluded: List[Dict[str, Any]]  # {id, reason}
    final_assemblage_hash: str
    token_usage_total: int
    success: bool
    failure_reason: Optional[str] = None
    # Phase 3.2.3: Track what's in working set vs pointer set
    working_set: List[str] = field(default_factory=list)  # IDs of items with full content
    pointer_set: List[str] = field(default_factory=list)  # IDs of items referenced but not included
    # Phase 3.2.4: Corpus snapshot for deterministic replay
    corpus_snapshot_id: Optional[str] = None  # Hash of CORTEX index + symbol registry state

class ContextAssembler:
    
    def __init__(self, token_estimator: Optional[Callable[[str], int]] = None):
        """
        Args:
            token_estimator: Function str -> int. Defaults to len(s)//4.
        """
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)

    def assemble(
        self,
        messages: List[ContextMessage],
        expansions: List[ContextExpansion],
        budget: ContextBudget,
        corpus_snapshot_id: Optional[str] = None
    ) -> Tuple[List[AssembledItem], AssemblyReceipt]:
        
        # 0. Initialize
        included_items = []
        excluded_items_info = []
        current_tokens = 0
        current_messages_count = 0
        current_expansions_count = 0
        
        # Helper: Deterministic sort
        # Primary: priority tier (handled by logic flow)
        # Secondary: recency (created_at DESC)
        # Tie-breaker: id ASC
        
        # 1. Identify Items
        system_msgs = [m for m in messages if m.source == "SYSTEM"]
        # Take the FIRST system message if multiple, or fail? 
        # Typically there's one. If multiple, assume the first one defined is the "main" one.
        # Deterministic: Sort by created_at, id.
        system_msgs.sort(key=lambda m: (m.created_at, m.id))
        system_prompt = system_msgs[0] if system_msgs else None
        
        user_msgs = [m for m in messages if m.source == "USER"]
        user_msgs.sort(key=lambda m: (m.created_at, m.id))
        latest_user_msg = user_msgs[-1] if user_msgs else None
        
        if not latest_user_msg:
             # Edge case: No user message? 
             # Allowed? "Mandatory: system prompt (if present), latest user message"
             # If no user message, maybe it's a pure system run?
             # Fail close if implied required. But let's assume it's valid to have none, though weird.
             pass

        # Recent Dialog: All messages EXCEPT system_prompt and latest_user_msg
        # Sort by Recency (Reverse Chronological) for selection
        other_messages = [
            m for m in messages 
            if m.id != (system_prompt.id if system_prompt else None)
            and m.id != (latest_user_msg.id if latest_user_msg else None)
        ]
        other_messages.sort(key=lambda m: (m.created_at, m.id), reverse=True) # Newest first
        
        # Expansions
        explicit_expansions = [e for e in expansions if e.is_explicit_reference]
        # Sort explicit: by symbol_id (stable)
        explicit_expansions.sort(key=lambda e: e.symbol_id)
        
        optional_expansions = [e for e in expansions if not e.is_explicit_reference]
        # Sort optional: by priority DESC, then symbol_id ASC
        optional_expansions.sort(key=lambda e: (-e.priority, e.symbol_id))

        # 2. Token Estimation & Truncation Prep
        # We need to truncate items *before* checking if they fit the TOTAL budget,
        # checking against their INDIVIDUAL budgets.
        
        def prepare_item(item: Union[ContextMessage, ContextExpansion]) -> AssembledItem:
            """
            Prepare item for assembly: estimate tokens, truncate if needed.
            
            Phase 3.2.1: Only HEAD truncation (preserve start, discard end).
            Uses binary search to find character cut point that fits max_tokens.
            """
            content = item.content
            is_msg = isinstance(item, ContextMessage)
            max_tokens = budget.max_tokens_per_message if is_msg else budget.max_tokens_per_expansion
            
            raw_tokens = self.token_estimator(content)
            final_content = content
            trimmed_start = False  # Always False in Phase 3.2.1 (HEAD truncation)
            trimmed_end = False
            
            if raw_tokens > max_tokens:
                # HEAD truncation: Binary search to find character cut point
                target = max_tokens
                low = 0
                high = len(content)
                best_cut = 0
                
                # 10 iterations sufficient for convergence on typical text
                for _ in range(10): 
                    mid = (low + high) // 2
                    s = content[:mid]
                    t = self.token_estimator(s)
                    if t <= target:
                        best_cut = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                
                final_content = content[:best_cut]
                trimmed_end = True
                
            est = self.token_estimator(final_content)
            
            # Phase 3.2.1: Expansions always assigned role="SYSTEM"
            role = item.source if is_msg else "SYSTEM"
            original_id = item.id if is_msg else item.symbol_id
            type_str = "message" if is_msg else "expansion"
            
            return AssembledItem(
                role=role,
                content=final_content,
                start_trimmed=trimmed_start,
                end_trimmed=trimmed_end,
                original_id=original_id,
                type=type_str,
                token_estimate=est
            )

        # 3. Selection
        
        final_payload_list = []
        
        available_tokens = budget.max_total_tokens - budget.reserve_response_tokens
        
        # Tier 1: Mandatory
        tier1_items = []
        if system_prompt:
            tier1_items.append(prepare_item(system_prompt))
        if latest_user_msg:
            tier1_items.append(prepare_item(latest_user_msg))
            
        cost_tier1 = sum(x.token_estimate for x in tier1_items)
        if cost_tier1 > available_tokens:
             return [], AssemblyReceipt(
                 budget_used=asdict(budget), items_included=[], items_excluded=[],
                 final_assemblage_hash="", token_usage_total=0, success=False,
                 failure_reason="Mandatory items exceed budget",
                 working_set=[], pointer_set=[], corpus_snapshot_id=corpus_snapshot_id
             )
        
        current_tokens += cost_tier1
        final_payload_list.extend(tier1_items)
        current_messages_count += len(tier1_items)
        
        # Tier 2 vs 3?
        # Priority Rules: 1. Mandatory, 2. Recent Dialog, 3. Explicit Expansions.
        # Wait. "2. Recent dialog ... 3. Explicit symbol expansions".
        # This means I fill dialog FIRST, then expansions?
        # If I fill dialog, I might starve expansions.
        # Is that the intent? "Priority Rules (NO DEVIATION)".
        # Yes.
        
        # Tier 2: Recent Dialog
        tier2_candidates = [prepare_item(m) for m in other_messages]
        # These are sorted Newest -> Oldest.
        # We add them until budget full or max messages.
        
        tier2_selected = []
        
        for item in tier2_candidates:
            if current_messages_count >= budget.max_messages:
                excluded_items_info.append({"id": item.original_id, "reason": "Max messages limit"})
                continue
                
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": item.original_id, "reason": "Token budget limit"})
                continue
                
            tier2_selected.append(item)
            current_tokens += item.token_estimate
            current_messages_count += 1
            
        # Tier 3: Explicit Expansions
        tier3_candidates = [prepare_item(e) for e in explicit_expansions]
        tier3_selected = []
        
        for item in tier3_candidates:
            if current_expansions_count >= budget.max_expansions:
                excluded_items_info.append({"id": item.original_id, "reason": "Max expansions limit"})
                continue
            
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": item.original_id, "reason": "Token budget limit"})
                continue
                
            tier3_selected.append(item)
            current_tokens += item.token_estimate
            current_expansions_count += 1
            
        # Tier 4: Optional Extras
        tier4_candidates = [prepare_item(e) for e in optional_expansions]
        # Only if budget remains.
        tier4_selected = []
        
        for item in tier4_candidates:
            if current_expansions_count >= budget.max_expansions:
                excluded_items_info.append({"id": item.original_id, "reason": "Max expansions limit"})
                continue
            
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": item.original_id, "reason": "Token budget limit"})
                continue
            
            tier4_selected.append(item)
            current_tokens += item.token_estimate
            current_expansions_count += 1

        # 4. Assembly & Ordering
        # "Ordering: Primary: priority tier? No, Wait."
        # "Ordering & Tie-breakers: Primary: priority tier, Secondary: recency"
        # The priorities were for SELECTION.
        # The Final Order in the context payload usually must be chronological to make sense to the model.
        # "Explicit ordering: all ordering decisions must have deterministic tie-breakers".
        
        # I need to construct the final list.
        # Structure:
        # 1. System Prompt (if any)
        # 2. Expansions (Explicit & Optional) - grouped?
        # 3. Dialog History (Oldest -> Newest)
        # 4. Latest User Message
        
        # This is a standard "Chat" topology.
        
        final_list = []
        
        # A. System (from Tier 1)
        if system_prompt:
             final_list.append([x for x in tier1_items if x.original_id == system_prompt.id][0])
            
        # B. Expansions (Tier 3 + Tier 4)
        # Sort by deterministic order?
        # Maybe explicit first, then optional?
        # Inside explicit: sorted by symbol_id (already done in candidate list)
        final_list.extend(tier3_selected)
        final_list.extend(tier4_selected)
        
        # C. Dialog History (Tier 2) - REVERSE (Oldest -> Newest)
        # tier2_selected matches `other_messages` which was Newest->Oldest.
        # So reverse it back to Oldest->Newest.
        final_list.extend(reversed(tier2_selected))
        
        # D. Latest User (Tier 1)
        if latest_user_msg:
             final_list.append([x for x in tier1_items if x.original_id == latest_user_msg.id][0])

        
        # 5. Receipt
        payload_str = json.dumps([asdict(x) for x in final_list], sort_keys=True)
        final_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        # Phase 3.2.3: Build working_set (included with content) and pointer_set (excluded but referenced)
        working_set = [x.original_id for x in final_list]
        pointer_set = [item["id"] for item in excluded_items_info]

        receipt = AssemblyReceipt(
            budget_used=asdict(budget),
            items_included=[x.original_id for x in final_list],
            items_excluded=excluded_items_info,
            final_assemblage_hash=final_hash,
            token_usage_total=current_tokens,
            success=True,
            working_set=working_set,
            pointer_set=pointer_set,
            corpus_snapshot_id=corpus_snapshot_id
        )

        return final_list, receipt
