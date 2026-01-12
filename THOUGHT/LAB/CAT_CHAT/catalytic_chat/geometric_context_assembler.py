"""
Geometric Context Assembler - I.2 CAT Chat Integration

Context assembler with geometric relevance scoring.
Extends ContextAssembler (pattern from GeometricCassette extending DatabaseCassette).

Preserves 4-tier priority system (CAT Chat relies on this).
Adds E-scoring as tie-breaker within each tier.

E-Scoring Strategy:
1. Compute E(query, item) for each item
2. Within each tier, sort by E DESC
3. Apply token budget as normal
4. Return geometric receipt with E distribution
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# Import base context assembler
from catalytic_chat.context_assembler import (
    ContextAssembler,
    ContextBudget,
    ContextMessage,
    ContextExpansion,
    AssembledItem,
    AssemblyReceipt
)

# Add CAPABILITY to path for imports
CAPABILITY_PATH = Path(__file__).parent.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

try:
    from geometric_reasoner import (
        GeometricReasoner,
        GeometricState,
    )
except ImportError:
    GeometricReasoner = None
    GeometricState = None


@dataclass
class GeometricAssemblyReceipt(AssemblyReceipt):
    """
    Extended receipt with quantum metrics.

    Inherits from AssemblyReceipt and adds:
    - query_Df: Participation ratio of query
    - mean_E: Mean E (Born rule) across included items
    - max_E: Maximum E value
    - gate_open: Whether mean_E exceeds threshold
    - E_distribution: Per-item E values
    - conversation_Df: Optional conversation state Df
    """
    query_Df: float = 0.0
    mean_E: float = 0.0
    max_E: float = 0.0
    gate_open: bool = False
    E_distribution: List[float] = field(default_factory=list)
    conversation_Df: Optional[float] = None
    E_threshold: float = 0.5


class GeometricContextAssembler(ContextAssembler):
    """
    Context assembler with geometric relevance scoring.

    Extends ContextAssembler (pattern from GeometricCassette extending DatabaseCassette).

    Preserves 4-tier priority system (CAT Chat relies on this):
    - Tier 1: Mandatory (system prompt, latest user message)
    - Tier 2: Recent Dialog (newest first)
    - Tier 3: Explicit Expansions
    - Tier 4: Optional Expansions

    Adds E-scoring as tie-breaker within each tier:
    - Compute E(query_state, item_state) for each item
    - Within each tier, sort by E DESC before token budget check
    - Track geometric metrics in receipt

    Usage:
        assembler = GeometricContextAssembler()

        items, receipt = assembler.assemble_with_geometry(
            messages=messages,
            expansions=expansions,
            budget=budget,
            query_state=query_state
        )

        print(f"Mean E: {receipt.mean_E:.3f}, Gate: {receipt.gate_open}")
    """

    def __init__(
        self,
        token_estimator: Optional[Callable[[str], int]] = None,
        model_name: str = 'all-MiniLM-L6-v2',
        E_threshold: float = 0.5
    ):
        """
        Initialize geometric context assembler.

        Args:
            token_estimator: Function str -> int (default: len(s)//4)
            model_name: Sentence transformer model for embeddings
            E_threshold: Threshold for E-gating (default 0.5)
        """
        super().__init__(token_estimator)

        # Lazy init (pattern from geometric_cassette.py:128-138)
        self._reasoner: Optional[GeometricReasoner] = None
        self._model_name = model_name
        self.E_threshold = E_threshold

        # Stats (pattern from geometric_cassette.py:119-126)
        self._geo_stats = {
            'assemblies': 0,
            'assemblies_geometric': 0,
            'embedding_calls': 0,
            'geometric_ops': 0
        }

    @property
    def reasoner(self) -> 'GeometricReasoner':
        """Lazy init (pattern from geometric_cassette.py:128-138)"""
        if self._reasoner is None:
            if GeometricReasoner is None:
                raise ImportError(
                    "GeometricReasoner not available. "
                    "Install: pip install sentence-transformers"
                )
            self._reasoner = GeometricReasoner(self._model_name)
        return self._reasoner

    def assemble_with_geometry(
        self,
        messages: List[ContextMessage],
        expansions: List[ContextExpansion],
        budget: ContextBudget,
        query_state: 'GeometricState',
        conversation_state: Optional['GeometricState'] = None
    ) -> Tuple[List[AssembledItem], GeometricAssemblyReceipt]:
        """
        Assemble context with E-scoring within tiers.

        E-scoring strategy:
        1. Compute E(query_state, item_state) for each item
        2. Within each tier, sort by E DESC before token budget check
        3. Apply token budget as normal
        4. Return geometric receipt with E distribution

        Args:
            messages: List of ContextMessage
            expansions: List of ContextExpansion
            budget: ContextBudget
            query_state: GeometricState from user query
            conversation_state: Optional accumulated conversation state

        Returns:
            Tuple of (items, GeometricAssemblyReceipt)
        """
        self._geo_stats['assemblies'] += 1
        self._geo_stats['assemblies_geometric'] += 1

        # Compute E values for all items
        message_E = self._compute_message_E_values(messages, query_state)
        expansion_E = self._compute_expansion_E_values(expansions, query_state)

        # Create E-sorted versions
        messages_with_E = self._attach_E_to_messages(messages, message_E)
        expansions_with_E = self._attach_E_to_expansions(expansions, expansion_E)

        # Run base assembly with E-enhanced sorting
        items, base_receipt = self._assemble_with_E_tiebreak(
            messages_with_E,
            expansions_with_E,
            budget
        )

        # Compute geometric metrics for included items
        included_E_values = self._get_E_for_included(items, message_E, expansion_E)
        mean_E = sum(included_E_values) / len(included_E_values) if included_E_values else 0.0
        max_E = max(included_E_values) if included_E_values else 0.0

        # Build geometric receipt
        geo_receipt = GeometricAssemblyReceipt(
            budget_used=base_receipt.budget_used,
            items_included=base_receipt.items_included,
            items_excluded=base_receipt.items_excluded,
            final_assemblage_hash=base_receipt.final_assemblage_hash,
            token_usage_total=base_receipt.token_usage_total,
            success=base_receipt.success,
            failure_reason=base_receipt.failure_reason,
            query_Df=query_state.Df,
            mean_E=mean_E,
            max_E=max_E,
            gate_open=mean_E >= self.E_threshold,
            E_distribution=included_E_values,
            conversation_Df=conversation_state.Df if conversation_state else None,
            E_threshold=self.E_threshold
        )

        return items, geo_receipt

    # ========================================================================
    # E-Value Computation
    # ========================================================================

    def _compute_message_E_values(
        self,
        messages: List[ContextMessage],
        query_state: 'GeometricState'
    ) -> Dict[str, float]:
        """Compute E(query, message) for each message."""
        E_values = {}

        for msg in messages:
            if msg.content and msg.content.strip():
                msg_state = self.reasoner.initialize(msg.content)
                self._geo_stats['embedding_calls'] += 1

                E = query_state.E_with(msg_state)
                self._geo_stats['geometric_ops'] += 1

                E_values[msg.id] = E

        return E_values

    def _compute_expansion_E_values(
        self,
        expansions: List[ContextExpansion],
        query_state: 'GeometricState'
    ) -> Dict[str, float]:
        """Compute E(query, expansion) for each expansion."""
        E_values = {}

        for exp in expansions:
            if exp.content and exp.content.strip():
                exp_state = self.reasoner.initialize(exp.content)
                self._geo_stats['embedding_calls'] += 1

                E = query_state.E_with(exp_state)
                self._geo_stats['geometric_ops'] += 1

                E_values[exp.symbol_id] = E

        return E_values

    def _attach_E_to_messages(
        self,
        messages: List[ContextMessage],
        E_values: Dict[str, float]
    ) -> List[Tuple[ContextMessage, float]]:
        """Attach E values to messages for sorting."""
        return [(msg, E_values.get(msg.id, 0.0)) for msg in messages]

    def _attach_E_to_expansions(
        self,
        expansions: List[ContextExpansion],
        E_values: Dict[str, float]
    ) -> List[Tuple[ContextExpansion, float]]:
        """Attach E values to expansions for sorting."""
        return [(exp, E_values.get(exp.symbol_id, 0.0)) for exp in expansions]

    def _get_E_for_included(
        self,
        items: List[AssembledItem],
        message_E: Dict[str, float],
        expansion_E: Dict[str, float]
    ) -> List[float]:
        """Get E values for included items."""
        E_values = []
        for item in items:
            if item.type == "message":
                E_values.append(message_E.get(item.original_id, 0.0))
            else:
                E_values.append(expansion_E.get(item.original_id, 0.0))
        return E_values

    # ========================================================================
    # E-Enhanced Assembly
    # ========================================================================

    def _assemble_with_E_tiebreak(
        self,
        messages_with_E: List[Tuple[ContextMessage, float]],
        expansions_with_E: List[Tuple[ContextExpansion, float]],
        budget: ContextBudget
    ) -> Tuple[List[AssembledItem], AssemblyReceipt]:
        """
        Run assembly with E as tie-breaker within tiers.

        Tier order preserved: Mandatory → Dialog → Explicit → Optional
        Within each tier: sort by E DESC
        """
        included_items = []
        excluded_items_info = []
        current_tokens = 0
        current_messages_count = 0
        current_expansions_count = 0

        # Extract messages and expansions
        messages = [m for m, _ in messages_with_E]
        E_lookup_msg = {m.id: e for m, e in messages_with_E}

        expansions = [e for e, _ in expansions_with_E]
        E_lookup_exp = {e.symbol_id: e_val for e, e_val in expansions_with_E}

        # Tier 1: Mandatory items (system prompt, latest user)
        system_msgs = [m for m in messages if m.source == "SYSTEM"]
        system_msgs.sort(key=lambda m: (m.created_at, m.id))
        system_prompt = system_msgs[0] if system_msgs else None

        user_msgs = [m for m in messages if m.source == "USER"]
        user_msgs.sort(key=lambda m: (m.created_at, m.id))
        latest_user_msg = user_msgs[-1] if user_msgs else None

        available_tokens = budget.max_total_tokens - budget.reserve_response_tokens

        # Prepare tier 1 items
        tier1_items = []
        if system_prompt:
            tier1_items.append(self._prepare_item(system_prompt, budget))
        if latest_user_msg:
            tier1_items.append(self._prepare_item(latest_user_msg, budget))

        cost_tier1 = sum(x.token_estimate for x in tier1_items)
        if cost_tier1 > available_tokens:
            return [], AssemblyReceipt(
                budget_used=asdict(budget),
                items_included=[],
                items_excluded=[],
                final_assemblage_hash="",
                token_usage_total=0,
                success=False,
                failure_reason="Mandatory items exceed budget"
            )

        current_tokens += cost_tier1
        included_items.extend(tier1_items)
        current_messages_count += len(tier1_items)

        # Tier 2: Recent dialog (sorted by recency, tie-break by E DESC)
        other_messages = [
            m for m in messages
            if m.id != (system_prompt.id if system_prompt else None)
            and m.id != (latest_user_msg.id if latest_user_msg else None)
        ]
        # Sort: newest first, then by E DESC for ties
        other_messages.sort(
            key=lambda m: (m.created_at, m.id, -E_lookup_msg.get(m.id, 0)),
            reverse=True
        )

        tier2_selected = []
        for msg in other_messages:
            if current_messages_count >= budget.max_messages:
                excluded_items_info.append({"id": msg.id, "reason": "Max messages limit"})
                continue

            item = self._prepare_item(msg, budget)
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": msg.id, "reason": "Token budget limit"})
                continue

            tier2_selected.append(item)
            current_tokens += item.token_estimate
            current_messages_count += 1

        # Tier 3: Explicit expansions (sorted by symbol_id, tie-break by E DESC)
        explicit_expansions = [e for e in expansions if e.is_explicit_reference]
        explicit_expansions.sort(
            key=lambda e: (e.symbol_id, -E_lookup_exp.get(e.symbol_id, 0))
        )

        tier3_selected = []
        for exp in explicit_expansions:
            if current_expansions_count >= budget.max_expansions:
                excluded_items_info.append({"id": exp.symbol_id, "reason": "Max expansions limit"})
                continue

            item = self._prepare_expansion_item(exp, budget)
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": exp.symbol_id, "reason": "Token budget limit"})
                continue

            tier3_selected.append(item)
            current_tokens += item.token_estimate
            current_expansions_count += 1

        # Tier 4: Optional expansions (sorted by priority DESC, E DESC)
        optional_expansions = [e for e in expansions if not e.is_explicit_reference]
        optional_expansions.sort(
            key=lambda e: (-e.priority, -E_lookup_exp.get(e.symbol_id, 0), e.symbol_id)
        )

        tier4_selected = []
        for exp in optional_expansions:
            if current_expansions_count >= budget.max_expansions:
                excluded_items_info.append({"id": exp.symbol_id, "reason": "Max expansions limit"})
                continue

            item = self._prepare_expansion_item(exp, budget)
            if current_tokens + item.token_estimate > available_tokens:
                excluded_items_info.append({"id": exp.symbol_id, "reason": "Token budget limit"})
                continue

            tier4_selected.append(item)
            current_tokens += item.token_estimate
            current_expansions_count += 1

        # Final ordering: System → Expansions → Dialog (oldest→newest) → Latest User
        final_list = []

        # A. System prompt
        if system_prompt:
            final_list.append([x for x in tier1_items if x.original_id == system_prompt.id][0])

        # B. Expansions (Tier 3 + Tier 4)
        final_list.extend(tier3_selected)
        final_list.extend(tier4_selected)

        # C. Dialog history (reverse to oldest→newest)
        final_list.extend(reversed(tier2_selected))

        # D. Latest user message
        if latest_user_msg:
            final_list.append([x for x in tier1_items if x.original_id == latest_user_msg.id][0])

        # Build receipt
        payload_str = json.dumps([asdict(x) for x in final_list], sort_keys=True)
        final_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        receipt = AssemblyReceipt(
            budget_used=asdict(budget),
            items_included=[x.original_id for x in final_list],
            items_excluded=excluded_items_info,
            final_assemblage_hash=final_hash,
            token_usage_total=current_tokens,
            success=True
        )

        return final_list, receipt

    def _prepare_item(
        self,
        msg: ContextMessage,
        budget: ContextBudget
    ) -> AssembledItem:
        """Prepare message item (reuse from base class logic)."""
        content = msg.content
        max_tokens = budget.max_tokens_per_message

        raw_tokens = self.token_estimator(content)
        final_content = content
        trimmed_end = False

        if raw_tokens > max_tokens:
            # HEAD truncation (binary search)
            low, high = 0, len(content)
            best_cut = 0
            for _ in range(10):
                mid = (low + high) // 2
                if self.token_estimator(content[:mid]) <= max_tokens:
                    best_cut = mid
                    low = mid + 1
                else:
                    high = mid - 1
            final_content = content[:best_cut]
            trimmed_end = True

        return AssembledItem(
            role=msg.source,
            content=final_content,
            start_trimmed=False,
            end_trimmed=trimmed_end,
            original_id=msg.id,
            type="message",
            token_estimate=self.token_estimator(final_content)
        )

    def _prepare_expansion_item(
        self,
        exp: ContextExpansion,
        budget: ContextBudget
    ) -> AssembledItem:
        """Prepare expansion item."""
        content = exp.content
        max_tokens = budget.max_tokens_per_expansion

        raw_tokens = self.token_estimator(content)
        final_content = content
        trimmed_end = False

        if raw_tokens > max_tokens:
            low, high = 0, len(content)
            best_cut = 0
            for _ in range(10):
                mid = (low + high) // 2
                if self.token_estimator(content[:mid]) <= max_tokens:
                    best_cut = mid
                    low = mid + 1
                else:
                    high = mid - 1
            final_content = content[:best_cut]
            trimmed_end = True

        return AssembledItem(
            role="SYSTEM",
            content=final_content,
            start_trimmed=False,
            end_trimmed=trimmed_end,
            original_id=exp.symbol_id,
            type="expansion",
            token_estimate=self.token_estimator(final_content)
        )

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict:
        """Return assembler statistics including geometric metrics."""
        return {
            **self._geo_stats,
            'reasoner_stats': self.reasoner.get_stats() if self._reasoner else {}
        }

    def supports_geometric(self) -> bool:
        """Check if assembler supports geometric operations."""
        return True
