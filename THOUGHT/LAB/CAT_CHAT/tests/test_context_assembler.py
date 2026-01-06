"""
Tests for CAT_CHAT Context Assembler
"""
import pytest
import json
from dataclasses import asdict
from catalytic_chat.context_assembler import (
    ContextAssembler, 
    ContextBudget, 
    ContextMessage, 
    ContextExpansion
)

def mock_token_estimator(text):
    # simple 1 char = 1 token for easy math
    return len(text)

@pytest.fixture
def assembler():
    return ContextAssembler(token_estimator=mock_token_estimator)

@pytest.fixture
def default_budget():
    return ContextBudget(
        max_total_tokens=100,
        reserve_response_tokens=10, # Available = 90
        max_messages=5,
        max_expansions=3,
        max_tokens_per_message=20,
        max_tokens_per_expansion=20
    )

def test_assembler_determinism(assembler, default_budget):
    messages = [
        ContextMessage(id="sys", source="SYSTEM", content="System", created_at="2025-01-01T00:00:00Z"),
        ContextMessage(id="u1", source="USER", content="User 1", created_at="2025-01-01T00:01:00Z")
    ]
    expansions = []
    
    items1, receipt1 = assembler.assemble(messages, expansions, default_budget)
    items2, receipt2 = assembler.assemble(messages, expansions, default_budget)
    
    assert receipt1.final_assemblage_hash == receipt2.final_assemblage_hash
    assert asdict(items1[0]) == asdict(items2[0])

def test_fail_closed_mandatory(assembler, default_budget):
    # System prompt > budget
    # budget avail = 90. max_per_msg = 20.
    # If msg > 20, it gets truncated to 20.
    # So to fail closed, truncated size must likely still be an issue? 
    # Or strict budget check?
    # The fail-closed condition is "if required elements cannot fit within budget".
    # But truncation happens first. So if truncated size fits, it passes.
    # Failure usually implies: Budget is so small even truncated mandatory items don't fit.
    
    tiny_budget = ContextBudget(
        max_total_tokens=5, # very small
        reserve_response_tokens=1, 
        max_messages=5,
        max_expansions=3,
        max_tokens_per_message=10, 
        max_tokens_per_expansion=10
    )
    
    messages = [
        ContextMessage(id="sys", source="SYSTEM", content="12345", created_at="2025-01-01T00:00:00Z"), # 5 tokens
        ContextMessage(id="u1", source="USER", content="67890", created_at="2025-01-01T00:01:00Z")   # 5 tokens
    ]
    # Total 10 tokens. Avail 4. 
    # Should fail.
    
    items, receipt = assembler.assemble(messages, [], tiny_budget)
    assert not receipt.success
    assert "Mandatory items exceed budget" in receipt.failure_reason

def test_priority_and_truncation(assembler, default_budget):
    # Budget: 90 avail.
    # Msg Limit: 20.
    
    # 1. System (Mandatory)
    sys = ContextMessage(id="sys", source="SYSTEM", content="A" * 10, created_at="T0") # 10t
    
    # 2. Latest User (Mandatory)
    last = ContextMessage(id="last", source="USER", content="B" * 50, created_at="T9") 
    # 50t > 20 limit. Truncated to 20t.
    # Total Mandatory = 10 + 20 = 30t. Remaining = 60t.
    
    # 3. Recent Dialog (Priority 2)
    # Newest First.
    hist1 = ContextMessage(id="h1", source="ASSISTANT", content="C" * 10, created_at="T8") # 10t
    hist2 = ContextMessage(id="h2", source="USER", content="D" * 30, created_at="T7")      # 30t->20t (trunc)
    hist3 = ContextMessage(id="h3", source="ASSISTANT", content="E" * 10, created_at="T6") # 10t
    
    # 4. Explicit Expansion (Priority 3)
    exp1 = ContextExpansion(symbol_id="@ex1", content="F" * 10, is_explicit_reference=True) # 10t
    
    # Messages list
    msgs = [sys, hist3, hist2, hist1, last] # shuffled order input
    exps = [exp1]
    
    # Expected Logic:
    # Mandatory: Sys(10) + Last(20) = 30 used. 60 left.
    # Tier 2 (Recent): 
    #   h1 (10) -> fit. Used 40. Left 50.
    #   h2 (20 truncated) -> fit. Used 60. Left 30.
    #   h3 (10) -> fit. Used 70. Left 20.
    # Tier 3 (Explicit):
    #   exp1 (10) -> fit. Used 80. Left 10.
    
    items, receipt = assembler.assemble(msgs, exps, default_budget)
    
    assert receipt.success
    assert receipt.token_usage_total == 80
    
    # Verify Content & Truncation
    # Last user msg should be "B"*20 (truncated 50->20)
    last_item = next(x for x in items if x.original_id == "last")
    assert len(last_item.content) < 25 # it's around 20 depending on binary search
    assert last_item.content.startswith("BBBB")
    assert last_item.role == "USER"
    
    # Verify Order in payload
    # Expected: System -> Expansions -> History (Oldest->Newest) -> Latest
    ids = [x.original_id for x in items]
    
    # Check subsequence
    assert ids[0] == "sys"
    assert "@ex1" in ids # Expansion
    assert ids[-1] == "last"
    
    # History check: h3 (Oldest), h2, h1 (Newest)
    # Because internal selection was h1, h2, h3 (Newest -> Oldest)
    # Logic reverses it back.
    # So we expect [sys, @ex1, h3, h2, h1, last]
    expected_order = ["sys", "@ex1", "h3", "h2", "h1", "last"]
    assert ids == expected_order

def test_starvation_of_expansions(assembler, default_budget):
    # Ensure High Priority Dialog starves Lower Priority Expansions if needed?
    # Wait, Prompt says: "1. Mandatory", "2. Recent dialog", "3. Explicit symbol".
    # So Dialog DOES starve expansions.
    
    # Budget 90.
    sys = ContextMessage(id="sys", source="SYSTEM", content="A"*10, created_at="T0") # 10
    last = ContextMessage(id="uLast", source="USER", content="B"*10, created_at="T9") # 10
    # Mandatory = 20. Left = 70.
    
    # Dialog (Priority 2) - Fill it up
    # 4 messages of 20 tokens = 80 tokens.
    # Only 3 will fit (3*20=60). Total 80 used. Left 10.
    hist = [
        ContextMessage(id=f"h{i}", source="USER", content="C"*20, created_at=f"T{i+1}")
        for i in range(4)
    ]
    # h3(Newest)..h0(Oldest).
    
    # Expansions (Priority 3)
    exp = ContextExpansion(symbol_id="@ex1", content="D"*15, is_explicit_reference=True) # 15t
    # Needs 15. Only 10 left. Should be excluded.
    
    items, receipt = assembler.assemble([sys, last] + hist, [exp], default_budget)
    
    ids = [x.original_id for x in items]
    assert "@ex1" not in ids
    
    # Check exclusions
    assert any(e['id'] == '@ex1' and e['reason'] == 'Token budget limit' for e in receipt.items_excluded)

def test_max_messages_limit(assembler):
    budget = ContextBudget(
        max_total_tokens=1000,
        reserve_response_tokens=0,
        max_messages=3, # Strict limit
        max_expansions=10,
        max_tokens_per_message=100,
        max_tokens_per_expansion=100
    )
    
    sys = ContextMessage(id="sys", source="SYSTEM", content="S", created_at="T0")
    last = ContextMessage(id="last", source="USER", content="L", created_at="T9")
    
    # 2 Mandatory messages consumed. 1 slot left.
    
    # 2 History messages
    h1 = ContextMessage(id="h1", source="USER", content="H1", created_at="T8") # Newest
    h2 = ContextMessage(id="h2", source="USER", content="H2", created_at="T7")
    
    items, receipt = assembler.assemble([sys, last, h1, h2], [], budget)
    
    ids = [x.original_id for x in items]
    assert "h1" in ids # Newest fits
    assert "h2" not in ids # Oldest excluded by max_messages
    assert len(items) == 3
