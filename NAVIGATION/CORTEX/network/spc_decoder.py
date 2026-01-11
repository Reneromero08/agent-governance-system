#!/usr/bin/env python3
"""
SPC Decoder - Semantic Pointer Compression decoder.

Implements deterministic pointer resolution per SPC_SPEC.md.

Pointer Types:
- SYMBOL_PTR: Single radical (e.g., "C", "I", "V") or CJK glyph (e.g., "法", "真")
- HASH_PTR: Content-addressed (e.g., "sha256:abc123...")
- COMPOSITE_PTR: Radical + number/operator/context (e.g., "C3", "I5:build", "C&I", "法.驗")

Operators (per SPC_SPEC Section 2.3):
- . PATH/ACCESS (法.驗)
- : CONTEXT/TYPE (C3:build)
- * ALL (C*)
- ! NOT/DENY (V!)
- ? CHECK/QUERY (J?)
- & AND/BIND (C&I)
- | OR/CHOICE (C|I)

Reference:
- LAW/CANON/SEMANTIC/SPC_SPEC.md
- LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md
- THOUGHT/LAB/FORMULA/research/questions/high_priority/q35_markov_blankets.md
- THOUGHT/LAB/FORMULA/research/questions/medium_priority/q33_conditional_entropy_semantic_density.md
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable


# Load codebook
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"

# CAS lookup callback (set by memory_cassette integration)
_cas_lookup_fn: Optional[Callable[[str], Optional[Dict]]] = None


class PointerType(Enum):
    """SPC pointer types."""
    SYMBOL_PTR = "symbol"
    HASH_PTR = "hash"
    COMPOSITE_PTR = "composite"


class ErrorCode(Enum):
    """Fail-closed error codes per SPC_SPEC Section 4."""
    E_CODEBOOK_MISMATCH = "E_CODEBOOK_MISMATCH"
    E_KERNEL_VERSION = "E_KERNEL_VERSION"
    E_TOKENIZER_MISMATCH = "E_TOKENIZER_MISMATCH"
    E_SYNTAX = "E_SYNTAX"
    E_UNKNOWN_SYMBOL = "E_UNKNOWN_SYMBOL"
    E_HASH_NOT_FOUND = "E_HASH_NOT_FOUND"
    E_AMBIGUOUS = "E_AMBIGUOUS"
    E_INVALID_OPERATOR = "E_INVALID_OPERATOR"
    E_INVALID_QUALIFIER = "E_INVALID_QUALIFIER"
    E_RULE_NOT_FOUND = "E_RULE_NOT_FOUND"
    E_CONTEXT_REQUIRED = "E_CONTEXT_REQUIRED"
    E_EXPANSION_FAILED = "E_EXPANSION_FAILED"
    E_SCHEMA_VIOLATION = "E_SCHEMA_VIOLATION"
    E_CAS_UNAVAILABLE = "E_CAS_UNAVAILABLE"


@dataclass
class FailClosed:
    """Fail-closed decode result."""
    error_code: ErrorCode
    error_detail: str
    pointer: str
    timestamp: str


@dataclass
class DecodeSuccess:
    """Successful decode result."""
    ir: Dict
    token_receipt: Dict


class SPCDecoder:
    """Deterministic SPC pointer decoder.

    Decodes SPC pointers to canonical IR per SPC_SPEC.md.
    Implements fail-closed semantics: any error returns FailClosed,
    never silent failure or best-effort decoding.

    Per Q35 (Markov Blankets): Decoding requires aligned blankets.
    Per Q33 (Semantic Density): CDR = concept_units / tokens(pointer).
    """

    KERNEL_VERSION = "1.0.0"
    TOKENIZER_ID = "tiktoken/o200k_base"

    # Regex patterns
    HASH_PATTERN = re.compile(r'^sha256:([a-f0-9]{16,64})$')

    # ASCII radicals per SPC_SPEC Appendix B
    ASCII_RADICALS = set("CIVLGSRAJP")

    # CJK glyphs per SPC_SPEC Appendix A (single-token symbols)
    CJK_GLYPHS = {
        "法": {"domain": "Law", "path": "LAW/CANON", "tokens": 1},
        "真": {"domain": "Truth", "path": "LAW/CANON/FOUNDATION/THE_SEMIOTIC_FOUNDATION_OF_TRUTH.md", "tokens": 1},
        "契": {"domain": "Contract", "path": "LAW/CANON/CONSTITUTION/CONTRACT.md", "tokens": 1},
        "恆": {"domain": "Invariant", "path": "LAW/CANON/CONSTITUTION/INVARIANTS.md", "tokens": 1},
        "驗": {"domain": "Verification", "path": "LAW/CANON/GOVERNANCE/VERIFICATION.md", "tokens": 1},
        "證": {"domain": "Receipt", "path": "NAVIGATION/RECEIPTS", "tokens": 1},
        "變": {"domain": "Catalytic", "path": "THOUGHT/LAB/CATALYTIC", "tokens": 1},
        "冊": {"domain": "Cortex", "path": "NAVIGATION/CORTEX/db", "tokens": 1},
        "試": {"domain": "Testbench", "path": "CAPABILITY/TESTBENCH", "tokens": 1},
        "查": {"domain": "Search", "path": "NAVIGATION/CORTEX/semantic", "tokens": 1},
        "道": {"domain": "Path", "path": None, "tokens": 1, "polysemic": True},  # Context-dependent
    }

    # Operators per SPC_SPEC Section 2.3
    OPERATORS = {
        ".": "PATH",      # Path/access
        ":": "CONTEXT",   # Context/type
        "*": "ALL",       # All
        "!": "NOT",       # Not/deny
        "?": "CHECK",     # Check/query
        "&": "AND",       # And/bind
        "|": "OR",        # Or/choice
    }

    # Extended pattern: radical + optional number + optional operator + optional context
    # Matches: C, C3, C*, C!, C?, C&I, C|I, C3:build, L.C.3
    RADICAL_PATTERN = re.compile(
        r'^([CIVLGSRAJP])(\d*)([*!?])?(?:([&|])([CIVLGSRAJP]))?(?::(\w+))?$'
    )

    # Path pattern: base.path.path (e.g., L.C.3, 法.驗)
    PATH_PATTERN = re.compile(r'^(.+?)\.(.+)$')

    def __init__(self, codebook_path: Path = None):
        """Initialize decoder.

        Args:
            codebook_path: Path to CODEBOOK.json (default: project CODEBOOK.json)
        """
        self.codebook_path = codebook_path or CODEBOOK_PATH
        self._codebook = None
        self._codebook_hash = None

    @property
    def codebook(self) -> Dict:
        """Lazy-load codebook."""
        if self._codebook is None:
            self._load_codebook()
        return self._codebook

    @property
    def codebook_hash(self) -> str:
        """Lazy-compute codebook hash."""
        if self._codebook_hash is None:
            self._load_codebook()
        return self._codebook_hash

    def _load_codebook(self):
        """Load and hash codebook."""
        with open(self.codebook_path, 'r', encoding='utf-8') as f:
            self._codebook = json.load(f)
        canonical = json.dumps(self._codebook, sort_keys=True, separators=(',', ':'))
        self._codebook_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def decode(
        self,
        pointer: str,
        context_keys: Dict = None,
        codebook_id: str = "ags-codebook",
        codebook_sha256: str = None,
        kernel_version: str = None
    ) -> Union[DecodeSuccess, FailClosed]:
        """Decode SPC pointer to canonical IR.

        Per CODEBOOK_SYNC_PROTOCOL, verifies sync_tuple before decoding.

        Args:
            pointer: The SPC pointer to decode
            context_keys: Context for disambiguation
            codebook_id: Expected codebook ID
            codebook_sha256: Expected codebook hash (fail-closed on mismatch)
            kernel_version: Expected kernel version

        Returns:
            DecodeSuccess with IR and token_receipt, or FailClosed with error
        """
        context_keys = context_keys or {}
        now = datetime.now(timezone.utc).isoformat()

        # Step 1: Verify codebook hash (Markov blanket check per Q35)
        if codebook_sha256 and codebook_sha256 != self.codebook_hash:
            return FailClosed(
                ErrorCode.E_CODEBOOK_MISMATCH,
                f"Expected {codebook_sha256[:16]}..., got {self.codebook_hash[:16]}...",
                pointer, now
            )

        # Step 2: Verify kernel version
        if kernel_version and kernel_version != self.KERNEL_VERSION:
            return FailClosed(
                ErrorCode.E_KERNEL_VERSION,
                f"Expected {kernel_version}, current is {self.KERNEL_VERSION}",
                pointer, now
            )

        # Step 3: Parse pointer type
        ptr_type, parsed = self._parse_pointer(pointer)
        if isinstance(parsed, FailClosed):
            return parsed

        # Step 4: Resolve based on type
        if ptr_type == PointerType.HASH_PTR:
            return self._resolve_hash_ptr(pointer, parsed, now)
        elif ptr_type == PointerType.SYMBOL_PTR:
            return self._resolve_symbol_ptr(pointer, parsed, context_keys, now)
        else:  # COMPOSITE_PTR
            return self._resolve_composite_ptr(pointer, parsed, context_keys, now)

    def _parse_pointer(self, pointer: str) -> Tuple[PointerType, Union[Dict, FailClosed]]:
        """Parse pointer and determine type.

        Handles:
        - HASH_PTR: sha256:abc123...
        - SYMBOL_PTR: C, I, V, 法, 真, etc.
        - COMPOSITE_PTR: C3, C*, C&I, C3:build, L.C.3, 法.驗
        """
        now = datetime.now(timezone.utc).isoformat()

        # Check for HASH_PTR
        hash_match = self.HASH_PATTERN.match(pointer)
        if hash_match:
            return PointerType.HASH_PTR, {"hash": hash_match.group(1)}

        # Check for CJK glyph (single character)
        if len(pointer) == 1 and pointer in self.CJK_GLYPHS:
            return PointerType.SYMBOL_PTR, {"glyph": pointer, "type": "cjk"}

        # Check for CJK path (e.g., 法.驗)
        if any(c in pointer for c in self.CJK_GLYPHS) and '.' in pointer:
            parts = pointer.split('.')
            return PointerType.COMPOSITE_PTR, {
                "type": "path",
                "parts": parts,
                "base": parts[0],
                "path": parts[1:]
            }

        # Check for ASCII path (e.g., L.C.3)
        if '.' in pointer and pointer[0] in self.ASCII_RADICALS:
            parts = pointer.split('.')
            return PointerType.COMPOSITE_PTR, {
                "type": "path",
                "parts": parts,
                "base": parts[0],
                "path": parts[1:]
            }

        # Check for RADICAL (SYMBOL_PTR or COMPOSITE_PTR)
        radical_match = self.RADICAL_PATTERN.match(pointer)
        if radical_match:
            radical = radical_match.group(1)
            number = radical_match.group(2)
            unary_op = radical_match.group(3)  # *, !, ?
            binary_op = radical_match.group(4)  # &, |
            second_radical = radical_match.group(5)
            context = radical_match.group(6)

            # Binary operation (C&I, C|I)
            if binary_op and second_radical:
                return PointerType.COMPOSITE_PTR, {
                    "type": "binary",
                    "radical": radical,
                    "operator": binary_op,
                    "operand": second_radical,
                    "context": context
                }

            # Unary operation (C*, C!, C?)
            if unary_op:
                return PointerType.COMPOSITE_PTR, {
                    "type": "unary",
                    "radical": radical,
                    "operator": unary_op,
                    "context": context
                }

            # Numbered rule (C3, I5)
            if number:
                return PointerType.COMPOSITE_PTR, {
                    "type": "numbered",
                    "radical": radical,
                    "number": int(number),
                    "context": context
                }

            # Context only (C:build) - rare but valid
            if context:
                return PointerType.COMPOSITE_PTR, {
                    "type": "context",
                    "radical": radical,
                    "context": context
                }

            # Simple radical
            return PointerType.SYMBOL_PTR, {"radical": radical, "type": "ascii"}

        # Unknown syntax
        return PointerType.SYMBOL_PTR, FailClosed(
            ErrorCode.E_SYNTAX,
            f"Pointer '{pointer}' does not match any known pattern",
            pointer, now
        )

    def _resolve_symbol_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve simple symbol pointer (ASCII radical or CJK glyph)."""

        # Handle CJK glyph
        if parsed.get("type") == "cjk":
            glyph = parsed["glyph"]
            entry = self.CJK_GLYPHS.get(glyph)

            if not entry:
                return FailClosed(
                    ErrorCode.E_UNKNOWN_SYMBOL,
                    f"CJK glyph '{glyph}' not found in registry",
                    pointer, now
                )

            # Check for polysemic symbol (requires context)
            if entry.get("polysemic"):
                context_type = context_keys.get("CONTEXT_TYPE")
                if not context_type:
                    return FailClosed(
                        ErrorCode.E_CONTEXT_REQUIRED,
                        f"Polysemic symbol '{glyph}' requires CONTEXT_TYPE key",
                        pointer, now
                    )
                # Resolve based on context (per SPC_SPEC Section 5.2)
                path = self._resolve_polysemic(glyph, context_type)
                if not path:
                    return FailClosed(
                        ErrorCode.E_AMBIGUOUS,
                        f"Unknown context '{context_type}' for symbol '{glyph}'",
                        pointer, now
                    )
                entry = {**entry, "path": path}

            ir = self._build_ir(pointer, "domain", {
                "glyph": glyph,
                "domain": entry["domain"],
                "path": entry["path"],
                "tokens": entry.get("tokens", 1)
            })

            receipt = self._build_receipt(pointer, ir, entry.get("path", ""))
            return DecodeSuccess(ir, receipt)

        # Handle ASCII radical
        radical = parsed.get("radical")
        if not radical:
            return FailClosed(
                ErrorCode.E_SYNTAX,
                f"No radical found in parsed pointer",
                pointer, now
            )

        radicals = self.codebook.get("radicals", {})

        if radical not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Radical '{radical}' not found in codebook",
                pointer, now
            )

        entry = radicals[radical]
        ir = self._build_ir(pointer, "domain", {
            "radical": radical,
            "domain": entry["domain"],
            "path": entry["path"],
            "tokens": entry.get("tokens", 1)
        })

        receipt = self._build_receipt(pointer, ir, entry.get("path", ""))
        return DecodeSuccess(ir, receipt)

    def _resolve_polysemic(self, glyph: str, context_type: str) -> Optional[str]:
        """Resolve polysemic symbol based on context.

        Per SPC_SPEC Section 5.2 - 道 (dào) example.
        """
        if glyph == "道":
            polysemic_map = {
                "CONTEXT_PATH": "LAW/CANON",
                "CONTEXT_PRINCIPLE": "LAW/CANON/FOUNDATION",
                "CONTEXT_METHOD": "CAPABILITY/SKILLS",
            }
            return polysemic_map.get(context_type)
        return None

    def _resolve_composite_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve composite pointer.

        Handles:
        - numbered: C3, I5 (rules/invariants)
        - unary: C*, C!, C? (all/not/check)
        - binary: C&I, C|I (and/or)
        - path: L.C.3, 法.驗 (hierarchical access)
        - context: C:build (contextual)
        """
        composite_type = parsed.get("type")

        # Path resolution (L.C.3, 法.驗)
        if composite_type == "path":
            return self._resolve_path_ptr(pointer, parsed, context_keys, now)

        # Binary operation (C&I, C|I)
        if composite_type == "binary":
            return self._resolve_binary_ptr(pointer, parsed, context_keys, now)

        # Unary operation (C*, C!, C?)
        if composite_type == "unary":
            return self._resolve_unary_ptr(pointer, parsed, context_keys, now)

        # Numbered rule (C3, I5)
        if composite_type == "numbered":
            return self._resolve_numbered_ptr(pointer, parsed, context_keys, now)

        # Context only (C:build)
        if composite_type == "context":
            return self._resolve_context_ptr(pointer, parsed, context_keys, now)

        # Fallback for old parsing format
        radical = parsed.get("radical")
        number = parsed.get("number")
        context = parsed.get("context")

        if number is not None:
            return self._resolve_numbered_ptr(pointer, {
                "type": "numbered",
                "radical": radical,
                "number": number,
                "context": context
            }, context_keys, now)

        # Radical with context only
        radicals = self.codebook.get("radicals", {})
        if radical not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Radical '{radical}' not found",
                pointer, now
            )
        entry = radicals[radical]
        ir = self._build_ir(pointer, "domain", {
            "radical": radical,
            "domain": entry["domain"],
            "path": entry["path"],
            "context": context
        })
        receipt = self._build_receipt(pointer, ir, entry.get("path", ""))
        return DecodeSuccess(ir, receipt)

    def _resolve_numbered_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve numbered rule pointer (C3, I5, etc.)."""
        radical = parsed["radical"]
        number = parsed["number"]
        context = parsed.get("context")

        if radical == "C":
            rules = self.codebook.get("contract_rules", {})
            rule_key = f"C{number}"
            if rule_key not in rules:
                return FailClosed(
                    ErrorCode.E_RULE_NOT_FOUND,
                    f"Contract rule '{rule_key}' not found",
                    pointer, now
                )
            rule = rules[rule_key]
            ir = self._build_ir(pointer, "contract_rule", {
                "id": rule_key,
                "summary": rule["summary"],
                "full": rule["full"],
                "context": context
            })
            receipt = self._build_receipt(pointer, ir, rule["full"])
            return DecodeSuccess(ir, receipt)

        elif radical == "I":
            invariants = self.codebook.get("invariants", {})
            inv_key = f"I{number}"
            if inv_key not in invariants:
                return FailClosed(
                    ErrorCode.E_RULE_NOT_FOUND,
                    f"Invariant '{inv_key}' not found",
                    pointer, now
                )
            inv = invariants[inv_key]
            ir = self._build_ir(pointer, "invariant", {
                "id": inv_key,
                "formal_id": inv.get("id"),
                "summary": inv["summary"],
                "full": inv["full"],
                "context": context
            })
            receipt = self._build_receipt(pointer, ir, inv["full"])
            return DecodeSuccess(ir, receipt)

        else:
            return FailClosed(
                ErrorCode.E_SYNTAX,
                f"Numbered rules only valid for C (contract) or I (invariant)",
                pointer, now
            )

    def _resolve_unary_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve unary operator pointer (C*, C!, C?)."""
        radical = parsed["radical"]
        operator = parsed["operator"]
        context = parsed.get("context")

        radicals = self.codebook.get("radicals", {})
        if radical not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Radical '{radical}' not found",
                pointer, now
            )

        entry = radicals[radical]
        op_meaning = self.OPERATORS.get(operator, "UNKNOWN")

        # C* = ALL contract rules
        if operator == "*":
            if radical == "C":
                all_rules = self.codebook.get("contract_rules", {})
                ir = self._build_ir(pointer, "rule_set", {
                    "radical": radical,
                    "operator": "ALL",
                    "rules": list(all_rules.keys()),
                    "count": len(all_rules),
                    "context": context
                })
                receipt = self._build_receipt(pointer, ir, f"All {len(all_rules)} contract rules")
                return DecodeSuccess(ir, receipt)
            elif radical == "I":
                all_invs = self.codebook.get("invariants", {})
                ir = self._build_ir(pointer, "rule_set", {
                    "radical": radical,
                    "operator": "ALL",
                    "rules": list(all_invs.keys()),
                    "count": len(all_invs),
                    "context": context
                })
                receipt = self._build_receipt(pointer, ir, f"All {len(all_invs)} invariants")
                return DecodeSuccess(ir, receipt)

        # C! = NOT/DENY, C? = CHECK/QUERY
        ir = self._build_ir(pointer, "operation", {
            "radical": radical,
            "operator": op_meaning,
            "domain": entry["domain"],
            "path": entry["path"],
            "context": context
        })
        receipt = self._build_receipt(pointer, ir, f"{op_meaning} {entry['domain']}")
        return DecodeSuccess(ir, receipt)

    def _resolve_binary_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve binary operator pointer (C&I, C|I)."""
        left = parsed["radical"]
        operator = parsed["operator"]
        right = parsed["operand"]
        context = parsed.get("context")

        radicals = self.codebook.get("radicals", {})

        if left not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Left radical '{left}' not found",
                pointer, now
            )
        if right not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Right radical '{right}' not found",
                pointer, now
            )

        left_entry = radicals[left]
        right_entry = radicals[right]
        op_meaning = self.OPERATORS.get(operator, "UNKNOWN")

        ir = self._build_ir(pointer, "binary_operation", {
            "operator": op_meaning,
            "left": {
                "radical": left,
                "domain": left_entry["domain"],
                "path": left_entry["path"]
            },
            "right": {
                "radical": right,
                "domain": right_entry["domain"],
                "path": right_entry["path"]
            },
            "context": context
        })
        receipt = self._build_receipt(
            pointer, ir,
            f"{left_entry['domain']} {op_meaning} {right_entry['domain']}"
        )
        return DecodeSuccess(ir, receipt)

    def _resolve_path_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve path pointer (L.C.3, 法.驗)."""
        parts = parsed["parts"]
        base = parsed["base"]
        path_parts = parsed["path"]

        # Resolve base (could be CJK or ASCII radical)
        if base in self.CJK_GLYPHS:
            base_entry = self.CJK_GLYPHS[base]
        else:
            radicals = self.codebook.get("radicals", {})
            if base not in radicals:
                return FailClosed(
                    ErrorCode.E_UNKNOWN_SYMBOL,
                    f"Base '{base}' not found",
                    pointer, now
                )
            base_entry = radicals[base]

        # Resolve path components
        resolved_path = [{"symbol": base, "domain": base_entry["domain"], "path": base_entry.get("path")}]

        for part in path_parts:
            # Check if it's a number (L.C.3 -> rule 3)
            if part.isdigit():
                resolved_path.append({"type": "index", "value": int(part)})
            # Check CJK
            elif part in self.CJK_GLYPHS:
                entry = self.CJK_GLYPHS[part]
                resolved_path.append({"symbol": part, "domain": entry["domain"], "path": entry.get("path")})
            # Check ASCII radical
            elif part in self.codebook.get("radicals", {}):
                entry = self.codebook["radicals"][part]
                resolved_path.append({"symbol": part, "domain": entry["domain"], "path": entry.get("path")})
            else:
                return FailClosed(
                    ErrorCode.E_UNKNOWN_SYMBOL,
                    f"Path component '{part}' not found",
                    pointer, now
                )

        # Build combined path
        combined_paths = [r.get("path") for r in resolved_path if r.get("path")]

        ir = self._build_ir(pointer, "path_access", {
            "parts": resolved_path,
            "combined_paths": combined_paths,
            "depth": len(resolved_path)
        })
        receipt = self._build_receipt(pointer, ir, " -> ".join(combined_paths))
        return DecodeSuccess(ir, receipt)

    def _resolve_context_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve context pointer (C:build)."""
        radical = parsed["radical"]
        context = parsed["context"]

        radicals = self.codebook.get("radicals", {})
        if radical not in radicals:
            return FailClosed(
                ErrorCode.E_UNKNOWN_SYMBOL,
                f"Radical '{radical}' not found",
                pointer, now
            )

        # Validate context
        valid_contexts = self.codebook.get("contexts", {})
        if context not in valid_contexts:
            return FailClosed(
                ErrorCode.E_INVALID_QUALIFIER,
                f"Unknown context '{context}'. Valid: {list(valid_contexts.keys())}",
                pointer, now
            )

        entry = radicals[radical]
        ir = self._build_ir(pointer, "contextual_domain", {
            "radical": radical,
            "domain": entry["domain"],
            "path": entry["path"],
            "context": context,
            "context_description": valid_contexts[context]
        })
        receipt = self._build_receipt(pointer, ir, f"{entry['domain']} in {context} context")
        return DecodeSuccess(ir, receipt)

    def _resolve_hash_ptr(
        self, pointer: str, parsed: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve hash pointer (CAS lookup).

        Per SPC_SPEC Section 2.2:
        - Format: sha256:<hex16-64>
        - Must exist in CAS or codebook
        - Hash collision → FAIL_CLOSED
        """
        hash_value = parsed["hash"]

        # Try CAS lookup via registered callback
        if _cas_lookup_fn is not None:
            try:
                result = _cas_lookup_fn(hash_value)
                if result is not None:
                    ir = self._build_ir(pointer, "cas_content", {
                        "hash": hash_value,
                        "hash_full": f"sha256:{hash_value}",
                        "content_type": result.get("type", "text"),
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata"),
                        "source": result.get("source", "memory_cassette")
                    })
                    receipt = self._build_receipt(
                        pointer, ir,
                        result.get("text", "")[:100] if result.get("text") else ""
                    )
                    return DecodeSuccess(ir, receipt)
            except Exception as e:
                return FailClosed(
                    ErrorCode.E_CAS_UNAVAILABLE,
                    f"CAS lookup error: {str(e)}",
                    pointer, now
                )

        # CAS not available
        return FailClosed(
            ErrorCode.E_HASH_NOT_FOUND,
            f"Hash '{hash_value[:16]}...' not found in CAS. "
            f"Use register_cas_lookup() to enable CAS integration.",
            pointer, now
        )

    def _build_ir(self, pointer: str, expansion_type: str, expansion: Dict) -> Dict:
        """Build canonical IR from expansion."""
        return {
            "job_id": f"spc-decode-{hashlib.sha256(pointer.encode()).hexdigest()[:8]}",
            "phase": 5,
            "task_type": "validation",
            "intent": f"Expanded from SPC pointer: {pointer}",
            "inputs": {
                "pointer": pointer,
                "expansion": {
                    "type": expansion_type,
                    **expansion
                }
            },
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": [],
            "determinism": "deterministic"
        }

    def _build_receipt(self, pointer: str, ir: Dict, expansion_text: str) -> Dict:
        """Build token receipt for decode operation.

        Per Q33: CDR = concept_units / tokens
        """
        # Estimate tokens (simplified - actual would use tiktoken)
        tokens_in = len(pointer)  # Pointer is typically 1-2 tokens
        tokens_out = len(expansion_text.split())  # Rough word count

        # Estimate concept_units (2 for simple rules per Q33)
        concept_units = 2 if ir["inputs"]["expansion"]["type"] in ("contract_rule", "invariant") else 1

        return {
            "operation": "spc_decode",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "concept_units": concept_units,
            "CDR": round(concept_units / tokens_in, 2) if tokens_in > 0 else 0,
            "compression_ratio": round(tokens_out / tokens_in, 2) if tokens_in > 0 else 0,
            "tokenizer": {
                "library": "tiktoken",
                "encoding": "o200k_base",
                "version": "0.5.1"
            }
        }


# Global decoder instance
_decoder: Optional[SPCDecoder] = None


def pointer_resolve(
    pointer: str,
    context_keys: Dict = None,
    codebook_sha256: str = None
) -> Dict:
    """Resolve SPC pointer to canonical IR.

    Convenience function wrapping SPCDecoder.

    Args:
        pointer: SPC pointer to resolve
        context_keys: Context for disambiguation
        codebook_sha256: Expected codebook hash (fail-closed on mismatch)

    Returns:
        On success: {"status": "SUCCESS", "ir": {...}, "token_receipt": {...}}
        On failure: {"status": "FAIL_CLOSED", "error_code": "...", ...}
    """
    global _decoder
    if _decoder is None:
        _decoder = SPCDecoder()

    result = _decoder.decode(pointer, context_keys, codebook_sha256=codebook_sha256)

    if isinstance(result, DecodeSuccess):
        return {
            "status": "SUCCESS",
            "ir": result.ir,
            "token_receipt": result.token_receipt
        }
    else:
        return {
            "status": "FAIL_CLOSED",
            "error_code": result.error_code.value,
            "error_detail": result.error_detail,
            "pointer": result.pointer,
            "timestamp_utc": result.timestamp
        }


# ============================================================================
# CAS Integration
# ============================================================================

def register_cas_lookup(lookup_fn: Callable[[str], Optional[Dict]]) -> None:
    """Register a CAS lookup function for HASH_PTR resolution.

    The lookup function should:
    - Accept a hash string (without 'sha256:' prefix)
    - Return a dict with keys: text, metadata, type, source
    - Return None if hash not found

    Example:
        def my_cas_lookup(hash_value: str) -> Optional[Dict]:
            memory = memory_cassette.memory_recall(hash_value)
            if memory:
                return {
                    "text": memory["text"],
                    "metadata": memory.get("metadata"),
                    "type": "memory",
                    "source": "memory_cassette"
                }
            return None

        register_cas_lookup(my_cas_lookup)
    """
    global _cas_lookup_fn
    _cas_lookup_fn = lookup_fn


def unregister_cas_lookup() -> None:
    """Unregister the CAS lookup function."""
    global _cas_lookup_fn
    _cas_lookup_fn = None


def is_cas_available() -> bool:
    """Check if CAS lookup is available."""
    return _cas_lookup_fn is not None


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    # Core classes
    "SPCDecoder",
    "PointerType",
    "ErrorCode",
    "DecodeSuccess",
    "FailClosed",
    # Main function
    "pointer_resolve",
    # CAS integration
    "register_cas_lookup",
    "unregister_cas_lookup",
    "is_cas_available",
]
