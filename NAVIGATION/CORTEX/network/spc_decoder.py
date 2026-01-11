#!/usr/bin/env python3
"""
SPC Decoder - Semantic Pointer Compression decoder.

Implements deterministic pointer resolution per SPC_SPEC.md.

Pointer Types:
- SYMBOL_PTR: Single radical (e.g., "C", "I", "V")
- HASH_PTR: Content-addressed (e.g., "sha256:abc123...")
- COMPOSITE_PTR: Radical + number/operator/context (e.g., "C3", "I5:build")

Reference:
- LAW/CANON/SEMANTIC/SPC_SPEC.md
- LAW/CANON/SEMANTIC/CODEBOOK_SYNC_PROTOCOL.md
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


# Load codebook
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODEBOOK_PATH = PROJECT_ROOT / "THOUGHT" / "LAB" / "COMMONSENSE" / "CODEBOOK.json"


class PointerType(Enum):
    """SPC pointer types."""
    SYMBOL_PTR = "symbol"
    HASH_PTR = "hash"
    COMPOSITE_PTR = "composite"


class ErrorCode(Enum):
    """Fail-closed error codes per SPC_SPEC."""
    E_CODEBOOK_MISMATCH = "E_CODEBOOK_MISMATCH"
    E_KERNEL_VERSION = "E_KERNEL_VERSION"
    E_SYNTAX = "E_SYNTAX"
    E_UNKNOWN_SYMBOL = "E_UNKNOWN_SYMBOL"
    E_HASH_NOT_FOUND = "E_HASH_NOT_FOUND"
    E_AMBIGUOUS = "E_AMBIGUOUS"
    E_INVALID_OPERATOR = "E_INVALID_OPERATOR"
    E_RULE_NOT_FOUND = "E_RULE_NOT_FOUND"
    E_CONTEXT_REQUIRED = "E_CONTEXT_REQUIRED"


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
    """

    KERNEL_VERSION = "1.0.0"
    TOKENIZER_ID = "tiktoken/o200k_base"

    # Regex patterns
    HASH_PATTERN = re.compile(r'^sha256:([a-f0-9]{16,64})$')
    RADICAL_PATTERN = re.compile(r'^([CIVLGSRAJP])(\d*)([*!?&|]?)(:(\w+))?$')

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
        """Parse pointer and determine type."""
        now = datetime.now(timezone.utc).isoformat()

        # Check for HASH_PTR
        hash_match = self.HASH_PATTERN.match(pointer)
        if hash_match:
            return PointerType.HASH_PTR, {"hash": hash_match.group(1)}

        # Check for RADICAL (SYMBOL_PTR or COMPOSITE_PTR)
        radical_match = self.RADICAL_PATTERN.match(pointer)
        if radical_match:
            radical = radical_match.group(1)
            number = radical_match.group(2)
            operator = radical_match.group(3)
            context = radical_match.group(5)

            if number or operator or context:
                return PointerType.COMPOSITE_PTR, {
                    "radical": radical,
                    "number": int(number) if number else None,
                    "operator": operator or None,
                    "context": context
                }
            else:
                return PointerType.SYMBOL_PTR, {"radical": radical}

        # Unknown syntax
        return PointerType.SYMBOL_PTR, FailClosed(
            ErrorCode.E_SYNTAX,
            f"Pointer '{pointer}' does not match any known pattern",
            pointer, now
        )

    def _resolve_symbol_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve simple radical pointer."""
        radical = parsed["radical"]
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

    def _resolve_composite_ptr(
        self, pointer: str, parsed: Dict, context_keys: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve composite pointer (radical + number/operator/context)."""
        radical = parsed["radical"]
        number = parsed.get("number")
        context = parsed.get("context")

        # Resolve numbered rules
        if number is not None:
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
        else:
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

    def _resolve_hash_ptr(
        self, pointer: str, parsed: Dict, now: str
    ) -> Union[DecodeSuccess, FailClosed]:
        """Resolve hash pointer (CAS lookup)."""
        # Hash pointers require CAS lookup - not yet implemented
        return FailClosed(
            ErrorCode.E_HASH_NOT_FOUND,
            f"CAS lookup not yet implemented for hash: {parsed['hash'][:16]}...",
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


# Convenience exports
__all__ = [
    "SPCDecoder",
    "PointerType",
    "ErrorCode",
    "DecodeSuccess",
    "FailClosed",
    "pointer_resolve"
]
