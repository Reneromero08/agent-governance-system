#!/usr/bin/env python3
"""
Receipt Verifier (Phase 6.2)

Verification tooling for cassette receipt chains.

Usage:
    from receipt_verifier import CassetteReceiptVerifier

    verifier = CassetteReceiptVerifier(cassette)
    result = verifier.verify_session_integrity(session_id)
    if result["valid"]:
        print(f"Session Merkle root: {result['merkle_root']}")
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import from cassette_receipt module
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"))

from cassette_receipt import (
    CassetteReceipt,
    receipt_from_dict,
    verify_receipt,
    verify_receipt_chain,
    compute_session_merkle_root,
)


class CassetteReceiptVerifier:
    """Verifier for cassette receipt chains.

    Provides methods to verify receipt integrity, chain linkage,
    and session Merkle roots.
    """

    def __init__(self, db_path: Path):
        """Initialize verifier.

        Args:
            db_path: Path to cassette SQLite database
        """
        self.db_path = db_path

    def get_receipts_by_session(self, session_id: str) -> List[CassetteReceipt]:
        """Get all receipts for a session in order.

        Args:
            session_id: Session identifier

        Returns:
            List of CassetteReceipt objects in receipt_index order
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute("""
                SELECT receipt_json FROM cassette_receipts
                WHERE session_id = ?
                ORDER BY receipt_index ASC
            """, (session_id,))

            receipts = []
            for row in cursor:
                data = json.loads(row['receipt_json'])
                receipts.append(receipt_from_dict(data))

            return receipts
        finally:
            conn.close()

    def get_all_receipts(self, limit: int = 1000) -> List[CassetteReceipt]:
        """Get all receipts in chain order.

        Args:
            limit: Maximum number of receipts to return

        Returns:
            List of CassetteReceipt objects in receipt_index order
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute("""
                SELECT receipt_json FROM cassette_receipts
                ORDER BY receipt_index ASC
                LIMIT ?
            """, (limit,))

            receipts = []
            for row in cursor:
                data = json.loads(row['receipt_json'])
                receipts.append(receipt_from_dict(data))

            return receipts
        finally:
            conn.close()

    def verify_session_integrity(self, session_id: str) -> Dict[str, Any]:
        """Verify all receipts for a session.

        Validates:
        - Receipt hash integrity
        - Chain linkage (parent_receipt_hash)
        - Contiguous receipt indices
        - Computed Merkle root matches stored

        Args:
            session_id: Session identifier

        Returns:
            Dict with {valid, errors, merkle_root, chain_length, stored_merkle_root}
        """
        receipts = self.get_receipts_by_session(session_id)

        if not receipts:
            return {
                "valid": True,
                "errors": [],
                "merkle_root": None,
                "chain_length": 0,
                "stored_merkle_root": None,
            }

        # Verify chain
        result = verify_receipt_chain(receipts, verify_hashes=True)

        # Get stored Merkle root from session
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                "SELECT merkle_root FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            stored_merkle_root = row['merkle_root'] if row else None
        finally:
            conn.close()

        # Compare computed vs stored Merkle root
        if stored_merkle_root and result["merkle_root"]:
            if stored_merkle_root != result["merkle_root"]:
                result["errors"].append(
                    f"Merkle root mismatch: stored={stored_merkle_root}, "
                    f"computed={result['merkle_root']}"
                )
                result["valid"] = False

        result["stored_merkle_root"] = stored_merkle_root
        return result

    def verify_record_exists(self, record_id: str) -> Dict[str, Any]:
        """Verify a record exists and has valid receipt.

        Args:
            record_id: Content hash of the record

        Returns:
            Dict with {exists, has_receipt, receipt_valid, receipt}
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Check if record exists in memories
            cursor = conn.execute(
                "SELECT hash FROM memories WHERE hash = ?",
                (record_id,)
            )
            record_exists = cursor.fetchone() is not None

            # Check if receipt exists
            cursor = conn.execute(
                "SELECT receipt_json FROM cassette_receipts WHERE record_id = ?",
                (record_id,)
            )
            receipt_row = cursor.fetchone()

            if not receipt_row:
                return {
                    "exists": record_exists,
                    "has_receipt": False,
                    "receipt_valid": None,
                    "receipt": None,
                }

            # Parse and verify receipt
            data = json.loads(receipt_row['receipt_json'])
            receipt = receipt_from_dict(data)
            receipt_valid = verify_receipt(receipt)

            return {
                "exists": record_exists,
                "has_receipt": True,
                "receipt_valid": receipt_valid,
                "receipt": receipt.to_dict(),
            }
        finally:
            conn.close()

    def get_chain_stats(self) -> Dict[str, Any]:
        """Get statistics about the receipt chain.

        Returns:
            Dict with {total_receipts, sessions_with_receipts, operations}
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Count total receipts
            cursor = conn.execute("SELECT COUNT(*) as count FROM cassette_receipts")
            total_receipts = cursor.fetchone()['count']

            # Count sessions with receipts
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT session_id) as count FROM cassette_receipts WHERE session_id IS NOT NULL"
            )
            sessions_with_receipts = cursor.fetchone()['count']

            # Count by operation type
            cursor = conn.execute("""
                SELECT operation, COUNT(*) as count FROM cassette_receipts
                GROUP BY operation
            """)
            operations = {row['operation']: row['count'] for row in cursor}

            return {
                "total_receipts": total_receipts,
                "sessions_with_receipts": sessions_with_receipts,
                "operations": operations,
            }
        finally:
            conn.close()

    def verify_full_chain(self) -> Dict[str, Any]:
        """Verify the entire receipt chain.

        Warning: This can be slow for large chains.

        Returns:
            Dict with {valid, errors, merkle_root, chain_length}
        """
        receipts = self.get_all_receipts(limit=100000)
        return verify_receipt_chain(receipts, verify_hashes=True)


def verify_cassette_receipts(db_path: Path, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to verify cassette receipts.

    Args:
        db_path: Path to cassette database
        session_id: Optional session to verify (None = verify all)

    Returns:
        Verification result
    """
    verifier = CassetteReceiptVerifier(db_path)

    if session_id:
        return verifier.verify_session_integrity(session_id)
    else:
        return verifier.verify_full_chain()


# ==============================================================================
# CLI
# ==============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify cassette receipt chains")
    parser.add_argument("db_path", type=Path, help="Path to cassette database")
    parser.add_argument("--session", type=str, help="Verify specific session")
    parser.add_argument("--stats", action="store_true", help="Show chain statistics")

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}")
        exit(1)

    verifier = CassetteReceiptVerifier(args.db_path)

    if args.stats:
        stats = verifier.get_chain_stats()
        print("Receipt Chain Statistics")
        print("=" * 40)
        print(f"Total receipts: {stats['total_receipts']}")
        print(f"Sessions with receipts: {stats['sessions_with_receipts']}")
        print("Operations:")
        for op, count in stats['operations'].items():
            print(f"  {op}: {count}")
    elif args.session:
        result = verifier.verify_session_integrity(args.session)
        print(f"Session: {args.session}")
        print(f"Valid: {result['valid']}")
        print(f"Chain length: {result['chain_length']}")
        if result['merkle_root']:
            print(f"Merkle root: {result['merkle_root'][:16]}...")
        if result['errors']:
            print("Errors:")
            for err in result['errors']:
                print(f"  - {err}")
    else:
        result = verifier.verify_full_chain()
        print("Full Chain Verification")
        print("=" * 40)
        print(f"Valid: {result['valid']}")
        print(f"Chain length: {result['chain_length']}")
        if result['merkle_root']:
            print(f"Merkle root: {result['merkle_root'][:16]}...")
        if result['errors']:
            print("Errors:")
            for err in result['errors']:
                print(f"  - {err}")
