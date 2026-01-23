#!/usr/bin/env python3
"""
Vector Migration Tool (Phase J.0.4)

Backfills embeddings for existing turns that don't have them.
Can process a single session or all sessions.

Usage:
    python -m catalytic_chat.migrate_vectors --session-id <id>
    python -m catalytic_chat.migrate_vectors --all
    python -m catalytic_chat.migrate_vectors --list  # List sessions needing migration
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from .embedding_engine import get_embedding_engine
from .vector_persistence import VectorPersistence
from .paths import get_cat_chat_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sessions_needing_migration(db_path: Path) -> List[str]:
    """Find all session IDs with turns lacking embeddings."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find sessions with turn_stored events that lack embeddings
    cursor = conn.execute("""
        SELECT DISTINCT se.session_id
        FROM session_events se
        LEFT JOIN session_event_embeddings see ON se.event_id = see.event_id
        WHERE se.event_type = 'turn_stored'
          AND see.event_id IS NULL
        ORDER BY se.session_id
    """)

    sessions = [row['session_id'] for row in cursor]
    conn.close()
    return sessions


def migrate_session(
    session_id: str,
    persistence: VectorPersistence,
    embedding_engine,
    batch_size: int = 50
) -> Dict[str, Any]:
    """Migrate embeddings for a single session.

    Returns stats dict with processed, failed, errors.
    """
    stats = {
        'session_id': session_id,
        'total': 0,
        'processed': 0,
        'failed': 0,
        'errors': []
    }

    # Get turns needing backfill
    turns = persistence.get_vectors_needing_backfill(session_id)
    stats['total'] = len(turns)

    if not turns:
        logger.info("Session %s: No turns need migration", session_id)
        return stats

    logger.info("Session %s: Migrating %d turns", session_id, len(turns))

    batch = []
    for turn in turns:
        try:
            # Extract content from payload
            payload = json.loads(turn['payload_json'])
            user_query = payload.get('user_query', '')
            assistant_response = payload.get('assistant_response', '')
            full_content = f"User: {user_query}\n\nAssistant: {assistant_response}"

            # Compute embedding
            embedding = embedding_engine.embed(full_content)

            # Queue for batch insert
            batch.append((
                turn['event_id'],
                session_id,
                turn['content_hash'],
                embedding
            ))

            # Flush batch when full
            if len(batch) >= batch_size:
                persistence.store_embeddings_batch(batch)
                stats['processed'] += len(batch)
                logger.info("  Processed %d/%d turns", stats['processed'], stats['total'])
                batch = []

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'event_id': turn['event_id'],
                'error': str(e)
            })
            logger.warning("  Failed to process turn %s: %s", turn['event_id'], e)

    # Flush remaining batch
    if batch:
        persistence.store_embeddings_batch(batch)
        stats['processed'] += len(batch)

    logger.info(
        "Session %s: Completed - %d processed, %d failed",
        session_id, stats['processed'], stats['failed']
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Backfill embeddings for existing sessions (Phase J.0.4)'
    )
    parser.add_argument(
        '--session-id',
        help='Specific session ID to migrate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate all sessions needing embeddings'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List sessions needing migration (no changes)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for embedding computation (default: 50)'
    )
    parser.add_argument(
        '--db-path',
        type=Path,
        help='Override database path'
    )

    args = parser.parse_args()

    # Validate args
    if not (args.session_id or args.all or args.list):
        parser.error('Must specify --session-id, --all, or --list')

    # Get database path
    db_path = args.db_path or get_cat_chat_db()

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    # List mode
    if args.list:
        sessions = get_sessions_needing_migration(db_path)
        if not sessions:
            print("No sessions need migration.")
        else:
            print(f"Sessions needing migration ({len(sessions)}):")
            for s in sessions:
                print(f"  - {s}")
        sys.exit(0)

    # Migration mode
    logger.info("Initializing embedding engine...")
    embedding_engine = get_embedding_engine()

    logger.info("Connecting to database: %s", db_path)
    persistence = VectorPersistence(db_path)
    persistence.ensure_schema()

    sessions_to_migrate = []
    if args.session_id:
        sessions_to_migrate = [args.session_id]
    elif args.all:
        sessions_to_migrate = get_sessions_needing_migration(db_path)

    if not sessions_to_migrate:
        logger.info("No sessions to migrate")
        sys.exit(0)

    logger.info("Migrating %d session(s)...", len(sessions_to_migrate))

    all_stats = []
    for session_id in sessions_to_migrate:
        stats = migrate_session(
            session_id=session_id,
            persistence=persistence,
            embedding_engine=embedding_engine,
            batch_size=args.batch_size
        )
        all_stats.append(stats)

    # Summary
    total_processed = sum(s['processed'] for s in all_stats)
    total_failed = sum(s['failed'] for s in all_stats)

    print()
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Sessions processed: {len(all_stats)}")
    print(f"Turns migrated: {total_processed}")
    print(f"Turns failed: {total_failed}")

    if total_failed > 0:
        print()
        print("Errors:")
        for stats in all_stats:
            for err in stats['errors']:
                print(f"  {err['event_id']}: {err['error']}")

    persistence.close()


if __name__ == '__main__':
    main()
