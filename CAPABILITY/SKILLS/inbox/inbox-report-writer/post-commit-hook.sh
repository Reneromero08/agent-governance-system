#!/bin/sh
# Post-commit hook to regenerate INBOX ledger
# This ensures the ledger is always up-to-date after commits

echo "📊 Regenerating INBOX ledger..."
python3 CAPABILITY/SKILLS/inbox/inbox-report-writer/generate_inbox_ledger.py --quiet 2>/dev/null || true

exit 0
