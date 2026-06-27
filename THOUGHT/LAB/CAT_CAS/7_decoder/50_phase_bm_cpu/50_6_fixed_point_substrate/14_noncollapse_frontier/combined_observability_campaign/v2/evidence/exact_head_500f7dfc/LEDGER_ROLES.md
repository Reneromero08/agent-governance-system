# Evidence Ledger Roles

`COMMANDS.jsonl` is the canonical complete command ledger for this evidence package. It contains the original snapshot/generation/Windows/target command rows plus the corrective verifier-recovery and snapshot-bridge reproduction rows.

`COMMAND_LEDGER.jsonl`, `target/target_COMMAND_LEDGER.jsonl`, and `cc/custody_COMMANDS.jsonl` are retained only as component ledgers. They are not complete by themselves; their rows are represented in `COMMANDS.jsonl`.

The canonical command row count is 55. Failed local reproduction attempts are retained with their nonzero exit codes because they were executed and explain why the successful reproduction uses Python `tarfile` extraction to preserve UTF-8 archive names.
