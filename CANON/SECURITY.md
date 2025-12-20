# Security Policy

This document outlines the security considerations for the Agent Governance System.  Because AGS may execute code and manage sensitive state, it is important to minimise risk.

## Principles

- **Least privilege** - Skills and tools should run with the minimum permissions required to perform their task.
- **Deterministic outputs** - Randomness and external network calls should be avoided or isolated to prevent non-reproducible behaviour.
- **No external side effects** - Skills must not perform irreversible actions (e.g. network requests, database writes) without explicit authorisation.
- **Auditability** - All changes should be traceable through fixtures, context records and the changelog.

## Reporting vulnerabilities

If you discover a vulnerability in this repository or its processes, please open an issue in the "Security" category or contact the maintainers privately. Do not disclose vulnerabilities publicly until they have been addressed.
