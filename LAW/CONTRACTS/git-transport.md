<!-- CONTENT_HASH: GENERATED -->

# Git Transport Policy (WSL + VS Code)

## Invariants
- HTTPS is the canonical transport for `origin` (`https://github.com/OWNER/REPO.git`).
- Do not switch remotes to SSH.
- Do not store plaintext tokens anywhere in the repo.
- Authentication should be non-interactive after first setup via a credential helper.

## Repo Guard
Use the repo guard to prevent SSH remote “flips”:

- `bash CAPABILITY/TOOLS/utilities/ensure_https_remote.sh .`

This script:
- Asserts `origin` exists
- Rewrites SSH `git@github.com:` / `ssh://git@github.com/...` to HTTPS
- Exits non-zero if it cannot derive an HTTPS URL

## WSL Authentication (Preferred)
Use Windows Git Credential Manager (GCM) from WSL when available:

- `git config --global credential.helper "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"`
- `git config --global credential.https://github.com.useHttpPath true`

## Doctor
Run the doctor to see current state and suggested fixes:

- `bash CAPABILITY/TOOLS/utilities/git_doctor.sh --repo .`
- Apply fixes: `bash CAPABILITY/TOOLS/utilities/git_doctor.sh --repo . --apply`

