---
name: coderabbit-comments
description: "Retrieve CodeRabbit review comments from the VS Code extension's local storage."
---
<!-- CONTENT_HASH: 6811f0b030c763a32c986394e8e77335d093b35d3fd666e02717f44c33d14e72 -->

**required_canon_version:** >=3.0.0


# Skill: coderabbit-comments

**Version:** 0.1.0

**Status:** Active



## Purpose

Read CodeRabbit review comments stored locally by the VS Code extension at
`%APPDATA%\Code\User\workspaceStorage\<workspace-hash>\coderabbit.coderabbit-vscode\`.
Allows agents to retrieve actionable review feedback without pushing commits or
installing the CodeRabbit CLI.

## Trigger

Use when a user mentions CodeRabbit comments they received through the VS Code
extension and wants an agent to review or act on the feedback.

## Inputs

- `action`: `"latest"` (default) or `"list"` or `"all"`
- `storage_path` (optional): explicit path to the review database JSON file.
  If omitted, the skill auto-discovers the correct path by scanning workspace
  storage directories.

## Outputs

Returns JSON with:
- `ok`: boolean
- `action`: the action performed
- `review_count`: number of reviews found
- `latest_review`: for `action=latest`, the most recent completed review with:
  - `id`: review UUID
  - `startedAt`: ISO timestamp
  - `comments`: array of actionable comments (severity != "none")
    - `filename`, `startLine`, `endLine`, `severity`, `comment`, `suggestions`, `resolution`
  - `file_count`: number of files reviewed
  - `files`: list of filenames reviewed
- `error`: error message if something went wrong

## Constraints

- Only works on Windows (reads from `%APPDATA%`)
- Requires CodeRabbit VS Code extension to have been used
- Cannot retrieve comments for reviews that have been purged from local storage
- Must not modify CANON or CONTEXT

## Fixtures

- `fixtures/basic/`: Tests with a mock storage file for deterministic validation
