# Source Reproduction Report V2

Status: frozen source reproduction only. Not canonical. No physical transfer.

Frozen source SHA: `6f39766e9cf622e2e41d178f8131bd8777b6cd1d`
Source worktree: `/tmp/ags-audio-source-6f39766-v2`
Source worktree status before runs: `clean`
Source worktree status after runs: `clean`
Venv link: `/run/media/reneshizzle/860_1/CCC 2.0/AI/agent-governance-system/.venv-linux`

## Classifications

- Candidate E: `SOURCE_NOT_REPRODUCED`
- Candidate F: `SOURCE_REPRODUCED`
- Candidate G: `SOURCE_REPRODUCED`
- Candidate H: `SOURCE_REPRODUCED`

## Normalization policy

The raw output directories and logs are preserved verbatim. For classification, `SHA256SUMS` files are not treated as scientific differences because the source qualifiers include absolute output-directory names in those manifests. Generated JSON files are compared after removing run-time scalar fields ending in `_elapsed_ms` or `_elapsed_ns`.

## Runs

- E run 1: rc=126, files=0, stdout=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, stderr=9157212899d8f3ec08ebe9934396f1f30c21e1ed961e6b65565a626bf0b24a23
- E run 2: rc=126, files=0, stdout=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855, stderr=9157212899d8f3ec08ebe9934396f1f30c21e1ed961e6b65565a626bf0b24a23
- F run 1: rc=0, files=4, stdout=1d9799986dfa68ce1b55cdeb800613ae5bb2ace353a32a373c2fc8031e7ed2f7, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- F run 2: rc=0, files=4, stdout=1d9799986dfa68ce1b55cdeb800613ae5bb2ace353a32a373c2fc8031e7ed2f7, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- G run 1: rc=0, files=75, stdout=de5f8c42f2347c64e15b018c16b79b8de182c301750c49b797460285db694d52, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- G run 2: rc=0, files=75, stdout=de5f8c42f2347c64e15b018c16b79b8de182c301750c49b797460285db694d52, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- H run 1: rc=0, files=4, stdout=08800f4676c6df975a765dcdcd01b9e41ab1e5d5e6526f3c653a611af6caf8ab, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- H run 2: rc=0, files=4, stdout=08800f4676c6df975a765dcdcd01b9e41ab1e5d5e6526f3c653a611af6caf8ab, stderr=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

## Controls

- E missing-argument control: rc=2
- F missing-argument control: rc=2
- G missing-argument control: rc=2
- H missing-argument control: rc=2

Payload hash: `8fb69931793b54848116a6f3288819f0d46f12048c08f61dc1f58a627d37aacf`
