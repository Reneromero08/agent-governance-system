# Experiment 2: Strict Slack-Space File Storage

## The OS Entropy Leak
When running files on a normal filesystem, creating lockfiles or temporary cache files forces the operating system to dynamically allocate blocks and modify directory tables. Even if the files are deleted, this represents a **hidden clean-space leak** inside the OS kernel.

## The Solution: Slack-Space Mapping
We pre-allocated `config.json` and `input.txt` to exactly 4,096 bytes (standard sector boundary) using random padding. The data-processing application was modified to store its lock status and intermediate data chunks in the padding bytes (slack space) of the existing files.

## Results
*   **Directory Hash Before Run:** `aa2fd202d2bbf75a1993a1bea1f218cd8c042347968c8ed2d1319cad98ecb428`
*   **Directory Hash After Run:** `aa2fd202d2bbf75a1993a1bea1f218cd8c042347968c8ed2d1319cad98ecb428` (100% Match)

### Verification Metrics:
*   No files were created or deleted in the filesystem directory.
*   All file sizes on disk remained exactly constant at **4,096 bytes** during execution.
*   The application output was successfully written to an external file.
*   All temporary changes to config and input files were dynamically cleaned up post-execution.
